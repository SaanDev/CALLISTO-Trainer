"""Train the burst/no-burst classifier."""

# NOTE: Vendored from H:\Burst Identifier (src/training/train.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from callisto_trainer.core.dataset import get_dataloaders
from callisto_trainer.core.metrics import compute_binary_metrics, find_best_threshold
from callisto_trainer.core.models.model_factory import create_model
from callisto_trainer.core.metadata_features import NUM_NUMERIC
from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.progress import emit_progress
from callisto_trainer.core.seed import set_seed


LOGGER = get_logger(__name__)


def _split_batch(batch):
    """Return (model_input_list, labels) for (image, label) or (image, label, meta) batches."""
    if len(batch) == 3:
        image, label, meta = batch
        return [image, meta], label
    image, label = batch
    return [image], label


def _move_inputs(inputs, device, channels_last):
    moved = []
    for position, tensor in enumerate(inputs):
        tensor = tensor.to(device, non_blocking=True)
        if position == 0 and channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        moved.append(tensor)
    return moved


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("performance", {}).get("device", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported performance.device: {requested}")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("performance.device is cuda, but PyTorch cannot see a CUDA GPU")
    return torch.device("cpu")


def _amp_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {dtype_name}")


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def _run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Any | None = None,
    channels_last: bool = False,
    include_predictions: bool = False,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    # Accumulate the loss on-device (one scalar tensor) and only sync to the host
    # once per epoch. The previous code called ``loss.cpu().item()`` every batch,
    # forcing a GPU->CPU stall on every single step.
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    sample_total = 0
    # Collect per-batch prediction arrays and concatenate once at the end, instead
    # of ``list.extend(tensor.tolist())`` which built millions of Python floats.
    prob_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for batch in dataloader:
        inputs, labels = _split_batch(batch)
        inputs = _move_inputs(inputs, device, channels_last)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with _autocast(device, use_amp, amp_dtype):
                logits = model(*inputs)
                logits = logits.reshape(labels.shape)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.shape[0]
        loss_total += loss.detach() * batch_size
        sample_total += batch_size
        prob_chunks.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        label_chunks.append(labels.detach().cpu().numpy())

    y_prob = np.concatenate(prob_chunks) if prob_chunks else np.asarray([], dtype=np.float32)
    y_true = np.concatenate(label_chunks) if label_chunks else np.asarray([], dtype=np.float32)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(loss_total.item() / sample_total) if sample_total else math.nan
    if include_predictions:
        metrics["_y_true"] = y_true.tolist()
        metrics["_y_prob"] = y_prob.tolist()
    return metrics


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Any | None = None,
    channels_last: bool = False,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Train for one epoch and return metrics."""
    return _run_epoch(
        model,
        dataloader,
        criterion,
        device,
        optimizer,
        threshold,
        use_amp,
        amp_dtype,
        scaler,
        channels_last,
        include_predictions,
    )


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate one epoch without gradient updates."""
    return _run_epoch(
        model,
        dataloader,
        criterion,
        device,
        None,
        threshold,
        use_amp,
        amp_dtype,
        None,
        channels_last,
        include_predictions,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    keep_epoch_copy: bool = True,
) -> Path:
    """Save model and optimizer state with Windows-safe alias handling.

    PyTorch checkpoint files can occasionally be locked by Windows security
    scanners, sync tools, or another Python process. To avoid crashing training
    when updating ``best.pt`` or ``last.pt``, first write a unique epoch file and
    then try to replace the alias with retries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    serializable_metrics = _strip_prediction_arrays(metrics)
    payload = {
        "epoch": epoch,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "metrics": serializable_metrics,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_path = path.with_name(f"{path.stem}_epoch_{epoch:03d}_{timestamp}_{time.time_ns()}.pt")
    temp_alias = (
        path.with_name(f".{path.stem}_epoch_{epoch:03d}_{time.time_ns()}.tmp")
        if keep_epoch_copy
        else unique_path
    )
    if keep_epoch_copy:
        torch.save(payload, unique_path)
    torch.save(payload, temp_alias)
    for attempt in range(6):
        try:
            temp_alias.replace(path)
            return unique_path if keep_epoch_copy else path
        except OSError as exc:
            if attempt == 5:
                LOGGER.warning(
                    "Could not update checkpoint alias %s because Windows kept it locked: %s. "
                    "Using unique checkpoint %s instead.",
                    path,
                    exc,
                    unique_path,
                )
                if keep_epoch_copy and temp_alias.exists():
                    temp_alias.unlink(missing_ok=True)
                return unique_path
            time.sleep(0.5 * (attempt + 1))

    return unique_path if keep_epoch_copy else path


def _monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    value = float(metrics.get(monitor, math.nan))
    if math.isnan(value):
        value = float(metrics.get("f1", math.nan))
    return value


def _strip_prediction_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_prediction_arrays(item)
            for key, item in value.items()
            if key not in {"_y_true", "_y_prob"}
        }
    if isinstance(value, list):
        return [_strip_prediction_arrays(item) for item in value]
    return value


def _training_label_counts(dataloader) -> Counter[int]:
    rows = getattr(dataloader.dataset, "rows", None)
    if rows is None:
        raise ValueError("Training dataset does not expose manifest rows for class balancing")
    return Counter(int(row["label_id"]) for row in rows)


class SmoothedBCEWithLogitsLoss(nn.Module):
    """``BCEWithLogitsLoss`` with label smoothing, which torch does not provide.

    Targets move from {0, 1} to {eps/2, 1 - eps/2}. Same purpose as the
    ``label_smoothing`` argument on CrossEntropyLoss: it removes the incentive to
    keep growing the logit on samples that are already right, which is what makes
    validation loss climb long after accuracy has settled.
    """

    def __init__(self, smoothing: float, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = float(smoothing)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(logits, smoothed)


def _binary_criterion(smoothing: float, pos_weight: torch.Tensor | None = None) -> nn.Module:
    if smoothing > 0:
        return SmoothedBCEWithLogitsLoss(smoothing, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _make_criterion(config: dict[str, Any], train_dataloader, device: torch.device) -> nn.Module:
    smoothing = float(config["training"].get("label_smoothing", 0.0))
    if smoothing:
        LOGGER.info("Label smoothing: %.3f", smoothing)

    balance_cfg = config["training"].get("class_balance", {})
    strategy = str(balance_cfg.get("strategy", "none")).lower()
    if strategy in {"none", "off", "false"}:
        LOGGER.info("Class balance: disabled")
        return _binary_criterion(smoothing)

    if strategy == "manual_pos_weight":
        pos_weight_value = float(balance_cfg["pos_weight"])
    elif strategy == "auto_pos_weight":
        counts = _training_label_counts(train_dataloader)
        negative_count = counts.get(0, 0)
        positive_count = counts.get(1, 0)
        if negative_count <= 0 or positive_count <= 0:
            raise ValueError(
                "auto_pos_weight requires both classes in the train split; "
                f"got counts={dict(counts)}"
            )
        pos_weight_value = negative_count / positive_count
        LOGGER.info(
            "Class balance: auto_pos_weight=%.4f from train counts No_Burst=%d Burst=%d",
            pos_weight_value,
            negative_count,
            positive_count,
        )
    else:
        raise ValueError(f"Unsupported training.class_balance.strategy: {strategy}")

    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    return _binary_criterion(smoothing, pos_weight=pos_weight)


def _config_with_threshold(config: dict[str, Any], threshold: float) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied["training"]["threshold"] = float(threshold)
    copied["training"]["auto_threshold"] = False
    return copied


def _prune_old_checkpoints(checkpoint_dir: Path, stem: str = "best", keep: int = 3) -> None:
    """Keep only the newest ``keep`` unique ``{stem}_epoch_*.pt`` files.

    The best-checkpoint alias (best.pt) is always preserved; this only trims the
    accumulating per-epoch copies so a long run does not fill the disk.
    """
    if keep is None or keep < 0:
        return
    epoch_files = sorted(
        checkpoint_dir.glob(f"{stem}_epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    stale = epoch_files[:-keep] if keep > 0 else epoch_files
    for path in stale:
        try:
            path.unlink()
        except OSError as exc:
            LOGGER.warning("Could not remove old checkpoint %s: %s", path, exc)


def fit(config: dict[str, Any]) -> dict[str, Any]:
    """Run training and checkpointing, optionally with early stopping.

    ``training.epochs`` is the schedule and, unless ``training.patience`` is
    positive, it is run in full. See :mod:`callisto_trainer.core.train_type` for
    why stopping early is opt-in.
    """
    seed = int(config["training"]["seed"])
    perf_cfg = config.get("performance", {})
    set_seed(
        seed,
        deterministic=bool(perf_cfg.get("deterministic", False)),
        benchmark=bool(perf_cfg.get("cudnn_benchmark", True)),
    )

    matmul_precision = perf_cfg.get("matmul_precision")
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))

    device = _device(config)
    use_amp = bool(perf_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = _amp_dtype(str(perf_cfg.get("amp_dtype", "float16")))
    channels_last = bool(perf_cfg.get("channels_last", False)) and device.type == "cuda"
    dataloaders = get_dataloaders(config)
    use_metadata = bool(config["model"].get("use_metadata", False))
    metadata_kwargs: dict[str, Any] = {}
    if use_metadata:
        station_vocab = config["model"].get("station_vocab", {})
        metadata_kwargs = dict(
            use_metadata=True,
            num_stations=len(station_vocab) + 1,
            num_numeric=NUM_NUMERIC,
            station_emb_dim=int(config["model"].get("station_emb_dim", 8)),
        )
        LOGGER.info("Model uses metadata conditioning (%d stations + %d numeric features)",
                    len(station_vocab) + 1, NUM_NUMERIC)
    model = create_model(
        config["model"]["name"],
        in_channels=int(config["model"]["in_channels"]),
        dropout=float(config["model"].get("dropout", 0.25)),
        pretrained=bool(config["model"].get("pretrained", False)),
        **metadata_kwargs,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if bool(perf_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        LOGGER.info("Compiling model with torch.compile")
        model = torch.compile(model)

    criterion = _make_criterion(config, dataloaders["train"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_name = str(config["training"].get("scheduler", "plateau")).lower()
    if scheduler_name in {"cosine", "cosine_annealing", "cosineannealinglr"}:
        # Smoothly decays LR over the run; often converges in fewer epochs than
        # plateau (faster wall-clock and frequently a better final score).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
        )
        scheduler_is_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        scheduler_is_plateau = True
    scaler = _make_grad_scaler(enabled=use_amp)

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_checkpoint_path = checkpoint_dir / "best.pt"
    epochs_without_improvement = 0
    stopped_early = False
    total_epochs = int(config["training"]["epochs"])
    threshold = float(config["training"]["threshold"])
    auto_threshold = bool(config["training"].get("auto_threshold", False))
    threshold_metric = str(config["training"].get("threshold_metric", "f1"))
    monitor = str(config["training"].get("monitor", "pr_auc"))
    # patience <= 0 disables early stopping: every configured epoch runs. Model
    # selection is unaffected either way -- best.pt still tracks the best
    # validation score, so a longer run can only find a better epoch, never
    # return a worse one.
    patience = int(config["training"]["patience"])
    early_stopping = patience > 0

    LOGGER.info("Training on %s", device)
    LOGGER.info(
        "CUDA speed options: amp=%s amp_dtype=%s channels_last=%s cudnn_benchmark=%s",
        use_amp,
        amp_dtype,
        channels_last,
        bool(perf_cfg.get("cudnn_benchmark", True)),
    )
    LOGGER.info(
        "Data sizes: train=%d val=%d test=%d",
        len(dataloaders["train"].dataset),
        len(dataloaders["val"].dataset),
        len(dataloaders["test"].dataset),
    )
    # Trainer addition: report to the GUI over stdout (see core/progress.py).
    emit_progress(
        {
            "event": "start",
            "task": "binary",
            "device": str(device),
            "classes": ["No_Burst", "Burst"],
            "total_epochs": total_epochs,
            "train_size": len(dataloaders["train"].dataset),
            "val_size": len(dataloaders["val"].dataset),
            "test_size": len(dataloaders["test"].dataset),
        }
    )

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            threshold,
            use_amp,
            amp_dtype,
            scaler,
            channels_last,
        )
        val_metrics = validate_one_epoch(
            model,
            dataloaders["val"],
            criterion,
            device,
            threshold,
            use_amp,
            amp_dtype,
            channels_last,
            include_predictions=auto_threshold,
        )

        tuned_threshold = threshold
        if auto_threshold:
            tuned_threshold, tuned_metrics = find_best_threshold(
                np.asarray(val_metrics["_y_true"]),
                np.asarray(val_metrics["_y_prob"]),
                metric=threshold_metric,
            )
            tuned_metrics["loss"] = val_metrics["loss"]
            tuned_metrics["default_threshold"] = threshold
            val_metrics = tuned_metrics

        score = _monitor_value(val_metrics, monitor)
        if scheduler_is_plateau:
            scheduler.step(score)
        else:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "decision_threshold": tuned_threshold,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(_strip_prediction_arrays(epoch_record))
        LOGGER.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f val_f1=%.4f val_pr_auc=%.4f threshold=%.3f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["f1"],
            val_metrics["pr_auc"],
            tuned_threshold,
        )
        emit_progress(
            {
                "event": "epoch",
                "task": "binary",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_pr_auc": val_metrics["pr_auc"],
                "score": score,
                "decision_threshold": tuned_threshold,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "is_best": score > best_score,
            }
        )

        checkpoint_config = _config_with_threshold(config, tuned_threshold)
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            checkpoint_config,
            {"train": train_metrics, "val": val_metrics, "decision_threshold": tuned_threshold},
            keep_epoch_copy=False,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_checkpoint_path = save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                checkpoint_config,
                {"train": train_metrics, "val": val_metrics, "decision_threshold": tuned_threshold},
                keep_epoch_copy=True,
            )
            _prune_old_checkpoints(
                checkpoint_dir,
                keep=int(config["training"].get("keep_checkpoints", 3)),
            )
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            LOGGER.info("Early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
            stopped_early = True
            break

    history_path = checkpoint_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    result = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_run": len(history),
        "total_epochs": total_epochs,
        "stopped_early": stopped_early,
        "history_path": str(history_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_alias": str(checkpoint_dir / "best.pt"),
    }
    emit_progress({"event": "finished", "task": "binary", **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the e-CALLISTO burst classifier")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    result = fit(config)
    LOGGER.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
