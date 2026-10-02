"""Train the burst-TYPE classifier (Type II / Type III / Other).

Stage 2 of the cascade. This mirrors ``src/training/train.py`` but targets a
3-way softmax head instead of a binary logit: CrossEntropyLoss (optionally with
inverse-frequency class weights), argmax predictions, and macro-F1 model
selection with no decision-threshold tuning. All the task-agnostic plumbing
(device/AMP setup, checkpoint writing, pruning) is imported from ``train.py`` so
it is not duplicated.
"""

# NOTE: Vendored from H:\Burst Identifier (src/training/train_type.py).
# Numerics are intentionally unchanged so preprocessed tensors stay
# byte-identical to the original pipeline. See tests/test_preprocess_parity.py.

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from callisto_trainer.core.dataset import get_dataloaders
from callisto_trainer.core.type_metrics import compute_multiclass_metrics
from callisto_trainer.core.models.model_factory import create_model, model_kwargs_from_config
from callisto_trainer.core.taxonomy import NO_BURST, NON_BURST_LABELS
from callisto_trainer.core.unified_metrics import merge_rejections, unified_region_metrics
from callisto_trainer.core.train_binary import (
    _amp_dtype,
    _autocast,
    _device,
    _make_grad_scaler,
    _move_inputs,
    _prune_old_checkpoints,
    _split_batch,
    _strip_prediction_arrays,
    save_checkpoint,
)
from callisto_trainer.core.config import load_config
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.core.progress import emit_progress
from callisto_trainer.core.seed import set_seed


LOGGER = get_logger(__name__)


def _ordered_class_names(config: dict[str, Any]) -> list[str]:
    """Return class names ordered by their label id (index 0 -> id 0)."""
    classes = config["data"]["classes"]
    return [name for name, _ in sorted(classes.items(), key=lambda item: int(item[1]))]


def _training_label_counts(dataloader) -> Counter[int]:
    rows = getattr(dataloader.dataset, "rows", None)
    if rows is None:
        raise ValueError("Training dataset does not expose manifest rows for class balancing")
    return Counter(int(row["label_id"]) for row in rows)


def _detection_balanced_weights(
    counts: Counter[int], class_names: list[str], exponent: float, background_weight: float
) -> list[float]:
    """Balance the burst types among themselves; keep background at full weight.

    Plain inverse frequency over every class treats background as just another
    class to be balanced away: the more background samples there are -- and
    there should be many, because that is what inference sees -- the less each
    one counts, until mistaking interference for a burst costs almost nothing.
    Here only the burst types are reweighted (inverse frequency, damped by
    ``exponent``, mean 1 across them), and No_Burst and RFI each keep
    ``background_weight``. Every rejection example then counts as much as an
    average burst example, and the decision threshold -- tuned afterwards to a
    false-alarm budget -- sets the operating point.
    """
    burst = [i for i, name in enumerate(class_names) if name not in NON_BURST_LABELS]
    total = sum(counts.get(i, 0) for i in burst)
    raw = {
        i: (total / (len(burst) * counts[i])) ** exponent for i in burst if counts.get(i, 0)
    }
    mean = sum(raw.values()) / len(raw) if raw else 1.0
    return [
        (raw[i] / mean) if i in raw else float(background_weight)
        for i in range(len(class_names))
    ]


def class_weights(
    config: dict[str, Any], counts: Counter[int], class_names: list[str]
) -> list[float] | None:
    """The loss weight of every class under ``training.class_balance``, or None.

    Separate from the loss so the post-training type-frequency correction (see
    ``core/type_priors.py``) can recover exactly what the model was fitted to.
    """
    num_classes = len(class_names)
    balance_cfg = config["training"].get("class_balance", {})
    strategy = str(balance_cfg.get("strategy", "none")).lower()
    if strategy in {"none", "off", "false"}:
        return None

    if strategy not in {"auto_class_weights", "auto_pos_weight", "detection_balanced"}:
        raise ValueError(
            f"Unsupported training.class_balance.strategy for the type model: {strategy}"
        )

    total = sum(counts.get(i, 0) for i in range(num_classes))
    if total == 0 or any(counts.get(i, 0) == 0 for i in range(num_classes)):
        raise ValueError(
            f"auto_class_weights requires every class in the train split; got counts={dict(counts)}"
        )

    # ``exponent`` damps the correction. At 1.0 this is plain inverse frequency,
    # which on a 2493/928/253/131 split puts a 19x spread between the commonest
    # and rarest class -- the optimizer then sees one rare sample as worth 19
    # common ones, and since the rare classes rest on only a few dozen *distinct*
    # hand-drawn bursts, the cheapest way to satisfy that pressure is to memorize
    # them. 0.5 (inverse square root) keeps the correction but cuts the spread to
    # ~4x. 0.0 disables weighting entirely.
    exponent = float(balance_cfg.get("exponent", 0.5))
    if exponent < 0:
        raise ValueError(f"class_balance.exponent must be >= 0, got {exponent}")

    if strategy == "detection_balanced":
        if not any(name in NON_BURST_LABELS for name in class_names):
            raise ValueError(
                "class_balance.strategy detection_balanced needs a No_Burst class; "
                f"got {class_names}"
            )
        return _detection_balanced_weights(
            counts, class_names, exponent, float(balance_cfg.get("background_weight", 1.0))
        )

    # weight[c] = (N / (K * count[c])) ** exponent, rescaled to mean 1.0.
    raw = [(total / (num_classes * counts[i])) ** exponent for i in range(num_classes)]
    mean = sum(raw) / num_classes
    return [value / mean for value in raw]


def _make_criterion(
    config: dict[str, Any],
    train_dataloader,
    num_classes: int,
    device: torch.device,
    class_names: list[str] | None = None,
) -> nn.Module:
    """CrossEntropyLoss, optionally weighted by inverse class frequency."""
    smoothing = float(config["training"].get("label_smoothing", 0.0))
    if smoothing:
        LOGGER.info("Label smoothing: %.3f", smoothing)

    names = class_names or [str(i) for i in range(num_classes)]
    strategy = str(config["training"].get("class_balance", {}).get("strategy", "none")).lower()
    if strategy in {"none", "off", "false"}:
        LOGGER.info("Class balance: disabled")
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

    counts = _training_label_counts(train_dataloader)
    weights = class_weights(config, counts, names)
    LOGGER.info(
        "Class balance: %s(exponent=%.2f)=%s from train counts %s",
        strategy,
        float(config["training"].get("class_balance", {}).get("exponent", 0.5)),
        {name: round(w, 4) for name, w in zip(names, weights)},
        {names[i]: counts.get(i, 0) for i in range(num_classes)},
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=smoothing)


def _run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Any | None = None,
    channels_last: bool = False,
    include_predictions: bool = False,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    sample_total = 0
    pred_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    # Trainer addition: the unified model is judged on how it *ranks* regions
    # (burst evidence), which needs the probabilities, not just the argmax.
    keep_probabilities = NO_BURST in class_names
    prob_chunks: list[np.ndarray] = []

    for batch in dataloader:
        inputs, labels = _split_batch(batch)
        inputs = _move_inputs(inputs, device, channels_last)
        labels = labels.to(device, non_blocking=True).long()

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with _autocast(device, use_amp, amp_dtype):
                logits = model(*inputs)
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
        pred_chunks.append(logits.detach().argmax(dim=1).cpu().numpy())
        label_chunks.append(labels.detach().cpu().numpy())
        if keep_probabilities:
            prob_chunks.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())

    y_pred = np.concatenate(pred_chunks) if pred_chunks else np.asarray([], dtype=np.int64)
    y_true = np.concatenate(label_chunks) if label_chunks else np.asarray([], dtype=np.int64)

    # Accuracy and macro-F1 as the operator reads them: RFI and No_Burst are one
    # "not a burst" (the loss and the ranking metrics below keep them apart).
    shown_true, shown_names = merge_rejections(y_true, class_names)
    shown_pred, _ = merge_rejections(y_pred, class_names)
    metrics = compute_multiclass_metrics(shown_true, shown_pred, shown_names)
    if keep_probabilities and prob_chunks:
        metrics.update(
            unified_region_metrics(y_true, np.concatenate(prob_chunks), class_names)
        )
    metrics["loss"] = float(loss_total.item() / sample_total) if sample_total else math.nan
    if include_predictions:
        metrics["_y_true"] = y_true.tolist()
        metrics["_y_pred"] = y_pred.tolist()
    return metrics


def fit_type(config: dict[str, Any]) -> dict[str, Any]:
    """Run burst-type training and checkpointing, optionally with early stopping.

    ``training.epochs`` means what it says: the full schedule runs unless
    ``training.patience`` is set to a positive number, which is what re-enables
    stopping on a plateau. Early stopping is off by default because a plateau in
    the validation score is not the same as convergence -- solar burst classes
    are heavily imbalanced, so the monitored macro-F1 routinely sits flat for a
    stretch of epochs and then improves again. Cutting the run there trades a
    real gain for saved time the operator did not ask to save.
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

    if bool(config["model"].get("use_metadata", False)):
        raise ValueError(
            "The multiclass model does not use the old station metadata branch; set "
            "model.use_metadata: false. Station and date reach the unified model "
            "through model.station_date."
        )

    class_names = _ordered_class_names(config)
    num_classes = len(class_names)

    device = _device(config)
    use_amp = bool(perf_cfg.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = _amp_dtype(str(perf_cfg.get("amp_dtype", "float16")))
    channels_last = bool(perf_cfg.get("channels_last", False)) and device.type == "cuda"
    dataloaders = get_dataloaders(config)

    # Trainer addition: an optional physics branch (measured drift rate, burst
    # extent and, for the region_v2 feature set, the interference features) fused
    # with the image backbone, over one or more views of the region. See
    # core/models/physics_model.py and core/region_inputs.py.
    branch_kwargs = model_kwargs_from_config(config["model"])
    if branch_kwargs.get("use_physics"):
        LOGGER.info(
            "Feature conditioning enabled: %s (%d features), views=%s",
            config["model"].get("feature_set") or "physics_v1",
            branch_kwargs["num_physics"],
            config["model"].get("views") or ["crop"],
        )
    if branch_kwargs.get("station_date"):
        LOGGER.info(
            "Station/date correction enabled: %d station slot(s), log-odds shift capped at %.2f",
            branch_kwargs["station_date"]["num_stations"],
            branch_kwargs["station_date"]["cap"],
        )

    model = create_model(
        config["model"]["name"],
        in_channels=int(config["model"]["in_channels"]),
        dropout=float(config["model"].get("dropout", 0.25)),
        pretrained=bool(config["model"].get("pretrained", False)),
        num_classes=num_classes,
        **branch_kwargs,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if bool(perf_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        LOGGER.info("Compiling model with torch.compile")
        model = torch.compile(model)

    criterion = _make_criterion(
        config, dataloaders["train"], num_classes, device, class_names=class_names
    )
    task = "unified" if NO_BURST in class_names else "type"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_name = str(config["training"].get("scheduler", "plateau")).lower()
    if scheduler_name in {"cosine", "cosine_annealing", "cosineannealinglr"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config["training"]["epochs"])
        )
        scheduler_is_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
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
    monitor = str(config["training"].get("monitor", "macro_f1"))
    # patience <= 0 disables early stopping: every configured epoch runs. Model
    # selection is unaffected either way -- best.pt still tracks the best
    # validation score, so a longer run can only find a better epoch, never
    # return a worse one.
    patience = int(config["training"]["patience"])
    early_stopping = patience > 0

    LOGGER.info("Training burst-type model on %s (classes=%s)", device, class_names)
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
            "task": task,
            "device": str(device),
            "classes": class_names,
            "total_epochs": total_epochs,
            "train_size": len(dataloaders["train"].dataset),
            "val_size": len(dataloaders["val"].dataset),
            "test_size": len(dataloaders["test"].dataset),
        }
    )

    for epoch in range(1, total_epochs + 1):
        train_metrics = _run_epoch(
            model,
            dataloaders["train"],
            criterion,
            device,
            class_names,
            optimizer=optimizer,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            channels_last=channels_last,
        )
        val_metrics = _run_epoch(
            model,
            dataloaders["val"],
            criterion,
            device,
            class_names,
            optimizer=None,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )

        score = float(val_metrics.get(monitor, val_metrics.get("macro_f1", math.nan)))
        if math.isnan(score):
            # A monitored metric can be undefined on a tiny split (no burst in
            # val, say); NaN never compares greater, so best.pt would never be
            # written. Fall back to the metric that is always defined.
            score = float(val_metrics.get("macro_f1", 0.0))
        if scheduler_is_plateau:
            scheduler.step(score)
        else:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(_strip_prediction_arrays(epoch_record))
        LOGGER.info(
            "Epoch %03d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )
        emit_progress(
            {
                "event": "epoch",
                "task": task,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_detection_ap": val_metrics.get("detection_ap"),
                "val_type_macro_f1": val_metrics.get("type_macro_f1"),
                "monitor": monitor,
                "score": score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "is_best": score > best_score,
            }
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            {"train": train_metrics, "val": val_metrics},
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
                config,
                {"train": train_metrics, "val": val_metrics},
                keep_epoch_copy=True,
            )
            _prune_old_checkpoints(
                checkpoint_dir, keep=int(config["training"].get("keep_checkpoints", 3))
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
    emit_progress({"event": "finished", "task": task, **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the e-CALLISTO burst-type classifier")
    parser.add_argument(
        "--config",
        default="configs/type_resnet18_rtx5060_8gb.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = fit_type(config)
    LOGGER.info("Type training complete: %s", result)


if __name__ == "__main__":
    main()
