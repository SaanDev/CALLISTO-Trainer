"""Export a trained checkpoint as a self-contained, portable bundle.

A bare ``best.pt`` is not enough to use a model correctly. It carries weights,
but not the knowledge that its input must be background-subtracted with
``plotutil_median_db``, mapped through a -1..8 dB window, resized to 224x224, and
— for the type model — cropped from a *region* rather than taken from a whole
file. Getting any of that wrong produces confident, wrong answers rather than an
error. The crop geometry the model was trained with is recorded in the card, so
a bundle stays reproducible even if the project's defaults change later.

So the bundle records everything needed to reproduce the input contract, plus a
standalone script that implements it, plus a TorchScript graph for use without
this codebase at all.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from callisto_trainer import __version__
from callisto_trainer.core.logging_utils import get_logger

LOGGER = get_logger(__name__)

BUNDLE_FORMAT_VERSION = 1


@dataclass
class ExportedBundle:
    """Result of one export."""

    directory: Path
    task: str
    files: list[str] = field(default_factory=list)
    torchscript_path: Path | None = None
    torchscript_error: str | None = None
    zip_path: Path | None = None

    def summary(self) -> str:
        parts = [f"bundle at {self.directory}"]
        if self.torchscript_path:
            parts.append("TorchScript included")
        elif self.torchscript_error:
            parts.append(f"TorchScript skipped ({self.torchscript_error})")
        if self.zip_path:
            parts.append(f"zipped to {self.zip_path.name}")
        return "; ".join(parts)


def _torch_load(path: Path, device: Any) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _ordered_class_names(config: dict[str, Any]) -> list[str]:
    classes = config.get("data", {}).get("classes", {}) or {}
    return [name for name, _ in sorted(classes.items(), key=lambda item: int(item[1]))]


def build_model_card(
    task: str,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    snapshot_info: dict[str, Any] | None,
    source_checkpoint: Path,
) -> dict[str, Any]:
    """The machine-readable contract a consumer needs to use this model right."""
    class_names = _ordered_class_names(config)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    preprocessing = config.get("preprocessing", {})
    crops = config.get("crops", {})
    # "unified" and "type" are both crop-based softmax models; only "binary" has
    # a sigmoid head and a decision threshold.
    is_multiclass = task in ("type", "unified")
    uses_physics = bool(model_cfg.get("use_physics", False))
    views = list(model_cfg.get("views") or ["crop"])
    inference = dict(config.get("inference", {}) or {})

    card: dict[str, Any] = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "exported_by": f"CALLISTO Trainer {__version__}",
        "task": task,
        "source_checkpoint": str(source_checkpoint),
        "trained_epoch": checkpoint.get("epoch"),
        "architecture": {
            "name": model_cfg.get("name"),
            "in_channels": int(model_cfg.get("in_channels", 1)),
            "num_classes": len(class_names) if is_multiclass else 1,
            "uses_metadata": bool(model_cfg.get("use_metadata", False)),
            "station_vocab_size": len(model_cfg.get("station_vocab", {}) or {}),
            "uses_physics": uses_physics,
            "views": views,
        },
        "classes": class_names if is_multiclass else ["No_Burst", "Burst"],
        "input": {
            # One channel per view, stacked: [views, height, width].
            "shape": [len(views), *config.get("data", {}).get("target_shape", [224, 224])],
            "views": views,
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "unit": "normalized dB above per-frequency background",
        },
        # The full input contract. Anything consuming this model must reproduce
        # these steps, in this order, or its predictions are meaningless.
        "preprocessing": {
            "orientation": "[frequency, time]; row 0 is the FIRST FITS row "
            "(normally the highest frequency)",
            "steps": [
                "replace NaN/Inf with the finite median",
                f"subtract per-frequency median over time, scale to dB "
                f"({preprocessing.get('background_method')})",
                f"map [{preprocessing.get('db_vmin')}, {preprocessing.get('db_vmax')}] dB "
                "to [0, 1] and clip",
                "bilinear resize to the input shape",
            ],
            "background_method": preprocessing.get("background_method"),
            "normalization": preprocessing.get("normalization"),
            "db_vmin": preprocessing.get("db_vmin"),
            "db_vmax": preprocessing.get("db_vmax"),
            "critical_note": (
                "Background subtraction MUST be computed over the whole file's time "
                "axis before any cropping. Cropping first makes a burst its own "
                "background and erases it."
            ),
        },
        "metrics": checkpoint.get("metrics", {}),
    }

    if is_multiclass:
        card["scope"] = {
            "operates_on": "a CROP around a single burst, not a whole file",
            "context_margin": crops.get("context_margin"),
            "min_rows": crops.get("min_rows"),
            "min_cols": crops.get("min_cols"),
            "warning": (
                "This model was trained on cropped burst regions. Running it on a "
                "whole-file spectrum is out of distribution and its output should "
                "not be trusted. Locate a region first, then crop it the same way."
            ),
        }
        card["output"] = {
            "head": "softmax",
            "interpretation": (
                "argmax over `classes` gives the burst type. For the unified model "
                "class 0 is No_Burst, and No_Burst and RFI both mean 'not a burst': "
                "`1 - P(No_Burst) - P(RFI)` is the burst evidence for a region, and "
                "the region is a burst when that reaches `burst_threshold`."
            ),
        }
        if task == "unified":
            card["output"]["non_burst_classes"] = [
                name for name in class_names if name in ("No_Burst", "RFI")
            ]
            card["output"]["burst_threshold"] = inference.get("burst_threshold")
            card["output"]["calibration"] = inference.get("calibration")
            card["region_finder"] = inference.get("region_finder")
            priors = inference.get("type_priors") or {}
            if priors.get("adjustment"):
                card["output"]["type_priors"] = {
                    "adjustment": priors.get("adjustment"),
                    "strength": priors.get("strength"),
                    "class_shares": priors.get("class_shares"),
                    "training_shares": priors.get("training_shares"),
                    "how_to_apply": (
                        "multiply each burst-type probability by "
                        "exp(strength * adjustment[type]), then rescale the burst types "
                        "so they sum to what they summed to before; leave No_Burst and "
                        "RFI unchanged. This corrects the type for how often each type "
                        "really occurs and never changes the burst evidence."
                    ),
                }
        if len(views) > 1:
            card["scope"]["context_min_pad_cols"] = int(crops.get("context_min_pad_cols", 240))
            card["scope"]["views"] = {
                "crop": "the region exactly as given",
                "context": (
                    "the full frequency band over the region's columns extended by "
                    f"max(region width, {crops.get('context_min_pad_cols', 240)}) samples "
                    "each side, block-max-pooled then bilinear-resized"
                ),
            }
            if "quiet_context" in views:
                from callisto_trainer.core.crops import QUIET_PERCENTILE

                card["scope"]["quiet_percentile"] = float(QUIET_PERCENTILE)
                card["scope"]["views"]["quiet_context"] = (
                    "the context view taken from the whole file normalized with each "
                    f"channel's {QUIET_PERCENTILE:g}th percentile as its background "
                    "instead of its median (same dB scale and window), so a continuum "
                    "lasting most of the file stays visible; see normalize_quiet() in "
                    "predict.py"
                )
        if uses_physics:
            from callisto_trainer.core.region_features import (
                FEATURE_SET_PHYSICS_V1,
                feature_names,
            )

            feature_set = str(model_cfg.get("feature_set") or FEATURE_SET_PHYSICS_V1)
            names = list(feature_names(feature_set))
            note = (
                "This model takes a SECOND input: measured burst physics for the "
                "region. Pass all zeros when no measurement is available -- the "
                "final 'measured' flag then reads 0, which is a state the model "
                "saw in training (about one region in seven yields no clean drift "
                "fit), so it degrades to the image-only case rather than being fed "
                "a fabricated drift rate of zero."
            )
            if feature_set != FEATURE_SET_PHYSICS_V1:
                note = (
                    "This model takes a SECOND input: the region's physics and "
                    "interference features, computed from the whole-file normalized "
                    "spectrum by callisto_trainer.core.region_features. They are always "
                    "measured in training, so all zeros is out of distribution: use "
                    "this project (CascadePredictor) to run the model on real files."
                )
            card["physics_input"] = {
                "required": True,
                "feature_set": feature_set,
                "shape": [len(names)],
                "feature_order": names,
                "note": note,
            }
    else:
        card["scope"] = {"operates_on": "a whole-file spectrum"}
        card["output"] = {
            "head": "sigmoid",
            "decision_threshold": float(training_cfg.get("threshold", 0.5)),
            "interpretation": (
                "probability >= decision_threshold means Burst. The threshold was "
                "tuned on the validation split; do not assume 0.5."
            ),
        }
        card["metadata_features"] = {
            "order": ["station_index", "freq_min", "freq_max", "freq_span",
                      "doy_sin", "doy_cos", "year"],
            "note": "frequency features are MHz/1000; station_index 0 means unknown",
        }

    if snapshot_info:
        card["training_data"] = {
            "snapshot": snapshot_info.get("kind"),
            "samples": snapshot_info.get("samples"),
            "class_counts": snapshot_info.get("class_counts"),
            "split_counts": snapshot_info.get("split_counts"),
            "event_leakage": snapshot_info.get("event_leakage"),
        }
    return card


STANDALONE_SCRIPT = '''"""Standalone inference for an exported CALLISTO Trainer model.

Depends only on numpy, astropy and torch -- none of the training project.

    python predict.py path/to/file.fit.gz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

BUNDLE = Path(__file__).resolve().parent
CARD = json.loads((BUNDLE / "model_card.json").read_text(encoding="utf-8"))

DB_VMIN = float(CARD["preprocessing"]["db_vmin"])
DB_VMAX = float(CARD["preprocessing"]["db_vmax"])
TARGET = tuple(CARD["input"]["shape"][1:])
PLOTUTIL_DB_SCALE = 2500.0 / 255.0 / 25.4


def read_spectrum(path):
    """Primary HDU as [frequency, time] float32."""
    with fits.open(path, memmap=False) as hdul:
        return np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))


def normalize(spectrum):
    """clean -> per-frequency median subtract -> dB window -> [0,1].

    Must be run on the WHOLE file before any cropping (see model_card.json).
    """
    data = np.asarray(spectrum, dtype=np.float32)
    finite = np.isfinite(data)
    if finite.any():
        data = np.where(finite, data, np.median(data[finite])).astype(np.float32)
    else:
        data = np.zeros_like(data)
    data = (data - np.median(data, axis=1, keepdims=True)) * np.float32(PLOTUTIL_DB_SCALE)
    return np.clip((data - DB_VMIN) / (DB_VMAX - DB_VMIN), 0.0, 1.0).astype(np.float32)


def normalize_quiet(spectrum):
    """As normalize(), but each channel's background is its quiet part.

    Needed only by a model whose views include "quiet_context".
    """
    percentile = float(CARD.get("scope", {}).get("quiet_percentile") or 10.0)
    data = np.asarray(spectrum, dtype=np.float32)
    finite = np.isfinite(data)
    if finite.any():
        data = np.where(finite, data, np.median(data[finite])).astype(np.float32)
    else:
        data = np.zeros_like(data)
    baseline = np.percentile(data, percentile, axis=1, keepdims=True).astype(np.float32)
    data = ((data - baseline).astype(np.float32) * np.float32(PLOTUTIL_DB_SCALE)).astype(np.float32)
    return np.clip((data - DB_VMIN) / (DB_VMAX - DB_VMIN), 0.0, 1.0).astype(np.float32)


def _resize_axis(data, new_size, axis):
    old = data.shape[axis]
    if old == new_size:
        return data.astype(np.float32, copy=False)
    pos = (np.arange(new_size, dtype=np.float64) * (old - 1) / (new_size - 1)
           if new_size > 1 and old > 1 else np.zeros(new_size))
    lo = np.clip(np.floor(pos).astype(np.intp), 0, old - 1)
    hi = np.minimum(lo + 1, old - 1)
    frac = (pos - lo).astype(np.float32)
    if axis == 1:
        return (data[:, lo] * (1 - frac[None, :]) + data[:, hi] * frac[None, :]).astype(np.float32)
    return (data[lo, :] * (1 - frac[:, None]) + data[hi, :] * frac[:, None]).astype(np.float32)


def resize(spectrum, target=TARGET):
    data = _resize_axis(np.asarray(spectrum, dtype=np.float32), int(target[1]), axis=1)
    return _resize_axis(data, int(target[0]), axis=0)


VIEWS = list(CARD.get("input", {}).get("views") or ["crop"])
CONTEXT_MIN_PAD = int(CARD.get("scope", {}).get("context_min_pad_cols") or 240)


def _max_pool(data, target, axis):
    factor = data.shape[axis] // max(1, int(target))
    if factor < 2:
        return data
    blocks = int(np.ceil(data.shape[axis] / factor))
    pad = blocks * factor - data.shape[axis]
    if pad:
        widths = [(0, 0), (0, 0)]
        widths[axis] = (0, pad)
        data = np.pad(data, widths, mode="edge")
    if axis == 1:
        return data.reshape(data.shape[0], blocks, factor).max(axis=2)
    return data.reshape(blocks, factor, data.shape[1]).max(axis=1)


def context_view(normalized, box):
    """Full band, the region's columns widened each side, max-pooled then resized."""
    r0, r1, c0, c1 = box
    n_time = normalized.shape[1]
    pad = max(c1 - c0, CONTEXT_MIN_PAD)
    width = min(c1 - c0 + 2 * pad, n_time)
    lo = min(max(0, c0 - pad), n_time - width)
    patch = normalized[:, lo:lo + width]
    patch = _max_pool(_max_pool(patch, TARGET[1], 1), TARGET[0], 0)
    return resize(patch)


def to_tensor(normalized, box=None, quiet=None):
    """Whole file, or one [row0, row1, col0, col1] region with the trained views.

    ``quiet`` is normalize_quiet() of the same file, required when the views
    include "quiet_context".
    """
    if box is None:
        return resize(normalized)[np.newaxis, np.newaxis]
    margin = float(CARD.get("scope", {}).get("context_margin") or 0.0)
    r0, r1, c0, c1 = box
    dr, dc = int(round((r1 - r0) * margin)), int(round((c1 - c0) * margin))
    r0, r1 = max(0, r0 - dr), min(normalized.shape[0], r1 + dr)
    c0, c1 = max(0, c0 - dc), min(normalized.shape[1], c1 + dc)
    layers = [resize(normalized[r0:r1, c0:c1])]
    if "context" in VIEWS:
        layers.append(context_view(normalized, box))
    if "quiet_context" in VIEWS:
        if quiet is None:
            raise ValueError("this model's views include quiet_context: pass quiet=normalize_quiet(spectrum)")
        layers.append(context_view(quiet, box))
    return np.stack(layers)[np.newaxis]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    model = torch.jit.load(str(BUNDLE / "model_scripted.pt"), map_location="cpu")
    model.eval()

    normalized = normalize(read_spectrum(sys.argv[1]))
    tensor = torch.from_numpy(to_tensor(normalized))
    if len(VIEWS) > 1:
        # A multi-view region model given a whole file: repeat the one view so
        # the shapes line up. The result is out of distribution (see below).
        tensor = tensor.repeat(1, len(VIEWS), 1, 1)

    inputs = [tensor]
    physics_spec = CARD.get("physics_input")
    if physics_spec:
        # Zeros mean "no measurement available", which the model was trained to
        # handle -- see the note in model_card.json. Measuring the drift rate
        # properly requires the project's burst_physics module.
        inputs.append(torch.zeros(1, int(physics_spec["shape"][0])))

    with torch.no_grad():
        logits = model(*inputs)

    if CARD["output"]["head"] == "sigmoid":
        probability = float(torch.sigmoid(logits.reshape(-1))[0])
        threshold = float(CARD["output"]["decision_threshold"])
        label = "Burst" if probability >= threshold else "No_Burst"
        print(json.dumps({"file": sys.argv[1], "predicted_label": label,
                          "burst_probability": probability,
                          "decision_threshold": threshold}, indent=2))
    else:
        probs = torch.softmax(logits.reshape(1, -1), dim=1)[0].numpy()
        names = CARD["classes"]
        best = int(probs.argmax())
        result = {"file": sys.argv[1], "burst_type": names[best],
                  "confidence": float(probs[best]),
                  "probabilities": {n: float(p) for n, p in zip(names, probs)}}
        # This model was trained on crops around a single burst. Handed a whole
        # recording it will still answer, confidently and meaninglessly, so say
        # so rather than let the number be quoted.
        if "CROP" in CARD.get("scope", {}).get("operates_on", ""):
            result["warning"] = (
                "Run on the WHOLE file, but this model expects a crop around one "
                "burst. Locate a region first and pass box=(row0,row1,col0,col1) "
                "to to_tensor(); this result is out of distribution."
            )
        if CARD.get("physics_input"):
            result["physics"] = "not measured (zeros passed; see model_card.json)"
            if CARD["physics_input"].get("feature_set", "physics_v1") != "physics_v1":
                result["warning"] = result.get("warning", "") + (
                    " This model's region features cannot be computed without the "
                    "CALLISTO Trainer project; zeros are out of distribution. Use the "
                    "project's CascadePredictor for real predictions."
                )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _readme(card: dict[str, Any]) -> str:
    task = card["task"]
    classes = ", ".join(card["classes"])
    lines = [
        f"# CALLISTO {task} model",
        "",
        f"Exported {card['exported_at']} by {card['exported_by']}.",
        "",
        "## What this predicts",
        "",
        f"Classes: {classes}",
        f"Input: {card['input']['shape']} float32 in {card['input']['value_range']}",
        "",
        "## Contents",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `model_card.json` | the full input contract, classes and metrics |",
        "| `weights.pt` | state dict for rebuilding the model in this project |",
        "| `checkpoint.pt` | the original checkpoint, config included |",
        "| `config.yaml` | the training configuration |",
        "| `model_scripted.pt` | TorchScript graph; runs without this project's code |",
        "| `predict.py` | standalone example using only numpy, astropy and torch |",
        "",
        "## Using it",
        "",
        "```bash",
        "python predict.py path/to/file.fit.gz",
        "```",
        "",
        "## The input contract",
        "",
        "The model is only valid on input prepared exactly this way:",
        "",
    ]
    lines += [f"{i}. {step}" for i, step in enumerate(card["preprocessing"]["steps"], 1)]
    lines += ["", f"**{card['preprocessing']['critical_note']}**", ""]

    # Softmax head means a crop-based model (type or unified); sigmoid means the
    # whole-file binary one. Keying off the head rather than the task name keeps
    # this correct as tasks are added.
    if card["output"]["head"] == "softmax":
        margin = card["scope"].get("context_margin")
        lines += [
            "## Scope",
            "",
            f"**{card['scope']['warning']}**",
            "",
        ]
        if margin:
            lines.append(
                f"Crops are expanded by a {margin:.0%} context margin on each side "
                "before resizing."
            )
        elif margin is not None:
            lines.append(
                "Crops are taken as the exact region given, with no context margin, "
                "then resized."
            )
        if card.get("physics_input"):
            lines += [
                "",
                "## Second input: measured physics",
                "",
                f"This model takes **two** inputs. The second is a "
                f"{card['physics_input']['shape'][0]}-element float32 vector:",
                "",
                "```",
                ", ".join(card["physics_input"]["feature_order"]),
                "```",
                "",
                card["physics_input"]["note"],
            ]
        lines.append("")
    else:
        lines += [
            "## Decision threshold",
            "",
            f"Use `{card['output']['decision_threshold']}`, tuned on the validation "
            "split. Do not assume 0.5.",
            "",
        ]

    metrics = card.get("metrics", {}).get("val", {})
    if metrics:
        interesting = ("accuracy", "macro_f1", "f1", "precision", "recall", "pr_auc", "roc_auc")
        rows = [f"- {k}: {v:.4f}" for k, v in metrics.items()
                if k in interesting and isinstance(v, (int, float))]
        if rows:
            lines += ["## Validation metrics at export", "", *rows, ""]
    return "\n".join(lines)


def export_torchscript(
    checkpoint: dict[str, Any], config: dict[str, Any], task: str, destination: Path
) -> tuple[Path | None, str | None]:
    """Trace the model to TorchScript. Returns ``(path, error)``.

    The metadata-conditioned binary model takes two inputs, so it is traced with
    both. Tracing can legitimately fail for some architectures; that is reported
    rather than aborting the export, because the rest of the bundle is still good.

    ``torch.jit.trace`` emits a DeprecationWarning on torch 2.12 in favour of
    ``torch.export``. TorchScript remains the format with the widest deployment
    support today, and the produced graph loads fine, so it is kept; revisit if a
    future torch removes it.
    """
    import torch

    from callisto_trainer.core.models.model_factory import create_model

    try:
        from callisto_trainer.core.models.model_factory import model_kwargs_from_config

        model_cfg = config.get("model", {})
        class_names = _ordered_class_names(config)
        uses_metadata = bool(model_cfg.get("use_metadata", False)) and task == "binary"

        # A physics-conditioned checkpoint nests its backbone one level down, so
        # the branch -- with its feature count and views -- has to be rebuilt or
        # load_state_dict rejects every weight.
        uses_physics = bool(model_cfg.get("use_physics", False))
        kwargs: dict[str, Any] = (
            model_kwargs_from_config(model_cfg) if uses_physics or uses_metadata else {}
        )

        model = create_model(
            model_cfg.get("name", "resnet18"),
            in_channels=int(model_cfg.get("in_channels", 1)),
            dropout=float(model_cfg.get("dropout", 0.25)),
            num_classes=len(class_names) if task in ("type", "unified") else 1,
            **kwargs,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        shape = config.get("data", {}).get("target_shape", [224, 224])
        views = len(model_cfg.get("views") or ["crop"])
        example: tuple[Any, ...] = (torch.zeros(1, views, int(shape[0]), int(shape[1])),)
        if uses_physics:
            example = (*example, torch.zeros(1, int(kwargs["num_physics"])))
        elif uses_metadata:
            from callisto_trainer.core.metadata_features import META_VECTOR_LEN

            example = (*example, torch.zeros(1, META_VECTOR_LEN))

        with torch.no_grad():
            scripted = torch.jit.trace(model, example, strict=False)
        scripted.save(str(destination))
        return destination, None
    except Exception as exc:
        LOGGER.warning("TorchScript export failed: %r", exc)
        return None, str(exc)


def export_model_bundle(
    checkpoint_path: str | Path,
    task: str,
    destination_root: str | Path,
    snapshot_dir: str | Path | None = None,
    include_torchscript: bool = True,
    make_zip: bool = False,
    name: str | None = None,
) -> ExportedBundle:
    """Write a self-contained bundle for ``checkpoint_path``."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(
            f"{checkpoint_path} has no embedded config, so its input contract "
            "cannot be recorded. Re-train with this app to produce an exportable "
            "checkpoint."
        )

    snapshot_info: dict[str, Any] | None = None
    if snapshot_dir:
        snapshot_json = Path(snapshot_dir) / "snapshot.json"
        if snapshot_json.exists():
            snapshot_info = json.loads(snapshot_json.read_text(encoding="utf-8"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(destination_root) / (name or f"callisto_{task}_model_{stamp}")
    directory.mkdir(parents=True, exist_ok=True)

    card = build_model_card(task, checkpoint, config, snapshot_info, checkpoint_path)
    bundle = ExportedBundle(directory=directory, task=task)

    (directory / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    with (directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    torch.save(checkpoint["model_state"], directory / "weights.pt")
    shutil.copy2(checkpoint_path, directory / "checkpoint.pt")
    (directory / "predict.py").write_text(STANDALONE_SCRIPT, encoding="utf-8")
    (directory / "README.md").write_text(_readme(card), encoding="utf-8")
    bundle.files = [
        "model_card.json", "config.yaml", "weights.pt",
        "checkpoint.pt", "predict.py", "README.md",
    ]

    if include_torchscript:
        path, error = export_torchscript(checkpoint, config, task, directory / "model_scripted.pt")
        bundle.torchscript_path = path
        bundle.torchscript_error = error
        if path is not None:
            bundle.files.append("model_scripted.pt")

    if make_zip:
        archive = shutil.make_archive(str(directory), "zip", root_dir=directory)
        bundle.zip_path = Path(archive)

    LOGGER.info("Exported %s model: %s", task, bundle.summary())
    return bundle
