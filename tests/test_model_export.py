"""Exported model bundles: contents, contract, TorchScript, and standalone parity.

The bundle re-implements preprocessing in a dependency-free script. That is the
whole point (it runs without this project) and also the main risk: if the copy
drifts from the real pipeline, the exported model silently receives different
input than it was trained on. These tests pin the two together.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from callisto_trainer.core.config import load_config
from callisto_trainer.core.crops import (
    CropConfig,
    PixelBox,
    crop_from_normalized,
    normalize_full_spectrum,
)
from callisto_trainer.core.fits_reader import read_fits_spectrum
from callisto_trainer.core.models.model_factory import create_model
from callisto_trainer.core.preprocess import preprocess_array
from callisto_trainer.services.model_export import (
    BUNDLE_FORMAT_VERSION,
    build_model_card,
    export_model_bundle,
)


def _make_checkpoint(tmp_path: Path, task: str) -> Path:
    """A small but genuine checkpoint: real architecture, real config, real weights."""
    config = load_config()
    is_type = task == "type"
    config["data"]["classes"] = (
        {"Type II": 0, "Type III": 1, "Other": 2} if is_type else {"No_Burst": 0, "Burst": 1}
    )
    config["model"].update(
        {
            "name": "simple_cnn",
            "in_channels": 1,
            "num_classes": 3 if is_type else 1,
            "use_metadata": not is_type,
            "station_vocab": {} if is_type else {"ALASKA-ANCHORAGE": 1, "BIR": 2},
        }
    )
    config["training"]["threshold"] = 0.37

    kwargs = {}
    if not is_type:
        from callisto_trainer.core.metadata_features import NUM_NUMERIC

        kwargs = dict(
            use_metadata=True, num_stations=3, num_numeric=NUM_NUMERIC, station_emb_dim=8
        )
    model = create_model(
        "simple_cnn", in_channels=1, num_classes=3 if is_type else 1, **kwargs
    )

    path = tmp_path / f"{task}_best.pt"
    torch.save(
        {
            "epoch": 7,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "config": config,
            "metrics": {"val": {"accuracy": 0.87, "macro_f1": 0.83, "f1": 0.81}},
        },
        path,
    )
    return path


def _load_bundle_script(directory: Path):
    """Import the bundle's predict.py as a module, the way a user would run it."""
    spec = importlib.util.spec_from_file_location(
        f"bundle_predict_{directory.name}", directory / "predict.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unified_physics_checkpoint(tmp_path: Path) -> Path:
    """A 4-class checkpoint with the physics branch, as training now produces."""
    from callisto_trainer.core.burst_physics import NUM_PHYSICS_FEATURES

    config = load_config()
    config["data"]["classes"] = {"No_Burst": 0, "Type II": 1, "Type III": 2, "Other": 3}
    config["model"].update(
        {
            "name": "simple_cnn",
            "in_channels": 1,
            "num_classes": 4,
            "use_metadata": False,
            "use_physics": True,
        }
    )
    model = create_model(
        "simple_cnn", in_channels=1, num_classes=4,
        use_physics=True, num_physics=NUM_PHYSICS_FEATURES,
    )
    path = tmp_path / "unified_physics.pt"
    torch.save(
        {"epoch": 5, "model_state": model.state_dict(), "config": config,
         "metrics": {"val": {"macro_f1": 0.77}}},
        path,
    )
    return path


@pytest.fixture
def unified_bundle(tmp_path: Path):
    return export_model_bundle(
        _unified_physics_checkpoint(tmp_path), "unified", tmp_path / "exports"
    )


def test_unified_bundle_exports_with_working_torchscript(unified_bundle) -> None:
    """Regression: the unified task was building a 1-class head and a bare CNN."""
    assert unified_bundle.torchscript_path is not None, unified_bundle.torchscript_error

    scripted = torch.jit.load(str(unified_bundle.torchscript_path), map_location="cpu")
    scripted.eval()
    from callisto_trainer.core.burst_physics import NUM_PHYSICS_FEATURES

    with torch.no_grad():
        output = scripted(torch.rand(2, 1, 224, 224), torch.zeros(2, NUM_PHYSICS_FEATURES))
    assert output.shape == (2, 4), "the unified head must stay 4-class"


def test_unified_card_describes_a_softmax_not_a_threshold(unified_bundle) -> None:
    card = json.loads((unified_bundle.directory / "model_card.json").read_text(encoding="utf-8"))

    assert card["classes"] == ["No_Burst", "Type II", "Type III", "Other"]
    assert card["output"]["head"] == "softmax"
    assert "decision_threshold" not in card["output"], (
        "a 4-class softmax model has no decision threshold"
    )
    assert card["architecture"]["num_classes"] == 4
    assert "No_Burst" in card["output"]["interpretation"]


def test_unified_card_documents_the_physics_input(unified_bundle) -> None:
    card = json.loads((unified_bundle.directory / "model_card.json").read_text(encoding="utf-8"))
    from callisto_trainer.core.burst_physics import NUM_PHYSICS_FEATURES, PHYSICS_FEATURES

    spec = card["physics_input"]
    assert spec["required"] is True
    assert spec["shape"] == [NUM_PHYSICS_FEATURES]
    assert spec["feature_order"] == list(PHYSICS_FEATURES)
    assert "measured" in spec["note"]
    assert card["architecture"]["uses_physics"] is True


def test_unified_readme_documents_both_inputs(unified_bundle) -> None:
    readme = (unified_bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "two** inputs" in readme or "two* inputs" in readme or "takes **two**" in readme
    assert "signed_log_drift" in readme
    assert "out of distribution" in readme


def test_unified_standalone_script_supplies_the_physics_input(unified_bundle) -> None:
    """The script must pass the second input, and flag whole-file misuse."""
    source = (unified_bundle.directory / "predict.py").read_text(encoding="utf-8")
    assert "physics_input" in source
    assert "torch.zeros(1, int(physics_spec" in source
    assert "out of distribution" in source


@pytest.fixture
def type_bundle(tmp_path: Path):
    checkpoint = _make_checkpoint(tmp_path, "type")
    return export_model_bundle(checkpoint, "type", tmp_path / "exports")


@pytest.fixture
def binary_bundle(tmp_path: Path):
    checkpoint = _make_checkpoint(tmp_path, "binary")
    return export_model_bundle(checkpoint, "binary", tmp_path / "exports")


# -- contents --------------------------------------------------------------


def test_bundle_contains_everything_needed(type_bundle) -> None:
    present = {path.name for path in type_bundle.directory.iterdir()}
    assert {
        "model_card.json",
        "config.yaml",
        "weights.pt",
        "checkpoint.pt",
        "predict.py",
        "README.md",
        "model_scripted.pt",
    } <= present


def test_weights_reload_into_the_architecture(type_bundle) -> None:
    config = yaml.safe_load((type_bundle.directory / "config.yaml").read_text(encoding="utf-8"))
    model = create_model("simple_cnn", in_channels=1, num_classes=3)
    state = torch.load(type_bundle.directory / "weights.pt", map_location="cpu")
    model.load_state_dict(state)  # must not raise
    assert config["data"]["classes"] == {"Type II": 0, "Type III": 1, "Other": 2}


def test_zip_is_created_on_request(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, "type")
    bundle = export_model_bundle(checkpoint, "type", tmp_path / "exports", make_zip=True)

    assert bundle.zip_path is not None and bundle.zip_path.exists()
    with zipfile.ZipFile(bundle.zip_path) as archive:
        assert "model_card.json" in archive.namelist()


def test_checkpoint_without_a_config_is_refused(tmp_path: Path) -> None:
    """A checkpoint with no config cannot record its input contract."""
    path = tmp_path / "bare.pt"
    torch.save({"model_state": {}}, path)

    with pytest.raises(ValueError, match="input contract"):
        export_model_bundle(path, "type", tmp_path / "exports")


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_model_bundle(tmp_path / "nope.pt", "type", tmp_path / "exports")


# -- model card ------------------------------------------------------------


def test_card_records_the_input_contract(type_bundle) -> None:
    card = json.loads((type_bundle.directory / "model_card.json").read_text(encoding="utf-8"))

    assert card["bundle_format_version"] == BUNDLE_FORMAT_VERSION
    assert card["task"] == "type"
    assert card["classes"] == ["Type II", "Type III", "Other"]
    assert card["input"]["shape"] == [1, 224, 224]
    assert card["preprocessing"]["db_vmin"] == -1.0
    assert card["preprocessing"]["db_vmax"] == 8.0
    assert len(card["preprocessing"]["steps"]) == 4
    assert "before any cropping" in card["preprocessing"]["critical_note"]


def test_type_card_warns_it_is_not_for_whole_files(type_bundle) -> None:
    card = json.loads((type_bundle.directory / "model_card.json").read_text(encoding="utf-8"))

    assert "CROP" in card["scope"]["operates_on"]
    assert "out of distribution" in card["scope"]["warning"]
    assert card["scope"]["context_margin"] == pytest.approx(0.0)
    assert card["output"]["head"] == "softmax"


def test_binary_card_carries_the_tuned_threshold(binary_bundle) -> None:
    card = json.loads((binary_bundle.directory / "model_card.json").read_text(encoding="utf-8"))

    assert card["output"]["head"] == "sigmoid"
    assert card["output"]["decision_threshold"] == pytest.approx(0.37)
    assert "do not assume 0.5" in card["output"]["interpretation"].lower()
    assert card["architecture"]["uses_metadata"] is True
    assert card["metadata_features"]["order"][0] == "station_index"


def test_card_includes_training_data_provenance(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "kind": "types",
                "samples": 240,
                "class_counts": {"Type II": 80, "Type III": 80, "Other": 80},
                "split_counts": {"train": 168, "val": 36, "test": 36},
                "event_leakage": 0,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = _make_checkpoint(tmp_path, "type")
    bundle = export_model_bundle(checkpoint, "type", tmp_path / "exports", snapshot_dir=snapshot)

    card = json.loads((bundle.directory / "model_card.json").read_text(encoding="utf-8"))
    assert card["training_data"]["samples"] == 240
    assert card["training_data"]["event_leakage"] == 0


def test_readme_documents_scope_and_metrics(type_bundle) -> None:
    readme = (type_bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "out of distribution" in readme, "the crop-scope warning must be prominent"
    assert "context margin" in readme
    assert "macro_f1" in readme
    assert "predict.py" in readme


def test_binary_readme_documents_the_threshold(binary_bundle) -> None:
    readme = (binary_bundle.directory / "README.md").read_text(encoding="utf-8")
    assert "0.37" in readme
    assert "Do not assume 0.5" in readme


# -- TorchScript -----------------------------------------------------------


def test_torchscript_matches_the_eager_model(type_bundle) -> None:
    scripted = torch.jit.load(str(type_bundle.directory / "model_scripted.pt"), map_location="cpu")
    scripted.eval()

    model = create_model("simple_cnn", in_channels=1, num_classes=3)
    model.load_state_dict(torch.load(type_bundle.directory / "weights.pt", map_location="cpu"))
    model.eval()

    sample = torch.rand(2, 1, 224, 224)
    with torch.no_grad():
        assert torch.allclose(scripted(sample), model(sample), atol=1e-5)


def test_torchscript_handles_the_metadata_branch(binary_bundle) -> None:
    from callisto_trainer.core.metadata_features import META_VECTOR_LEN

    assert binary_bundle.torchscript_path is not None, binary_bundle.torchscript_error
    scripted = torch.jit.load(str(binary_bundle.torchscript_path), map_location="cpu")
    scripted.eval()

    with torch.no_grad():
        output = scripted(torch.rand(1, 1, 224, 224), torch.zeros(1, META_VECTOR_LEN))
    assert output.reshape(-1).shape[0] == 1


def test_export_survives_a_torchscript_failure(tmp_path: Path, monkeypatch) -> None:
    """A tracing failure must not cost the rest of the bundle."""
    import callisto_trainer.services.model_export as module

    monkeypatch.setattr(
        module.torch.jit if hasattr(module, "torch") else torch.jit,
        "trace",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tracing exploded")),
    )
    checkpoint = _make_checkpoint(tmp_path, "type")
    bundle = export_model_bundle(checkpoint, "type", tmp_path / "exports")

    assert bundle.torchscript_path is None
    assert "tracing exploded" in (bundle.torchscript_error or "")
    assert (bundle.directory / "model_card.json").exists()
    assert (bundle.directory / "weights.pt").exists()


# -- standalone script parity ---------------------------------------------


def test_standalone_normalize_is_identical_to_the_pipeline(
    type_bundle, any_real_file
) -> None:
    """The dependency-free copy must not have drifted from the real thing."""
    script = _load_bundle_script(type_bundle.directory)
    spectrum, _ = read_fits_spectrum(any_real_file)

    assert np.array_equal(
        script.normalize(spectrum), normalize_full_spectrum(spectrum, load_config())
    )


def test_standalone_whole_file_tensor_matches_preprocess_array(
    type_bundle, any_real_file
) -> None:
    script = _load_bundle_script(type_bundle.directory)
    config = load_config()
    spectrum, _ = read_fits_spectrum(any_real_file)

    produced = script.to_tensor(script.normalize(spectrum))
    expected = preprocess_array(spectrum, config)

    assert produced.shape == (1, 1, 224, 224)
    assert np.array_equal(produced[0], expected)


def test_standalone_crop_matches_the_trained_crop(type_bundle, any_real_file) -> None:
    """A box handed to the bundle must yield exactly the training-time tensor."""
    script = _load_bundle_script(type_bundle.directory)
    config = load_config()
    spectrum, _ = read_fits_spectrum(any_real_file)

    box = (20, 70, 300, 520)
    normalized = normalize_full_spectrum(spectrum, config)
    expected = crop_from_normalized(
        normalized, PixelBox(*box), CropConfig.from_config(config)
    )
    produced = script.to_tensor(script.normalize(spectrum), box=box)

    assert np.array_equal(produced[0], expected)


def test_standalone_reads_its_settings_from_the_card(type_bundle) -> None:
    script = _load_bundle_script(type_bundle.directory)
    assert script.DB_VMIN == -1.0
    assert script.DB_VMAX == 8.0
    assert script.TARGET == (224, 224)


def test_standalone_script_has_no_project_imports(type_bundle) -> None:
    source = (type_bundle.directory / "predict.py").read_text(encoding="utf-8")
    assert "callisto_trainer" not in source
    for module in ("numpy", "astropy", "torch"):
        assert module in source


def test_build_model_card_is_pure(tmp_path: Path) -> None:
    """Card construction must not depend on files existing on disk."""
    config = load_config()
    config["data"]["classes"] = {"Type II": 0, "Type III": 1, "Other": 2}
    card = build_model_card(
        "type",
        {"epoch": 3, "metrics": {}},
        config,
        None,
        Path("nowhere/best.pt"),
    )
    assert card["task"] == "type"
    assert card["trained_epoch"] == 3
    assert "training_data" not in card
