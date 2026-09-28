"""The quiet-background view: long continua must survive preprocessing.

The median background makes a continuum that lasts most of a file its own
background. These tests pin that the quiet-part background keeps it bright, that
every place the unified model is fed builds the third view the same way, and
that nothing silently runs a three-view model without it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from callisto_trainer.core.config import load_config  # noqa: E402
from callisto_trainer.core.coords import SpectrumAxes  # noqa: E402
from callisto_trainer.core.crops import (  # noqa: E402
    VIEW_QUIET_CONTEXT,
    CropConfig,
    PixelBox,
    crop_context,
    normalize_full_spectrum,
    quiet_normalized_spectrum,
    region_views,
)
from callisto_trainer.core.region_features import feature_count  # noqa: E402
from callisto_trainer.core.region_inputs import (  # noqa: E402
    V2_FEATURE_SET,
    V3_VIEWS,
    RegionEncoder,
    RegionInputSpec,
)

SHAPE = (120, 2000)
CONTINUUM = (slice(20, 80), slice(0, 1400))   # 60 channels for 70% of the file


def _raw_with_continuum(seed: int = 0) -> np.ndarray:
    """Receiver digits: a flat background plus a continuum for 70% of the time."""
    rng = np.random.default_rng(seed)
    raw = 100.0 + rng.normal(0.0, 1.0, SHAPE)
    raw[CONTINUUM] += 15.0          # about +5.8 dB on the Plotutil scale
    return raw.astype(np.float32)


def test_the_median_background_flattens_a_long_continuum() -> None:
    raw = _raw_with_continuum()
    median_view = normalize_full_spectrum(raw, load_config())
    quiet_view = quiet_normalized_spectrum(raw, load_config())

    assert float(median_view[CONTINUUM].mean()) < 0.2, "sanity: the problem being fixed"
    assert float(quiet_view[CONTINUUM].mean()) > 0.6, "the continuum stays bright"
    assert float(quiet_view[:, 1500:].mean()) < 0.25, "and the quiet part stays dark"
    assert quiet_view.dtype == np.float32 and 0.0 <= quiet_view.min() <= quiet_view.max() <= 1.0


def test_a_short_burst_looks_the_same_in_both_views() -> None:
    rng = np.random.default_rng(1)
    raw = (100.0 + rng.normal(0.0, 1.0, SHAPE)).astype(np.float32)
    raw[30:70, 500:520] += 15.0
    config = load_config()
    median_view = normalize_full_spectrum(raw, config)
    quiet_view = quiet_normalized_spectrum(raw, config)
    burst = (slice(30, 70), slice(500, 520))
    assert abs(float(quiet_view[burst].mean()) - float(median_view[burst].mean())) < 0.1


def test_the_quiet_view_is_the_context_of_the_quiet_array() -> None:
    raw = _raw_with_continuum()
    config = load_config()
    normalized = normalize_full_spectrum(raw, config)
    quiet = quiet_normalized_spectrum(raw, config)
    box = PixelBox(30, 60, 600, 700)
    crop_cfg = CropConfig.from_config(config)

    views = region_views(normalized, box, crop_cfg, V3_VIEWS, quiet=quiet)
    assert views.shape == (3, 224, 224)
    assert np.array_equal(views[2], crop_context(quiet, box, crop_cfg)[0])
    with pytest.raises(ValueError, match="quiet"):
        region_views(normalized, box, crop_cfg, (VIEW_QUIET_CONTEXT,))


def test_the_encoder_builds_the_quiet_array_only_when_it_needs_it() -> None:
    config = load_config()
    raw = _raw_with_continuum()
    v3 = RegionEncoder(config, RegionInputSpec(views=V3_VIEWS, feature_set=V2_FEATURE_SET))
    v2 = RegionEncoder(config, RegionInputSpec(views=V3_VIEWS[:2], feature_set=V2_FEATURE_SET))
    assert v3.spec.needs_quiet and not v2.spec.needs_quiet
    assert v2.quiet(raw) is None
    quiet = v3.quiet(raw)
    assert np.array_equal(quiet, quiet_normalized_spectrum(raw, config))

    normalized = normalize_full_spectrum(raw, config)
    encoded = v3.encode(normalized, PixelBox(30, 60, 600, 700), None, None, quiet=quiet)
    assert encoded.image.shape == (3, 224, 224)
    assert encoded.features.shape == (feature_count(V2_FEATURE_SET),)


# -- a three-view model end to end ----------------------------------------------


def _v3_checkpoint(tmp_path: Path) -> Path:
    from callisto_trainer.core.models.model_factory import create_model
    from callisto_trainer.store.export import UNIFIED_CLASSES

    config = load_config()
    config["data"]["classes"] = dict(UNIFIED_CLASSES)
    config["model"].update(
        {"name": "simple_cnn", "in_channels": 1, "num_classes": len(UNIFIED_CLASSES),
         "use_metadata": False, "use_physics": True,
         "views": list(V3_VIEWS), "feature_set": V2_FEATURE_SET}
    )
    config["inference"] = {"burst_threshold": 0.5}
    model = create_model(
        "simple_cnn", in_channels=1, num_classes=len(UNIFIED_CLASSES), use_physics=True,
        num_physics=feature_count(V2_FEATURE_SET), num_views=3,
    )
    path = tmp_path / "unified_v3.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


def test_a_three_view_model_predicts_a_file(tmp_path: Path, any_real_file) -> None:
    from callisto_trainer.core.fits_reader import read_fits_spectrum_and_axes
    from callisto_trainer.core.inference import CascadePredictor, FileResult

    predictor = CascadePredictor(load_config(), unified_checkpoint=_v3_checkpoint(tmp_path))
    assert predictor.needs_quiet
    result = predictor.predict_file(any_real_file)
    assert result.error is None and result.regions_examined > 0

    spectrum, metadata = read_fits_spectrum_and_axes(any_real_file)
    normalized = normalize_full_spectrum(spectrum, load_config())
    with pytest.raises(ValueError, match="quiet"):
        predictor.predict_normalized(
            normalized, SpectrumAxes.from_metadata(metadata), FileResult("x", "x"), metadata
        )


def test_the_bundle_builds_the_quiet_view_as_training_does(tmp_path: Path, any_real_file) -> None:
    import importlib.util
    import sys

    from callisto_trainer.core.fits_reader import read_fits_spectrum
    from callisto_trainer.services.model_export import export_model_bundle

    bundle = export_model_bundle(_v3_checkpoint(tmp_path), "unified", tmp_path / "exports")
    card = json.loads((bundle.directory / "model_card.json").read_text(encoding="utf-8"))
    assert card["input"]["shape"] == [3, 224, 224]
    assert "quiet_context" in card["scope"]["views"]

    spec = importlib.util.spec_from_file_location("bundle_v3", bundle.directory / "predict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = load_config()
    spectrum, _ = read_fits_spectrum(any_real_file)
    quiet = quiet_normalized_spectrum(spectrum, config)
    assert np.array_equal(module.normalize_quiet(spectrum), quiet)
    box = (10, 60, 200, 320)
    tensor = module.to_tensor(module.normalize(spectrum), box, quiet=module.normalize_quiet(spectrum))
    assert tensor.shape == (1, 3, 224, 224)
    assert np.array_equal(
        tensor[0, 2], crop_context(quiet, PixelBox(*box), CropConfig.from_config(config))[0]
    )
    with pytest.raises(ValueError, match="quiet"):
        module.to_tensor(module.normalize(spectrum), box)


# -- labelling -------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_label_tab_shows_the_quiet_background(qapp, tmp_path: Path, axes_files) -> None:
    from callisto_trainer.services.importer import import_files
    from callisto_trainer.settings import AppSettings
    from callisto_trainer.ui.main_window import MainWindow

    settings = AppSettings(
        project_root=tmp_path,
        database_path=tmp_path / "data" / "annotations.db",
        display_cache_dir=tmp_path / "data" / "cache",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
    )
    window = MainWindow(settings)
    try:
        import_files(window.repository, [Path(p) for p in axes_files[:1]])
        tab = window.label_tab
        tab.refresh_queue(keep_selection=False)
        tab.queue.select_row(0)
        deadline = time.monotonic() + 5
        while tab._current_bundle is None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        bundle = tab._current_bundle
        assert bundle is not None and bundle.quiet is not None

        modes = []
        for _ in range(tab.view_mode.count()):
            tab._toggle_view_mode()
            modes.append(tab.canvas.view_mode)
        assert sorted(modes) == ["normalized", "quiet", "raw"], "V cycles through every view"
        tab.view_mode.setCurrentIndex(tab.view_mode.findData("quiet"))
        assert tab.canvas._display_array() is bundle.quiet

        tab._on_box_created(20, 70, 300, 520)
        box = window.repository.boxes_for_file(tab._current_file_id)[0]
        tab.panel.select_box(box.id)
        expected = crop_context(bundle.quiet, PixelBox(20, 70, 300, 520), tab.crop_config)
        assert np.array_equal(tab.panel.quiet_preview.image.image, expected[0])
    finally:
        window.label_tab.shutdown()
        window.close()
