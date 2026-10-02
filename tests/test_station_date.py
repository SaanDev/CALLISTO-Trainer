"""Station and date reach the unified model as a bounded correction.

The operator's rule: the model should take the station and the observation
month/year into account, but its answer must not rest on them. The correction is
capped so no log-odds -- burst vs not, or one type vs another -- moves by more
than ``cap``; with station and date unknown it is exactly off; and the same
station/date vector is built in training, calibration and prediction.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from callisto_trainer.core.config import load_config
from callisto_trainer.core.coords import SpectrumAxes
from callisto_trainer.core.metadata_features import (
    STATION_DATE_LEN,
    StationDateEncoder,
    fractional_year,
)
from callisto_trainer.core.region_features import feature_count
from callisto_trainer.core.taxonomy import NO_BURST, RFI, TYPE_II, TYPE_III

CLASSES = [NO_BURST, RFI, TYPE_II, TYPE_III]
N_FEATURES = feature_count("region_v2")
SHAPE = (200, 3600)
AXES = SpectrumAxes(time_s=np.arange(SHAPE[1]) * 0.25, freq_mhz=np.linspace(80.0, 20.0, SHAPE[0]))


def _rows(station: str, files: int, date: str = "2025-06-14") -> list[dict[str, str]]:
    return [{"file_path": f"{station}_{i}.fit.gz", "station": station, "date": date}
            for i in range(files)]


# -- encoding ---------------------------------------------------------------


def test_stations_need_enough_training_files_for_their_own_index() -> None:
    rows = _rows("BIR", 12) + _rows("glasgow", 10) + _rows("RARE", 3)
    # Several regions from one file count once.
    rows += [{"file_path": "RARE_0.fit.gz", "station": "RARE", "date": "2025-06-14"}] * 20
    encoder = StationDateEncoder.fit(rows, min_station_files=10)

    assert set(encoder.vocab) == {"BIR", "GLASGOW"}
    assert encoder.station_index("Glasgow ") == encoder.vocab["GLASGOW"]
    assert encoder.station_index("RARE") == 0, "rare stations share the unknown slot"
    assert encoder.station_index("NEVER-SEEN") == 0
    assert encoder.num_stations == 3


def test_date_is_month_on_a_cycle_and_a_clamped_year() -> None:
    rows = _rows("BIR", 10, "2025-01-03") + _rows("ALMATY", 10, "2026-09-20")
    encoder = StationDateEncoder.fit(rows)
    assert encoder.year_min == pytest.approx(fractional_year("2025-01-03")[0])
    assert encoder.year_max == pytest.approx(fractional_year("2026-09-20")[0])

    first = encoder.vector("BIR", "2025-01-03")
    last = encoder.vector("BIR", "2026-09-20")
    assert first[3] == pytest.approx(-1.0) and last[3] == pytest.approx(1.0)
    assert first[4] == 1.0

    # After the trained span: treated like its last month, not extrapolated.
    assert encoder.vector("BIR", "2031-09-01")[3] == pytest.approx(1.0)
    assert encoder.vector("BIR", "2019-09-01")[3] == pytest.approx(-1.0)

    # December sits next to January on the cycle; June is opposite.
    december, january, june = (encoder.vector("BIR", f"2025-{m}-01")[1:3] for m in ("12", "01", "06"))
    assert np.linalg.norm(december - january) < np.linalg.norm(december - june)


def test_unknown_station_and_date_encode_as_all_zeros() -> None:
    encoder = StationDateEncoder.fit(_rows("BIR", 10))
    for date in (None, "", "not a date", "2025-13-01"):
        vector = encoder.vector("SOMEWHERE-ELSE", date)
        assert vector.shape == (STATION_DATE_LEN,)
        assert not vector.any(), date


def test_encoder_round_trips_through_the_config() -> None:
    encoder = StationDateEncoder.fit(_rows("BIR", 10, "2025-03-01") + _rows("SSRT", 10, "2026-02-01"))
    again = StationDateEncoder.from_config({"enabled": True, **encoder.to_config()})
    for station, date in (("BIR", "2025-07-09"), ("ssrt", "2026-01-01"), ("X", None)):
        assert np.allclose(again.vector(station, date), encoder.vector(station, date), atol=1e-4)


# -- the correction ---------------------------------------------------------


def _model(cap: float = 1.0, num_stations: int = 4, **hide):
    from callisto_trainer.core.models.model_factory import create_model

    station_date = {"num_stations": num_stations, "cap": cap,
                    "drop_all": 0.0, "drop_station": 0.0, "drop_date": 0.0, **hide}
    return create_model(
        "simple_cnn", in_channels=1, num_classes=len(CLASSES), use_physics=True,
        num_physics=N_FEATURES, num_views=2, station_date=station_date,
    )


def _inputs(batch: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(batch, 2, 64, 64, generator=generator)
    region = torch.randn(batch, N_FEATURES, generator=generator)
    meta = torch.zeros(batch, STATION_DATE_LEN)
    meta[:, 0] = torch.randint(1, 4, (batch,), generator=generator).float()
    angle = torch.rand(batch, generator=generator) * 2 * math.pi
    meta[:, 1], meta[:, 2] = torch.sin(angle), torch.cos(angle)
    meta[:, 3] = torch.rand(batch, generator=generator) * 2 - 1
    meta[:, 4] = 1.0
    return image, region, meta


def _scramble(model: torch.nn.Module, scale: float = 25.0) -> None:
    """Give the correction large random weights: the worst case for the cap."""
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in model.station_date.parameters():
            parameter.copy_(torch.randn_like(parameter) * scale)


def test_correction_starts_at_zero() -> None:
    model = _model().eval()
    image, region, meta = _inputs()
    with torch.no_grad():
        with_meta = model(image, torch.cat([region, meta], dim=1))
        without = model(image, torch.cat([region, torch.zeros_like(meta)], dim=1))
    assert torch.allclose(with_meta, without), "training starts from the image model"


@pytest.mark.parametrize("cap", [0.5, 1.0, 2.0])
def test_no_log_odds_moves_by_more_than_the_cap(cap: float) -> None:
    model = _model(cap=cap).eval()
    _scramble(model)
    image, region, meta = _inputs(batch=64)
    with torch.no_grad():
        with_meta = model(image, torch.cat([region, meta], dim=1))
        without = model(image, torch.cat([region, torch.zeros_like(meta)], dim=1))

    shift = with_meta - without
    assert shift.abs().max() <= cap / 2 + 1e-5
    assert shift.abs().max() > cap / 4, "scrambled weights should push toward the cap"

    # Burst vs not a burst, on log-odds -- the decision Predict thresholds.
    burst = [CLASSES.index(TYPE_II), CLASSES.index(TYPE_III)]
    reject = [CLASSES.index(NO_BURST), CLASSES.index(RFI)]

    def log_odds(logits):
        return torch.logsumexp(logits[:, burst], 1) - torch.logsumexp(logits[:, reject], 1)

    assert (log_odds(with_meta) - log_odds(without)).abs().max() <= cap + 1e-5
    # And between any two types.
    pairwise = shift[:, :, None] - shift[:, None, :]
    assert pairwise.abs().max() <= cap + 1e-5


def test_unknown_station_and_date_turn_the_correction_off_exactly() -> None:
    model = _model().eval()
    _scramble(model)
    image, region, meta = _inputs()
    with torch.no_grad():
        image_only = model.classifier(
            torch.cat([_embed(model, image), model.feature_mlp(region)], dim=1)
        )
        unknown = model(image, torch.cat([region, torch.zeros_like(meta)], dim=1))
        station_only = meta.clone()
        station_only[:, 1:] = 0.0
        known_station = model(image, torch.cat([region, station_only], dim=1))
    assert torch.allclose(unknown, image_only, atol=1e-6)
    assert not torch.allclose(known_station, image_only), "a known station alone still counts"


def _embed(model, image):
    batch = image.shape[0]
    views = image.reshape(batch * model.num_views, 1, *image.shape[-2:])
    return model.backbone(views).flatten(1).reshape(batch, -1)


def test_training_hides_station_and_date() -> None:
    model = _model(drop_all=1.0).train()
    _scramble(model)
    image, region, meta = _inputs()
    model.eval()
    with torch.no_grad():
        image_only = model(image, torch.cat([region, torch.zeros_like(meta)], dim=1))
    # Only the correction in training mode, so dropout and batch norm elsewhere
    # stay out of the comparison.
    model.station_date.training = True
    with torch.no_grad():
        hidden = model(image, torch.cat([region, meta], dim=1))
    assert torch.allclose(hidden, image_only, atol=1e-6)


def test_old_checkpoints_keep_their_parameter_names() -> None:
    from callisto_trainer.core.models.model_factory import create_model

    plain = create_model("simple_cnn", in_channels=1, num_classes=4, use_physics=True,
                         num_physics=N_FEATURES, num_views=2)
    assert not any(key.startswith("station_date") for key in plain.state_dict())
    with_correction = _model()
    assert set(plain.state_dict()) < set(with_correction.state_dict())


@pytest.mark.parametrize("name", ["resnet34", "resnet50", "convnext_tiny"])
def test_new_backbones_build_with_the_correction(name: str) -> None:
    from callisto_trainer.core.models.model_factory import create_model

    model = create_model(
        name, in_channels=1, num_classes=len(CLASSES), pretrained=False, use_physics=True,
        num_physics=N_FEATURES, num_views=2,
        station_date={"num_stations": 3, "cap": 1.0},
    ).eval()
    image, region, meta = _inputs(batch=2)
    with torch.no_grad():
        logits = model(image, torch.cat([region, meta], dim=1))
    assert logits.shape == (2, len(CLASSES))


# -- training data ----------------------------------------------------------


def _manifest(tmp_path: Path) -> Path:
    rows = []
    for index in range(24):
        station = "BIR" if index < 12 else "GLASGOW"
        processed = tmp_path / f"s{index}.npz"
        np.savez(processed, spectrum=np.zeros((2, 8, 8), np.float32),
                 features=np.full(N_FEATURES, index, np.float32))
        rows.append({
            "file_path": f"{station}_{index}.fit.gz", "processed_path": str(processed),
            "label": NO_BURST, "label_id": 0, "station": station,
            "date": "2025-0{}-01".format(1 + index % 9),
            "split": "train" if index % 4 else "val",
        })
    rows.append({**rows[1], "split": "test"})
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dataloaders_fit_on_train_and_append_the_vector(tmp_path: Path) -> None:
    from callisto_trainer.core.dataset import get_dataloaders

    config = load_config()
    config["paths"]["manifest_path"] = str(_manifest(tmp_path))
    config["training"].update({"batch_size": 4, "num_workers": 0})
    config["augmentation"]["enabled"] = False
    config["model"].update({"use_metadata": False, "use_physics": True,
                            "feature_set": "region_v2",
                            "station_date": {"enabled": True, "min_station_files": 9}})
    loaders = get_dataloaders(config)

    section = config["model"]["station_date"]
    # BIR has 9 training files and GLASGOW 9 (every fourth goes to val).
    assert set(section["station_vocab"]) == {"BIR", "GLASGOW"}
    _, _, features = next(iter(loaders["val"]))
    assert features.shape[1] == N_FEATURES + STATION_DATE_LEN
    assert (features[:, N_FEATURES] > 0).all() and (features[:, -1] == 1).all()


# -- prediction -------------------------------------------------------------


def _checkpoint(tmp_path: Path, scramble: bool = True) -> Path:
    config = load_config()
    config["data"]["classes"] = {name: index for index, name in enumerate(CLASSES)}
    encoder = StationDateEncoder.fit(_rows("BIR", 10, "2025-01-01") + _rows("SSRT", 10, "2026-06-01"))
    config["model"].update(
        {"name": "simple_cnn", "in_channels": 1, "num_classes": len(CLASSES),
         "use_metadata": False, "use_physics": True, "views": ["crop", "context"],
         "feature_set": "region_v2",
         "station_date": {"enabled": True, "cap": 1.0, **encoder.to_config()}}
    )
    config["inference"] = {"burst_threshold": 0.5}
    model = _model(num_stations=encoder.num_stations)
    if scramble:
        _scramble(model, scale=5.0)
    path = tmp_path / "unified_station_date.pt"
    torch.save({"epoch": 1, "model_state": model.state_dict(), "config": config}, path)
    return path


def _spectrum() -> np.ndarray:
    rng = np.random.default_rng(0)
    array = np.clip(rng.normal(0.12, 0.03, SHAPE), 0.0, 1.0).astype(np.float32)
    for row in range(10, 150):
        col = int(400 + (row - 10) * 0.08)
        array[row, col:col + 6] = 0.8
    return array


def test_prediction_uses_the_files_station_and_date(tmp_path: Path) -> None:
    from callisto_trainer.core.inference import CascadePredictor, FileResult

    predictor = CascadePredictor(load_config(), unified_checkpoint=_checkpoint(tmp_path))
    assert predictor.encoder.spec.station_date is not None

    known =predictor.examine(_spectrum(), AXES, file_meta={"station": "BIR", "date": "2025-03-02"})
    unknown = predictor.examine(_spectrum(), AXES, file_meta=None)
    assert known and len(known) == len(unknown)
    for a, b in zip(known, unknown):
        # Station/date hidden is exactly the model with them unknown...
        assert a.image_only_evidence == pytest.approx(b.burst_evidence, abs=1e-5)
        assert b.image_only_evidence == pytest.approx(b.burst_evidence, abs=1e-5)
    # ...and a known station and date did move something.
    assert any(abs(a.burst_evidence - b.burst_evidence) > 1e-4 for a, b in zip(known, unknown))

    # predict_normalized takes them from the FITS metadata it is given.
    from_metadata = predictor.predict_normalized(
        _spectrum(), AXES, FileResult("x.fit.gz", "x.fit.gz"),
        {"station": "BIR", "date": "2025-03-02"},
    )
    assert from_metadata.burst_probability == pytest.approx(
        max(region.burst_evidence for region in known), abs=1e-6
    )


def test_file_report_lists_the_verdicts_station_and_date_changed() -> None:
    from callisto_trainer.core.file_eval import FileScore, file_level_report

    def score(name, is_burst, with_meta, without):
        s = FileScore(name, name, "BIR", is_burst)
        s.file_score = s.on_burst_score = with_meta
        s.image_only_file_score = s.image_only_on_burst_score = without
        return s

    scores = [
        score("added.fit.gz", False, 0.7, 0.4),     # flagged only because of station/date
        score("removed.fit.gz", False, 0.4, 0.7),
        score("steady.fit.gz", False, 0.2, 0.25),
        score("found.fit.gz", True, 0.6, 0.45),
        score("lost.fit.gz", True, 0.45, 0.6),
    ]
    effect = file_level_report(scores, 0.5)["station_date_effect"]
    assert effect["verdicts_changed"] == 4
    assert [e["file_path"] for e in effect["false_alarms_added"]] == ["added.fit.gz"]
    assert [e["file_path"] for e in effect["false_alarms_removed"]] == ["removed.fit.gz"]
    assert [e["file_path"] for e in effect["bursts_found_only_with_it"]] == ["found.fit.gz"]
    assert [e["file_path"] for e in effect["bursts_lost_to_it"]] == ["lost.fit.gz"]
    assert effect["image_only_false_alarms"] == 1 and effect["image_only_bursts_detected"] == 1


def test_file_report_has_no_effect_section_without_the_correction() -> None:
    from callisto_trainer.core.file_eval import FileScore, file_level_report

    s = FileScore("a", "a", "BIR", False)
    assert "station_date_effect" not in file_level_report([s], 0.5)


def test_bundle_documents_and_traces_the_station_date_input(tmp_path: Path) -> None:
    from callisto_trainer.services.model_export import export_model_bundle

    bundle = export_model_bundle(_checkpoint(tmp_path), "unified", tmp_path / "exports")
    import json

    card = json.loads((bundle.directory / "model_card.json").read_text(encoding="utf-8"))
    assert card["physics_input"]["shape"] == [N_FEATURES + STATION_DATE_LEN]
    assert card["physics_input"]["feature_order"][-STATION_DATE_LEN:] == [
        "station_index", "month_sin", "month_cos", "year", "date_known"
    ]
    assert set(card["station_date_input"]["station_vocab"]) == {"BIR", "SSRT"}
    assert card["station_date_input"]["max_log_odds_shift"] == 1.0

    assert bundle.torchscript_path is not None, bundle.torchscript_error
    scripted = torch.jit.load(str(bundle.torchscript_path), map_location="cpu").eval()
    _, region, meta = _inputs(batch=1)
    image = torch.rand(1, 2, 224, 224)
    with torch.no_grad():
        traced = scripted(image, torch.cat([region, meta], dim=1))
    assert traced.shape == (1, len(CLASSES))


class _StubPredictor:
    """Stands in for CascadePredictor in score_files: fixed regions per file."""

    def __init__(self, regions_by_file, station_date: bool) -> None:
        from types import SimpleNamespace

        self.config = load_config()
        encoder = StationDateEncoder.fit(_rows("BIR", 10)) if station_date else None
        self.encoder = SimpleNamespace(spec=SimpleNamespace(station_date=encoder))
        self.regions_by_file = regions_by_file
        self.seen_meta = []

    def quiet_for(self, spectrum):
        return None

    def examine(self, normalized, axes, rfi_channels=None, quiet=None, file_meta=None):
        self.seen_meta.append(file_meta)
        return self.regions_by_file.pop(0)


def _score(monkeypatch, regions_by_file, truths, station_date):
    import callisto_trainer.core.crops as crops
    import callisto_trainer.core.fits_reader as fits_reader
    from callisto_trainer.core.file_eval import score_files

    metadata = {"time_axis_s": AXES.time_s, "freq_axis_mhz": AXES.freq_mhz,
                "station": "BIR", "date": "2025-04-01"}
    monkeypatch.setattr(fits_reader, "read_fits_spectrum_and_axes",
                        lambda path: (np.zeros(SHAPE, np.float32), metadata))
    monkeypatch.setattr(crops, "normalize_full_spectrum", lambda spectrum, config: spectrum)
    predictor = _StubPredictor(regions_by_file, station_date)
    return score_files(predictor, truths), predictor


def _region(row0, col0, evidence, image_only=None):
    from callisto_trainer.core.inference import RegionResult

    return RegionResult(row0=row0, row1=row0 + 40, col0=col0, col1=col0 + 200, area=1, peak=1.0,
                        burst_evidence=evidence, image_only_evidence=image_only)


@pytest.mark.parametrize("station_date", [False, True])
def test_strays_are_only_the_regions_off_the_drawn_bursts(monkeypatch, station_date) -> None:
    """Regression: the image-only bookkeeping once swallowed the stray branch."""
    from callisto_trainer.core.file_eval import FileTruth

    truth = FileTruth("b.fit.gz", "b.fit.gz", "BIR", "burst", "test",
                      burst_boxes=[(0, 50, 0, 300, TYPE_III)])
    on = _region(0, 0, 0.9, 0.8 if station_date else None)
    off = _region(120, 2000, 0.6, 0.7 if station_date else None)
    (score,), predictor = _score(monkeypatch, [[on, off]], [truth], station_date)

    assert score.stray_scores == [pytest.approx(0.6)]
    assert score.on_burst_score == pytest.approx(0.9)
    assert predictor.seen_meta[0]["station"] == "BIR", "calibration passes the file's station"
    if station_date:
        assert score.image_only_on_burst_score == pytest.approx(0.8)
        assert score.image_only_file_score == pytest.approx(0.8)
    else:
        assert score.image_only_file_score is None


def test_a_file_with_no_candidates_is_scored_both_ways(monkeypatch) -> None:
    from callisto_trainer.core.file_eval import FileTruth, file_level_report

    truth = FileTruth("q.fit.gz", "q.fit.gz", "BIR", "no_burst", "test")
    (score,), _ = _score(monkeypatch, [[]], [truth], station_date=True)
    assert score.image_only_file_score == 0.0
    effect = file_level_report([score], 0.5)["station_date_effect"]
    assert effect["files"] == 1 and effect["image_only_false_alarms"] == 0
