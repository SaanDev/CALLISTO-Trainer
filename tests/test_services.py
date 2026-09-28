"""Import, caching and display decimation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from callisto_trainer.core.config import load_config
from callisto_trainer.services.cache import (
    DiskCache,
    SpectrumCache,
    decimate_for_display,
    load_bundle,
    preprocessing_signature,
)
from callisto_trainer.services.importer import find_fits_files, import_files, import_folders
from callisto_trainer.services.prep_worker import prefetch_window
from callisto_trainer.store.db import Database
from callisto_trainer.store.repository import AnnotationRepository


@pytest.fixture
def repo(tmp_path: Path) -> AnnotationRepository:
    return AnnotationRepository(Database(tmp_path / "annotations.db"))


@pytest.fixture
def config() -> dict:
    return load_config()


@pytest.fixture
def local_archive(tmp_path: Path, axes_files) -> Path:
    """Copy a few real files into a temp folder so imports do not touch H:."""
    archive = tmp_path / "archive" / "nested"
    archive.mkdir(parents=True)
    for source in axes_files[:3]:
        shutil.copy2(source, archive / source.name)
    return tmp_path / "archive"


# -- importer -------------------------------------------------------------


def test_find_fits_files_is_recursive_and_sorted(local_archive: Path) -> None:
    found = find_fits_files([local_archive])
    assert len(found) == 3
    assert all(p.name.endswith(".fit.gz") for p in found)
    assert found == sorted(found)


def test_find_fits_files_accepts_individual_files(local_archive: Path) -> None:
    one = next(local_archive.rglob("*.fit.gz"))
    assert find_fits_files([one]) == [one.resolve()]


def test_find_fits_files_ignores_missing_roots(tmp_path: Path) -> None:
    assert find_fits_files([tmp_path / "nope"]) == []


def test_import_reads_real_metadata(repo: AnnotationRepository, local_archive: Path) -> None:
    result = import_folders(repo, [local_archive])

    assert result.imported == 3
    assert result.failed == 0
    records = repo.files()
    assert len(records) == 3
    for record in records:
        assert record.station
        assert record.n_freq and record.n_time
        assert record.freq_axis_source == "axes_table"
        assert record.content_hash


def test_reimporting_the_same_folder_adds_nothing(
    repo: AnnotationRepository, local_archive: Path
) -> None:
    import_folders(repo, [local_archive])
    second = import_folders(repo, [local_archive])

    assert second.imported == 0
    assert second.duplicate_path == 3
    assert repo.total_files() == 3


def test_content_duplicates_are_detected_across_paths(
    repo: AnnotationRepository, local_archive: Path, tmp_path: Path
) -> None:
    """The same recording copied under a new name must not be imported twice."""
    import_folders(repo, [local_archive])

    elsewhere = tmp_path / "copy"
    elsewhere.mkdir()
    original = next(local_archive.rglob("*.fit.gz"))
    shutil.copy2(original, elsewhere / "renamed_copy.fit.gz")

    result = import_folders(repo, [elsewhere])
    assert result.imported == 0
    assert result.duplicate_content == 1
    assert repo.total_files() == 3


def test_unreadable_file_is_recorded_not_fatal(
    repo: AnnotationRepository, local_archive: Path, tmp_path: Path
) -> None:
    broken = local_archive / "broken.fit.gz"
    broken.write_bytes(b"this is not a FITS file at all")

    result = import_folders(repo, [local_archive])

    assert result.failed == 1
    assert result.imported == 3  # the good files still made it
    assert repo.total_files() == 4
    errored = [f for f in repo.files() if f.status == "error"]
    assert len(errored) == 1 and errored[0].file_name == "broken.fit.gz"


def test_import_progress_can_cancel(repo: AnnotationRepository, local_archive: Path) -> None:
    seen: list[str] = []

    def progress(index: int, total: int, name: str) -> bool:
        seen.append(name)
        return index < 1  # cancel once the second file comes up

    import_files(repo, find_fits_files([local_archive]), progress=progress)

    assert len(seen) == 2
    assert repo.total_files() == 1


def test_import_without_headers_is_still_usable(
    repo: AnnotationRepository, local_archive: Path
) -> None:
    result = import_files(repo, find_fits_files([local_archive]), read_headers=False)
    assert result.imported == 3
    assert all(record.n_freq is None for record in repo.files())


# -- caching --------------------------------------------------------------


def test_preprocessing_signature_tracks_the_db_window(config: dict) -> None:
    baseline = preprocessing_signature(config)
    assert preprocessing_signature(config) == baseline

    changed = load_config()
    changed["preprocessing"]["db_vmax"] = 12.0
    assert preprocessing_signature(changed) != baseline


def test_load_bundle_returns_axes_and_normalized_array(any_real_file, config: dict) -> None:
    bundle = load_bundle(1, any_real_file, config)

    assert bundle.normalized.ndim == 2
    assert bundle.normalized.dtype == np.float32
    assert 0.0 <= float(bundle.normalized.min())
    assert float(bundle.normalized.max()) <= 1.0
    assert bundle.axes.n_freq == bundle.normalized.shape[0]
    assert bundle.axes.n_time == bundle.normalized.shape[1]


def test_disk_cache_round_trip_and_reuse(any_real_file, config: dict, tmp_path: Path) -> None:
    disk = DiskCache(tmp_path / "cache")
    first = load_bundle(1, any_real_file, config, disk_cache=disk, content_key="abc")
    assert disk.size_bytes() > 0

    second = load_bundle(1, any_real_file, config, disk_cache=disk, content_key="abc")
    assert np.array_equal(first.normalized, second.normalized)


def test_disk_cache_survives_a_truncated_entry(tmp_path: Path) -> None:
    disk = DiskCache(tmp_path / "cache")
    disk.put("key", np.ones((4, 4), dtype=np.float32))
    (tmp_path / "cache" / "key.npy").write_bytes(b"truncated")

    assert disk.get("key") is None  # recovered, not raised
    assert not (tmp_path / "cache" / "key.npy").exists()


def test_disk_cache_prune_respects_budget(tmp_path: Path) -> None:
    disk = DiskCache(tmp_path / "cache", max_bytes=5000)
    for i in range(6):
        disk.put(f"key{i}", np.ones((10, 100), dtype=np.float32))  # ~4 KB each

    assert disk.size_bytes() > 5000
    assert disk.prune() > 0
    assert disk.size_bytes() <= 5000


def _bundle(file_id: int, cols: int):
    from callisto_trainer.core.coords import SpectrumAxes
    from callisto_trainer.services.cache import SpectrumBundle

    array = np.zeros((10, cols), dtype=np.float32)
    return SpectrumBundle(
        file_id=file_id,
        path=f"/tmp/{file_id}",
        normalized=array,
        axes=SpectrumAxes(np.arange(float(cols)), np.arange(10.0)),
        metadata={},
    )


def test_memory_cache_is_lru_and_byte_bounded() -> None:
    cache = SpectrumCache(max_bytes=10 * 10 * 4 * 3)  # room for ~3 bundles
    for i in range(3):
        cache.put(_bundle(i, 10))
    assert len(cache) == 3

    cache.get(0)  # touch 0 so 1 becomes least-recent
    cache.put(_bundle(3, 10))

    assert cache.contains(0)
    assert not cache.contains(1)
    assert cache.used_bytes <= cache.max_bytes


def test_oversized_bundle_is_still_kept() -> None:
    """One huge file must not evict itself into a permanent reload loop."""
    cache = SpectrumCache(max_bytes=16)
    cache.put(_bundle(1, 10_000))
    assert cache.contains(1)
    assert len(cache) == 1


# -- display decimation ---------------------------------------------------


def test_decimation_is_a_no_op_when_narrow_enough() -> None:
    array = np.zeros((10, 500), dtype=np.float32)
    out, factor = decimate_for_display(array, max_cols=2000)
    assert factor == 1
    assert out is array


def test_decimation_preserves_thin_bright_features() -> None:
    """A two-column burst must survive 40,000 -> 2,000 column reduction."""
    array = np.zeros((10, 39600), dtype=np.float32)
    array[:, 20000:20002] = 1.0

    pooled, factor = decimate_for_display(array, max_cols=2000)

    assert factor > 1
    assert pooled.shape[1] <= 2001
    assert float(pooled.max()) == pytest.approx(1.0), "max-pooling must keep the spike"
    # Mean pooling would have diluted it to ~1/factor; confirm that is not what happened.
    assert float(pooled.max()) > 1.0 / factor


def test_decimation_keeps_the_ragged_tail() -> None:
    array = np.zeros((4, 1005), dtype=np.float32)
    array[:, 1004] = 1.0  # in the remainder past the last full block

    pooled, _ = decimate_for_display(array, max_cols=100)
    assert float(pooled[:, -1].max()) == pytest.approx(1.0)


def test_decimation_on_a_real_wide_file(any_real_file, config: dict) -> None:
    bundle = load_bundle(1, any_real_file, config)
    pooled, factor = decimate_for_display(bundle.normalized, max_cols=1200)

    assert pooled.shape[0] == bundle.normalized.shape[0]
    assert pooled.shape[1] <= 1201
    assert pooled.dtype == np.float32
    if factor > 1:
        assert float(pooled.max()) == pytest.approx(float(bundle.normalized.max()), abs=1e-6)


# -- prefetch window ------------------------------------------------------


def test_prefetch_window_looks_mostly_forward() -> None:
    ids = list(range(100))
    assert prefetch_window(ids, 10, ahead=4, behind=1) == [9, 11, 12, 13, 14]


def test_prefetch_window_clamps_at_the_edges() -> None:
    ids = list(range(5))
    assert prefetch_window(ids, 0, ahead=4, behind=1) == [1, 2, 3, 4]
    assert prefetch_window(ids, 4, ahead=4, behind=1) == [3]
    assert prefetch_window([], 0) == []
