"""Batch import of FITS files into the annotation store.

Files are referenced in place -- nothing is copied. The archive here is ~587k
files and copying it would cost hundreds of gigabytes for no benefit. A cheap
content hash is recorded so duplicate imports are detected even when the same
recording arrives under a different path, and so annotations can be re-linked if
the archive moves.

Qt-free by design: progress is reported through a callback, which the UI wraps in
a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from callisto_trainer.core.fits_reader import read_fits_metadata
from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.store.repository import AnnotationRepository, content_hash

LOGGER = get_logger(__name__)

# e-CALLISTO ships gzipped FITS; plain .fit/.fits are accepted for convenience.
FITS_PATTERNS = ("*.fit.gz", "*.fits.gz", "*.fit", "*.fits")


@dataclass
class ImportResult:
    """Outcome of one import run."""

    scanned: int = 0
    imported: int = 0
    duplicate_path: int = 0
    duplicate_content: int = 0
    failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return self.duplicate_path + self.duplicate_content

    def summary(self) -> str:
        parts = [f"{self.imported} imported"]
        if self.skipped:
            parts.append(f"{self.skipped} already present")
        if self.failed:
            parts.append(f"{self.failed} unreadable")
        return ", ".join(parts)


def find_fits_files(
    roots: Sequence[str | Path], recursive: bool = True
) -> list[Path]:
    """Collect FITS paths under the given files or directories, de-duplicated."""
    found: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        root = Path(root)
        if root.is_file():
            candidates: Iterable[Path] = [root]
        elif root.is_dir():
            candidates = (
                path
                for pattern in FITS_PATTERNS
                for path in (root.rglob(pattern) if recursive else root.glob(pattern))
            )
        else:
            continue

        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)

    return sorted(found)


def import_files(
    repository: AnnotationRepository,
    paths: Sequence[str | Path],
    read_headers: bool = True,
    detect_content_duplicates: bool = True,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> ImportResult:
    """Import ``paths`` into the store.

    ``progress(index, total, name)`` is called per file; returning ``False``
    cancels the run and keeps everything imported so far. Unreadable files are
    recorded with an ``error`` status rather than aborting the batch, so one
    corrupt file in a 10,000-file folder does not cost the whole import.
    """
    result = ImportResult()
    known_paths = repository.existing_paths()
    known_hashes = repository.existing_hashes() if detect_content_duplicates else set()
    total = len(paths)

    for index, raw_path in enumerate(paths):
        path = Path(raw_path).resolve()
        result.scanned += 1

        if progress is not None and progress(index, total, path.name) is False:
            LOGGER.info("Import cancelled after %d files", index)
            break

        if str(path) in known_paths:
            result.duplicate_path += 1
            continue

        digest: str | None = None
        if detect_content_duplicates:
            try:
                digest = content_hash(path)
            except OSError as exc:
                result.failed += 1
                result.errors.append((str(path), repr(exc)))
                continue
            if digest in known_hashes:
                result.duplicate_content += 1
                continue

        metadata: dict[str, Any] = {"content_hash": digest}
        if read_headers:
            try:
                metadata.update(read_fits_metadata(path))
            except Exception as exc:
                # Keep the row so the file is visible and reviewable in the queue
                # instead of silently vanishing from the import.
                metadata["error"] = repr(exc)
                result.failed += 1
                result.errors.append((str(path), repr(exc)))

        file_id = repository.add_file(path, metadata)
        if file_id is None:
            result.duplicate_path += 1
            continue

        known_paths.add(str(path))
        if digest:
            known_hashes.add(digest)
        if not metadata.get("error"):
            result.imported += 1

    LOGGER.info("Import complete: %s", result.summary())
    return result


def import_folders(
    repository: AnnotationRepository,
    roots: Sequence[str | Path],
    recursive: bool = True,
    read_headers: bool = True,
    progress: Callable[[int, int, str], bool | None] | None = None,
) -> ImportResult:
    """Convenience wrapper: scan folders, then import everything found."""
    return import_files(
        repository,
        find_fits_files(roots, recursive=recursive),
        read_headers=read_headers,
        progress=progress,
    )
