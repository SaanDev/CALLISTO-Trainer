"""Background decoding of spectra with look-ahead prefetch.

The labelling loop is navigation-bound: the operator presses Next and expects the
next spectrum immediately, but decoding it takes 100-400 ms. So whenever a file
is displayed, the next few files in the queue are decoded on a thread pool and
parked in the cache. By the time Next is pressed the array is already in memory.

Only this module and the UI package touch Qt; the actual loading work lives in
:mod:`callisto_trainer.services.cache`, which stays headless-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from callisto_trainer.core.logging_utils import get_logger
from callisto_trainer.services.cache import (
    DiskCache,
    SpectrumBundle,
    SpectrumCache,
    load_bundle,
    preprocessing_signature,
)

LOGGER = get_logger(__name__)

# How many files ahead of the cursor to decode. Four covers a fast operator
# without thrashing the disk or holding too many large arrays resident.
DEFAULT_PREFETCH = 4


@dataclass(frozen=True)
class LoadRequest:
    """One file to decode."""

    file_id: int
    path: str
    content_hash: str | None = None


class _LoadSignals(QObject):
    """QRunnable cannot own signals, so they live on this helper."""

    finished = Signal(object)  # SpectrumBundle
    failed = Signal(int, str)  # file_id, error


class _LoadTask(QRunnable):
    def __init__(
        self,
        request: LoadRequest,
        config: dict[str, Any],
        disk_cache: DiskCache | None,
        signature: str,
        signals: _LoadSignals,
    ) -> None:
        super().__init__()
        self.request = request
        self.config = config
        self.disk_cache = disk_cache
        self.signature = signature
        self.signals = signals
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        request = self.request
        try:
            key = f"{request.content_hash}_{self.signature}" if request.content_hash else None
            bundle = load_bundle(
                request.file_id,
                request.path,
                self.config,
                disk_cache=self.disk_cache,
                content_key=key,
            )
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %r", request.path, exc)
            self.signals.failed.emit(request.file_id, str(exc))
            return
        self.signals.finished.emit(bundle)


class SpectrumLoader(QObject):
    """Decodes spectra off the UI thread and serves them from cache.

    ``loaded`` fires for every successful decode, whether it was requested
    explicitly or prefetched. Callers check :meth:`cached` first and only wait on
    the signal when the array is not yet resident.
    """

    loaded = Signal(object)  # SpectrumBundle
    failed = Signal(int, str)  # file_id, error

    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: str | None = None,
        max_workers: int = 3,
        memory_budget_bytes: int = 512 * 1024**2,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.signature = preprocessing_signature(config)
        self.cache = SpectrumCache(max_bytes=memory_budget_bytes)
        self.disk_cache = DiskCache(cache_dir) if cache_dir else None

        self.pool = QThreadPool(self)
        # Decoding is gzip + astropy: CPU-bound but GIL-releasing in the hot
        # loop. A small pool keeps the UI responsive without starving it.
        self.pool.setMaxThreadCount(max(1, int(max_workers)))

        self._signals = _LoadSignals()
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._in_flight: set[int] = set()

    # -- public API --------------------------------------------------------

    def cached(self, file_id: int) -> SpectrumBundle | None:
        return self.cache.get(file_id)

    def request(self, request: LoadRequest, force: bool = False) -> SpectrumBundle | None:
        """Return the bundle if resident, otherwise schedule a decode.

        When ``None`` is returned the caller should show a loading state and wait
        for :attr:`loaded`.
        """
        if not force:
            bundle = self.cache.get(request.file_id)
            if bundle is not None:
                return bundle
        self._schedule(request)
        return None

    def prefetch(self, requests: list[LoadRequest]) -> None:
        """Warm the cache for upcoming files; already-resident ones are skipped."""
        for request in requests:
            if self.cache.contains(request.file_id):
                continue
            self._schedule(request)

    def invalidate(self, file_id: int) -> None:
        self.cache.discard(file_id)

    def shutdown(self) -> None:
        """Stop accepting work and wait for running decodes to finish."""
        self.pool.clear()
        self.pool.waitForDone(5000)
        self.cache.clear()

    # -- internals ---------------------------------------------------------

    def _schedule(self, request: LoadRequest) -> None:
        if request.file_id in self._in_flight:
            return
        self._in_flight.add(request.file_id)
        self.pool.start(
            _LoadTask(request, self.config, self.disk_cache, self.signature, self._signals)
        )

    @Slot(object)
    def _on_finished(self, bundle: SpectrumBundle) -> None:
        self._in_flight.discard(bundle.file_id)
        self.cache.put(bundle)
        self.loaded.emit(bundle)

    @Slot(int, str)
    def _on_failed(self, file_id: int, message: str) -> None:
        self._in_flight.discard(file_id)
        self.failed.emit(file_id, message)


def prefetch_window(
    file_ids: list[int], current_index: int, ahead: int = DEFAULT_PREFETCH, behind: int = 1
) -> list[int]:
    """Ids to keep warm around the cursor.

    Slightly asymmetric: mostly forward, because operators move forward through
    the queue, but one step back so an immediate correction is also instant.
    """
    if not file_ids:
        return []
    start = max(0, current_index - behind)
    end = min(len(file_ids), current_index + ahead + 1)
    return [file_ids[i] for i in range(start, end) if i != current_index]
