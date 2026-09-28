"""Progress channel between a training subprocess and the GUI.

Training runs in its own process (CUDA and Qt in one process is a reliable way to
produce mysterious hangs, and a crashed run must not take the window down). The
two communicate through the child's stdout: each epoch emits one line of JSON
behind a distinctive prefix, and the parent picks those lines out while passing
everything else through to the log view.

A prefix rather than a separate pipe keeps the trainers runnable by hand from a
terminal, where the extra lines are simply visible alongside the normal logging.
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROGRESS_PREFIX = "@@CT_PROGRESS@@ "


def emit_progress(record: dict[str, Any]) -> None:
    """Write one progress record to stdout. Never raises."""
    try:
        sys.stdout.write(PROGRESS_PREFIX + json.dumps(record, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        # Progress reporting must never be able to kill a training run.
        pass


def parse_progress(line: str) -> dict[str, Any] | None:
    """Return the record encoded in ``line``, or ``None`` for ordinary output."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX):])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
