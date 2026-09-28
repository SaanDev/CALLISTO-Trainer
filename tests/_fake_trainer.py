"""A stand-in trainer used by tests to exercise the subprocess/progress path.

Emits the same event shapes as the real trainers plus some ordinary log output,
so the runner's line splitting and progress filtering are tested for real rather
than against a mock.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from callisto_trainer.core.progress import emit_progress  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument(
        "--report-env",
        action="store_true",
        help="Emit the inherited environment so the runner's setup can be checked.",
    )
    args = parser.parse_args()

    if args.report_env:
        import getpass
        import os

        # getpass.getuser() is what torch calls when picking its cache directory;
        # on Windows it needs USERNAME or it falls through to the Unix-only pwd
        # module. Reporting it here pins the runner's environment handling.
        try:
            user = getpass.getuser()
        except Exception as exc:
            user = f"FAILED: {exc!r}"
        emit_progress(
            {
                "event": "env",
                "user": user,
                "has_path": bool(os.environ.get("PATH")),
                "has_username": bool(os.environ.get("USERNAME") or os.environ.get("USER")),
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "unbuffered": os.environ.get("PYTHONUNBUFFERED", ""),
                "var_count": len(os.environ),
            }
        )
        return 0

    print("ordinary log line before start")
    emit_progress(
        {
            "event": "start",
            "task": "type",
            "device": "cpu",
            "classes": ["Type II", "Type III", "Other"],
            "total_epochs": args.epochs,
            "train_size": 51,
            "val_size": 11,
            "test_size": 10,
        }
    )

    for epoch in range(1, args.epochs + 1):
        print(f"chatter for epoch {epoch}")
        emit_progress(
            {
                "event": "epoch",
                "task": "type",
                "epoch": epoch,
                "total_epochs": args.epochs,
                "train_loss": 1.0 / epoch,
                "val_loss": 1.1 / epoch,
                "val_accuracy": 0.5 + epoch * 0.1,
                "val_macro_f1": 0.4 + epoch * 0.1,
                "score": 0.4 + epoch * 0.1,
                "learning_rate": 0.001,
                "is_best": True,
            }
        )
        time.sleep(0.02)

    if args.fail:
        print("exploding on purpose", file=sys.stderr)
        return 3

    emit_progress(
        {
            "event": "finished",
            "task": "type",
            "best_epoch": args.epochs,
            "best_score": 0.4 + args.epochs * 0.1,
            "best_alias": "/tmp/best.pt",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
