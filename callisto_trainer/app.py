"""Application entry point.

Run with:

    python -m callisto_trainer.app
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from callisto_trainer.settings import AppSettings

DARK_STYLESHEET = """
QWidget { background-color: #0d1117; color: #c9d1d9; }
QGroupBox {
    border: 1px solid #21262d; border-radius: 6px;
    margin-top: 10px; padding-top: 8px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8b949e; }
QPushButton {
    background-color: #21262d; border: 1px solid #30363d;
    border-radius: 5px; padding: 5px 10px;
}
QPushButton:hover { background-color: #30363d; }
QPushButton:pressed { background-color: #161b22; }
QPushButton:disabled { color: #6e7681; background-color: #161b22; }
QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #0d1117; border: 1px solid #30363d;
    border-radius: 5px; padding: 4px;
}
QListView, QListWidget, QTableView {
    background-color: #0d1117; border: 1px solid #21262d; border-radius: 5px;
}
QListView::item:selected, QListWidget::item:selected { background-color: #1f6feb; color: #ffffff; }
QTabWidget::pane { border: 1px solid #21262d; }
QTabBar::tab { background: #161b22; padding: 7px 16px; border: 1px solid #21262d; }
QTabBar::tab:selected { background: #0d1117; color: #58a6ff; border-bottom: 2px solid #1f6feb; }
QProgressBar { border: 1px solid #30363d; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background-color: #1f6feb; border-radius: 4px; }
QSplitter::handle { background-color: #21262d; }
QStatusBar { border-top: 1px solid #21262d; }
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CALLISTO Trainer")
    parser.add_argument("--settings", default=None, help="Path to configs/trainer.yaml")
    args = parser.parse_args(argv)

    settings = AppSettings.load(args.settings)
    settings.ensure_directories()

    application = QApplication(sys.argv)
    application.setApplicationName("CALLISTO Trainer")
    application.setStyle("Fusion")
    application.setStyleSheet(DARK_STYLESHEET)

    # Imported late so the stylesheet is in place before any widget is built.
    from callisto_trainer.ui.main_window import MainWindow

    window = MainWindow(settings)
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
