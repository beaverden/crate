"""Crate: the desktop app for the music library, on top of libcrate.

    crate        # opens the output folder used last (catalog.json), or asks for one
"""

import contextlib
import io
import sys

# qfluentwidgets prints an advert to stdout when first imported; keep it out of the terminal.
with contextlib.redirect_stdout(io.StringIO()):
    import qfluentwidgets

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from crate.window import MainWindow

__all__ = ["main"]

# Accent colour of buttons, selection, progress and genre chips.
THEME_COLOR = "#8B5CF6"


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Crate")
    app.setOrganizationName("music-library")
    qfluentwidgets.setThemeColor(THEME_COLOR)
    window = MainWindow(QSettings())
    window.show()
    raise SystemExit(app.exec())
