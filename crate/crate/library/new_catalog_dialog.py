from pathlib import Path

from libcrate import CATALOG_FILE, SETTINGS_FILE
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QGridLayout
from qfluentwidgets import BodyLabel, CaptionLabel, LineEdit, MessageBoxBase, PushButton, SubtitleLabel

__all__ = ["NewCatalogDialog"]

HINT = ("The catalog folder gets the catalog, its settings and the cover thumbnails; "
        "the music folder is scanned into it.")


class NewCatalogDialog(MessageBoxBase):
    """Pick the music folder (the library), the folder for its new catalog and, optionally, another
    catalog to copy the settings from; `library()`, `catalog_folder()` and `settings_from()` give
    the choice. A folder that holds a catalog already can't be picked: that one is opened instead."""

    def __init__(self, library: Path | None, catalog_folder: Path | None, settings_from: Path | None = None,
                 parent=None):
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel("New catalog"))
        self.hint = CaptionLabel(HINT)
        self.hint.setWordWrap(True)
        self.viewLayout.addWidget(self.hint)

        self._library = LineEdit()
        self._catalog = LineEdit()
        self._settings_from = LineEdit()
        self._settings_from.setPlaceholderText("Optional: another catalog; empty for the defaults")
        self._settings_from.setClearButtonEnabled(True)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        rows = (("Music folder", self._library, library, "Music folder"),
                ("Catalog folder", self._catalog, catalog_folder, "Folder for the new catalog"),
                ("Settings from", self._settings_from, settings_from, "Catalog to copy the settings from"))
        for row, (label, edit, value, title) in enumerate(rows):
            edit.setText(str(value) if value else "")
            edit.setMinimumWidth(380)
            edit.textChanged.connect(self._validate)
            browse = PushButton("Browse…")
            browse.clicked.connect(lambda checked=False, e=edit, t=title: self._browse(e, t))
            grid.addWidget(BodyLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(browse, row, 2)
        self.viewLayout.addLayout(grid)

        self.yesButton.setText("Create")
        self._validate()

    def library(self) -> Path:
        return Path(self._library.text().strip()).expanduser()

    def catalog_folder(self) -> Path:
        return Path(self._catalog.text().strip()).expanduser()

    def settings_from(self) -> Path | None:
        text = self._settings_from.text().strip()
        return Path(text).expanduser() if text else None

    def _browse(self, edit: LineEdit, title: str) -> None:
        start = edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            edit.setText(folder)

    def _validate(self) -> None:
        """Create needs an existing music folder, a catalog folder (created if needed) without a catalog,
        and, if given, a settings file or a catalog folder with one to copy."""
        problem = None
        if self._catalog.text().strip() and (self.catalog_folder() / CATALOG_FILE).is_file():
            problem = f"{self.catalog_folder()} holds a catalog already: Open it instead."
        elif (source := self.settings_from()) is not None:
            file = source / SETTINGS_FILE if source.is_dir() else source
            if not file.is_file():
                problem = f"No settings to copy: {file} doesn't exist."
        if problem:
            self.hint.setText(problem)
            self.hint.setTextColor(QColor(196, 43, 28), QColor(255, 153, 164))
        else:
            self.hint.setText(HINT)
            self.hint.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        self.yesButton.setEnabled(problem is None and bool(self._catalog.text().strip())
                                  and bool(self._library.text().strip()) and self.library().is_dir())
