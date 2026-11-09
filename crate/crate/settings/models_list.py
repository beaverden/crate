from pathlib import Path

from libcrate import models
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon, MessageBox, PushButton, SettingCard

from crate.settings.fetcher_section import size_text
from crate.tasks import DownloadModelsTask

__all__ = ["ModelsList"]

KIND_ICONS = {"essentia": FluentIcon.MUSIC, "clap": FluentIcon.CHAT, "text": FluentIcon.EDIT}


class ModelRow(SettingCard):
    """A model: what it is, its size, whether it's in the folder and what uses it; Download or Delete."""

    def __init__(self, model: models.Model, parent=None):
        super().__init__(KIND_ICONS.get(model.kind, FluentIcon.IOT), model.name, "", parent)
        self.model = model
        self.button = PushButton()
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.setFixedHeight(80)

    def show_state(self, folder: Path | None, users: list[str], downloading: str | None) -> None:
        installed = folder is not None and models.installed(folder, self.model.name)
        if downloading == self.model.name:
            state = "downloading…"
        elif installed:
            state = f"downloaded, {size_text(models.folder_size(folder, self.model.name))}"
        else:
            state = f"not downloaded, {size_text(self.model.size)}"
        needs = f"  ·  runs on {', '.join(self.model.requires)}" if self.model.requires else ""
        used = f"  ·  used by {', '.join(users)}" if users else "  ·  not used by any fetcher that's on"
        self.setContent(f"{self.model.about}.\n{state[0].upper()}{state[1:]}{needs}{used}")
        self.button.setText("Delete" if installed else "Download")
        self.button.setIcon(FluentIcon.DELETE if installed else FluentIcon.DOWNLOAD)
        self.button.setEnabled(folder is not None and downloading is None)


class ModelsList(QWidget):
    """Every model libcrate runs (libcrate.models), a row each, for the models folder: Download fetches a model and
    the ones it needs, in the background; Delete removes one, after asking (warning when a fetcher that's on uses
    it). Enrich downloads what the fetchers that are on need by itself; this is for doing it beforehand, or freeing
    the space.

    set_folder(folder) follows the folder setting as it's edited; set_users({model: users}) says what uses each.
    downloaded(names) when a download ended well, failed(message) when it didn't, deleted(name).
    """

    downloaded = Signal(list)
    failed = Signal(str)
    deleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.users: dict[str, list[str]] = {}
        self.task: DownloadModelsTask | None = None
        self._downloading: str | None = None
        self.rows = {name: ModelRow(model) for name, model in models.MODELS.items()}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for name, row in self.rows.items():
            row.button.clicked.connect(lambda checked=False, name=name: self._act(name))
            layout.addWidget(row)

    def set_folder(self, folder: Path | None) -> None:
        self.folder = folder
        self.refresh()

    def set_users(self, users: dict[str, list[str]]) -> None:
        self.users = users
        self.refresh()

    def refresh(self) -> None:
        folder = None if self.task is not None and self.task.folder != self.folder else self.folder
        for name, row in self.rows.items():
            row.show_state(folder, self.users.get(name, []), self._downloading if self.task else None)
        if self.task is not None:  # nothing else while it runs
            for row in self.rows.values():
                row.button.setEnabled(False)

    def download(self, name: str) -> None:
        if self.folder is None or self.task is not None:
            return
        self.task = DownloadModelsTask(self.folder, [name], self)
        self.task.model_done.connect(self._on_model)
        self.task.finished_ok.connect(lambda: self.downloaded.emit([name]))
        self.task.failed.connect(self.failed)
        self.task.finished.connect(self._on_finished)
        self.task.start()
        self.refresh()

    def delete(self, name: str) -> None:
        if self.folder is not None:
            models.delete(self.folder, name)
            self.refresh()
            self.deleted.emit(name)

    def _act(self, name: str) -> None:
        if self.folder is None or not models.installed(self.folder, name):
            self.download(name)
            return
        users = self.users.get(name, [])
        warning = f"\n\n{', '.join(users)} use{'s' if len(users) == 1 else ''} it: the next Enrich downloads it again." \
            if users else ""
        box = MessageBox(f"Delete {name}?", f"It frees {size_text(models.folder_size(self.folder, name))} in "
                         f"{self.folder}.{warning}", self.window())
        if box.exec():
            self.delete(name)

    def _on_model(self, name: str, status: str) -> None:
        self._downloading = name if status == "downloading" else None
        self.refresh()

    def _on_finished(self) -> None:
        self.task, self._downloading = None, None
        self.refresh()

    def shutdown(self) -> None:
        """Let a download finish (it can't be stopped halfway; what it fetches is only kept once complete)."""
        if self.task is not None:
            self.task.wait()
