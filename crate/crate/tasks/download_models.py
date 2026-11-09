from pathlib import Path

from libcrate import models
from libcrate.models.clap.models import ClapError
from PySide6.QtCore import QThread, Signal

__all__ = ["DownloadModelsTask"]


class DownloadModelsTask(QThread):
    """libcrate.models.download in a background thread: `names`, and the models they need, missing from `folder`.

    model_done(name, status) as each starts ("downloading") and ends ("downloaded"); finished_ok() or
    failed(message) at the end.
    """

    model_done = Signal(str, str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, folder: Path, names: list[str], parent=None):
        super().__init__(parent)
        self.folder, self.names = folder, names

    def run(self) -> None:
        try:
            models.download(self.folder, self.names, on_model=self.model_done.emit)
        except (models.ModelError, ClapError, OSError) as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit()
