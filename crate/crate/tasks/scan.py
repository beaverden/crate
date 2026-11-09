from pathlib import Path

from libcrate import Catalog, MetadataStore, ScanReport
from PySide6.QtCore import QThread, Signal

__all__ = ["ScanTask"]


class ScanTask(QThread):
    """`MetadataStore.scan` in a background thread: reads new and changed files, saves catalog.json.
    Given a library, `MetadataStore.init` instead: the default settings first, unless there are some
    (the library page writes them first, to report a problem with them before scanning).

    progress(done, total, rel_path) per file read; finished_ok(catalog, report) or failed(message) at the end.
    """

    progress = Signal(int, int, str)
    finished_ok = Signal(Catalog, ScanReport)
    failed = Signal(str)

    def __init__(self, store: MetadataStore, library: Path | None = None, parent=None):
        super().__init__(parent)
        self.store = store
        self.library = library

    def run(self) -> None:
        try:
            if self.library is None:
                catalog, report = self.store.scan(on_progress=self.progress.emit)
            else:
                catalog, report = self.store.init(self.library, on_progress=self.progress.emit)
        except (OSError, ValueError) as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(catalog, report)
