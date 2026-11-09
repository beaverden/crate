import threading

from libcrate import Catalog, MetadataStore, RunReport, SettingsError, steps
from libcrate.models.clap.models import ClapError
from libcrate.models.essentia.models import EssentiaError
from libcrate.services.llm.client import LLMError
from libcrate.models import ModelError
from PySide6.QtCore import QThread, Signal

__all__ = ["EnrichTask"]


class EnrichTask(QThread):
    """In a background thread: `MetadataStore.run`, the fetchers the catalog's settings turn on (or `names`), in the
    order they need, after downloading their models. stop() finishes what's in flight and starts nothing more.

    It works on a copy of the catalog, so the window can keep showing the original while fetchers write into the
    copy; finished_ok hands over the enriched copy.

    `steps` are what it goes through, as started() and progress() name them: a download per missing model
    ("download muq-mulan"), then the fetchers that can run, in order. The settings are read as it's made;
    unreadable ones (or an unknown fetcher in `names`) fail it when it runs.
    started(step, total, skipped) as each starts (the metadata fetchers of a stage all at once), with its tracks to
    do and those done already; progress(step, done, total, rel_path, outcome) per track, the outcome one of
    libcrate's OUTCOMES; finished_ok(catalog, report) at the end, a RunReport; or failed(message).
    """

    started = Signal(str, int, int)
    progress = Signal(str, int, int, str, str)
    finished_ok = Signal(Catalog, object)
    failed = Signal(str)

    def __init__(self, store: MetadataStore, catalog: Catalog, names: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.store = store
        self.catalog = catalog.model_copy(deep=True)
        self.names = names
        self._stop = threading.Event()
        try:
            self.settings, self._error = store.settings(), None
            self.steps = steps(self.settings, names)
        except (SettingsError, ValueError) as e:  # unreadable settings, an unknown fetcher
            self.settings, self._error, self.steps = None, str(e), []

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if self._error is not None:
            self.failed.emit(self._error)
            return
        try:
            report: RunReport = self.store.run(self.catalog, self.settings, self.names, stop=self._stop,
                                               on_start=self.started.emit, on_progress=self.progress.emit)
        except (ModelError, EssentiaError, ClapError, LLMError, ValueError) as e:  # can't be set up or downloaded
            self.failed.emit(str(e))
            return
        except OSError as e:
            self.failed.emit(f"could not save: {e}")
            return
        self.finished_ok.emit(self.catalog, report)
