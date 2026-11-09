from libcrate import FETCHERS, Catalog, EmbeddingFetcher, MetadataStore, SettingsError, progress
from PySide6.QtCore import QThread, Signal

__all__ = ["ProgressTask"]


class ProgressTask(QThread):
    """How far every fetcher is on the catalog with the saved settings (libcrate.progress), in a background thread:
    the description fetcher hashes each track's facts, and the embedding fetchers' tables are read.

    finished_ok({fetcher: FetcherProgress}) or failed(message) at the end.
    """

    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, store: MetadataStore, catalog: Catalog, parent=None):
        super().__init__(parent)
        self.store, self.catalog = store, catalog

    def run(self) -> None:
        try:
            settings = self.store.settings()
            found = {}
            for name, cls in FETCHERS.items():
                table = self.store.load_embeddings(name) if issubclass(cls, EmbeddingFetcher) else None
                found[name] = progress(self.catalog, settings, name, table)
        except (SettingsError, OSError) as e:
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(found)
