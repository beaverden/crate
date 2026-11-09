from libcrate import Catalog, MetadataStore, MusicMap, SettingsError
from PySide6.QtCore import QThread, Signal

from crate.library.feature import Unavailable, slot_embeddings

__all__ = ["MapTask"]


class MapTask(QThread):
    """`MetadataStore.music_map` with the map section of the catalog's settings, in a background thread: the
    saved map if the embeddings (those made with the fetcher's current settings) haven't changed, else a new one
    (several seconds).

    finished_ok(music_map, fetcher) or failed(message), a sentence or two, at the end.
    """

    finished_ok = Signal(MusicMap, str)
    failed = Signal(str)

    def __init__(self, store: MetadataStore, catalog: Catalog, parent=None):
        super().__init__(parent)
        self.store, self.catalog = store, catalog

    def run(self) -> None:
        try:
            settings = self.store.settings()
            name, table = slot_embeddings(self.store, self.catalog, settings, "map")
            music_map = self.store.music_map(name, clusters=settings.map.clusters, table=table)
        except Unavailable as e:
            self.failed.emit(f"{e.title[0].lower()}{e.title[1:]}. {e.message}")
            return
        except (OSError, SettingsError, ValueError) as e:
            self.failed.emit(f"{e}.")
            return
        self.finished_ok.emit(music_map, name)
