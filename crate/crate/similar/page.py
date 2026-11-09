from libcrate import Catalog, MetadataStore, SettingsError, Similar, TrackRecord
from qfluentwidgets import InfoBar

from crate.library.covers import CoverCache
from crate.library.feature import Unavailable, slot_embeddings
from crate.library.match_page import MatchPage

__all__ = ["SimilarPage"]

EMPTY = "Find tracks similar to one in the library: select it and press Similar, or right-click it."


class SimilarPage(MatchPage):
    """The tracks that sound most like a chosen one (libcrate.Similar), best match first, with how alike they are.

    find(store, catalog, track) lists them with the similar section of the catalog's settings.
    """

    def __init__(self, covers: CoverCache, parent=None):
        super().__init__("similar", "Similar", EMPTY, covers, parent)
        self.seed: TrackRecord | None = None

    def find(self, store: MetadataStore, catalog: Catalog, track: TrackRecord) -> bool:
        """List the tracks most like `track`, with the similar section of the catalog's settings; False (with a
        notification) when it can't: unreadable settings, no embeddings yet, or none for this track."""
        try:
            settings = store.settings()
        except SettingsError as e:
            self._notify(InfoBar.error, "Can't read the settings", str(e))
            return False
        try:
            name, table = slot_embeddings(store, catalog, settings, "similar")
        except Unavailable as e:
            self._notify(InfoBar.warning, e.title, e.message)
            return False
        except OSError as e:
            self._notify(InfoBar.error, "Can't read the embeddings", str(e))
            return False
        similar = Similar(catalog, table)
        if track.path not in similar:
            self._notify(InfoBar.warning, "This track has no embedding yet",
                         f"Press Enrich in the library to make its {name} embedding.")
            return False
        self.seed = track
        items = similar.find(track.path, **settings.similar.options())
        title = f"{', '.join(track.tags.artists)} – {track.tags.title or track.path}"
        self.show_matches(items, catalog.root, f"Like {title}  ·  {len(items) - 1} tracks  ·  {name}")
        return True
