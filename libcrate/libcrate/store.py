"""The catalog folder: everything libcrate knows about a library, and its settings, in one place.

    store = MetadataStore("~/music-metadata")
    catalog, report = store.init("/Volumes/KINGSTON/Spoti/All")   # default settings, then the first scan
    store.init(library, settings_from="~/other-metadata")          # or another catalog's settings
    catalog, report = store.scan()                                # later: new and changed files
    settings = store.settings()                                   # the catalog's settings.yaml
    store.run(catalog, settings)                                  # Enrich: the enabled fetchers, saved as it goes
    catalog = store.load()                                        # later, e.g. in the app
    paths, matrix = store.current_embeddings(catalog, settings, "discogs-effnet").matrix()
    table = store.current_embeddings(catalog, settings, "discogs-effnet")
    music_map = store.music_map("discogs-effnet", clusters=12, table=table)   # made once, then when they change

Layout:

    catalog.json                 the Catalog: tags, audio info, cover info, genres per source
    settings.yaml                the settings (libcrate.settings), readable only by the user
    embeddings/<model>/mean.parquet   each model's mean vector per track (libcrate.embeddings)
    embeddings/<model>/map.parquet    the music map made from them (libcrate.map)
"""

from pathlib import Path

from libcrate import features
from libcrate.catalog import Catalog, ScanReport
from libcrate.embeddings import EmbeddingTable
from libcrate.fetchers.base import MetadataFetcher
from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.fetchers.embeddings.embed import EmbedReport, embed
from libcrate.fetchers.enrich import EnrichReport, enrich
from libcrate.fetchers.run import RunReport
from libcrate.fetchers.run import run as run_fetchers
from libcrate.map import MusicMap
from libcrate.settings import SETTINGS_FILE, Settings, SettingsError, load_settings, save_settings
from libcrate.settings import settings_from as read_settings_from

__all__ = ["CATALOG_FILE", "EMBEDDINGS_FOLDER", "MAP_FILE", "MEAN_EMBEDDINGS_FILE", "SETTINGS_FILE", "MetadataStore"]

CATALOG_FILE = "catalog.json"
EMBEDDINGS_FOLDER = "embeddings"
# In each model's folder; a model's other files (per-frame vectors, ...) would go next to it.
MEAN_EMBEDDINGS_FILE = "mean.parquet"
MAP_FILE = "map.parquet"


class MetadataStore:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder).expanduser()
        if self.folder.exists() and not self.folder.is_dir():
            raise NotADirectoryError(f"{self.folder} is a file, not a folder")

    @property
    def catalog_path(self) -> Path:
        return self.folder / CATALOG_FILE

    @property
    def settings_path(self) -> Path:
        return self.folder / SETTINGS_FILE

    def settings(self) -> Settings:
        """This catalog's settings, over the defaults (all of them if it has no settings file)."""
        return load_settings(self.settings_path)

    def init(self, library: str | Path, settings_from: str | Path | None = None,
             **kwargs) -> tuple[Catalog, ScanReport]:
        """Start a catalog of `library` here: create_settings(settings_from), then scan.
        kwargs go to Catalog.scan, as for scan()."""
        self.create_settings(settings_from)
        return self.scan(library, **kwargs)

    def create_settings(self, settings_from: str | Path | None = None) -> bool:
        """Write the settings file, unless there is one; True if it was written.

        settings_from -> another catalog folder or settings file to copy the settings from
        (rewritten with the comments, over the defaults); without it, the defaults. Raises
        SettingsError if it can't be read, or if this catalog has settings already.
        """
        if settings_from is not None:
            if self.settings_path.exists():
                raise SettingsError(f"{self.settings_path} exists already; remove it to take the settings "
                                    f"from {settings_from}", self.settings_path)
            settings = read_settings_from(Path(settings_from))
        elif self.settings_path.exists():
            return False
        else:
            settings = Settings()
        self.folder.mkdir(parents=True, exist_ok=True)
        save_settings(settings, self.settings_path)
        return True

    def load(self) -> Catalog | None:
        """The saved catalog, or None if nothing was saved yet."""
        return Catalog.load(self.catalog_path) if self.catalog_path.is_file() else None

    def save(self, catalog: Catalog) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        catalog.save(self.catalog_path)

    def scan(self, library: str | Path | None = None, **kwargs) -> tuple[Catalog, ScanReport]:
        """Read new and changed files from `library` (default: the saved catalog's library) and save.

        kwargs go to Catalog.scan (workers, on_progress).
        """
        catalog = self.load()
        if library is None:
            if catalog is None:
                raise ValueError(f"no catalog in {self.folder} yet, so the library folder is needed")
            library = catalog.root
        catalog, report = Catalog.scan(Path(library), catalog, **kwargs)
        self.save(catalog)
        return catalog, report

    def run(self, catalog: Catalog, settings: Settings, names: list[str] | None = None, **kwargs) -> RunReport:
        """Enrich: run `names` (default: the enabled fetchers) on `catalog`, in the order they need, downloading
        their models first and saving as results come in. kwargs go to fetchers.run (on_start, on_progress, stop)."""
        return run_fetchers(self, catalog, settings, names, **kwargs)

    def enrich(self, catalog: Catalog, fetchers: list[MetadataFetcher], **kwargs) -> EnrichReport:
        """Run the metadata fetchers on `catalog`, saving it as results come in and at the end.

        kwargs go to enrich (on_progress, on_start, checkpoint_every, stop).
        """
        return enrich(catalog, fetchers, on_checkpoint=self.save, **kwargs)

    def embeddings_path(self, name: str) -> Path:
        return self.folder / EMBEDDINGS_FOLDER / name / MEAN_EMBEDDINGS_FILE

    def load_embeddings(self, name: str) -> EmbeddingTable:
        """Embedding fetcher `name`'s table, all of it; empty if none were computed yet."""
        return EmbeddingTable.load(self.embeddings_path(name))

    def current_embeddings(self, catalog: Catalog, settings: Settings, name: str) -> EmbeddingTable:
        """Embedding fetcher `name`'s rows for `catalog`'s tracks made with its current settings: what a feature
        can compare (features.current)."""
        return features.current(self.load_embeddings(name), catalog, settings, name)

    def music_map(self, name: str, *, clusters: int, table: EmbeddingTable | None = None) -> MusicMap:
        """The map of embedding fetcher `name`'s `table` (default: all its rows; a feature passes
        current_embeddings()): the saved one if it was made from the same embeddings and settings, else a new one
        (several seconds), which is saved. Raises ValueError with too few tracks."""
        table = self.load_embeddings(name) if table is None else table
        path = self.embeddings_path(name).with_name(MAP_FILE)
        saved = MusicMap.load(path)
        if saved is not None and saved.fingerprint == MusicMap.fingerprint_of(table, clusters):
            return saved
        music_map = MusicMap.compute(table, clusters=clusters)
        music_map.save(path)
        return music_map

    def embed(self, catalog: Catalog, fetchers: list[EmbeddingFetcher], **kwargs) -> EmbedReport:
        """Run the embedding fetchers on `catalog`'s tracks, saving their tables as results come in and at the end.

        kwargs go to embed (on_progress, on_start, checkpoint_every, stop).
        """
        tables = {f.name: self.load_embeddings(f.name) for f in fetchers}
        return embed(catalog, fetchers, tables, on_checkpoint=self._save_embeddings, **kwargs)

    def _save_embeddings(self, tables: dict[str, EmbeddingTable]) -> None:
        for name, table in tables.items():
            path = self.embeddings_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            table.save(path)
