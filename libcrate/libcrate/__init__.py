"""Music library metadata: tags, audio info, cover info, genres per source and embeddings, kept in a catalog folder.

    from libcrate import MetadataStore, Similar

    store = MetadataStore("~/music-metadata")
    catalog, report = store.init("/Volumes/KINGSTON/Spoti/All")   # writes settings.yaml, saves catalog.json
    catalog, report = store.scan()                                # later: incremental

    settings = store.settings()                                   # the catalog's settings.yaml
    store.run(catalog, settings, on_progress=...)                 # Enrich: the enabled fetchers, in order, saved as
                                                                  # it goes; stop=threading.Event() to cancel
    check(settings, "llm-descriptions").problems                  # what stands between a fetcher and running it
    progress(catalog, settings, "chosic-genres").to_do            # what the next Enrich does

    catalog = store.load()                                        # read-only use, e.g. the app
    catalog.tracks["Metronomy - The Light [GBMVH1900150].flac"].genre_names()
    table = store.current_embeddings(catalog, settings, settings.similar.embeddings)
    Similar(catalog, table).find(seed_path, **settings.similar.options())
    store.music_map(settings.map.embeddings, clusters=settings.map.clusters, table=table)   # tracks laid out
    Radio(catalog, table, [seed_path], **settings.radio.options()).next()

Nothing here prints or exits; errors are raised (see each function).
"""

from libcrate.catalog import Catalog, ScanError, ScanReport
from libcrate.embeddings import Embedding, EmbeddingTable, TrackMatch
from libcrate.fetchers import (
    DOWNLOAD, FETCHERS, KINDS, OUTCOMES, EmbeddingFetcher, EmbedReport, EnrichReport, Fetcher, FetcherCheck,
    FetcherProgress, Input, LookupNotFound, MetadataFetcher, RunReport, check, embed, enrich, fetcher_names,
    make_fetcher, needs_lookup, plan, progress, steps,
)
from libcrate.fetchers.descriptions import DescriptionFetcher
from libcrate.fetchers.genres import GenreFetcher
from libcrate.map import MusicMap
from libcrate.radio import Radio
from libcrate.similar import Similar
from libcrate.settings import Settings, SettingsError, about, load_settings
from libcrate.store import CATALOG_FILE, EMBEDDINGS_FOLDER, MEAN_EMBEDDINGS_FILE, SETTINGS_FILE, MetadataStore
from libcrate.text_search import TextSearch, fuse
from libcrate.track import (
    SCHEMA_HASH, AudioInfo, CoverInfo, Description, DescriptionSource, ExtractError, FetchedSource, FileStat, Genre,
    GenreSource, SourceStatus, Tags, TrackRecord,
)

__all__ = [
    "CATALOG_FILE", "DOWNLOAD", "EMBEDDINGS_FOLDER", "FETCHERS", "KINDS", "MEAN_EMBEDDINGS_FILE", "OUTCOMES",
    "SCHEMA_HASH", "SETTINGS_FILE", "AudioInfo", "Catalog", "CoverInfo", "Description", "DescriptionFetcher",
    "DescriptionSource", "EmbedReport", "Embedding", "EmbeddingFetcher", "EmbeddingTable", "EnrichReport",
    "ExtractError", "FetchedSource", "Fetcher", "FetcherCheck", "FetcherProgress", "FileStat", "Genre",
    "GenreFetcher", "GenreSource", "Input", "LookupNotFound", "MetadataFetcher", "MetadataStore", "MusicMap",
    "Radio", "RunReport", "ScanError", "ScanReport", "Settings", "SettingsError", "Similar", "SourceStatus", "Tags",
    "TextSearch", "TrackMatch", "TrackRecord", "about", "check", "embed", "enrich", "fetcher_names", "fuse",
    "load_settings", "make_fetcher", "needs_lookup", "plan", "progress", "steps",
]
