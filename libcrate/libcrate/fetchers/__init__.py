"""Fetchers fill a catalog from the services (libcrate.services) and local models (libcrate.models), one
subpackage per kind of data.

    libcrate.fetchers.genres         TrackRecord.genres, one result per fetcher
    libcrate.fetchers.descriptions   TrackRecord.descriptions, one result per fetcher
    libcrate.fetchers.embeddings     an EmbeddingTable per fetcher, next to the catalog

Every fetcher is a Fetcher (base.py), built from the settings by make_fetcher() (registry.py); genre and
description fetchers are MetadataFetchers, run by enrich() (enrich.py), embedding fetchers are run by embed()
(embeddings/embed.py). run() (run.py) runs the chosen ones in the order their inputs need; status.py says whether
one can run and how far it is.
"""

from libcrate.fetchers.base import KINDS, Fetcher, Input, LookupNotFound, MetadataFetcher
from libcrate.fetchers.embeddings import EmbeddingFetcher, EmbedReport, embed
from libcrate.fetchers.enrich import GIVE_UP_AFTER, EnrichReport, enrich, lookup, needs_lookup
from libcrate.fetchers.registry import FETCHERS, fetcher_names, make_fetcher
from libcrate.fetchers.run import DOWNLOAD, OUTCOMES, RunReport, plan, run, steps
from libcrate.fetchers.status import FetcherCheck, FetcherProgress, check, progress

__all__ = [
    "DOWNLOAD", "FETCHERS", "GIVE_UP_AFTER", "KINDS", "OUTCOMES", "EmbedReport", "EmbeddingFetcher", "EnrichReport",
    "Fetcher", "FetcherCheck", "FetcherProgress", "Input", "LookupNotFound", "MetadataFetcher", "RunReport", "check",
    "embed", "enrich", "fetcher_names", "lookup", "make_fetcher", "needs_lookup", "plan", "progress", "run", "steps",
]
