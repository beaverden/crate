"""The interface every genre source implements (see fetchers/enrich.py for how they're run)."""

from abc import abstractmethod

from libcrate.fetchers.base import LookupNotFound, MetadataFetcher
from libcrate.track import Genre, TrackRecord

__all__ = ["GenreFetcher", "LookupNotFound"]


class GenreFetcher(MetadataFetcher):
    kind = "genres"

    @abstractmethod
    def fetch(self, record: TrackRecord) -> list[Genre]:
        """Genres for the track; raises LookupNotFound if the source has none."""
