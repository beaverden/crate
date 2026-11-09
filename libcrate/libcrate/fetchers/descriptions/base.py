"""The interface every description source implements (see fetchers/enrich.py for how they're run)."""

from abc import abstractmethod

from libcrate.fetchers.base import MetadataFetcher
from libcrate.track import Description, TrackRecord

__all__ = ["DescriptionFetcher"]


class DescriptionFetcher(MetadataFetcher):
    kind = "descriptions"

    @abstractmethod
    def fetch(self, record: TrackRecord) -> Description:
        """The track's description; raises LookupNotFound if the source has none."""
