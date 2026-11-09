"""The interface every embedding fetcher implements (see embed.py for how they're run)."""

from abc import abstractmethod
from typing import ClassVar

from libcrate.embeddings import Embedding
from libcrate.fetchers.base import Fetcher
from libcrate.track import TrackRecord

__all__ = ["EmbeddingFetcher"]


class EmbeddingFetcher(Fetcher):
    """A fetcher whose results are vectors, one per track, kept in a table of their own next to the catalog
    (embeddings/<name>/), run by embed() (embed.py)."""

    kind = "embeddings"
    # What its vectors compare: "audio" (tracks by their sound), "clap" (the sound, and a description of it
    # embedded by the same model) or "text" (the tracks' descriptions, and a search query).
    space: ClassVar[str]

    @abstractmethod
    def key(self, record: TrackRecord) -> str | None:
        """The model and the settings (and, for descriptions, the text) the track's vector depends on; a changed
        key redoes the track. None: nothing to embed yet (no description)."""

    @abstractmethod
    def fetch(self, record: TrackRecord) -> Embedding:
        """The track's embedding; raises on failure (the track is retried next run)."""
