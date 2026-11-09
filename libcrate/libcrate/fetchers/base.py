"""The interface every fetcher implements, whatever it fetches (see run.py for how they're run).

A fetcher is built from the settings alone (its own section, settings.fetchers.get(name), and the services and
models it uses), which is cheap: it says what it needs and computes its keys without starting anything. open()
builds its client (a network session, a model in a child process), which is where a missing model or an unknown
provider shows up; a client given to the constructor (tests) is used as it is.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from libcrate.embeddings import EmbeddingTable
from libcrate.settings import FetcherSettings, Settings
from libcrate.track import TrackRecord

__all__ = ["KINDS", "Fetcher", "Input", "LookupNotFound", "MetadataFetcher"]

# What fetchers fetch, in the order they're listed: genres and descriptions go into TrackRecord fields of those
# names, embeddings into a table per fetcher next to the catalog.
KINDS = ("genres", "embeddings", "descriptions")


class LookupNotFound(Exception):
    """Raised by MetadataFetcher.fetch when the source has nothing for the track."""


@dataclass(frozen=True)
class Input:
    """Another fetcher whose results this one uses. Required: it has nothing to work on without them; otherwise
    it uses them when they're there, and does without (less well) when they aren't."""

    fetcher: str
    required: bool = False


class Fetcher(ABC):
    # Its name in the settings, the results and the app ("chosic-genres").
    name: ClassVar[str]
    # What it fetches: one of KINDS.
    kind: ClassVar[str]

    def __init__(self, settings: Settings | None = None, root: str | Path | None = None, *, client: Any = None):
        """
        settings -> all of them (default: the built-in ones): the fetcher's own section, and the services and
                    models it uses. A fetcher the settings don't know (a test's) gets FetcherSettings().
        root     -> the library folder (catalog.root), for fetchers that read the audio.
        client   -> a ready client (tests); otherwise open() builds one from the settings.
        """
        self.settings = settings or Settings()
        try:
            self.options: FetcherSettings = self.settings.fetchers.get(self.name)
        except KeyError:
            self.options = FetcherSettings()
        self.root = Path(root) if root is not None else None
        self.client = client

    @property
    def workers(self) -> int:
        """How many tracks are processed at once. 1 for sources with a request rate limit; more for ones that
        handle concurrent requests, or a model with several processes."""
        return 1

    def services(self) -> list[str]:
        """The services it asks (settings.services), which have to be set up."""
        return []

    def models(self) -> list[str]:
        """The models it runs (libcrate.models), downloaded before it runs."""
        return []

    def inputs(self) -> list[Input]:
        """The other fetchers whose results it uses; they run before it."""
        return []

    @abstractmethod
    def key(self, record: TrackRecord) -> str | None:
        """What the result depends on ("isrc:...", "spotify:...", a model and its settings),
        or None if the track can't be processed (no ID). A changed key redoes the track."""

    @abstractmethod
    def fetch(self, record: TrackRecord) -> Any:
        """The track's result (genres: a list of Genre; a description; an Embedding); raises LookupNotFound if
        the source has none (metadata fetchers), or any error on failure, retried on the next run."""

    def open(self, embeddings: Callable[[str], EmbeddingTable] | None = None) -> "Fetcher":
        """Get ready to fetch: build the client unless one was given. embeddings(name) loads another fetcher's
        embeddings, for fetchers that use them. Raises the client's error if it can't be set up."""
        if self.client is None:
            self.client = self.make_client()
        return self

    def make_client(self) -> Any:
        """The client, from the settings."""
        return None

    def close(self) -> None:
        """Release resources such as network sessions or a model's processes."""
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class MetadataFetcher(Fetcher):
    """A fetcher whose results go into the track record (TrackRecord.genres or .descriptions, by its kind), run by
    enrich() (enrich.py)."""
