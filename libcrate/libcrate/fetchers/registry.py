"""Every fetcher, by name, in the order they're listed (genres, embeddings, descriptions)."""

from pathlib import Path
from typing import Any

from libcrate.fetchers.base import Fetcher
from libcrate.fetchers.descriptions import LLMDescriptionFetcher
from libcrate.fetchers.embeddings import (
    DescriptionTextFetcher, DiscogsEffnetFetcher, DiscogsTrackFetcher, LaionClapMusicFetcher, MuqMulanFetcher,
)
from libcrate.fetchers.genres import ChosicGenreFetcher, EssentiaGenreFetcher, LLMGenreFetcher, MusicBrainzGenreFetcher
from libcrate.settings import Settings

__all__ = ["FETCHERS", "fetcher_names", "make_fetcher"]

FETCHERS: dict[str, type[Fetcher]] = {cls.name: cls for cls in (
    MusicBrainzGenreFetcher, ChosicGenreFetcher, EssentiaGenreFetcher, LLMGenreFetcher,
    DiscogsEffnetFetcher, DiscogsTrackFetcher, MuqMulanFetcher, LaionClapMusicFetcher, DescriptionTextFetcher,
    LLMDescriptionFetcher,
)}


def fetcher_names(kind: str | None = None) -> list[str]:
    """The fetchers' names, those of one kind ("genres", "embeddings", "descriptions") with `kind`."""
    return [name for name, cls in FETCHERS.items() if kind is None or cls.kind == kind]


def make_fetcher(name: str, settings: Settings | None = None, root: str | Path | None = None, *,
                 client: Any = None) -> Fetcher:
    """The fetcher called `name`, set up from `settings` (default: the built-in ones); cheap, as it starts
    nothing until open(). root -> the library folder (catalog.root), for fetchers that read the audio. Raises
    ValueError for an unknown name."""
    if name not in FETCHERS:
        raise ValueError(f"unknown fetcher {name!r}; choose from {', '.join(FETCHERS)}")
    return FETCHERS[name](settings or Settings(), root, client=client)
