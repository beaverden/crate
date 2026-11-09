"""Genre sources. Each is a GenreFetcher subclass; enrich() (fetchers/enrich.py) runs them over a catalog.

MusicBrainz and chosic look tracks up online; Essentia analyses the audio locally;
the llm fetcher asks a language model (paid API calls; Claude Sonnet 5.5 by default).
"""

from libcrate.fetchers.genres.base import GenreFetcher, LookupNotFound
from libcrate.fetchers.genres.chosic import ChosicGenreFetcher
from libcrate.fetchers.genres.essentia import EssentiaGenreFetcher
from libcrate.fetchers.genres.llm import LLMGenreFetcher, llm_service
from libcrate.fetchers.genres.musicbrainz import MusicBrainzGenreFetcher

__all__ = ["ChosicGenreFetcher", "EssentiaGenreFetcher", "GenreFetcher", "LLMGenreFetcher", "LookupNotFound",
           "MusicBrainzGenreFetcher", "llm_service"]
