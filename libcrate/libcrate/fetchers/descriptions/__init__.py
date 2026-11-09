"""Description sources: how a track sounds and feels, in words, for searching by text. Each is a DescriptionFetcher
subclass; enrich() (fetchers/enrich.py) runs them over a catalog, into TrackRecord.descriptions.

The llm-descriptions fetcher asks a language model (paid API calls), from facts about the track (tags, other
sources' genres, audio tags from a CLAP model: audio_tags.py) and what it knows.
"""

from libcrate.fetchers.descriptions.audio_tags import PHRASES, AudioTags
from libcrate.fetchers.descriptions.base import DescriptionFetcher
from libcrate.fetchers.descriptions.llm import FACT_SOURCES, LLMDescriptionFetcher

__all__ = ["FACT_SOURCES", "PHRASES", "AudioTags", "DescriptionFetcher", "LLMDescriptionFetcher"]
