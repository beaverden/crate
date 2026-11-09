"""Embedding fetchers. Each is an EmbeddingFetcher subclass; embed() (embed.py) runs them over a catalog.

discogs-effnet and discogs-track run Essentia's Discogs-EffNet models on the audio, locally; the CLAP models
(muq-mulan, laion-clap-music) run in the space descriptions are matched in. description-bge embeds the tracks'
descriptions (TrackRecord.descriptions) with a text model, for searching what they say; it runs after the
description fetchers (its inputs).
"""

from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.fetchers.embeddings.clap import ClapFetcher, LaionClapMusicFetcher, MuqMulanFetcher
from libcrate.fetchers.embeddings.description_text import DescriptionTextFetcher
from libcrate.fetchers.embeddings.discogs_effnet import DiscogsEffnetFetcher, DiscogsTrackFetcher
from libcrate.fetchers.embeddings.embed import EmbedReport, embed

__all__ = [
    "ClapFetcher", "DescriptionTextFetcher", "DiscogsEffnetFetcher", "DiscogsTrackFetcher", "EmbedReport",
    "EmbeddingFetcher", "LaionClapMusicFetcher", "MuqMulanFetcher", "embed",
]
