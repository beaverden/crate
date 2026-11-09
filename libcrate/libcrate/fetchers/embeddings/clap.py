from typing import ClassVar

import numpy as np

from libcrate.models.clap.client import ClapClient
from libcrate.models.clap.models import CLIP_SECONDS, CLIPS, MODELS
from libcrate.embeddings import Embedding
from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.track import TrackRecord

__all__ = ["ClapFetcher", "LaionClapMusicFetcher", "MuqMulanFetcher"]


class ClapFetcher(EmbeddingFetcher):
    """A CLAP model's audio embedding (512 dims): the mean of its 10 s clips' (from a quarter, half and three
    quarters of the way through the track), L2-normalised, in the space descriptions are embedded in too, so
    `model` embeds a search query for them (Describe). Runs locally, no network; as many tracks at once as the
    options' processes, each in its own process."""

    # The CLAP model (models.clap MODELS), named like the fetcher.
    model: ClassVar[str]
    space = "clap"

    @property
    def workers(self) -> int:
        return max(1, self.options.processes)

    def models(self) -> list[str]:
        return [self.model]

    def make_client(self) -> ClapClient:
        return ClapClient(self.settings.models.folder, self.model, processes=self.options.processes)

    def key(self, record: TrackRecord) -> str:
        return f"{self.model}:{MODELS[self.model].revision[:12]}:clips={len(CLIPS)}x{CLIP_SECONDS}s"

    def fetch(self, record: TrackRecord) -> Embedding:
        chunks = self.client.audio_embeddings(self.root / record.path)
        mean = chunks.mean(axis=0)
        return Embedding((mean / (np.linalg.norm(mean) + 1e-12)).astype(np.float32), n_frames=len(chunks))


class MuqMulanFetcher(ClapFetcher):
    name = model = "muq-mulan"


class LaionClapMusicFetcher(ClapFetcher):
    name = model = "laion-clap-music"
