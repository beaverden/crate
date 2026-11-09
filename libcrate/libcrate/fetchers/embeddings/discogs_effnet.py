import numpy as np

from libcrate.models.essentia.client import EMBEDDINGS, TRACK_EMBEDDINGS, EssentiaClient
from libcrate.models.essentia.models import EMBEDDING_MODEL, RESAMPLE_QUALITY, TRACK_EMBEDDING_MODEL
from libcrate.embeddings import Embedding
from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.track import TrackRecord

__all__ = ["DiscogsEffnetFetcher", "DiscogsTrackFetcher"]


class DiscogsEffnetFetcher(EmbeddingFetcher):
    """Essentia's Discogs-EffNet embeddings (1280 dims, the ones its genre model scores),
    averaged over the start of the track. Runs locally, no network; as many tracks at once as
    the options' processes, each in its own process."""

    name = "discogs-effnet"
    space = "audio"
    model = EMBEDDING_MODEL
    requests = EMBEDDINGS

    @property
    def workers(self) -> int:
        return max(1, self.options.processes)

    def models(self) -> list[str]:
        return list(self.requests)

    def make_client(self) -> EssentiaClient:
        return EssentiaClient(self.settings.models.folder, self.requests, max_seconds=self.options.max_seconds,
                              processes=self.options.processes)

    def key(self, record: TrackRecord) -> str:
        return f"{self.model}:resample={RESAMPLE_QUALITY}:seconds={self.options.max_seconds:g}"

    def fetch(self, record: TrackRecord) -> Embedding:
        frames = self._frames(str(self.root / record.path))
        return Embedding(frames.mean(axis=0).astype(np.float32), n_frames=len(frames))

    def _frames(self, path: str) -> np.ndarray:
        return self.client.embeddings(path)


class DiscogsTrackFetcher(DiscogsEffnetFetcher):
    """The same network trained for track similarity (1280 dims): contrastively, on Discogs
    metadata, so tracks of the same release end up close together."""

    name = "discogs-track"
    model = TRACK_EMBEDDING_MODEL
    requests = TRACK_EMBEDDINGS

    def _frames(self, path: str) -> np.ndarray:
        return self.client.track_embeddings(path)
