"""Tracks that sound like a description ("chill late-night electronic"), best match first.

    settings = store.settings()
    sound = make_fetcher(settings.describe.sound, settings)            # a CLAP embedding fetcher: its model
    search = TextSearch(catalog, store.current_embeddings(catalog, settings, sound.name))
    with ClapClient(settings.models.folder, sound.model, processes=1) as clap:
        [vector] = clap.query_embeddings(["chill late-night electronic"])
    for item in search.find(vector, count=settings.describe.count):
        item.track.tags.title, item.similarity

A CLAP model puts tracks and descriptions in one space, so the match is the cosine
similarity of the track's embedding (the mean of its chunks') and the description's,
as the model was trained (no centring, unlike similar tracks). Tracks without an
embedding are left out.

The same works for a text model's embeddings of the tracks' descriptions
(description-bge), with the query embedded by that model (ClapClient.query_embeddings).
fuse() combines such rankings, e.g. the sound's and the descriptions':

    sound = TextSearch(catalog, store.current_embeddings(catalog, settings, "muq-mulan")).find(sound_vector, count=...)
    words = TextSearch(catalog, store.current_embeddings(catalog, settings, "description-bge")).find(words_vector, ...)
    fuse([sound, words], count=50)
"""

import numpy as np

from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable, TrackMatch

__all__ = ["FUSE_K", "TextSearch", "fuse"]

# Reciprocal rank fusion's constant: the higher, the less the very top of each ranking outweighs the rest.
FUSE_K = 60


class TextSearch:
    """Tracks ranked by how well they match a description's vector, over one CLAP model's embeddings."""

    def __init__(self, catalog: Catalog, embeddings: EmbeddingTable):
        paths, vectors = embeddings.matrix()
        keep = [i for i, path in enumerate(paths) if path in catalog.tracks]
        self.tracks = [catalog.tracks[paths[i]] for i in keep]
        x = vectors[keep].astype(np.float32)
        self._x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) if len(x) else x

    def __len__(self) -> int:
        return len(self.tracks)

    def find(self, text: np.ndarray, *, count: int) -> list[TrackMatch]:
        """Up to `count` tracks, best match for `text` (a description's vector) first."""
        if not self.tracks:
            return []
        text = text / (np.linalg.norm(text) + 1e-12)
        similarity = self._x @ text
        return [TrackMatch(self.tracks[i], float(similarity[i])) for i in np.argsort(-similarity, kind="stable")[:count]]


def fuse(rankings: list[list[TrackMatch]], *, count: int) -> list[TrackMatch]:
    """Up to `count` tracks from several rankings of the same tracks (best first), by reciprocal rank fusion: each
    track scores 1 / (FUSE_K + its rank) in every ranking it's in, summed. A track missing from a ranking (no
    embedding there) scores nothing for it. Its similarity is its score over the best possible, 1 for a track
    first everywhere."""
    scores: dict[str, float] = {}
    tracks = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.track.path] = scores.get(item.track.path, 0.0) + 1 / (FUSE_K + rank)
            tracks[item.track.path] = item.track
    best = len(rankings) / (FUSE_K + 1)
    order = sorted(scores, key=lambda path: -scores[path])[:count]
    return [TrackMatch(tracks[path], scores[path] / best) for path in order]
