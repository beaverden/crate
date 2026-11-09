"""Similar tracks: the tracks of a catalog that sound most like a seed track, best match first.

    settings = store.settings()
    similar = Similar(catalog, store.current_embeddings(catalog, settings, settings.similar.embeddings))
    for item in similar.find("Metronomy - The Light [GBMVH1900150].flac", **settings.similar.options()):
        item.track.tags.title, item.similarity

Similarity is cosine similarity on mean-centred, L2-normalised embeddings
(centring stops a few "average" tracks from being everyone's neighbour). The
list is the closest tracks in order, nothing skipped, so it's the same every
time (Radio is the one that varies artists and leaves out other versions).

Tracks without an embedding in the table are left out.
"""

import numpy as np

from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable, TrackMatch, similarity_space

__all__ = ["Similar"]


class Similar:
    """Tracks ranked by similarity to a seed, over one embedding model's vectors of a catalog's tracks."""

    def __init__(self, catalog: Catalog, embeddings: EmbeddingTable):
        paths, vectors = embeddings.matrix()
        keep = [i for i, path in enumerate(paths) if path in catalog.tracks]
        self.tracks = [catalog.tracks[paths[i]] for i in keep]
        self._index = {track.path: i for i, track in enumerate(self.tracks)}
        self._x = similarity_space(vectors[keep])

    def __len__(self) -> int:
        return len(self.tracks)

    def __contains__(self, path: str) -> bool:
        return path in self._index

    def find(self, seed: str, *, count: int) -> list[TrackMatch]:
        """The track at path `seed`, then the `count` tracks most like it, best first. Raises KeyError if the
        seed has no embedding."""
        if seed not in self._index:
            raise KeyError(f"{seed} has no embedding")
        x, start = self._x, self._index[seed]
        similarity = x @ x[start]
        order = [i for i in np.argsort(-similarity, kind="stable") if i != start]
        return [TrackMatch(self.tracks[i], float(similarity[i])) for i in [start, *order[:count]]]
