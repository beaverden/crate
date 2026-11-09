"""A map of a catalog's tracks from one embedding model: similar-sounding tracks close together, in clusters.

    table = store.current_embeddings(catalog, settings, settings.map.embeddings)
    music_map = store.music_map(settings.map.embeddings, clusters=settings.map.clusters, table=table)   # cached
    music_map.paths, music_map.xy, music_map.clusters                         # one row per track

How it's laid out:
  1. The embeddings are L2-normalised and mean-centred, as for similar tracks
     (libcrate.embeddings.similarity_space), then reduced to 50 dimensions by PCA.
  2. t-SNE places them in 2-D, keeping each track's nearest neighbours (by cosine
     similarity) close; distances between far-apart regions mean less.
  3. k-means on the same vectors gives each track a cluster, for colouring.

It takes several seconds for a few thousand tracks, so the result is saved with a
fingerprint of what it was made from (each track's path and embedding key, and the
settings); MetadataStore.music_map() only recomputes when that changes.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from libcrate.embeddings import EmbeddingTable, similarity_space

__all__ = ["MIN_TRACKS", "MusicMap"]

# Fewer tracks than this make no map.
MIN_TRACKS = 10
# Bump when the layout changes, so saved maps are made again.
VERSION = 1
PCA_DIMENSIONS = 50
PERPLEXITY = 30.0  # t-SNE: about how many neighbours each track keeps close
_FINGERPRINT = b"fingerprint"


@dataclass
class MusicMap:
    paths: list[str]         # catalog paths, sorted
    xy: np.ndarray           # float32, shape (tracks, 2)
    clusters: np.ndarray     # int, shape (tracks,): 0 .. clusters-1
    fingerprint: str = ""    # what it was made from (fingerprint_of())

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def fingerprint_of(table: EmbeddingTable, clusters: int) -> str:
        """Changes when a track's embedding or the settings do."""
        digest = hashlib.sha256(f"v{VERSION} clusters={clusters}\n".encode())
        for path in table.matrix()[0]:
            digest.update(f"{path}\t{table.key(path)}\n".encode())
        return digest.hexdigest()

    @classmethod
    def compute(cls, table: EmbeddingTable, *, clusters: int) -> "MusicMap":
        """Lay out every track of `table`. Raises ValueError with fewer than MIN_TRACKS tracks."""
        paths, vectors = table.matrix()
        if len(paths) < MIN_TRACKS:
            raise ValueError(f"a map needs at least {MIN_TRACKS} tracks with embeddings, there are {len(paths)}")
        # Here rather than at the top: scikit-learn takes seconds to import, which every command and the app's
        # start would pay, while only making a map needs it.
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        x = similarity_space(vectors)
        reduced = PCA(n_components=min(PCA_DIMENSIONS, *x.shape), random_state=0).fit_transform(x)
        xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=min(PERPLEXITY, (len(x) - 1) / 3),
                  random_state=0).fit_transform(reduced)
        labels = KMeans(n_clusters=min(clusters, len(x)), n_init=10, random_state=0).fit_predict(x)
        return cls(paths, xy.astype(np.float32), labels.astype(np.int32), cls.fingerprint_of(table, clusters))

    @classmethod
    def load(cls, path: str | Path) -> "MusicMap | None":
        """The map saved at `path`, or None if there's none."""
        path = Path(path)
        if not path.is_file():
            return None
        data = pq.read_table(path)
        xy = np.column_stack([data["x"].to_numpy(), data["y"].to_numpy()]).astype(np.float32)
        fingerprint = (data.schema.metadata or {}).get(_FINGERPRINT, b"").decode()
        return cls(data["path"].to_pylist(), xy, data["cluster"].to_numpy().astype(np.int32), fingerprint)

    def save(self, path: str | Path) -> None:
        """Write atomically, so an interrupted save never leaves a truncated file."""
        path = Path(path)
        data = pa.table({
            "path": pa.array(self.paths, type=pa.string()),
            "x": pa.array(self.xy[:, 0], type=pa.float32()),
            "y": pa.array(self.xy[:, 1], type=pa.float32()),
            "cluster": pa.array(self.clusters, type=pa.int32()),
        }).replace_schema_metadata({_FINGERPRINT: self.fingerprint.encode()})
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(data, tmp)
        tmp.replace(path)
