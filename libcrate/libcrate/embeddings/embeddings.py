"""Embeddings: one vector per track from one model, kept in a Parquet file next to the catalog.

    table = EmbeddingTable.load(store.folder / "embeddings" / "discogs-effnet" / "mean.parquet")   # empty if missing
    table.get("Metronomy - The Light [GBMVH1900150].flac").vector    # float32, shape (1280,)
    paths, matrix = table.matrix()                                   # for similarity: one row per track

The file has one row per track: `path` (relative to the library root, like the
catalog's keys), `key` (the model and its settings; a changed key redoes the
track), `n_frames` (how many frames the vector averages) and `embedding`
(a fixed-size list of float32).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from libcrate.track import TrackRecord

__all__ = ["Embedding", "EmbeddingTable", "TrackMatch", "similarity_space"]


def similarity_space(vectors: np.ndarray) -> np.ndarray:
    """The vectors L2-normalised, mean-centred and normalised again (float32), so that a dot product is
    a cosine similarity; centring stops a few "average" tracks from being everyone's neighbour."""
    x = vectors.astype(np.float32)
    if len(x):
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x -= x.mean(axis=0)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


@dataclass(frozen=True)
class TrackMatch:
    """A track and how well it matches what was asked (similar tracks, a description)."""

    track: TrackRecord
    similarity: float  # cosine similarity: 1 for a perfect match, lower the less alike


@dataclass(frozen=True)
class Embedding:
    vector: np.ndarray  # float32, 1-D
    n_frames: int = 1   # how many frames (stretches of audio) the vector is the mean of


class EmbeddingTable:
    """One model's embeddings: relative path -> (key, Embedding)."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, Embedding]] = {}

    # --- files ----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingTable":
        """The table saved at `path`, or an empty one if there's no file yet."""
        table = cls()
        path = Path(path)
        if not path.is_file():
            return table
        data = pq.read_table(path)
        if data.num_rows == 0:
            return table
        column = data["embedding"].combine_chunks()
        vectors = column.values.to_numpy().reshape(len(column), -1).astype(np.float32)
        for rel, key, n_frames, vector in zip(
                data["path"].to_pylist(), data["key"].to_pylist(), data["n_frames"].to_pylist(), vectors):
            table._rows[rel] = (key, Embedding(vector, n_frames))
        return table

    def save(self, path: str | Path) -> None:
        """Write atomically, so an interrupted save never leaves a truncated file."""
        path = Path(path)
        rows = sorted(self._rows.items())
        # An empty table still needs a list size Arrow accepts.
        vectors = np.stack([e.vector for _, (_, e) in rows]) if rows else np.empty((0, 1))
        data = pa.table({
            "path": pa.array([rel for rel, _ in rows], type=pa.string()),
            "key": pa.array([key for _, (key, _) in rows], type=pa.string()),
            "n_frames": pa.array([e.n_frames for _, (_, e) in rows], type=pa.int32()),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel(), type=pa.float32()), list_size=vectors.shape[1]),
        })
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(data, tmp, compression="zstd")
        tmp.replace(path)

    # --- rows -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def key(self, rel: str) -> str | None:
        """What the track's embedding was made with, or None if it has none."""
        row = self._rows.get(rel)
        return row[0] if row else None

    def get(self, rel: str) -> Embedding | None:
        row = self._rows.get(rel)
        return row[1] if row else None

    def put(self, rel: str, key: str, embedding: Embedding) -> None:
        self._rows[rel] = (key, embedding)

    def keep_only(self, paths) -> None:
        """Drop the rows of tracks not in `paths` (gone from the catalog)."""
        keep = set(paths)
        self._rows = {rel: row for rel, row in self._rows.items() if rel in keep}

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """Paths, sorted, and their vectors stacked in the same order: shape (tracks, dims)."""
        paths = sorted(self._rows)
        if not paths:
            return [], np.empty((0, 0), dtype=np.float32)
        return paths, np.stack([self._rows[rel][1].vector for rel in paths])
