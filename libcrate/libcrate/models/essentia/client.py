"""Client for Essentia's audio models: everything they can tell about a track from its audio.

    from libcrate.models.essentia import EssentiaClient

    with EssentiaClient(models_dir, GENRES, max_seconds=300, processes=4) as essentia:   # load on the first file
        essentia.genres("song.flac")[:3]           # [('Electronic---Deep House', 0.41), ...]
        essentia.embeddings("song.flac").shape     # (frames, 1280)
        essentia.track_embeddings("song.flac")     # (frames, 1280), for track similarity
                                                   # leaving `with` (or close()) unloads them

The audio is decoded to 16 kHz mono and turned into Discogs-EffNet embeddings,
which classification heads score. Available today:

  genres       the genre_discogs400 head: one score in [0, 1] per Discogs style
               ("Parent---Style"), averaged over the track
  embeddings   the Discogs-EffNet embeddings themselves, one row per frame
  track_embeddings
               the same network trained for track similarity (tracks of the same
               release close together), one row per frame

Adding another head (moods, for example) means a method on Models (models.py),
its name in Models.REQUESTS, and a method here that calls self._request().

The models run in child processes (worker.py, through models.workers.WorkerPool),
up to `processes` of them, each analysing one file at a time; they start on the
first request that needs them and stop on close(). TensorFlow never returns its
memory (about 1.2 GB after one track) while its process lives, so this is the only
way to unload it; the calling process never imports TensorFlow at all.

Requirements: the models' folders in `models_dir` (models.MODEL_FILES); libcrate.models downloads them
(crate: Settings › Models).
"""

from pathlib import Path

import numpy as np

from libcrate.models.essentia.models import (
    EMBEDDING_MODEL, GENRE_MODEL, TRACK_EMBEDDING_MODEL, EssentiaError, check_requirements,
)
from libcrate.models.workers import WorkerPool

__all__ = ["EMBEDDINGS", "GENRES", "TRACK_EMBEDDINGS", "EssentiaClient", "EssentiaError"]

# What each request needs: the models to check for before starting.
GENRES = (EMBEDDING_MODEL, GENRE_MODEL)
EMBEDDINGS = (EMBEDDING_MODEL,)
TRACK_EMBEDDINGS = (TRACK_EMBEDDING_MODEL,)


class EssentiaClient:
    def __init__(self, models_dir: Path, models: tuple[str, ...], *, max_seconds: float, processes: int):
        """
        models_dir  -> the models folder, a subfolder per model (models.MODEL_FILES).
        models      -> the models the requests to come need (GENRES, EMBEDDINGS, TRACK_EMBEDDINGS).
        max_seconds -> only the first N seconds are analysed; much faster, and enough for genre.
        processes   -> how many files can be analysed at once, each in its own process
                       holding its own copy of the models (about 1.2 GB each).

        Raises EssentiaError right away if one of `models` is missing; they load on their first request.
        """
        check_requirements(models_dir, models)
        self.models_dir = Path(models_dir).expanduser()
        self.max_seconds = max_seconds
        self._pool = WorkerPool("libcrate.models.essentia.worker", [self.models_dir, max_seconds],
                                processes=processes, error=EssentiaError, label="Essentia")
        self.processes = self._pool.processes

    def genres(self, path: str | Path) -> list[tuple[str, float]]:
        """[(Discogs style label, score)] for all 400 styles, highest first."""
        return [tuple(pair) for pair in self._request("genres", path)]

    def embeddings(self, path: str | Path) -> np.ndarray:
        """Discogs-EffNet embeddings, shape (frames, 1280): one row per ~1 s of audio."""
        return self._request("embeddings", path)

    def track_embeddings(self, path: str | Path) -> np.ndarray:
        """Discogs-EffNet track-similarity embeddings, shape (frames, 1280): one row per ~1 s of audio."""
        return self._request("track_embeddings", path)

    def _request(self, name: str, path: str | Path):
        """Have a free child process run Models.<name>(path) and return its result."""
        return self._pool.request(name, str(path))

    def close(self) -> None:
        """Stop the child processes, which frees the models' memory. Later requests start new ones."""
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
