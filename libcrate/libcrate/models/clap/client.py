"""Client for the CLAP models: a track's sound and descriptions of music, as vectors in one space.

    from libcrate.models.clap import ClapClient

    with ClapClient(models_dir, "muq-mulan", processes=2) as clap:   # loads on first use
        clap.audio_embeddings("song.flac").shape            # (3, 512): one per 10 s clip, L2-normalised
        clap.text_embeddings(["chill late-night electronic"])   # (1, 512), L2-normalised
        clap.query_embeddings(["chill late-night electronic"])  # the same for a search query (see models.TEXT_MODELS)

A text model (models.TEXT_MODELS: "bge-base") runs the same way, for text only.

A track and a description match as well as their vectors' dot product says.
The model runs in child processes (worker.py, through models.workers.WorkerPool),
so torch never loads in the caller; each holds its own copy (1.5-3 GB), and they share the
computer's cores.

Requirements: the model's folder in `models_dir`; libcrate.models downloads it (crate: Settings › Models).
"""

import os
from pathlib import Path

import numpy as np

from libcrate.models.clap.models import ClapError, check_requirements
from libcrate.models.workers import WorkerPool

__all__ = ["ClapClient", "ClapError"]


class ClapClient:
    def __init__(self, models_dir: Path, model: str, *, processes: int):
        """
        models_dir -> folder with the models (models.MODELS), one subfolder each.
        model      -> which one (a name in models.MODELS).
        processes  -> how many requests can run at once, each in its own process.

        Raises ClapError right away if the model is unknown or missing; it loads on the first request.
        """
        self.folder = check_requirements(models_dir, model)
        self.model = model
        processes = max(1, processes)
        # Each process gets its share of the cores: torch would otherwise give each all of them, and
        # processes fighting over the cores are slower than fewer would be.
        threads = max(1, (os.cpu_count() or 1) // processes)
        self._pool = WorkerPool("libcrate.models.clap.worker", [model, self.folder, threads],
                                processes=processes, error=ClapError, label="CLAP")
        self.processes = self._pool.processes

    def audio_embeddings(self, path: str | Path) -> np.ndarray:
        """One vector per clip of the track (models.load_clips): shape (clips, dims)."""
        return self._pool.request("audio_embeddings", str(path))

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        """One vector per text: shape (texts, dims)."""
        return self._pool.request("text_embeddings", list(texts))

    def query_embeddings(self, texts: list[str]) -> np.ndarray:
        """text_embeddings of search queries: a text model's query prefix first (the same as text_embeddings for
        CLAP models)."""
        return self._pool.request("query_embeddings", list(texts))

    def close(self) -> None:
        """Stop the child processes, which frees the model's memory. Later requests start new ones."""
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
