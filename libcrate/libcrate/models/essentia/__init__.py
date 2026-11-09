"""Essentia's audio models, run in a child process."""

from libcrate.models.essentia.client import EMBEDDINGS, GENRES, TRACK_EMBEDDINGS, EssentiaClient
from libcrate.models.essentia.models import MODEL_FILES, EssentiaError

__all__ = ["EMBEDDINGS", "GENRES", "MODEL_FILES", "TRACK_EMBEDDINGS", "EssentiaClient", "EssentiaError"]
