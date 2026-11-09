"""A text-only embedding model (BAAI's bge), for matching descriptions of tracks to what's searched for.

Imports torch and transformers: only the child process (worker.py) imports this.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

__all__ = ["BgeText"]

BATCH = 32
MAX_TOKENS = 512


class BgeText:
    """text(texts): one L2-normalised vector per text (the first token's, as bge is trained). No audio."""

    def __init__(self, folder: Path):
        self._tokenizer = AutoTokenizer.from_pretrained(folder)
        self._model = AutoModel.from_pretrained(folder).eval()

    def audio(self, clips: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("a text model has no audio side")

    def text(self, texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), BATCH):
                tokens = self._tokenizer(texts[start:start + BATCH], padding=True, truncation=True,
                                         max_length=MAX_TOKENS, return_tensors="pt")
                first = self._model(**tokens).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(first, dim=-1).numpy())
        return np.concatenate(out)
