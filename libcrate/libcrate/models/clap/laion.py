"""LAION's CLAP, converted into transformers' CLAP format (convert.py): music and descriptions in one space.

Imports torch and transformers: only the child process (worker.py) imports this.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

__all__ = ["LaionClap"]

SAMPLE_RATE = 48000


class LaionClap:
    """audio(clips) and text(texts): one vector each (512 dimensions for LAION's music model)."""

    def __init__(self, folder: Path):
        self._model = ClapModel.from_pretrained(folder).eval()
        self._processor = ClapProcessor.from_pretrained(folder)

    def audio(self, clips: list[np.ndarray]) -> np.ndarray:
        """`clips`: mono audio at SAMPLE_RATE, 10 s each (or less: the processor repeats a short one)."""
        with torch.no_grad():
            inputs = self._processor(audio=clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            return self._model.get_audio_features(**inputs).pooler_output.numpy()

    def text(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self._processor(text=list(texts), return_tensors="pt", padding=True)
            return self._model.get_text_features(**inputs).pooler_output.numpy()
