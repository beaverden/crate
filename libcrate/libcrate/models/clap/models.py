"""The CLAP models: audio and text in one space, so a track's sound can be matched to a description of it.

Each model has a folder, MODELS_DIR/<name>/; libcrate.models downloads them (with each
spec's download()), and nothing here does on its own.

Available today:

  muq-mulan          Tencent's MuQ-MuLan (large), trained on music and descriptions of
                     it, 512 dimensions: the release's weights as they are (2.65 GB,
                     CC BY-NC 4.0: not for commercial use) and XLM-RoBERTa's tokenizer,
                     run by our own rebuild of the network (muq_mulan.py). It told
                     genres apart best in a comparison on a 2,000-track library
                     (TASKS.md), techno from psytrance especially.
  laion-clap-music   LAION's CLAP trained on music (music_audioset_epoch_15_esc_90.14),
                     512 dimensions. The copy on Hugging Face (laion/larger_clap_music)
                     isn't this checkpoint and maps every text to nearly the same
                     vector, so LAION's original release is downloaded (2.35 GB) and
                     converted (convert.py) into transformers' CLAP format, a 780 MB
                     folder; its config and tokenizer files come from the Hugging Face
                     copy, which has the right architecture.

Next to them, TEXT_MODELS are text-only models, for searching what's been written about
tracks (their descriptions) rather than their sound; they live in the same folder, are
downloaded with them, and run the same way (text requests only):

  bge-base           BAAI's bge-base-en-v1.5 (440 MB), 768 dimensions. A search query
                     gets its query_prefix first, as bge is trained.

A track is embedded from CLIPS: CLIP_SECONDS from a quarter, half and three quarters
of the way through it, one vector each: the comparison's set-up, and on the CPU about
3 s a track for MuQ-MuLan, where the whole of a track would take ten times as long.

Another model is one more entry in MODELS: its files, sample rate, download() and
load(), which returns an object whose audio(clips) and text(texts) give its vectors.

Importing this is cheap: torch and transformers are only imported when Models is
created, which only the child process (worker.py) does.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

__all__ = ["CLIPS", "CLIP_SECONDS", "MODELS", "TEXT_MODELS", "ClapError", "HuggingFaceText", "LaionCheckpoint", "Models",
           "MuqMulanRelease", "check_requirements", "load_clips"]

# Where a track's clips start, as fractions of its length (each centred there), and how long they are.
CLIPS = (0.25, 0.5, 0.75)
CLIP_SECONDS = 10
# A track shorter than the clips together is cut into CLIP_SECONDS pieces instead; a shorter
# last piece than this is left out, unless it's the whole track.
MIN_CLIP_SECONDS = 2


class ClapError(Exception):
    """A model is missing or broken, or a file can't be analysed."""


@dataclass(frozen=True)
class MuqMulanRelease:
    """Tencent's MuQ-MuLan release, used as it is."""

    name: str
    repo: str            # Hugging Face repo with the weights
    revision: str
    text_repo: str       # Hugging Face repo with the text model's config and tokenizer
    text_revision: str

    sample_rate: ClassVar[int] = 24000
    query_prefix: ClassVar[str] = ""
    files: ClassVar[tuple[str, ...]] = ("pytorch_model.bin", "text/config.json", "text/tokenizer.json",
                                        "text/tokenizer_config.json", "text/sentencepiece.bpe.model")

    def download(self, folder: Path) -> None:
        """Fill `folder` (empty) with the model."""
        hf_hub_download(self.repo, self.files[0], revision=self.revision, local_dir=folder)
        for file in self.files[1:]:
            hf_hub_download(self.text_repo, Path(file).name, revision=self.text_revision, local_dir=folder / "text")
        for bookkeeping in (folder / ".cache", folder / "text" / ".cache"):  # hf_hub_download's
            shutil.rmtree(bookkeeping, ignore_errors=True)

    def load(self, folder: Path):
        from libcrate.models.clap.muq_mulan import MuqMulan  # torch: in the child process only
        return MuqMulan(folder)


@dataclass(frozen=True)
class LaionCheckpoint:
    """One of LAION's original CLAP checkpoints, converted into transformers' format."""

    name: str
    checkpoint: str      # file in LAION's release
    revision: str        # of LAION's release
    files_from: str      # Hugging Face repo with the matching config and tokenizer files
    files_revision: str

    sample_rate: ClassVar[int] = 48000
    query_prefix: ClassVar[str] = ""
    files: ClassVar[tuple[str, ...]] = ("model.safetensors", "config.json", "preprocessor_config.json",
                                        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                                        "special_tokens_map.json")

    def download(self, folder: Path) -> None:
        """Fill `folder` (empty) with the converted model."""
        for name in self.files[1:]:
            hf_hub_download(self.files_from, name, revision=self.files_revision, local_dir=folder)
        original = hf_hub_download("lukewys/laion_clap", self.checkpoint, revision=self.revision,
                                   local_dir=folder / ".original")
        # In a child process: converting loads the 2.35 GB checkpoint and torch, neither of which
        # should stay in the caller (the app).
        done = subprocess.run([sys.executable, "-m", "libcrate.models.clap.convert", str(original), str(folder)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            last = (done.stderr.strip().splitlines() or ["no output"])[-1]
            raise ClapError(f"could not convert {self.checkpoint}: {last}")
        shutil.rmtree(folder / ".original")
        shutil.rmtree(folder / ".cache", ignore_errors=True)  # hf_hub_download's bookkeeping

    def load(self, folder: Path):
        from libcrate.models.clap.laion import LaionClap  # torch: in the child process only
        return LaionClap(folder)


@dataclass(frozen=True)
class HuggingFaceText:
    """A text-only model from Hugging Face, used as it is."""

    name: str
    repo: str
    revision: str
    query_prefix: str    # put before a search query (not before the texts searched)

    files: ClassVar[tuple[str, ...]] = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                                        "special_tokens_map.json", "vocab.txt")

    def download(self, folder: Path) -> None:
        """Fill `folder` (empty) with the model."""
        for name in self.files:
            hf_hub_download(self.repo, name, revision=self.revision, local_dir=folder)
        shutil.rmtree(folder / ".cache", ignore_errors=True)  # hf_hub_download's bookkeeping

    def load(self, folder: Path):
        from libcrate.models.clap.text import BgeText  # torch: in the child process only
        return BgeText(folder)


# The first is Describe's default.
MODELS = {
    "muq-mulan": MuqMulanRelease(
        name="muq-mulan", repo="OpenMuQ/MuQ-MuLan-large", revision="2e01c796b71dca71b45251384c04cd7b237c9020",
        text_repo="FacebookAI/xlm-roberta-base", text_revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    ),
    "laion-clap-music": LaionCheckpoint(
        name="laion-clap-music", checkpoint="music_audioset_epoch_15_esc_90.14.pt",
        revision="b3708341862f581175dba5c356a4ebf74a9b6651",
        files_from="laion/larger_clap_music", files_revision="a0b4534a14f58e20944452dff00a22a06ce629d1",
    ),
}


TEXT_MODELS = {
    "bge-base": HuggingFaceText(
        name="bge-base", repo="BAAI/bge-base-en-v1.5", revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


def _all() -> dict:
    """Every model the folder holds, CLAP and text."""
    return {**MODELS, **TEXT_MODELS}


def check_requirements(models_dir: Path, name: str) -> Path:
    """The model's folder; raises ClapError if it's unknown or not downloaded yet. Cheap: imports no torch."""
    if name not in _all():
        raise ClapError(f"unknown CLAP model {name!r}; choose from {', '.join(_all())}")
    folder = Path(models_dir).expanduser() / name
    if not all((folder / f).is_file() for f in _all()[name].files):
        raise ClapError(f"missing model {name} in {models_dir} (Settings › Models)")
    return folder


def load_clips(path: str | Path, sample_rate: int) -> list[np.ndarray]:
    """The track's CLIPS, mono, at `sample_rate` (float32); a track shorter than all of them together is cut into
    CLIP_SECONDS pieces instead (the last one left out if under MIN_CLIP_SECONDS, unless it's the only one)."""
    try:
        info = soundfile.info(str(path))
        rate, frames, size = info.samplerate, info.frames, CLIP_SECONDS * info.samplerate
        if frames >= len(CLIPS) * size:
            starts = [min(max(int(frames * at) - size // 2, 0), frames - size) for at in CLIPS]
        else:
            starts = [s for s in range(0, frames, size) if s == 0 or frames - s >= MIN_CLIP_SECONDS * rate]
        clips = [soundfile.read(str(path), start=start, frames=size, dtype="float32", always_2d=True)[0].mean(axis=1)
                 for start in starts]
    except (soundfile.LibsndfileError, RuntimeError) as e:
        raise ClapError(f"could not decode {path}: {e}") from e
    if not clips or len(clips[0]) == 0:
        raise ClapError(f"no audio in {path}")
    if rate != sample_rate:
        common = gcd(sample_rate, rate)
        clips = [resample_poly(c, sample_rate // common, rate // common) for c in clips]
    return [c.astype(np.float32) for c in clips]


def _normalized(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class Models:
    """A loaded model. Imports torch and transformers, so only the child process (worker.py) uses this."""

    # The methods the client may call through the child process.
    REQUESTS = ("audio_embeddings", "text_embeddings", "query_embeddings")

    def __init__(self, name: str, folder: Path, threads: int):
        """threads -> how many CPU threads torch uses here (the cores shared among the processes)."""
        import torch  # here rather than at the top: seconds and hundreds of MB to import

        torch.set_num_threads(threads)
        self.spec = _all()[name]
        self._model = self.spec.load(folder)

    def audio_embeddings(self, path: str) -> np.ndarray:
        """One L2-normalised vector per clip of the track (load_clips): shape (clips, dims)."""
        return _normalized(self._model.audio(load_clips(path, self.spec.sample_rate))).astype(np.float32)

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        """One L2-normalised vector per text: shape (texts, dims)."""
        return _normalized(self._model.text(list(texts))).astype(np.float32)

    def query_embeddings(self, texts: list[str]) -> np.ndarray:
        """text_embeddings of search queries: with the model's query_prefix (bge's), if it has one."""
        return self.text_embeddings([self.spec.query_prefix + t for t in texts])
