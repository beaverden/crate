"""Convert one of LAION's original CLAP checkpoints into transformers' CLAP format.

    python -m libcrate.models.clap.convert CHECKPOINT.pt FOLDER

FOLDER already holds the matching config.json (from the Hugging Face copy, which has the
architecture right); this adds model.safetensors. Run in a child process by
LaionCheckpoint.download(), so the caller never loads torch or the 2.35 GB checkpoint.

The renaming follows transformers' former convert_clap_original_pytorch_to_hf.py
(Apache-2.0), checked against LAION's laion_clap package: the converted model gives
the same text embeddings and the same search results on test_music/.
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import scalar
from transformers import ClapConfig, ClapModel

__all__ = ["convert", "rename"]

# Original name part -> transformers' (in this order).
RENAMES = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}
_SEQUENTIAL = re.compile(r".*sequential.(\d+).*")
_PROJECTION = re.compile(r".*_projection.(\d+).*")
# What the original keeps that transformers doesn't use: buffers it works out itself (the spectrogram, attention
# masks), and heads only used in training (HTSAT's classifier, the contrastive loss's transforms).
_NOT_IN_TRANSFORMERS = re.compile(r"spectrogram_extractor|logmel_extractor|attn_mask|tscam_conv|encoder\.head\.|_transform\.")
# The checkpoint pickles numpy scalars (its epoch counters), under numpy 1's module name.
_SAFE_GLOBALS = [(scalar, "numpy.core.multiarray.scalar"), np.dtype,
                 *(getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType"))]


def rename(state_dict: dict) -> dict:
    """The original weights under transformers' names; the audio attention's qkv split into query, key, value."""
    out = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        for old, new in RENAMES.items():
            if old in key:
                key = key.replace(old, new)
        if match := _SEQUENTIAL.match(key):
            layer = match.group(1)
            key = key.replace(f"sequential.{layer}.", f"layers.{int(layer) // 3}.linear.")
        elif match := _PROJECTION.match(key):
            layer = int(match.group(1))  # nn.Sequential(Linear, ReLU, Linear): 0 and 2
            key = key.replace(f"_projection.{layer}.", f"_projection.linear{1 if layer == 0 else 2}.")
        if "qkv" in key:
            third = value.size(0) // 3
            for i, part in enumerate(("query", "key", "value")):
                out[key.replace("qkv", part)] = value[i * third:(i + 1) * third]
        else:
            out[key] = value
    return out


def convert(checkpoint: Path, folder: Path) -> None:
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ClapModel(ClapConfig.from_pretrained(folder))
    result = model.load_state_dict(rename(original["state_dict"]), strict=False)
    missing = [k for k in result.missing_keys if not k.endswith("token_type_ids")]  # a buffer, set by transformers
    unexpected = [k for k in result.unexpected_keys if not _NOT_IN_TRANSFORMERS.search(k)]
    if missing or unexpected:
        raise SystemExit(f"weights don't match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.save_pretrained(folder)


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
