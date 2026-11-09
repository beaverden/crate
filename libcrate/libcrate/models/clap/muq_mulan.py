# Rebuilt from Tencent's MuQ (https://github.com/tencent-ailab/MuQ), under this licence:
#
# MIT License
#
# Copyright (c) Tencent.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tencent's MuQ-MuLan: music and descriptions of it in one 512-dimension space.

The network, for running it only (no training), rebuilt from the `muq` package
(https://github.com/tencent-ailab/MuQ, MIT licence; the weights are CC BY-NC 4.0),
whose own code needs transformers 4. Same weights, same numbers:

  audio (24 kHz, 10 s clips) -> mel spectrogram (dB, normalised) -> 2 residual conv blocks
        -> 12-layer conformer (transformers' Wav2Vec2ConformerEncoder) -> linear 1024->768
        -> mean over time -> linear 768->512 -> L2 norm
  text -> XLM-RoBERTa base -> 8-layer transformer of MuLan's own -> mean over tokens
        -> linear 768->512 -> L2 norm

The folder holds the release's pytorch_model.bin (both halves, 2.65 GB) and, in text/,
XLM-RoBERTa's config and tokenizer. Each half is built on its first use, so a process that
only embeds tracks (Enrich) never holds the text half, and the reverse (Describe).

Imports torch and transformers: only the child process (worker.py) imports this.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
from transformers.audio_utils import mel_filter_bank
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig, Wav2Vec2ConformerEncoder,
)

__all__ = ["MuqMulan"]

SAMPLE_RATE = 24000
WEIGHTS = "pytorch_model.bin"
TEXT_FOLDER = "text"
# The mel spectrogram MuQ was trained on, and its mean and spread over the training set.
N_FFT, HOP, N_MELS = 2048, 240, 128
MEL_MEAN, MEL_STD = 6.768444971712967, 18.417922652295623


class MuqMulan:
    """audio(clips) and text(texts): one L2-normalised 512-dimension vector each."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        # Memory-mapped, and used in place (_load): only the half that gets built is read, and processes
        # holding the same half share its memory.
        self._weights = torch.load(self.folder / WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
        self._audio: _AudioTower | None = None
        self._text: _TextTower | None = None
        self._tokenizer = None

    def audio(self, clips: list[np.ndarray]) -> np.ndarray:
        """`clips`: mono audio at SAMPLE_RATE, 10 s each (or less). Shape (clips, 512)."""
        if self._audio is None:
            self._audio = _load(_AudioTower(), self._weights, "mulan.audio.model.model.", {
                "mulan.audio.proj.": "proj.", "mulan.audio_to_latents.": "to_latent.",
            })
        with torch.no_grad():  # one at a time: as fast as together, with less memory at the peak
            return np.concatenate([self._audio(torch.from_numpy(c[None].astype(np.float32))).numpy() for c in clips])

    def text(self, texts: list[str]) -> np.ndarray:
        """Shape (texts, 512). One text at a time: MuLan averages over every token, padding included."""
        if self._text is None:
            config = XLMRobertaConfig.from_pretrained(self.folder / TEXT_FOLDER)
            self._text = _load(_TextTower(config), self._weights, None, {
                "mulan.text.model.": "roberta.", "mulan.text.transformer.": "transformer.",
                "mulan.text_to_latents.": "to_latent.",
            })
            self._tokenizer = AutoTokenizer.from_pretrained(self.folder / TEXT_FOLDER)
        with torch.no_grad():
            return np.concatenate([self._text(self._tokenizer([t], return_tensors="pt")).numpy() for t in texts])


def _load(module: nn.Module, weights: dict, audio_prefix: str | None, renames: dict[str, str]) -> nn.Module:
    """`module` with its weights from the release's: names starting with a key of `renames` get its value
    instead, and with `audio_prefix` the MuQ encoder's go to `muq.`. Everything the module has must be there."""
    if audio_prefix:
        renames = {audio_prefix: "muq.", **renames}
    state = {}
    for name, tensor in weights.items():
        for old, new in renames.items():
            if name.startswith(old):
                state[new + name[len(old):]
                      .replace("conv.weight_g", "conv.parametrizations.weight.original0")
                      .replace("conv.weight_v", "conv.parametrizations.weight.original1")] = tensor
    unused = set(state) - set(module.state_dict())
    for name in unused:  # what the release keeps for training: the RVQ, MuQ's prediction head, RoBERTa's pooler
        del state[name]
    module.load_state_dict(state, strict=True, assign=True)  # use the mapped tensors: no copy
    return module.eval()


# -- audio ----------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: tuple[int, int]):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.conv3 = nn.Conv2d(inputs, outputs, 3, padding=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(outputs)

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(self.bn3(self.conv3(x)) + out)


class _Subsampling(nn.Module):
    """Two residual conv blocks: a quarter of the frames, then a linear layer to the conformer's width."""

    def __init__(self, channels=512, width=1024, bands=N_MELS):
        super().__init__()
        self.conv = nn.Sequential(_ResidualBlock(1, channels, (2, 2)), _ResidualBlock(channels, channels, (2, 2)))
        self.linear = nn.Linear(channels * bands // 4, width)

    def forward(self, mel):  # (batch, bands, frames)
        x = self.conv(mel.unsqueeze(1))  # (batch, channels, bands / 4, frames / 4)
        return self.linear(x.permute(0, 3, 1, 2).flatten(2))  # (batch, frames / 4, width)


class _Muq(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _Subsampling()
        self.conformer = Wav2Vec2ConformerEncoder(Wav2Vec2ConformerConfig(
            hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, intermediate_size=4096,
            hidden_act="swish", conv_depthwise_kernel_size=31, position_embeddings_type="rotary",
            rotary_embedding_base=10000, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
            layer_norm_eps=1e-5, max_source_positions=5000, attn_implementation="eager",
        ))
        filters = mel_filter_bank(num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=N_MELS, min_frequency=0,
                                  max_frequency=SAMPLE_RATE / 2, sampling_rate=SAMPLE_RATE, norm=None, mel_scale="htk")
        self.register_buffer("mel_filters", torch.from_numpy(filters.T).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def mel(self, audio):
        """torchaudio's MelSpectrogram (power) then AmplitudeToDB, as MuQ uses them, less the last frame."""
        spec = torch.stft(audio, N_FFT, HOP, window=self.window, center=True, pad_mode="reflect", return_complex=True)
        mel = self.mel_filters @ spec.abs().pow(2)
        return 10 * torch.log10(mel.clamp(min=1e-10))[..., :-1]

    def forward(self, audio):
        x = self.conv((self.mel(audio) - MEL_MEAN) / MEL_STD)
        return self.conformer(x).last_hidden_state


class _AudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.muq = _Muq()
        self.proj = nn.Linear(1024, 768)
        self.to_latent = nn.Linear(768, 512)

    def forward(self, audio):
        return F.normalize(self.to_latent(self.proj(self.muq(audio)).mean(dim=1)), dim=-1)


# -- text -----------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    """Layer norm without a bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.learned_gamma)


class _Attention(nn.Module):
    """Pre-norm attention with L2-normalised, learnt-scale queries and keys."""

    def __init__(self, dim=768, heads=8, dim_head=64, scale=8):
        super().__init__()
        self.heads, self.scale = heads, scale
        self.norm = _LayerNorm(dim)
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(dim, heads * dim_head * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim, bias=False))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v))  # (batch, heads, n, d)
        q = F.normalize(q, dim=-1) * self.q_scale
        k = F.normalize(k, dim=-1) * self.k_scale
        attention = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.to_out((attention @ v).transpose(1, 2).flatten(2))


class _GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def _feed_forward(dim=768, mult=4) -> nn.Sequential:
    hidden = int(dim * mult * 2 / 3)
    return nn.Sequential(_LayerNorm(dim), nn.Linear(dim, hidden * 2, bias=False), _GEGLU(), nn.Identity(),
                         nn.Linear(hidden, dim, bias=False))


class _Transformer(nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        self.layers = nn.ModuleList(nn.ModuleList([_Attention(), _feed_forward()]) for _ in range(depth))

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x


class _TextTower(nn.Module):
    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.transformer = _Transformer()
        self.to_latent = nn.Linear(768, 512)

    def forward(self, tokens):
        x = self.roberta(**tokens).last_hidden_state
        return F.normalize(self.to_latent(self.transformer(x).mean(dim=1)), dim=-1)
