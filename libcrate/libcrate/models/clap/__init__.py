"""CLAP models (audio and text in one space) and text-only models for descriptions, run in a child process."""

from libcrate.models.clap.client import ClapClient
from libcrate.models.clap.models import MODELS, TEXT_MODELS, ClapError

__all__ = ["MODELS", "TEXT_MODELS", "ClapClient", "ClapError"]
