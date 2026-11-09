import hashlib

from libcrate.models.clap.client import ClapClient
from libcrate.models.clap.models import TEXT_MODELS
from libcrate.embeddings import Embedding
from libcrate.fetchers.base import Input
from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.track import TrackRecord

__all__ = ["DescriptionTextFetcher"]


class DescriptionTextFetcher(EmbeddingFetcher):
    """A text model's embedding of the track's descriptions (TrackRecord.description_text(): each found description
    with its keywords), L2-normalised, for searching them by what they say; `model` embeds the search query. Tracks
    without a description are left out; a changed description is embedded again. Runs locally, no network, in one
    process (a small model, given all the cores)."""

    name = "description-bge"
    space = "text"
    model = "bge-base"
    # The description fetchers whose descriptions it embeds.
    descriptions = ("llm-descriptions",)

    def models(self) -> list[str]:
        return [self.model]

    def inputs(self) -> list[Input]:
        return [Input(name, required=True) for name in self.descriptions]

    def make_client(self) -> ClapClient:
        return ClapClient(self.settings.models.folder, self.model, processes=1)

    def key(self, record: TrackRecord) -> str | None:
        text = record.description_text()
        if not text:
            return None
        return f"{self.model}:{TEXT_MODELS[self.model].revision[:12]}:text={hashlib.sha256(text.encode()).hexdigest()[:12]}"

    def fetch(self, record: TrackRecord) -> Embedding:
        [vector] = self.client.text_embeddings([record.description_text()])
        return Embedding(vector)
