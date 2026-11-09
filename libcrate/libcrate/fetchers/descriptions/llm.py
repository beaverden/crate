import hashlib
from collections.abc import Callable

from libcrate.models.clap import MODELS as CLAP_MODELS
from libcrate.models.clap import ClapClient
from libcrate.services.llm.client import DESCRIPTION_PROMPT_HASH, LLMClient, describe_track
from libcrate.embeddings import EmbeddingTable
from libcrate.fetchers.base import Input, LookupNotFound
from libcrate.fetchers.descriptions.audio_tags import AudioTags
from libcrate.fetchers.descriptions.base import DescriptionFetcher
from libcrate.fetchers.genres.llm import llm_service
from libcrate.track import Description, SourceStatus, TrackRecord

__all__ = ["FACT_SOURCES", "LLMDescriptionFetcher"]

# Genre fetchers whose results are given as facts, and how the facts name them: not llm-genres, the same kind of
# model's own knowledge. The names are part of the facts, so of the key: changing one redoes every description.
FACT_SOURCES = {"musicbrainz-genres": "musicbrainz", "chosic-genres": "chosic", "essentia-genres": "essentia"}


class LLMDescriptionFetcher(DescriptionFetcher):
    """A language model's description of how the track sounds and feels (paid API calls), from facts: the tags,
    the other sources' genres and, with audio tags, phrases a CLAP model says fit the track; plus what the model
    knows of the track or artist. Every track with an artist and a title gets one, known to the model or not.

    The key holds a hash of the facts (but the audio tags, which shift a little as the library grows), so a track is
    described again when its tags or genres change, e.g. after another genre source ran on it. The audio tags come
    from the options' audio_tags_from embeddings, once they're computed (open()).
    """

    name = "llm-descriptions"

    def __init__(self, *args, audio_tags: AudioTags | None = None, **kwargs):
        """audio_tags -> ready audio tags (tests); otherwise open() makes them from the embeddings, if there are any."""
        super().__init__(*args, **kwargs)
        self.audio_tags = audio_tags

    @property
    def workers(self) -> int:
        return max(1, self.options.workers)

    @property
    def model(self) -> str:
        return llm_service(self.settings, self.options.service).model

    @property
    def uses_audio_tags(self) -> bool:
        """Audio tags are asked for, from embeddings a fetcher that's on makes (so its model is needed)."""
        source = self.options.audio_tags_from
        try:
            on = self.settings.fetchers.get(source).enabled
        except KeyError:
            on = False
        return self.options.audio_tags > 0 and on

    def services(self) -> list[str]:
        return [self.options.service]

    def models(self) -> list[str]:
        source = self.options.audio_tags_from
        return [source] if self.uses_audio_tags and source in CLAP_MODELS else []

    def inputs(self) -> list[Input]:
        inputs = [Input(name) for name in FACT_SOURCES]
        return inputs + [Input(self.options.audio_tags_from)] if self.options.audio_tags > 0 else inputs

    def make_client(self) -> LLMClient:
        service = llm_service(self.settings, self.options.service)
        return LLMClient(service.model, **service.model_kwargs(self.settings.keys))

    def open(self, embeddings: Callable[[str], EmbeddingTable] | None = None) -> "LLMDescriptionFetcher":
        super().open(embeddings)
        source = self.options.audio_tags_from
        if self.audio_tags is None and self.uses_audio_tags and embeddings is not None and source in CLAP_MODELS:
            table = embeddings(source)
            if len(table):
                clap = ClapClient(self.settings.models.folder, source, processes=1)
                self.audio_tags = AudioTags(table, clap, count=self.options.audio_tags)
        return self

    def facts(self, record: TrackRecord) -> str:
        """The facts the model gets, but the audio tags: "Name: value" lines."""
        lines = [describe_track(record.tags)]
        if record.tags.genre:
            lines.append(f"Genre tags: {', '.join(record.tags.genre)}")
        for source, label in FACT_SOURCES.items():
            result = record.genres.get(source)
            if result and result.status == SourceStatus.FOUND and result.genres:
                lines.append(f"Genres ({label}): " + ", ".join(
                    f"{g.name} ({g.confidence:.0%})" if g.confidence is not None else g.name for g in result.genres))
        return "\n".join(lines)

    def key(self, record: TrackRecord) -> str | None:
        if not (record.tags.artists and record.tags.title):
            return None
        facts = hashlib.sha256(self.facts(record).encode()).hexdigest()[:12]
        return f"{self.model}:prompt={DESCRIPTION_PROMPT_HASH}:facts={facts}"

    def fetch(self, record: TrackRecord) -> Description:
        facts = self.facts(record)
        if self.audio_tags is not None and (tags := self.audio_tags.tags(record.path)):
            facts += f"\nAudio analysis tags (automatic, may be wrong): {', '.join(tags)}"
        guess = self.client.describe(facts)
        if guess.description is None:
            raise LookupNotFound("the model had nothing to say about this track")
        return guess.description

    def close(self) -> None:
        super().close()
        if self.audio_tags is not None:
            self.audio_tags.close()
