from libcrate.services.llm.client import GENRE_PROMPT_HASH, LLMClient
from libcrate.fetchers.genres.base import GenreFetcher, LookupNotFound
from libcrate.settings import LLMService, Settings
from libcrate.track import Genre, TrackRecord
from libcrate.util import split_label

__all__ = ["LLMGenreFetcher", "llm_service"]


def llm_service(settings: Settings, name: str) -> LLMService:
    """The language model service called `name`; ValueError if there's none (or it isn't a language model)."""
    service = getattr(settings.services, name, None)
    if not isinstance(service, LLMService):
        raise ValueError(f"no language model service called {name!r} (services)")
    return service


class LLMGenreFetcher(GenreFetcher):
    """Genres a language model knows for the track, from its tags (paid API calls).

    Names are lowercased Discogs styles with their parent and the model's
    confidence, like Essentia's. Tracks the model doesn't know are not_found,
    and so are tracks where no style reaches min_confidence.
    """

    name = "llm-genres"

    @property
    def workers(self) -> int:
        return max(1, self.options.workers)

    @property
    def model(self) -> str:
        return llm_service(self.settings, self.options.service).model

    def services(self) -> list[str]:
        return [self.options.service]

    def make_client(self) -> LLMClient:
        service = llm_service(self.settings, self.options.service)
        return LLMClient(service.model, **service.model_kwargs(self.settings.keys))

    def key(self, record: TrackRecord) -> str | None:
        # Nothing to ask about without an artist and a title. A new model, prompt or
        # threshold redoes tracks.
        if not (record.tags.artists and record.tags.title):
            return None
        return f"{self.model}:prompt={GENRE_PROMPT_HASH}:min={self.options.min_confidence}"

    def fetch(self, record: TrackRecord) -> list[Genre]:
        guess = self.client.genres(record.tags, max_genres=self.options.max_genres)
        if not guess.known:
            raise LookupNotFound("the model doesn't know this track")
        if not guess.styles:
            detail = f" (invalid labels: {', '.join(guess.invalid_labels)})" if guess.invalid_labels else ""
            raise LookupNotFound(f"the model gave no Discogs style{detail}")
        genres = []
        min_confidence = self.options.min_confidence
        for label, confidence in guess.styles:  # most confident first
            if confidence < min_confidence:
                break
            parent, style = split_label(label)
            genres.append(Genre(name=style.lower(), parent=parent, confidence=round(confidence, 4)))
        if not genres:
            best_label, best = guess.styles[0]
            raise LookupNotFound(f"no style above {min_confidence:.0%} (best: {best_label} {best:.0%})")
        return genres
