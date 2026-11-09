"""Client for language models, through LangChain: what a model knows about a track from its tags.

    from libcrate.services.llm import LLMClient

    llm = LLMClient("anthropic:claude-sonnet-5-5", effort="low")
    guess = llm.genres(record.tags, max_genres=3)
    guess.known, guess.styles          # True, [('Electronic---Deep House', 0.8), ...]

Any chat model LangChain's init_chat_model() supports works: "anthropic:...",
"openai:...", "google_genai:...", once that provider's LangChain package is
installed and its API key is set (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).
Extra keyword arguments go to init_chat_model (effort, temperature, ...).

Each question is a fixed system prompt, a pydantic answer schema and the track's
tags, sent through _ask(); the system prompt is what providers cache between
requests (marked explicitly for Anthropic; OpenAI and Gemini cache long repeated
prefixes on their own). Adding a question (moods, for example) means a prompt,
a schema and a method that calls _ask(). Available today:

  genres    Discogs styles ("Parent---Style"), the same 400 labels Essentia's genre
            model uses, so the two are comparable. The model says whether it knows
            the track at all: an unknown track gets no styles rather than a guess
            from the title's words.
  describe  A description of how the track sounds and feels, with search keywords,
            written from facts about it (its tags, genres from other sources, tags
            from an audio analysis) plus what the model knows of the track or its
            artist, if anything. Every track gets one, known or not; in an experiment,
            searching such descriptions beat searching knowledge-only ones.

"""

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from libcrate.track import Description, Tags

__all__ = ["DESCRIPTION_PROMPT_HASH", "DISCOGS_STYLES", "GENRE_PROMPT_HASH", "DescriptionGuess", "GenreGuess",
           "LLMClient", "LLMError", "Usage", "describe_track"]

DISCOGS_STYLES: list[str] = files("libcrate.services.llm").joinpath("discogs_styles.txt").read_text().split("\n")[:-1]

GENRE_PROMPT = f"""You identify the music style of tracks from their tags (artist, title, album, year, label).

Answer only from what you actually know about this artist or track:
- If you recognise the artist or the track well enough to judge its style, set "known" to true and give up to 3 styles, most fitting first.
- If you don't, set "known" to false and give no styles. Do not guess from the words of the title, album or label.
- Use only labels from the list below, copied exactly ("Parent---Style").
- "confidence" is how sure you are that the style fits, from 0 to 1.

Styles:
{chr(10).join(DISCOGS_STYLES)}"""

_LABELS_BY_KEY = {label.lower(): label for label in DISCOGS_STYLES}

DESCRIPTION_PROMPT = """You describe music tracks for a search engine: how each sounds and feels, so people can find it by
describing what they'd like to hear. Cover genre and scene, mood, energy and tempo, vocals and their language,
instruments and production, era, and where it fits (late night, workout, party, ...). Write plainly, no praise.

You get facts about the track: its tags, genres from several sources, and tags from an automatic audio analysis.
The audio tags are the analysis' best guesses and are often partly wrong: use those that agree with each other and
with the genres, and ignore the rest. If you recognise the track or its artist, add what you know (scene, era,
language, themes) and set "recognised" to true; if you don't, set it to false and describe the track from the facts
alone, without inventing details. Never guess anything from the words of the title, album or label.

Describe only the music. Never mention the facts, the tags or the analysis, whether they agree, or what you left
out: the description is searched word by word, so a mood you rule out would make the track match it.

Give 2-4 sentences, 8-15 short search keywords (e.g. "female vocals", "melancholic", "deep house", "Romanian"),
and the language of the lyrics if you know it."""


class LLMError(Exception):
    """The model can't be created, or a request failed."""


class StyleGuess(BaseModel):
    label: str = Field(description='A Discogs style from the list, e.g. "Electronic---Deep House"')
    confidence: float = Field(description="0 to 1")


class GenreAnswer(BaseModel):
    known: bool = Field(description="Whether you recognise the artist or track well enough to judge its style")
    styles: list[StyleGuess] = Field(description="Up to 3 styles, most fitting first; empty when not known")


class DescriptionAnswer(BaseModel):
    recognised: bool = Field(description="Whether you know this track, or its artist well, from your own knowledge")
    language: str | None = Field(description="Language of the lyrics, or null if instrumental or you can't tell")
    description: str = Field(description="2-4 sentences; empty only if the facts say nothing about the music")
    keywords: list[str] = Field(description="8-15 short search keywords")


def prompt_hash(prompt: str, answer: type[BaseModel]) -> str:
    """A question's system prompt and answer schema, hashed: part of its fetcher's key, so a change to either
    redoes the results."""
    question = prompt + json.dumps(answer.model_json_schema(), sort_keys=True)
    return hashlib.sha256(question.encode()).hexdigest()[:12]


GENRE_PROMPT_HASH = prompt_hash(GENRE_PROMPT, GenreAnswer)
DESCRIPTION_PROMPT_HASH = prompt_hash(DESCRIPTION_PROMPT, DescriptionAnswer)


@dataclass
class Usage:
    input: int = 0           # all prompt tokens, cached ones included
    output: int = 0          # output tokens, reasoning included
    cache_read: int = 0
    cache_creation: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input += other.input
        self.output += other.output
        self.cache_read += other.cache_read
        self.cache_creation += other.cache_creation
        return self


@dataclass
class GenreGuess:
    known: bool
    styles: list[tuple[str, float]]  # (Discogs label, confidence), most fitting first
    invalid_labels: list[str] = field(default_factory=list)  # answers outside the style list, dropped
    usage: Usage = field(default_factory=Usage)


def describe_track(tags: Tags) -> str:
    lines = [
        ("Artist", ", ".join(tags.artists)),
        ("Title", tags.title),
        ("Album", tags.album),
        ("Year", tags.year),
        ("Label", tags.label or tags.publisher),
    ]
    return "\n".join(f"{name}: {value}" for name, value in lines if value)


@dataclass
class DescriptionGuess:
    description: Description | None  # None when the model had nothing to say
    usage: Usage = field(default_factory=Usage)


class LLMClient:
    def __init__(self, model: str, *, chat=None, **model_kwargs):
        """
        model        -> LangChain model string, e.g. "anthropic:claude-sonnet-5-5".
        chat         -> a ready LangChain chat model (tests); otherwise init_chat_model(model, **model_kwargs).

        Defaults: the services.llm section of the settings (libcrate.settings).
        """
        self.model = model
        if chat is None:
            try:
                chat = init_chat_model(model, **model_kwargs)
            except Exception as e:  # unknown provider, missing provider package or API key
                raise LLMError(f"could not create model {model!r}: {e}") from e
        self._chat = chat
        self._structured = {}  # answer schema -> the chat model bound to it

    def genres(self, tags: Tags, *, max_genres: int) -> GenreGuess:
        """The Discogs styles the model knows the track by, at most `max_genres`, most confident first."""
        answer, usage = self._ask(GENRE_PROMPT, GenreAnswer, describe_track(tags))
        valid, invalid = [], []
        for s in answer.styles if answer.known else []:
            label = _LABELS_BY_KEY.get(s.label.strip().lower())
            if label is None:
                invalid.append(s.label)
            else:
                valid.append((label, min(max(s.confidence, 0.0), 1.0)))
        # The model doesn't reliably list the most confident style first, so sort before keeping the top ones.
        styles, seen = [], set()
        for label, confidence in sorted(valid, key=lambda lc: -lc[1]):
            if label not in seen:
                seen.add(label)
                styles.append((label, confidence))
        return GenreGuess(known=answer.known, styles=styles[:max_genres], invalid_labels=invalid, usage=usage)

    def describe(self, facts: str) -> DescriptionGuess:
        """A description of the track `facts` are about (lines of "Name: value", e.g. from describe_track plus genres)."""
        answer, usage = self._ask(DESCRIPTION_PROMPT, DescriptionAnswer, facts)
        if not answer.description.strip():
            return DescriptionGuess(None, usage)
        keywords = list(dict.fromkeys(k.strip() for k in answer.keywords if k.strip()))
        language = (answer.language or "").strip() or None
        return DescriptionGuess(Description(text=answer.description.strip(), keywords=keywords, language=language,
                                            recognised=answer.recognised), usage)

    def _ask[T: BaseModel](self, system_prompt: str, schema: type[T], text: str) -> tuple[T, Usage]:
        """Ask the model `text` under a fixed `system_prompt`, for an answer shaped like `schema`."""
        if self._chat is None:
            raise LLMError("the client is closed")
        if schema not in self._structured:
            # json_schema: the provider's native structured output. Also avoids forced tool
            # calls, which recent Claude models reject.
            self._structured[schema] = self._chat.with_structured_output(schema, method="json_schema",
                                                                          include_raw=True)
        block = {"type": "text", "text": system_prompt}
        if getattr(self._chat, "_llm_type", "") == "anthropic-chat":
            block["cache_control"] = {"type": "ephemeral"}  # Anthropic caches only marked prefixes
        try:
            out = self._structured[schema].invoke([SystemMessage(content=[block]), HumanMessage(content=text)])
        except Exception as e:  # provider errors (network, rate limit, refusal) all look different
            raise LLMError(f"{type(e).__name__}: {e}") from e
        answer = out.get("parsed")
        if answer is None:
            raise LLMError(f"unparseable answer: {out.get('parsing_error')}")
        return answer, _usage(out.get("raw"))

    def close(self) -> None:
        """Drop the chat model; requests fail afterwards. LangChain shares its HTTP connections
        between all chat models in the process and closes them itself, so this doesn't touch them."""
        self._chat = None
        self._structured.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _usage(message) -> Usage:
    meta = getattr(message, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    # LangChain reports cache writes either as cache_creation or, when the provider splits
    # them by cache lifetime, under these keys with cache_creation set to 0.
    written = sum(details.get(k) or 0 for k in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"))
    return Usage(
        input=meta.get("input_tokens", 0),
        output=meta.get("output_tokens", 0),
        cache_read=details.get("cache_read", 0) or 0,
        cache_creation=written,
    )
