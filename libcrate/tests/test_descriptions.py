"""Descriptions: the llm description fetcher (fake chat model, fake CLAP client), its audio tags, enrich() into
TrackRecord.descriptions, the description embeddings, and fusing rankings. No API calls, no models."""

import numpy as np
import pytest

from libcrate.catalog import Catalog
from libcrate.services.llm.client import (
    DESCRIPTION_PROMPT, DESCRIPTION_PROMPT_HASH, DescriptionAnswer, LLMClient, prompt_hash,
)
from libcrate.embeddings import Embedding, EmbeddingTable, TrackMatch
from libcrate.fetchers import make_fetcher
from libcrate.fetchers.descriptions import PHRASES, AudioTags, LLMDescriptionFetcher
from libcrate.fetchers.embeddings import DescriptionTextFetcher, embed
from libcrate.fetchers.enrich import enrich, needs_lookup
from libcrate.settings import Settings
from libcrate.text_search import FUSE_K, fuse
from libcrate.track import Description, DescriptionSource, Genre, GenreSource, SourceStatus

ONE = "Artist A - One [AAA111].flac"
TWO = "Artist B - Two [BBB222].flac"
THREE = "sub/Artist C - Three.flac"


class FakeStructured:
    def __init__(self, answers):
        self.answers = answers  # title -> DescriptionAnswer
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        title = messages[1].content.split("Title: ")[1].split("\n")[0]
        answer = self.answers.get(title, DescriptionAnswer(recognised=False, language=None, description="",
                                                           keywords=[]))
        return {"parsed": answer, "raw": None, "parsing_error": None}


class FakeChat:
    _llm_type = "fake-chat"

    def __init__(self, answers):
        self.structured = FakeStructured(answers)

    def with_structured_output(self, schema, **kwargs):
        return self.structured


def answer(text="Warm deep house for late nights.", keywords=("deep house", "late night", "deep house"),
           language=" English ", recognised=True):
    return DescriptionAnswer(recognised=recognised, language=language, description=f" {text} ", keywords=list(keywords))


def client(answers):
    return LLMClient("anthropic:claude-sonnet-5-5", chat=FakeChat(answers))


class FakeClap:
    """A CLAP model's text side: each phrase's vector points along its own axis of `dims`."""

    def __init__(self, dims):
        self.dims, self.calls, self.closed = dims, 0, False

    def text_embeddings(self, texts):
        self.calls += 1
        return np.eye(len(texts), self.dims, dtype=np.float32)

    def close(self):
        self.closed = True


@pytest.fixture
def catalog(library):
    catalog, _ = Catalog.scan(library)
    return catalog


def test_client_cleans_the_answer():
    guess = client({"The Light": answer()}).describe("Artist: Metronomy\nTitle: The Light")
    assert guess.description == Description(text="Warm deep house for late nights.",
                                            keywords=["deep house", "late night"], language="English", recognised=True)
    assert client({}).describe("Artist: X\nTitle: Nothing").description is None
    assert "Never guess anything from the words of the title" in DESCRIPTION_PROMPT


def test_prompt_hash_follows_the_prompt_and_the_answer():
    assert prompt_hash(DESCRIPTION_PROMPT, DescriptionAnswer) == DESCRIPTION_PROMPT_HASH
    assert prompt_hash(DESCRIPTION_PROMPT + " ", DescriptionAnswer) != DESCRIPTION_PROMPT_HASH

    class Other(DescriptionAnswer):
        mood: str

    assert prompt_hash(DESCRIPTION_PROMPT, Other) != DESCRIPTION_PROMPT_HASH


def test_facts_key_and_fetch(catalog):
    record = catalog.tracks[ONE]
    record.genres["chosic-genres"] = GenreSource(status=SourceStatus.FOUND, genres=[Genre(name="house")])
    record.genres["llm-genres"] = GenreSource(status=SourceStatus.FOUND, genres=[Genre(name="pop")])
    llm = client({record.tags.title: answer()})
    fetcher = LLMDescriptionFetcher(Settings(), client=llm)

    facts = fetcher.facts(record)
    assert f"Title: {record.tags.title}" in facts and "Genres (chosic): house" in facts
    assert "pop" not in facts  # the llm-genres are the model's own knowledge
    key = fetcher.key(record)
    assert key.startswith(f"anthropic:claude-sonnet-5-5:prompt={DESCRIPTION_PROMPT_HASH}:facts=")
    record.genres["essentia-genres"] = GenreSource(status=SourceStatus.FOUND,
                                            genres=[Genre(name="deep house", confidence=0.47)])
    assert "Genres (essentia): deep house (47%)" in fetcher.facts(record) and fetcher.key(record) != key  # redone

    assert fetcher.fetch(record).text == "Warm deep house for late nights."
    record.tags.title = None
    assert fetcher.key(record) is None  # nothing to go on


def test_audio_tags_are_facts_but_not_in_the_key(catalog):
    record = catalog.tracks[ONE]
    table = EmbeddingTable()
    dims = len(PHRASES)
    for n, path in enumerate(sorted(catalog.tracks)):
        table.put(path, "k", Embedding(np.eye(dims, dtype=np.float32)[n * 10]))  # each fits one phrase best
    clap = FakeClap(dims)
    tags = AudioTags(table, clap, count=2)
    chat = FakeChat({record.tags.title: answer()})
    fetcher = LLMDescriptionFetcher(Settings(), client=LLMClient("m", chat=chat), audio_tags=tags)
    key = fetcher.key(record)

    fetcher.fetch(record)
    sent = chat.structured.calls[0][1].content
    assert sent.endswith(f"Audio analysis tags (automatic, may be wrong): {PHRASES[0]}, {PHRASES[1]}")
    assert clap.calls == 1 and clap.closed  # the phrases once, then the model's process is stopped
    assert tags.tags("not in the table.flac") == [] and fetcher.key(record) == key


def test_enrich_writes_descriptions_and_redoes_changed_facts(catalog):
    title = catalog.tracks[ONE].tags.title
    fetcher = LLMDescriptionFetcher(Settings(), client=client({title: answer()}))
    report = enrich(catalog, [fetcher])
    one = catalog.tracks[ONE].descriptions["llm-descriptions"]
    assert one.status == SourceStatus.FOUND and one.description.language == "English"
    assert catalog.tracks[TWO].descriptions["llm-descriptions"].status == SourceStatus.NOT_FOUND  # the model had nothing
    assert catalog.tracks[ONE].genres.keys() == catalog.tracks[ONE].genres.keys()  # genres untouched
    assert report.looked_up["llm-descriptions"][SourceStatus.FOUND] == 1

    assert not needs_lookup(catalog.tracks[ONE], fetcher)
    catalog.tracks[ONE].genres["chosic-genres"] = GenreSource(genres=[Genre(name="disco")])
    assert needs_lookup(catalog.tracks[ONE], fetcher)


def test_descriptions_survive_a_rescan_and_the_json(catalog, library):
    catalog.tracks[ONE].descriptions["llm-descriptions"] = DescriptionSource(description=Description(text="Calm.", keywords=["x"]))
    assert catalog.tracks[ONE].description_text() == "Calm.\nKeywords: x"
    assert catalog.tracks[TWO].description_text() == ""
    again = Catalog.model_validate_json(catalog.model_dump_json())
    assert again.tracks[ONE].descriptions["llm-descriptions"].description.text == "Calm."
    (library / ONE).touch()
    rescanned, _ = Catalog.scan(library, again)
    assert rescanned.tracks[ONE].descriptions["llm-descriptions"].description.text == "Calm."


class FakeTextClient:
    model, processes = "bge-base", 1

    def __init__(self):
        self.texts = []

    def text_embeddings(self, texts):
        self.texts += texts
        return np.ones((len(texts), 4), dtype=np.float32)

    def close(self):
        pass


def test_description_embeddings_follow_the_descriptions(catalog):
    fetcher = DescriptionTextFetcher(Settings(), client=FakeTextClient())
    catalog.tracks[ONE].descriptions["llm-descriptions"] = DescriptionSource(description=Description(text="Calm."))
    catalog.tracks[TWO].descriptions["llm-descriptions"] = DescriptionSource(status=SourceStatus.NOT_FOUND)
    table = EmbeddingTable()
    report = embed(catalog, [fetcher], {"description-bge": table})
    assert report.computed["description-bge"] == 1 and report.skipped["description-bge"] == 2  # no description
    assert fetcher.client.texts == ["Calm."] and fetcher.key(catalog.tracks[ONE]).startswith("bge-base:a5beb1e3e68b:text=")

    embed(catalog, [fetcher], {"description-bge": table})
    assert fetcher.client.texts == ["Calm."]  # unchanged: not again
    catalog.tracks[ONE].descriptions["llm-descriptions"] = DescriptionSource(description=Description(text="Loud."))
    embed(catalog, [fetcher], {"description-bge": table})
    assert fetcher.client.texts == ["Calm.", "Loud."]


def test_fuse(catalog):
    one, two, three = (catalog.tracks[p] for p in (ONE, TWO, THREE))
    sound = [TrackMatch(one, 0.3), TrackMatch(two, 0.2), TrackMatch(three, 0.1)]
    words = [TrackMatch(two, 0.8), TrackMatch(one, 0.7)]  # three has no description
    fused = fuse([sound, words], count=10)
    assert [m.track.path for m in fused] == [ONE, TWO, THREE]  # one: ranks 1 and 2; two: 2 and 1, a tie kept in order
    assert fused[0].similarity == pytest.approx((1 / (FUSE_K + 1) + 1 / (FUSE_K + 2)) / (2 / (FUSE_K + 1)))
    assert fuse([sound], count=1)[0].similarity == pytest.approx(1.0)


def test_audio_tags_come_from_the_embeddings_once_computed(catalog, tmp_path):
    settings = Settings.model_validate({"models": {"folder": str(tmp_path)},
                                        "fetchers": {"embeddings": {"muq-mulan": {"enabled": True}}}})
    fetcher = make_fetcher("llm-descriptions", settings)
    assert isinstance(fetcher, LLMDescriptionFetcher) and fetcher.workers == 8
    assert fetcher.models() == ["muq-mulan"]  # its text side, for the phrases
    assert make_fetcher("llm-descriptions", Settings()).models() == []  # muq-mulan off: no audio tags
    fetcher.client = "a client"  # no model to make
    assert fetcher.open(lambda name: EmbeddingTable()).audio_tags is None  # none computed yet: no CLAP model needed

    off = Settings.model_validate({"fetchers": {"descriptions": {"llm-descriptions": {"audio_tags": 0}}}})
    assert make_fetcher("llm-descriptions", off).models() == []
    assert [put.fetcher for put in make_fetcher("llm-descriptions", off).inputs()] == [
        "musicbrainz-genres", "chosic-genres", "essentia-genres"]
