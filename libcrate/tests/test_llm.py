"""LLM client and fetcher: a fake chat model for the logic, no API calls."""

import json

import pytest

from libcrate.catalog import Catalog
from libcrate.services.llm.client import (
    DISCOGS_STYLES, GENRE_PROMPT, GENRE_PROMPT_HASH, GenreAnswer, LLMClient, LLMError, StyleGuess,
    Usage, _usage, describe_track,
)
from libcrate.fetchers.genres import LLMGenreFetcher
from libcrate.fetchers.enrich import lookup
from libcrate.settings import Settings
from libcrate.store import MetadataStore
from libcrate.track import Genre, GenreSource, SourceStatus, Tags, TrackRecord


class FakeAI:
    def __init__(self, usage=None):
        self.usage_metadata = usage


class FakeStructured:
    def __init__(self, answers):
        self.answers = answers  # title -> GenreAnswer | Exception | None (unparseable)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        title = messages[1].content.split("Title: ")[1].split("\n")[0]
        answer = self.answers.get(title, GenreAnswer(known=False, styles=[]))
        if isinstance(answer, Exception):
            raise answer
        usage = {"input_tokens": 2400, "output_tokens": 60,
                 "input_token_details": {"cache_read": 2300, "cache_creation": 0}}
        return {"parsed": answer, "raw": FakeAI(usage), "parsing_error": None if answer else "bad json"}


class FakeChat:
    def __init__(self, answers=None, llm_type="fake-chat"):
        self._llm_type = llm_type
        self.structured = FakeStructured(answers or {})
        self.structured_args = None

    def with_structured_output(self, schema, **kwargs):
        self.structured_args = (schema, kwargs)
        return self.structured


def answer(*styles, known=True):
    return GenreAnswer(known=known, styles=[StyleGuess(label=l, confidence=c) for l, c in styles])


TAGS = Tags(title="The Light", artists=["Metronomy"], album="Metronomy Forever", year=2019, label="Because Music")


# -- prompt and parsing --------------------------------------------------------

def test_style_list():
    assert len(DISCOGS_STYLES) == 400
    assert DISCOGS_STYLES[0] == "Blues---Boogie Woogie"
    assert all(label in GENRE_PROMPT for label in DISCOGS_STYLES)


def test_describe_track():
    assert describe_track(TAGS) == (
        "Artist: Metronomy\nTitle: The Light\nAlbum: Metronomy Forever\nYear: 2019\nLabel: Because Music")
    assert describe_track(Tags(title="X", artists=["A", "B"], publisher="P")) == "Artist: A, B\nTitle: X\nLabel: P"


def test_uses_native_structured_output():
    chat = FakeChat({"The Light": answer(known=False)})
    LLMClient("fake:model", chat=chat).genres(TAGS, max_genres=3)
    schema, kwargs = chat.structured_args
    assert schema is GenreAnswer
    assert kwargs == {"method": "json_schema", "include_raw": True}


def test_each_answer_schema_is_bound_once():
    chat = FakeChat({"The Light": answer(known=False)})
    client = LLMClient("m", chat=chat)
    client.genres(TAGS, max_genres=3)
    bound = client._structured[GenreAnswer]
    client.genres(TAGS, max_genres=3)
    assert client._structured == {GenreAnswer: bound}


@pytest.mark.parametrize("llm_type, cached", [("anthropic-chat", True), ("openai-chat", False)])
def test_cache_marker_only_for_anthropic(llm_type, cached):
    chat = FakeChat({"The Light": answer(("Electronic---Synth-pop", 0.9))}, llm_type=llm_type)
    LLMClient("m", chat=chat).genres(TAGS, max_genres=3)
    system = chat.structured.calls[0][0].content[0]
    assert system["text"] == GENRE_PROMPT
    assert ("cache_control" in system) is cached


def test_guess_cleans_answers():
    chat = FakeChat({"The Light": answer(
        ("electronic---synth-pop", 0.9),       # case fixed to the canonical label
        ("Electronic---Synth-pop", 0.5),       # duplicate dropped
        ("Electronic---Made Up", 0.8),         # not a Discogs style: dropped and reported
        ("Electronic---Electro", 1.7),         # confidence clamped
        ("Electronic---Downtempo", 0.4),
        ("Electronic---Nu-Disco", 0.3),        # beyond max_genres
    )})
    guess = LLMClient("m", chat=chat).genres(TAGS, max_genres=3)
    assert guess.known
    # sorted by confidence before keeping the top 3: the model's own order isn't reliable
    assert guess.styles == [("Electronic---Electro", 1.0), ("Electronic---Synth-pop", 0.9),
                            ("Electronic---Downtempo", 0.4)]
    assert guess.invalid_labels == ["Electronic---Made Up"]
    assert guess.usage == Usage(input=2400, output=60, cache_read=2300, cache_creation=0)


def test_most_confident_styles_are_kept():
    chat = FakeChat({"The Light": answer(
        ("Pop---Vocal", 0.3), ("Rock---Pop Rock", 0.6), ("Funk / Soul---Soul", 0.35), ("Pop---Ballad", 0.5))})
    guess = LLMClient("m", chat=chat).genres(TAGS, max_genres=3)
    assert guess.styles == [("Rock---Pop Rock", 0.6), ("Pop---Ballad", 0.5), ("Funk / Soul---Soul", 0.35)]


@pytest.mark.parametrize("details, written", [
    ({"cache_read": 0, "cache_creation": 5160}, 5160),
    ({"cache_read": 0, "cache_creation": 0, "ephemeral_5m_input_tokens": 5160, "ephemeral_1h_input_tokens": 0}, 5160),
    ({}, 0),
])
def test_cache_writes_are_counted(details, written):
    usage = _usage(FakeAI({"input_tokens": 5325, "output_tokens": 68, "input_token_details": details}))
    assert usage.cache_creation == written


def test_unknown_drops_any_styles():
    chat = FakeChat({"The Light": answer(("Electronic---House", 0.2), known=False)})
    guess = LLMClient("m", chat=chat).genres(TAGS, max_genres=3)
    assert (guess.known, guess.styles) == (False, [])


@pytest.mark.parametrize("response, match", [
    (None, "unparseable answer: bad json"),
    (RuntimeError("HTTP 529 overloaded"), "RuntimeError: HTTP 529 overloaded"),
])
def test_guess_errors(response, match):
    chat = FakeChat({"The Light": response})
    with pytest.raises(LLMError, match=match):
        LLMClient("m", chat=chat).genres(TAGS, max_genres=3)


def test_bad_model_string_is_an_llm_error():
    with pytest.raises(LLMError, match="could not create model"):
        LLMClient("no-such-provider:model")


def test_pilot_parses_options(llm_pilot):
    assert llm_pilot.parse_options(["effort=low", "temperature=0", "max_tokens=512", "x=a=b"]) == {
        "effort": "low", "temperature": 0, "max_tokens": 512, "x": "a=b"}
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        llm_pilot.parse_options(["effort"])


def test_real_anthropic_request_offline(monkeypatch):
    """The request ChatAnthropic would send, intercepted before the network: the style list
    is cache-marked, structured output and effort go in output_config, no forced tool call."""
    langchain_anthropic = pytest.importorskip("langchain_anthropic")
    captured = {}

    def fake_create(self, payload):
        captured.update(payload)
        raise RuntimeError("intercepted")

    monkeypatch.setattr(langchain_anthropic.ChatAnthropic, "_create", fake_create)
    client = LLMClient("anthropic:claude-sonnet-5-5", api_key="sk-test", effort="low")
    with pytest.raises(LLMError, match="intercepted"):
        client.genres(TAGS, max_genres=3)

    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["output_config"]["effort"] == "low"
    assert "format" in captured["output_config"]
    assert "tool_choice" not in captured and "tools" not in captured


# -- fetcher -------------------------------------------------------------------

@pytest.fixture
def record(tmp_path, make_flac):
    path = make_flac(tmp_path / "s.flac", {"ARTIST": "Metronomy", "TITLE": "The Light"})
    return TrackRecord.from_file(path, tmp_path)


def fetcher(answers, min_confidence=0.5):
    client = LLMClient("anthropic:claude-sonnet-5-5", chat=FakeChat(answers))
    settings = Settings.model_validate({"fetchers": {"genres": {"llm-genres": {"min_confidence": min_confidence}}}})
    return LLMGenreFetcher(settings, client=client)


def test_fetcher_genres(record):
    f = fetcher({"The Light": answer(("Electronic---Synth-pop", 0.91234), ("Pop---Indie Pop", 0.6))})
    assert f.fetch(record) == [
        Genre(name="synth-pop", parent="Electronic", confidence=0.9123),
        Genre(name="indie pop", parent="Pop", confidence=0.6),
    ]


def test_fetcher_key(record, tmp_path, make_flac):
    assert fetcher({}).key(record) == f"anthropic:claude-sonnet-5-5:prompt={GENRE_PROMPT_HASH}:min=0.5"
    untitled = TrackRecord.from_file(make_flac(tmp_path / "u.flac", {"ARTIST": "Metronomy"}), tmp_path)
    assert fetcher({}).key(untitled) is None  # nothing to ask about


@pytest.mark.parametrize("answers, error", [
    ({}, "the model doesn't know this track"),
    ({"The Light": answer(("Electronic---Made Up", 0.9))}, "the model gave no Discogs style (invalid labels: Electronic---Made Up)"),
])
def test_fetcher_not_found(record, answers, error):
    result = lookup(record, fetcher(answers))
    assert (result.status, result.error) == (SourceStatus.NOT_FOUND, error)


def test_fetcher_error_is_retried_status(record):
    result = lookup(record, fetcher({"The Light": RuntimeError("rate limited")}))
    assert result.status == SourceStatus.ERROR


def test_min_confidence(record):
    f = fetcher({"The Light": answer(("Electronic---Synth-pop", 0.8), ("Pop---Indie Pop", 0.5), ("Pop---Europop", 0.2))})
    assert [g.name for g in f.fetch(record)] == ["synth-pop", "indie pop"]  # 0.5 kept, 0.2 dropped


def test_nothing_confident_enough_is_not_found(record):
    result = lookup(record, fetcher({"The Light": answer(("Electronic---Hardstyle", 0.0), ("Pop---Europop", 0.3))}))
    assert (result.status, result.error) == (SourceStatus.NOT_FOUND, "no style above 50% (best: Pop---Europop 30%)")


def test_fetcher_key_follows_threshold(record):
    stricter = fetcher({}, min_confidence=0.7)
    assert stricter.key(record).endswith(":min=0.7")


# -- pilot -----------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [("Synth-Pop", "synthpop"), ("Drum & Bass", "drumbass"), ("80S", "80s")])
def test_normalize(llm_pilot, name, expected):
    assert llm_pilot.normalize(name) == expected


@pytest.mark.parametrize("guessed, reference, expected", [
    (["Synth-pop"], ["synthpop", "indie"], (True, True)),
    (["Deep House"], ["house"], (False, True)),
    (["Techno"], ["jazz"], (False, False)),
    ([], ["jazz"], (False, False)),
])
def test_compare(llm_pilot, guessed, reference, expected):
    assert llm_pilot.compare(guessed, reference) == expected


@pytest.fixture
def pilot_catalog(library, tmp_path):
    catalog, _ = Catalog.scan(library)
    catalog.tracks["Artist A - One [AAA111].flac"].genres["musicbrainz-genres"] = GenreSource(genres=[Genre(name="Pop")])
    catalog.tracks["Artist B - Two [BBB222].flac"].genres["chosic-genres"] = GenreSource(genres=[Genre(name="techno")])
    path = tmp_path / "out"
    MetadataStore(path).save(catalog)
    return path


def test_pick_tracks_needs_reference_genres(llm_pilot, pilot_catalog):
    pick_tracks = llm_pilot.pick_tracks
    catalog = MetadataStore(pilot_catalog).load()
    picked = pick_tracks(catalog, 10, seed=0)
    # One has a musicbrainz result, Two a chosic one; Three has neither.
    assert sorted(t.path for t in picked) == ["Artist A - One [AAA111].flac", "Artist B - Two [BBB222].flac"]
    assert [t.path for t in pick_tracks(catalog, 1, seed=3)] == [t.path for t in pick_tracks(catalog, 1, seed=3)]


def test_pilot_end_to_end(llm_pilot, pilot_catalog, tmp_path, capsys, monkeypatch):
    pilot_mod = llm_pilot
    answers = {"One": answer(("Pop---Indie Pop", 0.9)), "Two": answer(known=False)}
    monkeypatch.setattr(pilot_mod, "LLMClient",
                        lambda model, **options: LLMClient(model, chat=FakeChat(answers)))
    report = tmp_path / "report.json"

    assert pilot_mod.main([str(pilot_catalog), "-m", "fake:model", "-n", "5", "--report", str(report),
                           "-o", "effort=low", "-o", "api_key=sk-secret"]) == 0

    err = capsys.readouterr().err
    assert "▸ Asking fake:model about 2 tracks" in err
    assert "known        1/2 (50%)" in err
    assert "    loosely    1/1 (100%)" in err  # "indie pop" contains "pop"
    assert "agreement, not accuracy" in err
    data = json.loads(report.read_text())
    assert data["summary"]["known"] == 1 and data["summary"]["unknown"] == 1
    assert data["summary"]["by_confidence"]["≥ 80%"] == {"tracks": 1, "exact": 0}
    assert data["summary"]["usage"] == {"input": 4800, "output": 120, "cache_read": 4600, "cache_creation": 0}
    assert data["options"] == {"effort": "low"}  # the API key never reaches the report
    assert "sk-secret" not in report.read_text()


def test_pilot_without_reference_tracks(llm_pilot, library, tmp_path, capsys):
    pilot_mod = llm_pilot
    catalog, _ = Catalog.scan(library)
    for t in catalog.tracks.values():
        t.genres.clear()
    path = tmp_path / "out"
    MetadataStore(path).save(catalog)
    assert pilot_mod.main([str(path), "-m", "fake:model"]) == 2
    assert "no tracks with chosic-genres or musicbrainz-genres genres" in capsys.readouterr().err


def test_pilot_gives_up_when_every_call_fails(llm_pilot, library, tmp_path, capsys, monkeypatch):
    pilot_mod = llm_pilot
    catalog, _ = Catalog.scan(library)
    for t in catalog.tracks.values():  # all three tracks have reference genres
        t.genres["chosic-genres"] = GenreSource(status=SourceStatus.FOUND, genres=[Genre(name="pop")])
    path = tmp_path / "out"
    MetadataStore(path).save(catalog)
    failing = {title: RuntimeError("no API key") for title in ("One", "Two", "Three")}
    monkeypatch.setattr(pilot_mod, "LLMClient",
                        lambda model, **options: LLMClient(model, chat=FakeChat(failing)))

    assert pilot_mod.main([str(path), "-m", "fake:model", "-n", "3"]) == 1
    err = capsys.readouterr().err
    assert err.count("no API key") == 3
    assert "the first 3 tracks all failed, stopping" in err


def test_closed_classifier_refuses_to_guess():
    chat = FakeChat({"The Light": answer(("Pop---Indie Pop", 0.9))})
    with LLMClient("m", chat=chat) as client:
        assert client.genres(TAGS, max_genres=3).known
    with pytest.raises(LLMError, match="closed"):
        client.genres(TAGS, max_genres=3)


def test_closing_the_fetcher_closes_the_classifier():
    client = LLMClient("m", chat=FakeChat())
    with LLMGenreFetcher(Settings(), client=client):
        pass
    with pytest.raises(LLMError, match="closed"):
        client.genres(TAGS, max_genres=3)
