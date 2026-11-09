"""Enrich as a whole (fetchers/run.py): the order fetchers run in, what stops one from running (status.py), how far
each is, the models downloaded first, and the features' slots (features.py). Fake fetchers, no network."""

import dataclasses

import numpy as np
import pytest

import libcrate.fetchers.registry as registry
import libcrate.models.registry as models_registry
from libcrate import features
from libcrate.catalog import Catalog
from libcrate.models.essentia.models import EMBEDDING_MODEL
from libcrate.embeddings import Embedding, EmbeddingTable
from libcrate.fetchers import DOWNLOAD, EmbeddingFetcher, Input, LookupNotFound, check, plan, progress, run, steps
from libcrate.fetchers.descriptions import DescriptionFetcher
from libcrate.fetchers.genres import GenreFetcher
from libcrate.models import installed
from libcrate.settings import Settings
from libcrate.store import MetadataStore
from libcrate.track import Description, Genre, GenreSource, SourceStatus

ONE = "Artist A - One [AAA111].flac"


def enabled(*names, **extra) -> Settings:
    """The settings with `names` on, and `extra` sections (e.g. keys={...})."""
    data = {"fetchers": {"genres": {}, "embeddings": {}, "descriptions": {}}, **extra}
    for name in names:
        kind = registry.FETCHERS[name].kind
        data["fetchers"][kind][name] = {"enabled": True}
    return Settings.model_validate(data)


@pytest.fixture
def catalog(library):
    return Catalog.scan(library)[0]


# -- the order -----------------------------------------------------------------------

def test_plan_follows_the_inputs():
    settings = enabled("description-bge", "llm-descriptions", "muq-mulan", "chosic-genres", "essentia-genres",
                       "discogs-effnet")
    assert plan(settings) == ["chosic-genres", "essentia-genres", "discogs-effnet", "muq-mulan", "llm-descriptions",
                              "description-bge"]
    assert plan(settings, ["description-bge", "chosic-genres"]) == ["chosic-genres", "description-bge"]
    assert plan(Settings()) == []  # nothing chosen


# -- what stops a fetcher --------------------------------------------------------------

def test_a_language_model_needs_its_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert check(enabled("llm-genres"), "llm-genres").problems == [
        "no API key for anthropic: set keys.anthropic, or ANTHROPIC_API_KEY"]
    assert check(enabled("llm-genres", keys={"anthropic": "sk-1"}), "llm-genres").ready
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert check(enabled("llm-genres"), "llm-genres").ready
    other = Settings.model_validate({"fetchers": {"genres": {"llm-genres": {"service": "chosic"}}}})
    assert "no language model" not in " ".join(check(other, "llm-genres").problems)  # chosic exists: no key to need


def test_inputs_required_and_optional(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    folder = {"models": {"folder": str(tmp_path)}}
    bge = check(enabled("description-bge", **folder), "description-bge")
    assert bge.problems == ["needs llm-descriptions, which is off"] and bge.downloads == ["bge-base"]
    assert check(enabled("description-bge", "llm-descriptions", **folder), "description-bge").ready

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert check(enabled("description-bge", "llm-descriptions", **folder), "description-bge").problems == [
        "needs llm-descriptions, which can't run"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    described = check(enabled("llm-descriptions", "chosic-genres", **folder), "llm-descriptions")
    assert described.ready and described.downloads == []  # no audio tags while muq-mulan is off
    assert described.notes == ["uses musicbrainz-genres when it's on", "uses essentia-genres when it's on",
                               "uses muq-mulan when it's on"]
    tagged = check(enabled("llm-descriptions", "muq-mulan", **folder), "llm-descriptions")
    assert tagged.downloads == ["muq-mulan"]  # its text side, for the audio tags' phrases


def test_downloads_include_what_a_model_needs(tmp_path):
    essentia = check(enabled("essentia-genres", models={"folder": str(tmp_path)}), "essentia-genres")
    assert essentia.ready and essentia.downloads == [EMBEDDING_MODEL, "genre_discogs400-discogs-effnet-1"]
    assert essentia.download_size > 20_000_000


def test_steps_name_the_downloads_first(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = enabled("discogs-effnet", "chosic-genres", "llm-genres", models={"folder": str(tmp_path)})
    assert steps(settings) == [DOWNLOAD.format(EMBEDDING_MODEL), "chosic-genres", "discogs-effnet"]  # llm: no key


# -- how far a fetcher is ----------------------------------------------------------------

def test_progress_of_a_metadata_fetcher(catalog):
    settings = Settings()
    records = sorted(catalog.tracks)
    assert progress(catalog, settings, "chosic-genres").cant == 2  # only one track has a Spotify ID
    one = catalog.tracks[ONE]
    one.genres["chosic-genres"] = GenreSource(key=registry.make_fetcher("chosic-genres").key(one))
    assert progress(catalog, settings, "chosic-genres").done == 1

    assert progress(catalog, settings, "essentia-genres").todo == 3
    key = registry.make_fetcher("essentia-genres").key(one)
    catalog.tracks[records[0]].genres["essentia-genres"] = GenreSource(key=key)
    catalog.tracks[records[1]].genres["essentia-genres"] = GenreSource(key="older settings")
    catalog.tracks[records[2]].genres["essentia-genres"] = GenreSource(status=SourceStatus.ERROR, key=key)
    counts = progress(catalog, settings, "essentia-genres")
    assert (counts.done, counts.redo, counts.errors, counts.todo, counts.to_do) == (1, 1, 1, 0, 2)


def test_progress_of_an_embedding_fetcher(catalog):
    records = sorted(catalog.tracks)
    effnet = registry.make_fetcher("discogs-effnet")
    table = EmbeddingTable()
    table.put(records[0], effnet.key(None), Embedding(np.ones(2, dtype=np.float32)))
    table.put(records[1], "another model", Embedding(np.ones(2, dtype=np.float32)))
    counts = progress(catalog, Settings(), "discogs-effnet", table)
    assert (counts.done, counts.redo, counts.todo) == (1, 1, 1)
    assert progress(catalog, Settings(), "description-bge").cant == 3  # nothing described yet


# -- running -----------------------------------------------------------------------------

class FakeGenres(GenreFetcher):
    name = "chosic-genres"

    def key(self, record):
        return "k"

    def fetch(self, record):
        if record.tags.title == "Three":
            raise LookupNotFound("nothing")
        return [Genre(name="pop")]


class FakeEmbeddings(EmbeddingFetcher):
    name, space = "discogs-effnet", "audio"

    def models(self):
        return [EMBEDDING_MODEL]

    def key(self, record):
        return "v1"

    def fetch(self, record):
        return Embedding(np.ones(2, dtype=np.float32))


class FakeDescriptions(DescriptionFetcher):
    name = "llm-descriptions"

    def inputs(self):
        return [Input("chosic-genres"), Input("discogs-effnet")]

    def key(self, record):
        return f"genres={record.genre_names('chosic-genres')}"

    def fetch(self, record):
        return Description(text=f"{', '.join(record.genre_names('chosic-genres')) or 'unknown'} music")


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """The fake fetchers in the registry; downloading a model writes its files."""
    for fake in (FakeGenres, FakeEmbeddings, FakeDescriptions):
        monkeypatch.setitem(registry.FETCHERS, fake.name, fake)

    def fetch(folder):
        for name in models_registry.MODELS[EMBEDDING_MODEL].files:
            (folder / name).write_text("model")

    monkeypatch.setitem(models_registry.MODELS, EMBEDDING_MODEL,
                        dataclasses.replace(models_registry.MODELS[EMBEDDING_MODEL], fetch=fetch))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    return tmp_path / "models"


def test_run_downloads_then_runs_in_order(library, tmp_path, fakes):
    store = MetadataStore(tmp_path / "out")
    catalog, _ = store.scan(library)
    settings = enabled("llm-descriptions", "discogs-effnet", "chosic-genres", models={"folder": str(fakes)})
    started, outcomes = [], []

    report = store.run(catalog, settings, on_start=lambda step, total, skipped: started.append(step),
                       on_progress=lambda step, done, total, rel, outcome: outcomes.append((step, outcome)))

    assert started == [DOWNLOAD.format(EMBEDDING_MODEL), "chosic-genres", "discogs-effnet", "llm-descriptions"]
    assert report.downloaded == [EMBEDDING_MODEL] and installed(fakes, EMBEDDING_MODEL)
    assert (DOWNLOAD.format(EMBEDDING_MODEL), "downloaded") in outcomes
    assert sorted(o for s, o in outcomes if s == "chosic-genres") == ["found", "found", "not found"]
    assert [o for s, o in outcomes if s == "discogs-effnet"] == ["computed"] * 3
    assert report.enriched.looked_up["llm-descriptions"][SourceStatus.FOUND] == 3
    assert report.embedded.computed == {"discogs-effnet": 3} and not report.interrupted

    saved = store.load().tracks[ONE]
    assert saved.descriptions["llm-descriptions"].description.text == "pop music"  # after the genres
    assert len(store.load_embeddings("discogs-effnet")) == 3


def test_run_skips_what_cant_run(library, tmp_path, fakes, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    store = MetadataStore(tmp_path / "out")
    catalog, _ = store.scan(library)
    report = store.run(catalog, enabled("chosic-genres", "llm-genres", "description-bge"))
    assert report.not_run == {"llm-genres": "no API key for anthropic: set keys.anthropic, or ANTHROPIC_API_KEY",
                              "description-bge": "needs llm-descriptions, which is off"}
    assert report.enriched.looked_up["chosic-genres"][SourceStatus.FOUND] == 2 and report.embedded is None


# -- features ------------------------------------------------------------------------------

def test_feature_slots_and_their_problems():
    settings = enabled("muq-mulan")
    slots = features.slots(settings)
    assert [slot.label for slot in slots] == ["similar", "map", "describe (sound)", "describe (words)", "radio"]
    assert [features.problem(settings, slot) for slot in slots] == [None] * 5
    off = "Enrich doesn't keep the {} embeddings up to date while it's off: turn it on in Fetchers"
    assert [features.note(settings, slot) for slot in slots] == [
        off.format("discogs-effnet"), off.format("discogs-effnet"), None, off.format("description-bge"),
        off.format("discogs-effnet")]
    wrong = Settings.model_validate({"describe": {"sound": "discogs-effnet", "words": None}})
    [sound, words] = features.slots(wrong, "describe")
    assert features.problem(wrong, sound) == "the sound of describe can't be discogs-effnet"
    assert features.note(wrong, sound) is None  # the problem says enough
    assert features.problem(wrong, words) is None and features.note(wrong, words) is None  # optional, and empty


def test_used_by(monkeypatch):
    settings = enabled("muq-mulan", "llm-descriptions")
    assert features.used_by(settings, "muq-mulan") == ["describe (sound)", "llm-descriptions"]
    assert features.used_by(settings, "discogs-effnet") == ["similar", "map", "radio"]


def test_features_only_use_rows_of_the_current_settings(catalog):
    records = sorted(catalog.tracks)
    table = EmbeddingTable()
    key = registry.make_fetcher("muq-mulan").key(None)
    table.put(records[0], key, Embedding(np.ones(2, dtype=np.float32)))
    table.put(records[1], "another revision", Embedding(np.ones(2, dtype=np.float32)))
    table.put("gone.flac", key, Embedding(np.ones(2, dtype=np.float32)))
    assert features.current(table, catalog, Settings(), "muq-mulan").matrix()[0] == [records[0]]
