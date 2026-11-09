import numpy as np
import pytest
from libcrate import Description, DescriptionSource, Embedding, EmbeddingTable, MetadataStore, make_fetcher
from libcrate.models.clap import ClapError
from PySide6.QtCore import QCoreApplication

import crate.describe.page as page_module
from crate.describe import DescribePage
from crate.library.covers import CoverCache
from crate.library.model import TRACK_ROLE

MODEL = "muq-mulan"
VECTORS = {"The Light": [1.0, 0.0], "TEMPO": [0.6, 0.8], "Waterloo": [0.0, 1.0]}
# The fake model's vector for a description: "chill" points at The Light, anything else at Waterloo.
TEXTS = {"chill": [1.0, 0.1]}


class FakeClap:
    made = []

    def __init__(self, models, model, *, processes):
        self.model, self.folder, self.processes = model, models / model, processes
        self.texts, self.closed = [], False
        FakeClap.made.append(self)

    def text_embeddings(self, texts):
        self.texts += texts
        if texts == ["boom"]:
            raise ClapError("the CLAP process stopped while working on ['boom'] (exit code -9)")
        return np.array([TEXTS.get(t, [0.0, 1.0]) for t in texts], dtype=np.float32)

    def query_embeddings(self, texts):
        return self.text_embeddings(texts)

    def close(self):
        self.closed = True


@pytest.fixture
def store(library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    store.scan(library)
    return store


def save_embeddings(store, name=MODEL, vectors=VECTORS):
    table = EmbeddingTable()
    for track in store.load().tracks.values():
        key = make_fetcher(name).key(track)  # what the fetcher would make now: a feature uses only those
        table.put(track.path, key, Embedding(np.array(vectors[track.tags.title], dtype=np.float32)))
    path = store.embeddings_path(name)
    path.parent.mkdir(parents=True)
    table.save(path)


@pytest.fixture
def page(qapp, monkeypatch):
    monkeypatch.setattr(FakeClap, "made", [])
    monkeypatch.setattr(page_module, "ClapClient", FakeClap)
    page = DescribePage(CoverCache(40, 4))
    notices = page.notices = []
    page._notify = lambda kind, title, content, duration=0: notices.append(title)
    yield page
    page.shutdown()


def describe(page, text):
    page.box.setText(text)
    started = page.describe()
    while page.task is not None:
        page.task.wait()
        QCoreApplication.processEvents()
    return started


def titles(page):
    return [page.model.index(r, 0).data(TRACK_ROLE).tags.title for r in range(page.model.rowCount())]


def test_describe_lists_the_best_matches(page, store):
    save_embeddings(store)
    page.set_catalog(store, store.load())
    assert not page.play_button.isEnabled()

    assert describe(page, "chill")
    assert titles(page) == ["The Light", "TEMPO", "Waterloo"]
    assert page.model.index(0, 1).data() == "100%"  # cosine 0.995
    assert page.summary.text() == "“chill”  ·  3 tracks  ·  muq-mulan"
    assert page.play_button.isEnabled() and page.box.isEnabled()

    played = []
    page.play_requested.connect(lambda tracks, row, root: played.append((len(tracks), row, root)))
    page.play_button.click()
    assert played == [(3, 0, store.load().root)]

    assert describe(page, "loud")
    assert titles(page)[0] == "Waterloo"
    assert len(FakeClap.made) == 1 and FakeClap.made[0].texts == ["chill", "loud"]  # one model process, kept


def test_describe_follows_the_settings(page, store):
    save_embeddings(store)
    store.settings_path.write_text("describe:\n  count: 1\n")
    page.set_catalog(store, store.load())
    describe(page, "chill")
    assert titles(page) == ["The Light"]


def test_describe_needs_a_catalog_embeddings_and_the_model(page, store, tmp_path, monkeypatch):
    page.box.setText("chill")
    assert not page.describe() and page.notices == ["No catalog open"]

    page.set_catalog(store, store.load())
    assert not page.describe() and page.notices[-1] == "No muq-mulan embeddings yet"

    save_embeddings(store)

    def missing(*args, **kwargs):
        raise ClapError("missing CLAP model muq-mulan in /x (Settings › CLAP › Download models)")

    monkeypatch.setattr(page_module, "ClapClient", missing)
    assert not page.describe() and page.notices[-1] == "The model isn't there"
    assert page.box.isEnabled()


def test_a_failed_description_says_so(page, store):
    save_embeddings(store)
    page.set_catalog(store, store.load())
    assert describe(page, "boom")
    assert page.notices == ["Couldn't describe"] and page.model.rowCount() == 0 and page.box.isEnabled()


def test_shutdown_stops_the_model(page, store):
    save_embeddings(store)
    page.set_catalog(store, store.load())
    describe(page, "chill")
    page.shutdown()
    assert FakeClap.made[0].closed


def test_start_radio_from_the_best_matches(page, store, monkeypatch):
    save_embeddings(store)
    page.set_catalog(store, store.load())
    assert not page.radio_button.isEnabled()
    monkeypatch.setattr(page_module, "RADIO_SEEDS", 2)
    describe(page, "chill")
    requests = []
    page.radio_requested.connect(lambda seeds, name: requests.append(([t.tags.title for t in seeds], name)))
    page.radio_button.click()
    assert requests == [(["The Light", "TEMPO"], "“chill”")]


def test_describe_matches_the_descriptions_too(page, store):
    save_embeddings(store)
    catalog = store.load()
    for track in catalog.tracks.values():
        track.descriptions["llm-descriptions"] = DescriptionSource(description=Description(text=track.tags.title))
    store.save(catalog)
    # In the descriptions' space "chill" points at Waterloo: the opposite of the sound's ranking.
    save_embeddings(store, "description-bge", {"The Light": [0.0, 1.0], "TEMPO": [0.6, 0.8], "Waterloo": [1.0, 0.0]})
    page.set_catalog(store, store.load())
    describe(page, "chill")

    assert [c.model for c in FakeClap.made] == ["muq-mulan", "bge-base"]  # a client per model
    assert set(titles(page)) == {"The Light", "TEMPO", "Waterloo"} and titles(page)[-1] == "TEMPO"  # fused: the middle
    assert page.summary.text() == "“chill”  ·  3 tracks  ·  muq-mulan + description-bge"
    assert page.model.index(0, 1).data() == "98%"  # first in one ranking, third in the other

    store.settings_path.write_text("describe:\n  words: null\n")  # the sound only
    describe(page, "chill")
    assert titles(page)[0] == "The Light" and [c.closed for c in FakeClap.made] == [False, True]
