"""Embeddings: the Parquet table, running models over a catalog (fetchers/embeddings/embed.py) with fake
models, the Discogs-EffNet fetcher with a fake client, the store, and the real model."""

import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from libcrate.catalog import Catalog
from libcrate.models.essentia.client import EMBEDDINGS, TRACK_EMBEDDINGS, EssentiaClient
from libcrate.models.essentia.models import (
    EMBEDDING_MODEL, MODEL_FILES, TRACK_EMBEDDING_MODEL, EssentiaError, check_requirements,
)
from libcrate.embeddings import Embedding, EmbeddingTable
from libcrate.fetchers.embeddings import (
    DiscogsEffnetFetcher, DiscogsTrackFetcher, EmbeddingFetcher, embed,
)
from libcrate.fetchers import fetcher_names, make_fetcher
from libcrate.settings import Settings
from libcrate.store import MetadataStore

ONE = "Artist A - One [AAA111].flac"
TWO = "Artist B - Two [BBB222].flac"
THREE = "sub/Artist C - Three.flac"


def vec(*values) -> np.ndarray:
    return np.array(values, dtype=np.float32)


class FakeModel(EmbeddingFetcher):
    """Embeds a track as [len(title), 1, 0]; titles in `fail` raise."""

    name = "fake"
    workers = 1  # set by a test

    def __init__(self, name="fake", version="v1", fail=(), on_fetch=None):
        self.name, self.version, self.fail, self.on_fetch = name, version, set(fail), on_fetch
        self.calls = []
        self.closed = False

    def key(self, record):
        return f"fake:{self.version}"

    def fetch(self, record):
        self.calls.append(record.path)
        if self.on_fetch:
            self.on_fetch(record)
        if record.tags.title in self.fail:
            raise RuntimeError(f"cannot embed {record.tags.title}")
        return Embedding(vec(len(record.tags.title), 1, 0), n_frames=7)

    def close(self):
        self.closed = True


@pytest.fixture
def catalog(library):
    return Catalog.scan(library)[0]


# -- the table -------------------------------------------------------------------

def test_table_round_trip(tmp_path):
    table = EmbeddingTable()
    table.put("b.flac", "k2", Embedding(vec(3, 4), n_frames=2))
    table.put("a.flac", "k1", Embedding(vec(1, 2)))
    table.save(tmp_path / "t.parquet")

    loaded = EmbeddingTable.load(tmp_path / "t.parquet")

    assert len(loaded) == 2
    assert (loaded.key("a.flac"), loaded.key("b.flac"), loaded.key("c.flac")) == ("k1", "k2", None)
    assert loaded.get("b.flac").n_frames == 2 and loaded.get("c.flac") is None
    paths, matrix = loaded.matrix()
    assert paths == ["a.flac", "b.flac"]
    assert matrix.dtype == np.float32 and matrix.tolist() == [[1, 2], [3, 4]]
    schema = pq.read_schema(tmp_path / "t.parquet")
    assert schema.names == ["path", "key", "n_frames", "embedding"]
    assert str(schema.field("embedding").type) == "fixed_size_list<element: float>[2]"


def test_missing_table_is_empty(tmp_path):
    table = EmbeddingTable.load(tmp_path / "nothing.parquet")
    assert len(table) == 0
    assert table.matrix()[0] == []


def test_empty_table_round_trip(tmp_path):
    EmbeddingTable().save(tmp_path / "t.parquet")
    assert len(EmbeddingTable.load(tmp_path / "t.parquet")) == 0
    assert not (tmp_path / "t.parquet.tmp").exists()


def test_keep_only(tmp_path):
    table = EmbeddingTable()
    for rel in ("a", "b", "c"):
        table.put(rel, "k", Embedding(vec(1)))
    table.keep_only({"a", "c", "d"})
    assert table.matrix()[0] == ["a", "c"]


# -- running models --------------------------------------------------------------

def test_embed_computes_every_track(catalog):
    model = FakeModel()
    tables = {"fake": EmbeddingTable()}

    report = embed(catalog, [model], tables)

    assert (report.computed, report.skipped, report.errors) == ({"fake": 3}, {"fake": 0}, 0)
    assert tables["fake"].get(ONE).vector.tolist() == [3, 1, 0]
    assert (tables["fake"].key(ONE), tables["fake"].get(ONE).n_frames) == ("fake:v1", 7)


def test_second_run_only_redoes_failures_and_new_keys(catalog):
    tables = {"fake": EmbeddingTable()}
    first = embed(catalog, [FakeModel(fail={"Two"})], tables)
    assert first.failed == {"fake": [(TWO, "RuntimeError: cannot embed Two")]}
    assert tables["fake"].key(TWO) is None  # failures aren't stored

    again = FakeModel()
    report = embed(catalog, [again], tables)
    assert again.calls == [TWO]
    assert (report.computed, report.skipped) == ({"fake": 1}, {"fake": 2})

    upgraded = FakeModel(version="v2")
    embed(catalog, [upgraded], tables)
    assert len(upgraded.calls) == 3


def test_tracks_gone_from_the_catalog_are_dropped(catalog):
    table = EmbeddingTable()
    table.put("deleted.flac", "fake:v1", Embedding(vec(1, 1, 1)))
    embed(catalog, [FakeModel()], {"fake": table})
    assert table.matrix()[0] == [ONE, TWO, THREE]


def test_models_run_one_after_another_into_their_own_tables(catalog):
    calls = []
    a, b = (FakeModel(name, on_fetch=lambda r, n=name: calls.append(n)) for name in ("a", "b"))
    tables = {"a": EmbeddingTable(), "b": EmbeddingTable()}

    report = embed(catalog, [a, b], tables)

    assert report.computed == {"a": 3, "b": 3}
    assert calls == ["a"] * 3 + ["b"] * 3
    assert len(tables["a"]) == len(tables["b"]) == 3


def test_workers_compute_tracks_concurrently(catalog):
    two_at_once = threading.Barrier(2, timeout=5)  # the first two tracks wait for each other
    model = FakeModel(on_fetch=lambda r: two_at_once.wait() if r.path in (ONE, TWO) else None)
    model.workers = 2
    table = EmbeddingTable()

    report = embed(catalog, [model], {"fake": table})

    assert report.computed == {"fake": 3}
    assert table.matrix()[0] == [ONE, TWO, THREE]


def test_checkpoints_and_progress(catalog):
    checkpoints, progress = [], []
    embed(catalog, [FakeModel(fail={"Two"})], {"fake": EmbeddingTable()},
          on_checkpoint=lambda t: checkpoints.append(len(t["fake"])), checkpoint_every=2,
          on_progress=lambda *a: progress.append(a))
    assert checkpoints == [1, 2]  # after 2 tracks (one failed), then at the end
    assert progress == [
        ("fake", 1, 3, ONE, None),
        ("fake", 2, 3, TWO, "RuntimeError: cannot embed Two"),
        ("fake", 3, 3, THREE, None),
    ]


def test_start_callback(catalog):
    started = []
    tables = {"a": EmbeddingTable(), "b": EmbeddingTable()}
    embed(catalog, [FakeModel("a")], tables, on_start=lambda *a: started.append(a))
    embed(catalog, [FakeModel("a"), FakeModel("b")], tables, on_start=lambda *a: started.append(a))
    assert started == [("a", 3, 0), ("a", 0, 3), ("b", 3, 0)]  # nothing to do is reported too


def test_stop_keeps_finished_tracks(catalog):
    stop = threading.Event()
    table = EmbeddingTable()
    report = embed(catalog, [FakeModel(on_fetch=lambda r: stop.set())], {"fake": table}, stop=stop)
    assert report.interrupted
    assert len(table) == 1


def test_gives_up_when_first_tracks_all_fail(catalog):
    report = embed(catalog, [FakeModel(fail={"One", "Two", "Three"})], {"fake": EmbeddingTable()})
    assert report.gave_up == {"fake": "RuntimeError: cannot embed Three"}
    assert report.errors == 3


# -- store ---------------------------------------------------------------------

def test_store_embed_saves_and_loads(library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    catalog, _ = store.scan(library)

    store.embed(catalog, [FakeModel()])

    assert store.embeddings_path("fake") == tmp_path / "out" / "embeddings" / "fake" / "mean.parquet"
    paths, matrix = store.load_embeddings("fake").matrix()
    assert paths == [ONE, TWO, THREE]
    assert matrix[:, 0].tolist() == [3, 3, 5]


# -- Discogs-EffNet --------------------------------------------------------------

class FakeEssentia:
    max_seconds = 300.0
    processes = 2

    def __init__(self):
        self.paths = []
        self.closed = False

    def embeddings(self, path):
        self.paths.append(path)
        return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    def track_embeddings(self, path):
        self.paths.append(("track", path))
        return np.array([[0, 2], [4, 6]], dtype=np.float32)

    def close(self):
        self.closed = True


def test_discogs_effnet_averages_frames(catalog, library):
    client = FakeEssentia()
    settings = Settings.model_validate({"fetchers": {"embeddings": {"discogs-effnet": {"processes": 2}}}})
    with DiscogsEffnetFetcher(settings, library, client=client) as fetcher:
        embedding = fetcher.fetch(catalog.tracks[THREE])
    assert fetcher.workers == 2  # one track per model process
    assert client.paths == [str(library / THREE)]
    assert (embedding.vector.tolist(), embedding.vector.dtype, embedding.n_frames) == ([3, 4], np.float32, 3)
    assert fetcher.key(None) == f"{EMBEDDING_MODEL}:resample=4:seconds=300"
    assert client.closed


def test_discogs_track_uses_the_track_model(catalog, library):
    client = FakeEssentia()
    with DiscogsTrackFetcher(Settings(), library, client=client) as fetcher:
        embedding = fetcher.fetch(catalog.tracks[THREE])
    assert client.paths == [("track", str(library / THREE))]
    assert (embedding.vector.tolist(), embedding.n_frames) == ([2, 4], 2)
    assert fetcher.key(None) == f"{TRACK_EMBEDDING_MODEL}:resample=4:seconds=300"
    assert client.closed


def test_essentia_embedding_fetchers_from_the_settings(tmp_path):
    models = tmp_path / "models"
    for model in (EMBEDDING_MODEL, TRACK_EMBEDDING_MODEL):
        (models / model).mkdir(parents=True)
        for name in MODEL_FILES[model]:
            (models / model / name).write_text("{}")
    settings = Settings.model_validate({"models": {"folder": str(models)}, "fetchers": {"embeddings": {
        "discogs-effnet": {"max_seconds": 60, "processes": 3}}}})
    with make_fetcher("discogs-effnet", settings, tmp_path) as f:  # checks the model files, loads nothing
        assert isinstance(f, DiscogsEffnetFetcher) and f.models() == [EMBEDDING_MODEL]
        assert (f.root, f.client.models_dir, f.client.max_seconds, f.workers) == (tmp_path, models, 60, 3)
    with make_fetcher("discogs-track", settings, tmp_path) as f:  # needs only the track model
        assert isinstance(f, DiscogsTrackFetcher) and f.workers == 4 and f.models() == [TRACK_EMBEDDING_MODEL]
    assert fetcher_names("embeddings") == ["discogs-effnet", "discogs-track", "muq-mulan", "laion-clap-music",
                                           "description-bge"]
    with pytest.raises(EssentiaError, match="missing Essentia models"):
        make_fetcher("discogs-effnet", Settings.model_validate({"models": {"folder": str(tmp_path / "none")}})).open()


# -- real model ------------------------------------------------------------------

TEST_MUSIC = Path(__file__).resolve().parents[2] / "test_music"
REAL_FILE = TEST_MUSIC / "Metronomy - The Light [GBMVH1900150].flac"
MODEL_DIR = Settings().models.folder


def _real_model_available():
    try:
        check_requirements(MODEL_DIR, (EMBEDDING_MODEL, TRACK_EMBEDDING_MODEL))
    except EssentiaError:
        return False
    return REAL_FILE.is_file()


real_model = pytest.mark.skipif(not _real_model_available(), reason="essentia, its models or test_music/ not available")


@pytest.mark.essentia
@real_model
def test_real_embeddings(tmp_path):
    out = tmp_path / "out"
    catalog, _ = Catalog.scan(TEST_MUSIC)
    catalog.tracks = {REAL_FILE.name: catalog.tracks[REAL_FILE.name]}  # one track is enough
    MetadataStore(out).save(catalog)

    store = MetadataStore(out)
    with make_fetcher("discogs-effnet", Settings(), TEST_MUSIC) as effnet:
        report = store.embed(catalog, [effnet])
    assert report.computed == {"discogs-effnet": 1} and not report.errors

    table = MetadataStore(out).load_embeddings("discogs-effnet")
    embedding = table.get(REAL_FILE.name)
    assert embedding.vector.shape == (1280,) and embedding.n_frames > 200

    # Same as asking the client directly, and the same embeddings the genre model scores.
    with EssentiaClient(MODEL_DIR, EMBEDDINGS, max_seconds=300, processes=1) as client:
        frames = client.embeddings(REAL_FILE)
    assert np.allclose(frames.mean(axis=0), embedding.vector, atol=1e-5)


@pytest.mark.essentia
@real_model
def test_real_track_embeddings():
    with EssentiaClient(MODEL_DIR, EMBEDDINGS + TRACK_EMBEDDINGS, max_seconds=60, processes=1) as client:
        plain, track = client.embeddings(REAL_FILE), client.track_embeddings(REAL_FILE)
    assert track.shape == plain.shape and track.shape[1] == 1280
    assert not np.allclose(track, plain)  # another model, not the genre one again
