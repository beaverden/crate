"""Essentia client and fetcher: a fake client for the logic, the real models for smoke tests."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import libcrate.models.workers as workers_module
from libcrate.catalog import Catalog
from libcrate.models.essentia.client import EMBEDDINGS, GENRES, EssentiaClient
from libcrate.models.essentia.models import EMBEDDING_MODEL, GENRE_MODEL, MODEL_FILES, EssentiaError, check_requirements
from libcrate.fetchers import make_fetcher
from libcrate.fetchers.genres import EssentiaGenreFetcher
from libcrate.fetchers.enrich import lookup
from libcrate.settings import Settings
from libcrate.store import MetadataStore
from libcrate.track import Genre, SourceStatus, TrackRecord

DEFAULTS = Settings().fetchers.get("essentia-genres")
MODEL_DIR = Settings().models.folder


def fetcher(root, client, **settings):
    """An EssentiaGenreFetcher with the default settings, except `settings`."""
    options = Settings.model_validate({"fetchers": {"genres": {"essentia-genres": settings}}})
    return EssentiaGenreFetcher(options, root, client=client)

SCORES = [
    ("Electronic---Deep House", 0.41234),
    ("Electronic---House", 0.30001),
    ("Electronic---Tech House", 0.26),
    ("Electronic---Minimal", 0.25999),
    ("Pop---Synth-pop", 0.1),
]


class FakeClient:
    processes = 1

    def __init__(self, scores=SCORES):
        self.scores_ = scores
        self.paths = []

    def genres(self, path):
        self.paths.append(path)
        return self.scores_

    def close(self):
        self.closed = True


@pytest.fixture
def record(tmp_path, make_flac):
    return TrackRecord.from_file(make_flac(tmp_path / "sub" / "s.flac"), tmp_path)


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "models"
    for model, files in MODEL_FILES.items():
        (d / model).mkdir(parents=True)
        for name in files:
            (d / model / name).write_text("{}")
    return d


def test_check_requirements_missing_files(tmp_path):
    with pytest.raises(EssentiaError, match=fr"missing Essentia models in .*: {EMBEDDING_MODEL}, {GENRE_MODEL} "
                                            r"\(Settings › Models\)"):
        check_requirements(tmp_path, GENRES)


def test_check_requirements_ok(model_dir):
    check_requirements(model_dir, GENRES)
    (model_dir / GENRE_MODEL / f"{GENRE_MODEL}.json").unlink()
    check_requirements(model_dir, EMBEDDINGS)  # embeddings don't need the genre head
    with pytest.raises(EssentiaError, match=f": {GENRE_MODEL} "):
        check_requirements(model_dir, GENRES)


def test_fetch_filters_and_formats(tmp_path, record):
    client = FakeClient()
    f = fetcher(tmp_path, client)

    genres = f.fetch(record)

    assert client.paths == [tmp_path / "sub" / "s.flac"]  # root + relative path
    assert genres == [  # top 3 at >= 15%, lowercased style, parent kept, rounded
        Genre(name="deep house", parent="Electronic", confidence=0.4123),
        Genre(name="house", parent="Electronic", confidence=0.3),
        Genre(name="tech house", parent="Electronic", confidence=0.26),
    ]


def test_fetch_settings(tmp_path, record):
    f = fetcher(tmp_path, FakeClient(), min_confidence=0.3, max_genres=5)
    assert [g.name for g in f.fetch(record)] == ["deep house", "house"]


def test_nothing_confident_enough_is_not_found(tmp_path, record):
    f = fetcher(tmp_path, FakeClient([("Pop---Synth-pop", 0.12), ("Rock---Indie", 0.1)]))
    result = lookup(record, f)
    assert result.status == SourceStatus.NOT_FOUND
    assert result.error == "no genre above 15% (best: Pop---Synth-pop 12%)"


def test_analysis_failure_is_an_error(tmp_path, record):
    class Broken(FakeClient):
        def genres(self, path):
            raise EssentiaError(f"could not decode {path}")

    result = lookup(record, fetcher(tmp_path, Broken()))
    assert result.status == SourceStatus.ERROR
    assert "could not decode" in result.error


def test_closing_the_fetcher_closes_the_classifier(tmp_path):
    client = FakeClient()
    with fetcher(tmp_path, client):
        pass
    assert client.closed


def test_classifier_checks_requirements_before_starting_a_process(tmp_path):
    with pytest.raises(EssentiaError, match="missing Essentia models"):
        EssentiaClient(tmp_path, GENRES, max_seconds=300, processes=1)


class FakeProcess:
    """Stands in for a model process: records how many requests run at once."""

    made, running, most_at_once = [], 0, 0
    lock = threading.Lock()

    def __init__(self, pool):
        self.stopped = False
        self.requests = 0
        FakeProcess.made.append(self)

    def request(self, name, path):
        with FakeProcess.lock:
            FakeProcess.running += 1
            FakeProcess.most_at_once = max(FakeProcess.most_at_once, FakeProcess.running)
        time.sleep(0.02)
        with FakeProcess.lock:
            FakeProcess.running -= 1
        self.requests += 1
        if path == "bad.flac":
            raise EssentiaError("could not decode bad.flac")
        return [["Pop---Synth-pop", 0.5]]

    def stop(self):
        self.stopped = True


def test_requests_spread_over_at_most_n_processes(model_dir, monkeypatch):
    monkeypatch.setattr(FakeProcess, "made", [])
    monkeypatch.setattr(workers_module, "_Worker", FakeProcess)
    client = EssentiaClient(model_dir, GENRES, max_seconds=300, processes=3)
    assert FakeProcess.made == []  # nothing starts before a request

    threads = [threading.Thread(target=client.genres, args=(f"{i}.flac",)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(FakeProcess.made) == 3 and FakeProcess.most_at_once == 3
    assert sum(p.requests for p in FakeProcess.made) == 12
    with pytest.raises(EssentiaError, match="could not decode"):
        client.genres("bad.flac")
    client.genres("ok.flac")  # a failed request gives its process back
    client.close()
    assert all(p.stopped for p in FakeProcess.made)


def test_key_follows_model_and_settings(tmp_path, record):
    a = fetcher(tmp_path, FakeClient())
    b = fetcher(tmp_path, FakeClient(), min_confidence=0.3)
    assert a.key(record) == f"{GENRE_MODEL}:resample=4:min=0.15:top=3"
    assert a.key(record) != b.key(record)  # changing the threshold re-analyses


# -- real models ---------------------------------------------------------------

TEST_MUSIC = Path(__file__).resolve().parents[2] / "test_music"
REAL_FILE = TEST_MUSIC / "Metronomy - The Light [GBMVH1900150].flac"


def _real_models_available():
    try:
        check_requirements(MODEL_DIR, GENRES)
    except EssentiaError:
        return False
    return REAL_FILE.is_file()


@pytest.fixture(scope="module")
def real_fetcher():
    with fetcher(TEST_MUSIC, EssentiaClient(MODEL_DIR, GENRES, max_seconds=DEFAULTS.max_seconds, processes=1)) as f:
        yield f


@pytest.mark.essentia
@pytest.mark.skipif(not _real_models_available(), reason="essentia, its models or test_music/ not available")
@pytest.mark.parametrize("name, expected", [
    ("Metronomy - The Light [GBMVH1900150].flac", [("downtempo", 0.2802), ("chillwave", 0.1831), ("nu-disco", 0.1503)]),
    ("NASAYA - TEMPO [GBCFB2100500].flac",
     [("deep house", 0.2393), ("tropical house", 0.1947), ("progressive house", 0.175)]),
    ("Melissa Laveaux - yemaya [FR9W12566789].flac", [("tropical house", 0.3024), ("deep house", 0.2191), ("house", 0.1826)]),
    ("Omar Dahl - K’ami [US83Z2301234].flac", [("tropical house", 0.6253), ("deep house", 0.4932), ("house", 0.2021)]),
])
def test_real_models_give_known_genres(real_fetcher, name, expected):
    """Values from resampleQuality=4 (quality 1 gives Metronomy chillwave first; 4, like the best, 0, downtempo)."""
    record = TrackRecord.from_file(TEST_MUSIC / name, TEST_MUSIC)

    fetched = real_fetcher.fetch(record)

    assert [g.name for g in fetched] == [n for n, _ in expected]
    assert [g.confidence for g in fetched] == pytest.approx([c for _, c in expected], abs=1e-3)


@pytest.mark.essentia
@pytest.mark.skipif(not _real_models_available(), reason="essentia, its models or test_music/ not available")
def test_real_enrich_with_essentia(tmp_path):
    out = tmp_path / "out"
    catalog, _ = Catalog.scan(TEST_MUSIC)
    rel = REAL_FILE.name
    catalog.tracks = {rel: catalog.tracks[rel]}  # one track to analyse
    store = MetadataStore(out)
    store.save(catalog)

    with make_fetcher("essentia-genres", Settings(), TEST_MUSIC) as essentia:
        report = store.enrich(catalog, [essentia])
    assert report.errors == 0

    result = MetadataStore(out).load().tracks[rel].genres["essentia-genres"]
    assert result.status == SourceStatus.FOUND and result.key is not None
    assert [g.name for g in result.genres] == ["downtempo", "chillwave", "nu-disco"]


real_models = pytest.mark.skipif(not _real_models_available(), reason="essentia, its models or test_music/ not available")


@pytest.mark.essentia
@real_models
def test_real_models_run_in_a_child_process_and_unload(tmp_path):
    client = EssentiaClient(MODEL_DIR, GENRES, max_seconds=DEFAULTS.max_seconds, processes=1)
    assert client._pool.workers == []  # nothing loaded until there's a file to analyse

    with client:
        assert client.genres(REAL_FILE)[0][0] == "Electronic---Downtempo"
        [model_process] = client._pool.workers
        process = model_process._process
        assert process.poll() is None

        (tmp_path / "bad.flac").write_bytes(b"nope")
        with pytest.raises(EssentiaError, match="could not decode"):
            client.genres(tmp_path / "bad.flac")
        assert model_process._process is process  # a bad file doesn't cost a reload

        with pytest.raises(EssentiaError, match="unknown request 'open'"):
            client._request("open", REAL_FILE)  # only Models.REQUESTS run in the child

    assert model_process._process is None and process.poll() is not None


@pytest.mark.essentia
@real_models
def test_real_models_analyse_in_parallel_processes():
    with EssentiaClient(MODEL_DIR, EMBEDDINGS, max_seconds=30, processes=2) as client:
        results = [None, None]

        def analyse(i):
            results[i] = client.embeddings(REAL_FILE)

        threads = [threading.Thread(target=analyse, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(client._pool.workers) == 2
        assert len({p._process.pid for p in client._pool.workers}) == 2
        assert (results[0] == results[1]).all()


@pytest.mark.essentia
@real_models
def test_real_models_recover_from_a_crashed_process():
    with EssentiaClient(MODEL_DIR, GENRES, max_seconds=DEFAULTS.max_seconds, processes=1) as client:
        client.genres(REAL_FILE)
        client._pool.workers[0]._process.kill()

        with pytest.raises(EssentiaError, match="the Essentia process stopped while working on"):
            client.genres(REAL_FILE)
        assert client.genres(REAL_FILE)[0][0] == "Electronic---Downtempo"  # a new process


@pytest.mark.essentia
@real_models
def test_real_models_never_load_tensorflow_in_the_caller():
    code = ("import sys; from libcrate.models.essentia.client import EssentiaClient as C\n"
            f"with C({str(MODEL_DIR)!r}, {GENRES!r}, max_seconds=300, processes=1) as c: c.genres({str(REAL_FILE)!r})\n"
            "print('essentia' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
