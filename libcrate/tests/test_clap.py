"""CLAP: audio clips, the model download and conversion, the client's fetcher, text search, and the real models."""

from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

import dataclasses

import libcrate.models.registry as registry
from libcrate.catalog import Catalog
from libcrate.models.clap.client import ClapClient
from libcrate.models.clap.convert import rename
from libcrate.models.clap.models import (
    MODELS, ClapError, check_requirements, load_clips,
)
from libcrate.embeddings import Embedding, EmbeddingTable
from libcrate.fetchers import fetcher_names, make_fetcher
from libcrate.fetchers.embeddings import ClapFetcher, MuqMulanFetcher
from libcrate.models import ModelError, download, installed
from libcrate.settings import Settings
from libcrate.text_search import TextSearch

MODEL = "muq-mulan"


def complete(folder: Path, files=MODELS[MODEL].files) -> Path:
    for name in files:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("{}")
    return folder


# -- audio -------------------------------------------------------------------------

def write(path: Path, seconds: float, rate=44100, channels=2) -> Path:
    """A file whose samples are the time in minutes (0.5 at 30 s), the left channel twice the right."""
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate / 60
    soundfile.write(path, np.stack([t * 4 / 3, t * 2 / 3][:channels], axis=1), rate, subtype="FLOAT")
    return path


def test_clips_from_through_the_track(tmp_path):
    clips = load_clips(write(tmp_path / "a.wav", 60), 24000)
    assert [len(c) for c in clips] == [10 * 24000] * 3 and all(c.dtype == np.float32 for c in clips)
    # 10 s centred on 15, 30 and 45 s: mono (the channels' mean), resampled
    assert [round(float(c[len(c) // 2]) * 60, 1) for c in clips] == [15.0, 30.0, 45.0]
    assert np.allclose(clips[0][1000], (10 + 1000 / 24000) / 60, atol=1e-4)


@pytest.mark.parametrize("seconds, lengths", [
    (25, [10, 10, 5]),  # shorter than the clips together: cut into pieces
    (21, [10, 10]),     # a 1 s last piece is left out
    (1, [1]),           # unless it's the whole track
])
def test_short_tracks_in_pieces(tmp_path, seconds, lengths):
    clips = load_clips(write(tmp_path / "a.wav", seconds, rate=48000), 48000)
    assert [len(c) / 48000 for c in clips] == lengths


def test_unreadable_audio(tmp_path):
    (tmp_path / "bad.flac").write_bytes(b"nope")
    with pytest.raises(ClapError, match="could not decode"):
        load_clips(tmp_path / "bad.flac", 24000)
    with pytest.raises(ClapError, match="no audio"):
        load_clips(write(tmp_path / "empty.wav", 0), 24000)


# -- models ------------------------------------------------------------------------

def test_check_requirements(tmp_path):
    with pytest.raises(ClapError, match=r"missing model muq-mulan .*\(Settings › Models\)"):
        check_requirements(tmp_path, MODEL)
    with pytest.raises(ClapError, match="unknown CLAP model 'other'"):
        check_requirements(tmp_path, "other")
    complete(tmp_path / MODEL)
    assert check_requirements(tmp_path, MODEL) == tmp_path / MODEL and installed(tmp_path, MODEL)
    (tmp_path / MODEL / "text" / "tokenizer.json").unlink()  # a file in its text/ folder counts too
    assert not installed(tmp_path, MODEL)


def fake_model(fail=None):
    """MuQ-MuLan's registry entry, downloading by writing its files (or failing with `fail`)."""
    calls = []

    def fetch(folder):
        calls.append(folder)
        (folder / "config.json").write_text("{}")
        if fail:
            raise fail
        complete(folder)

    return dataclasses.replace(registry.MODELS[MODEL], fetch=fetch), calls


def test_download_builds_in_a_part_folder(tmp_path, monkeypatch):
    model, calls = fake_model()
    monkeypatch.setitem(registry.MODELS, MODEL, model)
    assert download(tmp_path, [MODEL]) == [MODEL]
    assert calls == [tmp_path / f"{MODEL}.part"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [MODEL] and installed(tmp_path, MODEL)
    assert download(tmp_path, [MODEL]) == [] and len(calls) == 1


@pytest.mark.parametrize("error, kind, match", [
    (OSError("disk full"), ModelError, "could not download muq-mulan: OSError: disk full"),
    (ClapError("could not convert x.pt: boom"), ClapError, "could not convert x.pt: boom"),
])
def test_failed_download_leaves_nothing(tmp_path, monkeypatch, error, kind, match):
    monkeypatch.setitem(registry.MODELS, MODEL, fake_model(fail=error)[0])
    with pytest.raises(kind, match=match):
        download(tmp_path, [MODEL])
    assert list(tmp_path.iterdir()) == []


def test_rename_to_transformers():
    qkv = torch.arange(6.0).reshape(6, 1)
    renamed = rename({
        "module.text_branch.encoder.layer.0.attention.self.query.weight": torch.zeros(1),
        "module.audio_branch.layers.0.blocks.0.attn.qkv.weight": qkv,
        "module.audio_branch.layers.0.blocks.0.mlp.fc1.weight": torch.zeros(1),
        "module.text_projection.0.weight": torch.zeros(1),
        "module.audio_projection.2.bias": torch.zeros(1),
        "module.audio_branch.bn0.weight": torch.zeros(1),
    })
    base = "audio_model.audio_encoder.layers.0.blocks.0."
    assert set(renamed) == {
        "text_model.encoder.layer.0.attention.self.query.weight",
        f"{base}attention.self.query.weight", f"{base}attention.self.key.weight", f"{base}attention.self.value.weight",
        f"{base}intermediate.dense.weight",
        "text_projection.linear1.weight", "audio_projection.linear2.bias",
        "audio_model.audio_encoder.batch_norm.weight",
    }
    assert renamed[f"{base}attention.self.value.weight"].flatten().tolist() == [4.0, 5.0]


# -- the fetcher ---------------------------------------------------------------------

class FakeClap:
    model, processes = MODEL, 2

    def __init__(self):
        self.paths, self.closed = [], False

    def audio_embeddings(self, path):
        self.paths.append(path)
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def close(self):
        self.closed = True


def test_fetcher_averages_chunks(library):
    catalog, _ = Catalog.scan(library)
    record = next(iter(catalog.tracks.values()))
    client = FakeClap()
    settings = Settings.model_validate({"fetchers": {"embeddings": {MODEL: {"processes": 2}}}})
    with MuqMulanFetcher(settings, library, client=client) as fetcher:
        embedding = fetcher.fetch(record)
    assert (fetcher.name, fetcher.workers) == (MODEL, 2) and client.paths == [library / record.path]
    assert np.allclose(embedding.vector, [0.7071, 0.7071], atol=1e-4) and embedding.n_frames == 2
    assert fetcher.key(None) == f"{MODEL}:2e01c796b71d:clips=3x10s"
    assert client.closed


def test_make_clap_fetcher(tmp_path):
    assert MODEL in fetcher_names("embeddings")
    settings = Settings.model_validate({"models": {"folder": str(tmp_path)},
                                        "fetchers": {"embeddings": {MODEL: {"processes": 3}}}})
    assert make_fetcher(MODEL, settings).models() == [MODEL]
    with pytest.raises(ClapError, match="missing model"):
        make_fetcher(MODEL, settings, tmp_path).open()
    complete(tmp_path / MODEL)
    with make_fetcher(MODEL, settings, tmp_path) as fetcher:  # checks the folder, loads nothing
        assert isinstance(fetcher, ClapFetcher)
        assert (fetcher.client.model, fetcher.workers) == (MODEL, 3)


# -- text search ---------------------------------------------------------------------

def test_text_search_ranks_by_cosine(library):
    catalog, _ = Catalog.scan(library)
    paths = sorted(catalog.tracks)
    table = EmbeddingTable()
    for path, vector in zip(paths, ([1.0, 0.0], [0.6, 0.8], [0.0, 2.0])):
        table.put(path, "k", Embedding(np.array(vector, dtype=np.float32)))
    table.put("gone.flac", "k", Embedding(np.ones(2, dtype=np.float32)))
    search = TextSearch(catalog, table)
    assert len(search) == 3

    items = search.find(np.array([0.0, 3.0]), count=2)
    assert [i.track.path for i in items] == [paths[2], paths[1]]
    assert [round(i.similarity, 3) for i in items] == [1.0, 0.8]  # cosine: lengths don't count
    assert TextSearch(catalog, EmbeddingTable()).find(np.ones(2), count=5) == []


# -- the real models ------------------------------------------------------------------

TEST_MUSIC = Path(__file__).resolve().parents[2] / "test_music"
MODELS_DIR = Settings().models.folder
QUERIES = ["chill relaxed electronic music", "energetic dance music with a strong beat", "1980s pop song",
           "acoustic folk with singing", "dark heavy techno", "upbeat funk with brass"]
# The same queries' similarities with each model's own code (Tencent's muq package; LAION's laion_clap
# package and the original checkpoint): ours must match them.
ORIGINAL = {
    "muq-mulan": np.array([
        [1.00, 0.46, 0.18, 0.00, 0.26, 0.08],
        [0.46, 1.00, 0.30, -0.02, 0.50, 0.28],
        [0.18, 0.30, 1.00, 0.23, 0.16, 0.23],
        [0.00, -0.02, 0.23, 1.00, 0.20, 0.02],
        [0.26, 0.50, 0.16, 0.20, 1.00, -0.09],
        [0.08, 0.28, 0.23, 0.02, -0.09, 1.00],
    ]),
    "laion-clap-music": np.array([
        [1.00, 0.27, 0.16, 0.15, 0.33, 0.27],
        [0.27, 1.00, 0.06, 0.40, 0.56, 0.35],
        [0.16, 0.06, 1.00, -0.04, 0.11, 0.30],
        [0.15, 0.40, -0.04, 1.00, 0.27, 0.08],
        [0.33, 0.56, 0.11, 0.27, 1.00, 0.16],
        [0.27, 0.35, 0.30, 0.08, 0.16, 1.00],
    ]),
}


@pytest.fixture(scope="module", params=list(MODELS))
def clap(request):
    if not installed(MODELS_DIR, request.param) or not TEST_MUSIC.is_dir():
        pytest.skip(f"the {request.param} model or test_music/ not available")
    with ClapClient(MODELS_DIR, request.param, processes=1) as client:
        yield client


@pytest.mark.clap
def test_real_text_embeddings_match_the_original(clap):
    texts = clap.text_embeddings(QUERIES)
    assert texts.shape == (6, 512) and np.allclose(np.linalg.norm(texts, axis=1), 1, atol=1e-5)
    assert np.allclose(texts @ texts.T, ORIGINAL[clap.model], atol=0.006)


@pytest.mark.clap
def test_real_descriptions_find_the_right_tracks(clap):
    catalog, _ = Catalog.scan(TEST_MUSIC)
    table = EmbeddingTable()
    for path in catalog.tracks:
        clips = clap.audio_embeddings(TEST_MUSIC / path)
        assert clips.shape[1] == 512
        table.put(path, "k", Embedding(clips.mean(axis=0)))
    search = TextSearch(catalog, table)
    texts = clap.text_embeddings(["1980s pop song", "upbeat funk with brass"])
    assert search.find(texts[0], count=1)[0].track.tags.title.startswith("Believe It or Not")
    assert search.find(texts[1], count=1)[0].track.tags.title == "Jamarama"


@pytest.mark.clap
def test_real_text_model_finds_the_matching_description():
    if not installed(MODELS_DIR, "bge-base"):
        pytest.skip("the bge-base model isn't downloaded")
    descriptions = ["Dark, hard-hitting techno with a pounding kick, for peak-time club sets.",
                    "A soft acoustic folk song with gentle female vocals and fingerpicked guitar.",
                    "Upbeat funk with a brass section and a slap bassline."]
    with ClapClient(MODELS_DIR, "bge-base", processes=1) as client:
        docs = client.text_embeddings(descriptions)
        queries = client.query_embeddings(["dark driving techno", "quiet folk with a woman singing", "funky brass"])
    assert docs.shape == (3, 768) and np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-5)
    assert list((queries @ docs.T).argmax(axis=1)) == [0, 1, 2]
