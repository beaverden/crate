"""The models registry (libcrate.models): what's installed, what a model needs, downloading (fake HTTP)."""

import curl_cffi.requests
import pytest

from libcrate import models
from libcrate.models.essentia.models import EMBEDDING_MODEL, GENRE_MODEL, MODEL_FILES, TRACK_EMBEDDING_MODEL


class FakeDownload:
    def __init__(self, body=b"model", status_code=200, length=None):
        self.body, self.status_code = body, status_code
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        self.closed = False

    def iter_content(self):
        yield self.body[:2]
        yield self.body[2:]

    def close(self):
        self.closed = True


@pytest.fixture
def http(monkeypatch):
    """Serves FakeDownload responses by URL (the URL itself by default); records the URLs requested."""
    state = {"responses": {}, "calls": []}

    def get(url, stream=False, timeout=None):
        state["calls"].append(url)
        answer = state["responses"].get(url, FakeDownload(url.encode()))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(curl_cffi.requests, "get", get)
    return state


def test_every_model_is_described():
    for name, model in models.MODELS.items():
        assert model.name == name and model.kind in ("essentia", "clap", "text") and model.about and model.size > 0
        assert all(required in models.MODELS for required in model.requires)


def test_needed_puts_required_models_first():
    assert models.needed([GENRE_MODEL]) == [EMBEDDING_MODEL, GENRE_MODEL]
    assert models.needed([EMBEDDING_MODEL, GENRE_MODEL, "muq-mulan"]) == [EMBEDDING_MODEL, GENRE_MODEL, "muq-mulan"]
    with pytest.raises(models.ModelError, match="unknown model 'nope'"):
        models.needed(["nope"])


def test_download_with_what_it_needs(tmp_path, http):
    seen = []
    target = tmp_path / "a" / "models"
    assert models.missing(target, [GENRE_MODEL]) == [EMBEDDING_MODEL, GENRE_MODEL]

    done = models.download(target, [GENRE_MODEL], on_model=lambda name, status: seen.append((name, status)))

    assert done == [EMBEDDING_MODEL, GENRE_MODEL]
    assert seen == [(EMBEDDING_MODEL, "downloading"), (EMBEDDING_MODEL, "downloaded"),
                    (GENRE_MODEL, "downloading"), (GENRE_MODEL, "downloaded")]
    assert sorted(p.name for p in target.iterdir()) == sorted([EMBEDDING_MODEL, GENRE_MODEL])  # no .part folders
    for model in (EMBEDDING_MODEL, GENRE_MODEL):
        for name, url in MODEL_FILES[model].items():
            assert (target / model / name).read_bytes() == url.encode()
        assert models.installed(target, model) and models.folder_size(target, model) > 0
    assert models.missing(target, [GENRE_MODEL]) == []


def test_installed_models_are_left_alone(tmp_path, http):
    models.download(tmp_path, [EMBEDDING_MODEL])
    http["calls"].clear()
    assert models.download(tmp_path, [EMBEDDING_MODEL, TRACK_EMBEDDING_MODEL]) == [TRACK_EMBEDDING_MODEL]
    assert http["calls"] == list(MODEL_FILES[TRACK_EMBEDDING_MODEL].values())


def test_stop_between_models(tmp_path, http):
    assert models.download(tmp_path, [GENRE_MODEL], stop=lambda: bool(http["calls"])) == [EMBEDDING_MODEL]


@pytest.mark.parametrize("response, match", [
    (FakeDownload(b"short", length=999), "incomplete download of .*: 5 of 999 bytes"),
    (FakeDownload(b"", status_code=404), "HTTP 404"),
])
def test_failures_leave_nothing(tmp_path, http, response, match):
    http["responses"][MODEL_FILES[EMBEDDING_MODEL][f"{EMBEDDING_MODEL}.pb"]] = response
    with pytest.raises(models.ModelError, match=match):
        models.download(tmp_path, [EMBEDDING_MODEL])
    assert list(tmp_path.iterdir()) == []  # neither the model nor a .part folder
    assert response.closed


def test_network_error(tmp_path, http):
    http["responses"][MODEL_FILES[EMBEDDING_MODEL][f"{EMBEDDING_MODEL}.pb"]] = curl_cffi.requests.RequestsError("timed out")
    with pytest.raises(models.ModelError, match="could not download .*timed out"):
        models.download(tmp_path, [EMBEDDING_MODEL])


def test_delete(tmp_path, http):
    models.download(tmp_path, [EMBEDDING_MODEL])
    models.delete(tmp_path, EMBEDDING_MODEL)
    assert not models.installed(tmp_path, EMBEDDING_MODEL) and models.folder_size(tmp_path, EMBEDDING_MODEL) == 0
