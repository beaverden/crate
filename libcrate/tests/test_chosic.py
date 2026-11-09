"""Chosic client tests against a fake HTTP session (no network)."""

import pytest

import libcrate.services.chosic.client as chosic_mod
from libcrate.services.chosic.client import CHOSIC_TOOLS_API, ChosicClient, ChosicError, ChosicNotFound
from libcrate.settings import Settings


def make_client(**settings):
    """A ChosicClient with the default settings, except `settings`."""
    defaults = Settings()
    options = {"max_genres": defaults.fetchers.get("chosic-genres").max_genres}
    return ChosicClient(**{**defaults.services.chosic.model_dump(), **options, **settings})


TRACK_ID = "53iuhJlwXhSER5J2IYYv1W"
ARTIST_ID = "artist-1"


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Routes GETs by API path; the handshake POST always succeeds unless overridden."""

    def __init__(self):
        self.headers = {}
        self.routes: dict[str, list[FakeResponse]] = {}
        self.calls: list[tuple[str, str]] = []
        self.handshake = FakeResponse(200, {"success": True})
        self.closed = False

    def add(self, path, *responses):
        self.routes.setdefault(path, []).extend(responses)

    def post(self, url, timeout=None):
        self.calls.append(("POST", url.removeprefix(CHOSIC_TOOLS_API)))
        if isinstance(self.handshake, Exception):
            raise self.handshake
        return self.handshake

    def get(self, url, timeout=None):
        path = url.removeprefix(CHOSIC_TOOLS_API)
        self.calls.append(("GET", path))
        queue = self.routes.get(path)
        if not queue:
            return FakeResponse(404, text="not found")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def close(self):
        self.closed = True


def track(artist_id=ARTIST_ID):
    return FakeResponse(200, {"artists": [{"id": artist_id}]})


def artist(*genres):
    return FakeResponse(200, {"genres": list(genres)})


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(chosic_mod.time, "sleep", calls.append)
    return calls


@pytest.fixture
def session(monkeypatch, sleeps):
    s = FakeSession()
    monkeypatch.setattr(chosic_mod.requests, "Session", lambda **kwargs: s)
    return s


@pytest.fixture
def client(session):
    return make_client()


def test_get_genres(client, session):
    session.add(f"/tracks/{TRACK_ID}", track())
    session.add(f"/artists/{ARTIST_ID}", artist("pop", "Pop", " ", "jazz rock"))

    assert client.get_genres(TRACK_ID) == ["pop", "jazz rock"]  # deduped, blanks dropped
    assert session.calls == [("POST", "/handshake/"), ("GET", f"/tracks/{TRACK_ID}"), ("GET", f"/artists/{ARTIST_ID}")]


def test_max_genres(session):
    session.add(f"/tracks/{TRACK_ID}", track())
    session.add(f"/artists/{ARTIST_ID}", artist("a", "b", "c"))
    assert make_client(max_genres=2).get_genres(TRACK_ID) == ["a", "b"]


def test_artist_genres_are_cached(client, session):
    session.add(f"/tracks/{TRACK_ID}", track())
    session.add("/tracks/other", track())
    session.add(f"/artists/{ARTIST_ID}", artist("pop"))

    client.get_genres(TRACK_ID)
    client.get_genres("other")

    assert session.calls.count(("GET", f"/artists/{ARTIST_ID}")) == 1


def test_expired_session_rehandshakes_once(client, session):
    session.add(f"/tracks/{TRACK_ID}", FakeResponse(403), track())
    session.add(f"/artists/{ARTIST_ID}", artist("pop"))

    assert client.get_genres(TRACK_ID) == ["pop"]
    assert [c for c in session.calls if c[0] == "POST"] == [("POST", "/handshake/")] * 2


@pytest.mark.parametrize("track_resp, artist_resp, error, match", [
    (FakeResponse(200, {"artists": []}), None, ChosicNotFound, "did not contain artist ID"),
    (track(), artist(), ChosicNotFound, "did not contain genres"),
    (FakeResponse(404), None, ChosicNotFound, "no entry for /tracks/"),
    (FakeResponse(500, text="boom"), None, ChosicError, r"HTTP 500 \(boom\)"),
    (FakeResponse(200, None), None, ChosicError, "failed to decode"),
])
def test_errors(client, session, track_resp, artist_resp, error, match):
    session.add(f"/tracks/{TRACK_ID}", track_resp)
    if artist_resp:
        session.add(f"/artists/{ARTIST_ID}", artist_resp)
    with pytest.raises(ChosicError, match=match) as e:
        client.get_genres(TRACK_ID)
    assert type(e.value) is error


def test_empty_track_id(client):
    with pytest.raises(ChosicError, match="empty track ID"):
        client.get_genres("  ")


def test_failed_handshake(client, session):
    session.handshake = FakeResponse(200, {"success": False})
    with pytest.raises(ChosicError, match="unsuccessful"):
        client.get_genres(TRACK_ID)


def test_network_error_in_handshake_is_wrapped(client, session):
    session.handshake = chosic_mod.requests.RequestsError("connection reset")
    with pytest.raises(ChosicError, match="handshake request failed: connection reset"):
        client.get_genres(TRACK_ID)


def test_network_error_in_request_is_wrapped(client, session):
    def boom(url, timeout=None):
        raise chosic_mod.requests.RequestsError("timed out")
    session.get = boom
    with pytest.raises(ChosicError, match="chosic request failed: timed out"):
        client.get_genres(TRACK_ID)


def test_requests_are_spaced(client, session, sleeps, monkeypatch):
    now = iter([100.0, 100.0, 100.5, 101.5, 103.0, 103.0])  # (check, stamp) per request
    monkeypatch.setattr(chosic_mod.time, "monotonic", lambda: next(now))
    session.add(f"/tracks/{TRACK_ID}", track())
    session.add(f"/artists/{ARTIST_ID}", artist("pop"))

    client.get_genres(TRACK_ID)  # handshake, track, artist

    assert sleeps == [pytest.approx(1.0)]  # only the track request came too soon


def test_context_manager_closes_session(session):
    with make_client():
        pass
    assert session.closed
