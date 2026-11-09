"""MusicBrainz client tests against a fake HTTP session (no network)."""

import json

import pytest

import libcrate.services.musicbrainz.client as mb_mod
from libcrate.services.musicbrainz.client import (
    MB_API, MusicBrainzClient, MusicBrainzError, MusicBrainzNotFound,
)
from libcrate.settings import Settings


def make_client(**settings):
    """A MusicBrainzClient with the default settings, except `settings`."""
    defaults = Settings()
    options = defaults.fetchers.get("musicbrainz-genres").model_dump(exclude={"enabled"})
    return MusicBrainzClient(**{**defaults.services.musicbrainz.model_dump(), **options, **settings})


REC_ID = "11111111-1111-1111-1111-111111111111"

RECORDING = {
    "id": REC_ID,
    "genres": [{"name": "pop", "count": 3}, {"name": "jazz rock", "count": 1}],
    "releases": [
        {"release-group": {"id": "rg-single", "genres": [{"name": "pop", "count": 1}]}},
        {"release-group": {"id": "rg-album", "genres": [{"name": "soft rock", "count": 4}, {"name": "Pop", "count": 2}]}},
    ],
    "artist-credit": [
        {"artist": {"id": "a1", "genres": [{"name": "yacht rock", "count": 2}]}},
        {"artist": {"id": "a1", "genres": [{"name": "yacht rock", "count": 2}]}},  # same artist twice
        {"artist": {"id": "a2", "genres": [{"name": "funk", "count": 1}, {"count": 9}]}},  # nameless genre ignored
    ],
}


class FakeResponse:
    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Serves queued responses per URL path and records every request."""

    def __init__(self):
        self.routes: dict[str, list[FakeResponse]] = {}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.headers: dict[str, str] = {}

    def add(self, path, *responses):
        self.routes.setdefault(path, []).extend(responses)

    def get(self, url, params=None, timeout=None):
        path = url.removeprefix(MB_API)
        self.calls.append((path, params))
        queue = self.routes.get(path)
        if not queue:
            return FakeResponse(404, {"error": "not found"})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def close(self):
        self.closed = True


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(mb_mod.time, "sleep", calls.append)
    return calls


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def client(session, sleeps):
    c = make_client()
    c._session = session
    return c


def test_user_agent_header():
    with make_client() as c:
        assert c._session.headers["User-Agent"] == "libcrate/0.1.0"
        assert c._session.headers["Accept"] == "application/json"


def test_get_genres_detailed(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(200, RECORDING))

    d = client.get_genres_detailed(recording_mbid=REC_ID)

    assert d["recording"] == [("pop", 3), ("jazz rock", 1)]
    assert d["release_group"] == [("soft rock", 4), ("Pop", 2)]  # the group with the most votes
    assert d["artist"] == [("yacht rock", 2), ("funk", 1)]  # each artist once
    assert session.calls == [(f"/recording/{REC_ID}", {"inc": mb_mod.MB_INC, "fmt": "json"})]


def test_get_genres_pools_and_ranks(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(200, RECORDING))
    # pop 3 + Pop 2 merge case-insensitively (last spelling seen wins) = 5,
    # soft rock 4, yacht rock 2, jazz rock 1, funk 1
    assert client.get_genres(recording_mbid=REC_ID) == ["Pop", "soft rock", "yacht rock", "jazz rock", "funk"]


def test_pool_keeps_vote_counts(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(200, RECORDING))
    detailed = client.get_genres_detailed(recording_mbid=REC_ID)
    assert client.pool(detailed) == [("Pop", 5), ("soft rock", 4), ("yacht rock", 2), ("jazz rock", 1), ("funk", 1)]


@pytest.mark.parametrize("kwargs, expected", [
    ({"max_genres": 2}, ["Pop", "soft rock"]),
    ({"min_count": 2}, ["Pop", "soft rock", "yacht rock"]),
    ({"min_ratio": 0.5}, ["Pop", "soft rock"]),  # >= 2.5 votes
    ({"use_artist": False}, ["Pop", "soft rock", "jazz rock"]),
    ({"use_release_group": False}, ["pop", "yacht rock", "jazz rock", "funk"]),
])
def test_pool_options(session, sleeps, kwargs, expected):
    c = make_client(**kwargs)
    c._session = session
    session.add(f"/recording/{REC_ID}", FakeResponse(200, RECORDING))
    assert c.get_genres(recording_mbid=REC_ID) == expected


def test_no_genres_returns_empty(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(200, {"id": REC_ID}))
    assert client.get_genres(recording_mbid=REC_ID) == []


def test_isrc_is_resolved_to_recording(client, session):
    session.add("/isrc/USEE10251052", FakeResponse(200, {"recordings": [{"id": REC_ID}]}))
    session.add(f"/recording/{REC_ID}", FakeResponse(200, RECORDING))

    assert client.get_genres(isrc=" usee1-0251052 ")[0] == "Pop"  # normalised before lookup
    assert [path for path, _ in session.calls] == ["/isrc/USEE10251052", f"/recording/{REC_ID}"]


def test_unknown_isrc(client, session):
    session.add("/isrc/XX0000000000", FakeResponse(200, {"recordings": []}))
    with pytest.raises(MusicBrainzNotFound) as e:
        client.get_genres(isrc="XX0000000000")
    assert e.value.status_code == 404


def test_needs_an_id(client):
    with pytest.raises(MusicBrainzError, match="need recording_mbid or isrc"):
        client.get_genres()


def test_responses_are_cached(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(200, RECORDING))
    client.get_genres(recording_mbid=REC_ID)
    client.get_genres(recording_mbid=REC_ID)
    assert len(session.calls) == 1


def test_503_is_retried_with_backoff(client, session, sleeps):
    session.add(f"/recording/{REC_ID}", FakeResponse(503, text="slow down"), FakeResponse(200, RECORDING))

    assert client.get_genres(recording_mbid=REC_ID)[0] == "Pop"
    assert len(session.calls) == 2
    assert 2 in sleeps  # first backoff


def test_503_gives_up_after_three_attempts(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(503, text="slow down"))
    with pytest.raises(MusicBrainzError) as e:
        client.get_genres(recording_mbid=REC_ID)
    assert e.value.status_code == 503
    assert len(session.calls) == 3


def test_404(client):
    with pytest.raises(MusicBrainzNotFound, match=f"no MusicBrainz recording with ID {REC_ID}") as e:
        client.get_genres(recording_mbid=REC_ID)
    assert e.value.status_code == 404


def test_other_http_error(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(500, text="boom"))
    with pytest.raises(MusicBrainzError, match=r"HTTP 500 \(boom\)") as e:
        client.get_genres(recording_mbid=REC_ID)
    assert e.value.status_code == 500


def test_bad_json(client, session):
    session.add(f"/recording/{REC_ID}", FakeResponse(200, None, text="<html>"))
    with pytest.raises(MusicBrainzError, match="bad JSON"):
        client.get_genres(recording_mbid=REC_ID)


def test_network_error_is_wrapped(client, session):
    def boom(*args, **kwargs):
        raise mb_mod.requests.RequestsError("connection reset")
    session.get = boom
    with pytest.raises(MusicBrainzError, match="connection reset"):
        client.get_genres(recording_mbid=REC_ID)


def test_throttle_waits_between_requests(client, monkeypatch, sleeps):
    now = iter([100.0, 100.0, 100.2, 101.0])  # check, stamp, check, stamp
    monkeypatch.setattr(mb_mod.time, "monotonic", lambda: next(now))

    client._throttle()  # first request: last was long ago, no wait
    client._throttle()  # 0.2s later: wait the remaining 0.8s

    assert sleeps == [pytest.approx(0.8)]


def test_context_manager_closes_session(session, sleeps):
    with make_client() as c:
        c._session = session
    assert session.closed
