"""The online fetchers, with fake clients (no network)."""

import pytest

from libcrate.catalog import Catalog
from libcrate.services.chosic.client import ChosicError, ChosicNotFound
from libcrate.services.musicbrainz.client import MusicBrainzError, MusicBrainzNotFound
from libcrate.fetchers.genres import ChosicGenreFetcher, GenreFetcher, MusicBrainzGenreFetcher
from libcrate.fetchers.enrich import lookup
from libcrate.settings import Settings
from libcrate.track import Genre, SourceStatus, TrackRecord

ONE = "Artist A - One [AAA111].flac"     # has a Spotify URL in COMMENT
TWO = "Artist B - Two [BBB222].flac"     # no IDs


@pytest.fixture
def catalog(library):
    return Catalog.scan(library)[0]


def test_fetcher_is_abstract():
    with pytest.raises(TypeError):
        GenreFetcher(Settings())


class FakeMBClient:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls, self.closed = result, error, [], False

    def get_genres_detailed(self, recording_mbid=None, isrc=None):
        self.calls.append((recording_mbid, isrc))
        if self.error:
            raise self.error
        return self.result

    def pool(self, detailed):
        return detailed

    def close(self):
        self.closed = True


def test_musicbrainz_fetcher(tmp_path, make_flac):
    rec = TrackRecord.from_file(make_flac(tmp_path / "s.flac", {"ISRC": "us-ee1-02-51052"}), tmp_path)
    client = FakeMBClient(result=[("pop", 5), ("jazz rock", 1)])
    f = MusicBrainzGenreFetcher(Settings(), client=client)

    assert f.key(rec) == "isrc:USEE10251052"
    assert f.fetch(rec) == [Genre(name="pop", votes=5), Genre(name="jazz rock", votes=1)]
    assert client.calls == [(None, "us-ee1-02-51052")]

    with f:
        pass
    assert client.closed


def test_musicbrainz_fetcher_prefers_mbid(tmp_path, make_flac):
    rec = TrackRecord.from_file(make_flac(tmp_path / "s.flac", {"ISRC": "X", "MUSICBRAINZ_TRACKID": "ABC-1"}), tmp_path)
    assert MusicBrainzGenreFetcher(Settings(), client=FakeMBClient()).key(rec) == "mbid:abc-1"


@pytest.mark.parametrize("client, outcome", [
    (FakeMBClient(error=MusicBrainzNotFound("no recording", 404)), SourceStatus.NOT_FOUND),
    (FakeMBClient(result=[]), SourceStatus.NOT_FOUND),  # recording exists, nobody voted
    (FakeMBClient(error=MusicBrainzError("HTTP 503", 503)), SourceStatus.ERROR),
])
def test_musicbrainz_fetcher_outcomes(tmp_path, make_flac, client, outcome):
    rec = TrackRecord.from_file(make_flac(tmp_path / "s.flac", {"ISRC": "USEE10251052"}), tmp_path)
    assert lookup(rec, MusicBrainzGenreFetcher(Settings(), client=client)).status == outcome


class FakeChosicClient:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.closed = result, error, False

    def get_genres(self, track_id):
        if self.error:
            raise self.error
        return self.result

    def close(self):
        self.closed = True


@pytest.mark.parametrize("client, outcome, genres", [
    (FakeChosicClient(result=["pop", "disco"]), SourceStatus.FOUND, ["pop", "disco"]),
    (FakeChosicClient(error=ChosicNotFound("no genres")), SourceStatus.NOT_FOUND, []),
    (FakeChosicClient(error=ChosicError("HTTP 500")), SourceStatus.ERROR, []),
])
def test_chosic_fetcher(catalog, client, outcome, genres):
    f = ChosicGenreFetcher(Settings(), client=client)
    rec = catalog.tracks[ONE]
    assert f.key(rec) == "spotify:1111111111111111111111"
    result = lookup(rec, f)
    assert (result.status, [g.name for g in result.genres]) == (outcome, genres)


def test_chosic_fetcher_needs_spotify_id(catalog):
    assert ChosicGenreFetcher(Settings(), client=FakeChosicClient()).key(catalog.tracks[TWO]) is None
