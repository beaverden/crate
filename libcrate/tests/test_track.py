from pathlib import Path

import pytest
from pydantic import create_model

from libcrate.catalog import Catalog
from libcrate.track import SCHEMA_HASH, Description, DescriptionSource, ExtractError, Genre, GenreSource, TrackRecord
from libcrate.track.track import _schema_hash


@pytest.mark.parametrize("name, expected", [
    ("song.flac", True), ("song.FLAC", True), ("song.mp3", False), ("song.m4a", False),
    ("._song.flac", False), (".hidden.flac", False),
    ("cover.jpg", False), ("notes.txt", False),
])
def test_is_audio_file(name, expected):
    assert TrackRecord.is_audio_file(Path(name)) is expected


def test_flac_tags_are_normalised(tmp_path, make_flac):
    path = make_flac(tmp_path / "lib" / "song.flac", {
        "TITLE": "The Light",
        "ARTIST": ["Metronomy", "Guest"],
        "ALBUM": "Metronomy Forever",
        "ALBUMARTIST": "Metronomy",
        "COMPOSER": ["Joseph Mount", "Martin Hadley"],
        "DATE": "2019-09-13",
        "TRACKNUMBER": "9",
        "TRACKTOTAL": "17",
        "DISCNUMBER": "1/2",
        "ISRC": "GBMVH1900150",
        "BARCODE": "5060686501654",
        "ORGANIZATION": "Because Music",
        "PUBLISHER": "Because Music",
        "COPYRIGHT": "2019 Because Music",
        "COMMENT": "https://open.spotify.com/track/64R2n9O7Aif7YP3ivkOWYS",
        "MUSICBRAINZ_TRACKID": "b1a9c0e9-0000-0000-0000-000000000000",
        "BPM": "120",
        "LYRICS": "[00:20.92] Help me please",
        "GENRE": ["electronic", "pop"],
        "MOOD": "happy",
    })
    t = TrackRecord.from_file(path, tmp_path / "lib").tags

    assert t.title == "The Light"
    assert t.artists == ["Metronomy", "Guest"]
    assert t.album == "Metronomy Forever"
    assert t.album_artists == ["Metronomy"]
    assert t.composers == ["Joseph Mount", "Martin Hadley"]
    assert (t.date, t.year) == ("2019-09-13", 2019)
    assert (t.track_number, t.track_total) == (9, 17)
    assert (t.disc_number, t.disc_total) == (1, 2)  # total from "1/2"
    assert t.isrc == "GBMVH1900150"
    assert t.upc == "5060686501654"  # BARCODE fallback
    assert t.label == "Because Music"  # ORGANIZATION fallback
    assert t.publisher == "Because Music"
    assert t.copyright == "2019 Because Music"
    assert t.spotify_track_id == "64R2n9O7Aif7YP3ivkOWYS"
    assert t.musicbrainz_track_id == "b1a9c0e9-0000-0000-0000-000000000000"
    assert t.bpm == 120.0
    assert t.lyrics == "[00:20.92] Help me please"
    assert t.genre == ["electronic", "pop"]
    assert t.extra == {"mood": ["happy"]}  # unmapped tags are kept, lowercased


@pytest.mark.parametrize("tags, expected", [
    ({"COMMENT": "https://open.spotify.com/track/64R2n9O7Aif7YP3ivkOWYS"}, "64R2n9O7Aif7YP3ivkOWYS"),
    ({"COMMENT": "Bought on Bandcamp"}, None),  # a non-Spotify comment is just a comment
    ({"SPOTIFY_TRACK_ID": "64R2n9O7Aif7YP3ivkOWYS"}, "64R2n9O7Aif7YP3ivkOWYS"),  # fallback tag
    ({"COMMENT": "https://open.spotify.com/track/1111111111111111111111",
      "SPOTIFY_TRACK_ID": "64R2n9O7Aif7YP3ivkOWYS"}, "1111111111111111111111"),  # comment first
    ({"COMMENT": "nice", "SPOTIFY_TRACK_ID": "not-an-id"}, None),
])
def test_spotify_track_id(tmp_path, make_flac, tags, expected):
    t = TrackRecord.from_file(make_flac(tmp_path / "s.flac", tags), tmp_path).tags
    assert t.spotify_track_id == expected
    assert t.comment == tags.get("COMMENT")


def test_isrc_tag_wins_over_filename(tmp_path, make_flac):
    path = make_flac(tmp_path / "Artist - Song [GBMVH1900150].flac", {"ISRC": "USEE10251052"})
    assert TrackRecord.from_file(path, tmp_path).tags.isrc == "USEE10251052"


def test_isrc_falls_back_to_filename(tmp_path, make_flac):
    path = make_flac(tmp_path / "Óperentzia - 60 Pengő [QZZ792501624].flac", {"TITLE": "60 Pengő"})
    assert TrackRecord.from_file(path, tmp_path).tags.isrc == "QZZ792501624"


def test_explicit_totals_win_over_slash_form(tmp_path, make_flac):
    path = make_flac(tmp_path / "s.flac", {"TRACKNUMBER": "3/12", "TOTALTRACKS": "14"})
    t = TrackRecord.from_file(path, tmp_path).tags
    assert (t.track_number, t.track_total) == (3, 14)


def test_bad_numbers_become_none(tmp_path, make_flac):
    path = make_flac(tmp_path / "s.flac", {"TRACKNUMBER": "A1", "DATE": "unknown", "BPM": "fast"})
    t = TrackRecord.from_file(path, tmp_path).tags
    assert t.track_number is None
    assert t.year is None
    assert t.bpm is None
    assert t.date == "unknown"


def test_genre_tags_are_tags_not_results(tmp_path, make_flac):
    """Genres come from the fetchers only: the file's GENRE tag is a tag, and any other genre-ish tag an extra one."""
    path = make_flac(tmp_path / "s.flac", {"GENRE": ["Pop", "80S"], "GENRE_ESSENTIA": ["soft rock (47.40%)"]})
    r = TrackRecord.from_file(path, tmp_path)

    assert r.genres == {} and r.descriptions == {}
    assert r.tags.genre == ["Pop", "80S"]
    assert r.tags.extra["genre_essentia"] == ["soft rock (47.40%)"]


def test_audio_info_and_cover(tmp_path, make_flac, png):
    path = make_flac(tmp_path / "s.flac", {"TITLE": "x"}, seconds=2.5, cover=png)
    r = TrackRecord.from_file(path, tmp_path)

    assert r.audio.format == "flac"
    assert r.audio.duration == 2.5
    assert r.audio.sample_rate == 44100
    assert r.audio.channels == 2
    assert r.audio.bits_per_sample == 16
    assert r.cover is not None
    assert (r.cover.mime, r.cover.size, r.cover.width, r.cover.height) == ("image/png", len(png), 1, 1)


def test_file_without_tags_or_cover(tmp_path, make_flac):
    path = make_flac(tmp_path / "s.flac")
    r = TrackRecord.from_file(path, tmp_path)
    assert r.tags.title is None
    assert r.tags.artists == []
    assert r.tags.extra == {}
    assert r.genres == {}
    assert r.cover is None


def test_record_path_and_stat(tmp_path, make_flac):
    path = make_flac(tmp_path / "lib" / "sub" / "s.flac")
    r = TrackRecord.from_file(path, tmp_path / "lib")
    st = path.stat()
    assert r.path == "sub/s.flac"
    assert (r.stat.size, r.stat.mtime_ns) == (st.st_size, st.st_mtime_ns)


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "bad.flac"
    path.write_bytes(b"definitely not audio")
    with pytest.raises(ExtractError):
        TrackRecord.from_file(path, tmp_path)


def test_genre_names(library):
    catalog, _ = Catalog.scan(library)
    track = catalog.tracks["Artist A - One [AAA111].flac"]
    track.genres["chosic"] = GenreSource(genres=[Genre(name="POP"), Genre(name="disco")])
    track.genres["essentia"] = GenreSource(genres=[Genre(name="deep house", confidence=0.4)])
    track.genres["musicbrainz"] = GenreSource(genres=[Genre(name="Pop")])

    assert track.genre_names("musicbrainz") == ["Pop"]
    assert track.genre_names("essentia") == ["deep house"]
    assert track.genre_names("never-queried") == []
    # all sources in the order they were added, deduplicated case-insensitively (first spelling wins)
    assert track.genre_names() == ["POP", "disco", "deep house"]


def _model(doc, field_type):
    return create_model("Record", __doc__=doc, x=(field_type, ...))


def test_schema_hash_follows_the_shape_not_the_docs():
    one, other = _model("One doc.", int), _model("Another doc.", int)
    assert one.model_json_schema()["description"] == "One doc."
    assert _schema_hash(one) == _schema_hash(other)
    assert _schema_hash(_model("One doc.", int)) != _schema_hash(_model("One doc.", str))
    assert len(SCHEMA_HASH) == 12


def test_is_stale(tmp_path, make_flac, touch_later):
    path = make_flac(tmp_path / "s.flac")
    record = TrackRecord.from_file(path, tmp_path)
    assert not record.is_stale(path)

    touch_later(path)
    assert record.is_stale(path)  # changed on disk
    fresh = TrackRecord.from_file(path, tmp_path)
    assert fresh.model_copy(update={"schema_hash": "0123456789ab"}).is_stale(path)  # another shape
    assert fresh.is_stale(tmp_path / "gone.flac")


def test_keep_results(tmp_path, make_flac):
    path = make_flac(tmp_path / "s.flac", {"TITLE": "x"})
    old = TrackRecord.from_file(path, tmp_path)
    old.genres["chosic"] = GenreSource(genres=[Genre(name="disco")], key="spotify:x")
    old.descriptions["llm"] = DescriptionSource(description=Description(text="Calm."))
    new = TrackRecord.from_file(path, tmp_path)

    new.keep_results(old)

    assert new.genre_names("chosic") == ["disco"] and new.description_text() == "Calm."
