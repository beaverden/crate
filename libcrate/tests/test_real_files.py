"""Smoke test against the real tagged files in the repo's test_music/ (read-only)."""

from pathlib import Path

import pytest

from libcrate.track import TrackRecord

TEST_MUSIC = Path(__file__).resolve().parents[2] / "test_music"
FILES = sorted(p for p in TEST_MUSIC.glob("*") if TrackRecord.is_audio_file(p)) if TEST_MUSIC.is_dir() else []


@pytest.mark.skipif(not FILES, reason="test_music/ not available")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem[:40])
def test_real_file(path):
    r = TrackRecord.from_file(path, TEST_MUSIC)

    assert r.tags.title
    assert r.tags.artists
    assert r.tags.isrc and r.tags.isrc in path.name
    assert r.tags.spotify_track_id and len(r.tags.spotify_track_id) == 22
    assert r.audio.format == "flac" and r.audio.duration > 0
    assert r.cover is not None
    assert r.genres == {}  # only the fetchers give genres
