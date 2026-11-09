import shutil

import pytest

from libcrate.catalog import Catalog
from libcrate.track import SCHEMA_HASH, GenreSource, SourceStatus, TrackRecord

ONE = "Artist A - One [AAA111].flac"
TWO = "Artist B - Two [BBB222].flac"
THREE = "sub/Artist C - Three.flac"


@pytest.fixture
def extracted(monkeypatch):
    """Records which files scan_library actually re-extracts."""
    calls = []
    real = TrackRecord.from_file

    def spy(path, root):
        calls.append(path.relative_to(root).as_posix())
        return real(path, root)

    monkeypatch.setattr(TrackRecord, "from_file", spy)
    return calls


def test_find_audio_files_skips_junk(library):
    names = [p.relative_to(library).as_posix() for p in Catalog.find_audio_files(library)]
    assert names == [ONE, TWO, THREE]


def test_first_scan_adds_everything(library):
    catalog, report = Catalog.scan(library)

    assert sorted(report.added) == [ONE, TWO, THREE]
    assert report.updated == report.removed == report.failed == []
    assert report.unchanged == 0
    assert list(catalog.tracks) == [ONE, TWO, THREE]
    assert catalog.root == str(library.resolve())
    assert catalog.tracks[ONE].tags.spotify_track_id == "1111111111111111111111"
    assert catalog.errors == []


def test_rescan_without_changes_extracts_nothing(library, extracted):
    catalog, _ = Catalog.scan(library)
    extracted.clear()

    again, report = Catalog.scan(library, catalog)

    assert extracted == []
    assert report.unchanged == 3
    assert again.tracks == catalog.tracks


def test_changed_file_is_reextracted(library, retag, extracted):
    catalog, _ = Catalog.scan(library)
    retag(library / TWO, album="New Album")
    extracted.clear()

    catalog, report = Catalog.scan(library, catalog)

    assert extracted == [TWO]
    assert report.updated == [TWO]
    assert report.unchanged == 2
    assert catalog.tracks[TWO].tags.album == "New Album"


def test_new_file_is_added(library, make_flac, extracted):
    catalog, _ = Catalog.scan(library)
    make_flac(library / "Artist D - Four.flac", {"title": "Four"})
    extracted.clear()

    catalog, report = Catalog.scan(library, catalog)

    assert extracted == ["Artist D - Four.flac"]
    assert report.added == ["Artist D - Four.flac"]
    assert catalog.tracks["Artist D - Four.flac"].tags.title == "Four"


def test_deleted_file_is_removed(library):
    catalog, _ = Catalog.scan(library)
    (library / THREE).unlink()

    catalog, report = Catalog.scan(library, catalog)

    assert report.removed == [THREE]
    assert THREE not in catalog.tracks


def test_other_schema_is_reextracted(library, extracted):
    catalog, _ = Catalog.scan(library)
    catalog.tracks[ONE].schema_hash = "0123456789ab"
    extracted.clear()

    catalog, report = Catalog.scan(library, catalog)

    assert extracted == [ONE]
    assert catalog.tracks[ONE].schema_hash == SCHEMA_HASH


def test_input_catalog_is_not_modified(library):
    catalog, _ = Catalog.scan(library)
    before = catalog.model_dump()
    (library / ONE).unlink()

    Catalog.scan(library, catalog)

    assert catalog.model_dump() == before


def test_corrupt_new_file_is_reported(library):
    (library / "broken.flac").write_bytes(b"not a flac")

    catalog, report = Catalog.scan(library)

    assert [e.path for e in report.failed] == ["broken.flac"]
    assert [e.path for e in catalog.errors] == ["broken.flac"]
    assert "broken.flac" not in catalog.tracks


def test_file_that_breaks_keeps_last_good_record(library, touch_later):
    catalog, _ = Catalog.scan(library)
    good = catalog.tracks[TWO]
    (library / TWO).write_bytes(b"truncated")
    touch_later(library / TWO)

    catalog, report = Catalog.scan(library, catalog)

    assert [e.path for e in report.failed] == [TWO]
    assert catalog.tracks[TWO] == good
    assert report.removed == []


def test_failed_file_is_retried_next_scan(library, make_flac):
    (library / "late.flac").write_bytes(b"still copying")
    catalog, _ = Catalog.scan(library)
    make_flac(library / "late.flac", {"title": "Late"})

    catalog, report = Catalog.scan(library, catalog)

    assert report.added == ["late.flac"]
    assert catalog.errors == []


def test_moved_library_keeps_records(library, tmp_path, extracted):
    """Keys are relative, so a remounted/copied library doesn't trigger a rescan."""
    catalog, _ = Catalog.scan(library)
    moved = tmp_path / "elsewhere"
    shutil.copytree(library, moved)  # copy2 preserves mtimes
    extracted.clear()

    catalog, report = Catalog.scan(moved, catalog)

    assert extracted == []
    assert report.unchanged == 3
    assert catalog.root == str(moved.resolve())


def test_progress_callback(library):
    seen = []
    Catalog.scan(library, on_progress=lambda done, total, rel: seen.append((done, total, rel)))
    assert [(d, t) for d, t, _ in seen] == [(1, 3), (2, 3), (3, 3)]
    assert sorted(rel for _, _, rel in seen) == [ONE, TWO, THREE]


def test_missing_root_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        Catalog.scan(tmp_path / "nope")


# -- saving and queries ---------------------------------------------------------

def test_json_round_trip(library, tmp_path):
    catalog, _ = Catalog.scan(library)
    out = tmp_path / "library.json"

    catalog.save(out)

    assert Catalog.load(out) == catalog
    assert not out.with_name("library.json.tmp").exists()


def test_save_overwrites_existing(library, tmp_path):
    out = tmp_path / "library.json"
    out.write_text("old")
    catalog, _ = Catalog.scan(library)
    catalog.save(out)
    assert Catalog.load(out) == catalog


def test_missing_source(library):
    catalog, _ = Catalog.scan(library)
    catalog.tracks["Artist B - Two [BBB222].flac"].genres["chosic"] = GenreSource(status=SourceStatus.NOT_FOUND)

    # Two was queried (nothing found), so not missing.
    assert [t.path for t in catalog.missing_source("chosic")] == [
        "Artist A - One [AAA111].flac", "sub/Artist C - Three.flac",
    ]
