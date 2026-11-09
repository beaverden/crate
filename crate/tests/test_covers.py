import os
import time

from libcrate import Catalog

from crate.library.covers import CoverCache

LIGHT = "Metronomy - The Light [GBMVH1900150].flac"


def wait_for(qapp, cache, track):
    loaded = []
    cache.loaded.connect(lambda: loaded.append(True))
    if cache.get(track) is None and track.cover is not None:
        while not loaded:
            qapp.processEvents()
    return cache.get(track)


def wait_until(condition):
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_loads_thumbnails_in_the_background(qapp, library, tmp_path):
    catalog, _ = Catalog.scan(library)
    cache = CoverCache(40, 4)
    cache.set_library(catalog.root, tmp_path / "thumbnails")

    thumbnail = wait_for(qapp, cache, catalog.tracks[LIGHT])
    assert thumbnail is not None
    assert (thumbnail.width(), thumbnail.devicePixelRatio()) == (80, 2)
    # Tracks without a cover aren't read at all.
    assert cache.get(catalog.tracks["Abba - Waterloo.flac"]) is None
    cache.shutdown()


def test_saved_thumbnails_are_used_on_the_next_start(qapp, library, tmp_path):
    catalog, _ = Catalog.scan(library)
    folder = tmp_path / "thumbnails"
    cache = CoverCache(40, 4)
    cache.set_library(catalog.root, folder)
    wait_for(qapp, cache, catalog.tracks[LIGHT])
    cache.shutdown()
    assert [p.suffix for p in folder.iterdir()] == [".jpg"]

    (library / LIGHT).unlink()  # only the saved thumbnail is left
    cache = CoverCache(40, 4)
    cache.set_library(catalog.root, folder)
    assert wait_for(qapp, cache, catalog.tracks[LIGHT]) is not None
    cache.shutdown()


def test_prune_drops_thumbnails_of_changed_files(qapp, library, tmp_path):
    catalog, _ = Catalog.scan(library)
    folder = tmp_path / "thumbnails"
    cache = CoverCache(40, 4)
    cache.set_library(catalog.root, folder)
    wait_for(qapp, cache, catalog.tracks[LIGHT])
    old = next(folder.iterdir())

    stat = (library / LIGHT).stat()
    os.utime(library / LIGHT, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    catalog, _ = Catalog.scan(library, catalog)
    cache.prune(catalog)
    wait_until(lambda: not old.exists())

    assert wait_for(qapp, cache, catalog.tracks[LIGHT]) is not None  # made again for the changed file
    assert len(list(folder.iterdir())) == 1
    cache.shutdown()


def test_unreadable_file_and_unwritable_folder(qapp, library, tmp_path):
    catalog, _ = Catalog.scan(library)
    cache = CoverCache(40, 4)
    cache.set_library("/nonexistent", tmp_path / "thumbnails")
    assert wait_for(qapp, cache, catalog.tracks[LIGHT]) is None

    blocker = tmp_path / "file"
    blocker.write_text("")
    cache.set_library(catalog.root, blocker / "thumbnails")  # can't be created: not saved, still shown
    assert wait_for(qapp, cache, catalog.tracks[LIGHT]) is not None
    cache.shutdown()


def test_nothing_after_shutdown(qapp, library, tmp_path):
    """An enrich still running at closing shows its catalog after the covers have shut down."""
    catalog, _ = Catalog.scan(library)
    cache = CoverCache(40, 4)
    cache.set_library(catalog.root, tmp_path / "thumbnails")
    cache.shutdown()
    cache.prune(catalog)
    assert cache.get(catalog.tracks[LIGHT]) is None
