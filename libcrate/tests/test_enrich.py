"""Running genre fetchers over a catalog (fetchers/enrich.py), with fake fetchers (no network, no models)."""

import json
import threading

import pytest

from libcrate.catalog import Catalog
from libcrate.fetchers.genres import GenreFetcher, LookupNotFound
from libcrate.fetchers.enrich import enrich, needs_lookup
from libcrate.track import Genre, GenreSource, SourceStatus

ONE = "Artist A - One [AAA111].flac"     # spotify 111...
TWO = "Artist B - Two [BBB222].flac"     # no IDs
THREE = "sub/Artist C - Three.flac"      # no IDs


class FakeFetcher(GenreFetcher):
    """Looks tracks up by title; `answers` maps title -> genres list | Exception."""

    name = "fake"

    def __init__(self, name, answers=None, keyed_by="title"):
        self.name = name
        self.answers = answers or {}
        self.keyed_by = keyed_by
        self.calls = []
        self.closed = False

    def key(self, record):
        value = getattr(record.tags, self.keyed_by)
        return f"{self.keyed_by}:{value}" if value else None

    def fetch(self, record):
        self.calls.append(record.path)
        answer = self.answers.get(record.tags.title, LookupNotFound("nothing"))
        if isinstance(answer, Exception):
            raise answer
        return [Genre(name=g) for g in answer]

    def close(self):
        self.closed = True


@pytest.fixture
def catalog(library):
    return Catalog.scan(library)[0]


def test_statuses_are_recorded(catalog):
    f = FakeFetcher("web", {"One": ["pop", "disco"], "Two": RuntimeError("HTTP 503")})

    report = enrich(catalog, [f])

    one, two, three = (catalog.tracks[p].genres["web"] for p in (ONE, TWO, THREE))
    assert (one.status, [g.name for g in one.genres]) == (SourceStatus.FOUND, ["pop", "disco"])
    assert (one.key, one.error) == ("title:One", None)
    assert one.fetched_at is not None
    assert (two.status, two.error) == (SourceStatus.ERROR, "RuntimeError: HTTP 503")
    assert (three.status, three.error) == (SourceStatus.NOT_FOUND, "nothing")
    assert report.looked_up["web"] == {SourceStatus.FOUND: 1, SourceStatus.ERROR: 1, SourceStatus.NOT_FOUND: 1}
    assert report.errors == 1
    assert not report.interrupted


def test_track_without_id_is_not_found_without_a_request(catalog):
    f = FakeFetcher("web", keyed_by="isrc")  # none of the fixture tracks has an ISRC

    enrich(catalog, [f])

    assert f.calls == []
    result = catalog.tracks[ONE].genres["web"]
    assert result.status == SourceStatus.NOT_FOUND
    assert result.key is None
    assert "no ID web can look it up by" in result.error


def test_second_run_only_retries_errors(catalog):
    enrich(catalog, [FakeFetcher("web", {"One": ["pop"], "Two": RuntimeError("timeout")})])
    f = FakeFetcher("web", {"One": ["changed"], "Two": ["house"]})

    report = enrich(catalog, [f])

    assert f.calls == [TWO]
    assert catalog.tracks[TWO].genres["web"].status == SourceStatus.FOUND
    assert catalog.tracks[ONE].genre_names("web") == ["pop"]  # found result kept
    assert report.skipped["web"] == 2


def test_changed_id_is_looked_up_again(library, retag):
    catalog, _ = Catalog.scan(library)
    enrich(catalog, [FakeFetcher("web", {"One": ["pop"]})])
    retag(library / ONE, title="One (Remastered)")
    catalog, _ = Catalog.scan(library, catalog)
    f = FakeFetcher("web", {"One (Remastered)": ["rock"]})

    enrich(catalog, [f])

    assert f.calls == [ONE]
    assert catalog.tracks[ONE].genres["web"].key == "title:One (Remastered)"
    assert catalog.tracks[ONE].genre_names("web") == ["rock"]


def test_results_without_a_key_are_redone(catalog):
    """A result no fetcher made (key None, e.g. from an old catalog) is looked up again, and then replaced."""
    catalog.tracks[TWO].genres["web"] = GenreSource(genres=[Genre(name="old")])
    f = FakeFetcher("web", {"Two": ["house"]})

    enrich(catalog, [f])

    assert TWO in f.calls and catalog.tracks[TWO].genre_names("web") == ["house"]


def test_results_survive_a_rescan(library, retag):
    catalog, _ = Catalog.scan(library)
    enrich(catalog, [FakeFetcher("web", {"One": ["pop"]})])
    retag(library / ONE, album="New Album")

    catalog, report = Catalog.scan(library, catalog)

    assert report.updated == [ONE]
    assert catalog.tracks[ONE].tags.album == "New Album"
    assert catalog.tracks[ONE].genre_names("web") == ["pop"]


def test_checkpoints(catalog):
    saves = []
    enrich(catalog, [FakeFetcher("web")], checkpoint_every=2, on_checkpoint=lambda c: saves.append(c))
    assert len(saves) == 2  # after 2 lookups, then at the end
    assert all(s is catalog for s in saves)


def test_progress_callback(catalog):
    seen = []
    enrich(catalog, [FakeFetcher("web", {"One": ["pop"]})],
           on_progress=lambda src, done, total, rel, res: seen.append((src, done, total, rel, res.status)))
    assert seen == [
        ("web", 1, 3, ONE, SourceStatus.FOUND),
        ("web", 2, 3, TWO, SourceStatus.NOT_FOUND),
        ("web", 3, 3, THREE, SourceStatus.NOT_FOUND),
    ]


def test_start_callback(catalog):
    started = []
    web = FakeFetcher("web", {"One": ["pop"]})
    enrich(catalog, [web], on_start=lambda *a: started.append(a))
    enrich(catalog, [web, FakeFetcher("other")], on_start=lambda *a: started.append(a))
    assert started[0] == ("web", 3, 0)
    assert sorted(started[1:]) == [("other", 3, 0), ("web", 0, 3)]  # nothing to do is reported too


def test_stop_event(catalog):
    stop = threading.Event()
    stop.set()
    f = FakeFetcher("web", {"One": ["pop"]})

    report = enrich(catalog, [f], stop=stop)

    assert f.calls == []
    assert report.interrupted
    assert "web" not in catalog.tracks[ONE].genres


def test_stop_mid_run_keeps_finished_lookups(catalog):
    stop = threading.Event()

    def stop_after_first(src, done, total, rel, res):
        stop.set()

    report = enrich(catalog, [FakeFetcher("web", {"One": ["pop"]})], on_progress=stop_after_first, stop=stop)

    assert report.interrupted
    assert catalog.tracks[ONE].genre_names("web") == ["pop"]
    assert "web" not in catalog.tracks[TWO].genres


def test_fetchers_run_in_parallel(catalog):
    """Both fetchers block until the other has started: only possible if they run concurrently."""
    started = {"a": threading.Event(), "b": threading.Event()}

    class Blocking(FakeFetcher):
        def fetch(self, record):
            started[self.name].set()
            other = "b" if self.name == "a" else "a"
            assert started[other].wait(timeout=5)
            return super().fetch(record)

    report = enrich(catalog, [Blocking("a", {"One": ["x"]}), Blocking("b", {"One": ["y"]})])

    assert catalog.tracks[ONE].genre_names("a") == ["x"]
    assert catalog.tracks[ONE].genre_names("b") == ["y"]
    assert report.errors == 0


def test_needs_lookup(catalog):
    f = FakeFetcher("chosic")
    record = catalog.tracks[ONE]
    assert needs_lookup(record, f)  # no chosic result
    record.genres["chosic"] = GenreSource(status=SourceStatus.NOT_FOUND, key=f.key(record))
    assert not needs_lookup(record, f)  # asked already, with the same key


def test_missing_source_counts_errors(catalog):
    enrich(catalog, [FakeFetcher("web", {"One": ["pop"], "Two": RuntimeError("x")})])
    assert [t.path for t in catalog.missing_source("web")] == [TWO]


# -- schema upgrade ----------------------------------------------------------


def test_old_catalog_is_upgraded(library, tmp_path):
    """A catalog written in an older shape (no schema hash, results with fields since dropped) loads, rescans into
    the new shape and keeps its results; one no fetcher keyed is looked up again."""
    out = tmp_path / "lib.json"
    catalog, _ = Catalog.scan(library)
    data = json.loads(catalog.model_dump_json())
    for track in data["tracks"].values():
        del track["schema_hash"]
        track["schema_version"] = 1
    data["tracks"][TWO]["genres"] = {"chosic": {"status": "found", "origin": "tags", "genres": [{"name": "house"}]}}
    out.write_text(json.dumps(data))

    old = Catalog.load(out)
    new, report = Catalog.scan(library, old)

    assert sorted(report.updated) == [ONE, TWO, THREE]
    assert new.tracks[TWO].genre_names("chosic") == ["house"]
    assert needs_lookup(new.tracks[TWO], FakeFetcher("chosic"))


# -- workers and giving up -------------------------------------------------------

@pytest.fixture
def big_catalog(library, make_flac):
    for i in range(7):
        make_flac(library / f"Artist X - Song {i}.flac", {"ARTIST": "X", "TITLE": f"Song {i}"})
    return Catalog.scan(library)[0]  # 10 tracks


def test_workers_run_lookups_concurrently(big_catalog):
    """Each lookup waits until 4 are in flight: only possible with 4 workers."""
    barrier = threading.Barrier(4, timeout=5)

    class Concurrent(FakeFetcher):
        workers = 4

        def fetch(self, record):
            barrier.wait()
            return [Genre(name="x")]

    for rel in list(big_catalog.tracks)[8:]:  # keep 8, a multiple of 4, so every wait completes
        del big_catalog.tracks[rel]

    report = enrich(big_catalog, [Concurrent("web")])

    assert report.looked_up["web"][SourceStatus.FOUND] == 8


def test_one_worker_is_sequential_by_default(big_catalog):
    active, peak = 0, 0
    lock = threading.Lock()

    class Counting(FakeFetcher):
        def fetch(self, record):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            with lock:
                active -= 1
            return [Genre(name="x")]

    assert Counting("web").workers == 1
    enrich(big_catalog, [Counting("web")])
    assert peak == 1


def test_gives_up_when_first_lookups_all_fail(big_catalog):
    broken = FakeFetcher("web", {t.tags.title: RuntimeError("no API key") for t in big_catalog.tracks.values()})
    healthy = FakeFetcher("other", {t.tags.title: ["x"] for t in big_catalog.tracks.values()})

    report = enrich(big_catalog, [broken, healthy])

    assert len(broken.calls) == 3
    assert report.gave_up == {"web": "RuntimeError: no API key"}
    assert report.looked_up["web"][SourceStatus.ERROR] == 3
    assert report.looked_up["other"][SourceStatus.FOUND] == 10  # the other fetcher is unaffected
    assert sum("web" in t.genres for t in big_catalog.tracks.values()) == 3  # the rest stays to do


def test_does_not_give_up_on_mixed_results(big_catalog):
    titles = [t.tags.title for t in big_catalog.tracks.values()]
    answers = {title: RuntimeError("timeout") for title in titles}
    answers[titles[1]] = ["x"]  # one success among the first three
    f = FakeFetcher("web", answers)

    report = enrich(big_catalog, [f])

    assert report.gave_up == {}
    assert len(f.calls) == 10
