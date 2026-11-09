from libcrate import MetadataStore, SourceStatus
from PySide6.QtCore import QCoreApplication

from crate.tasks import EnrichTask, ScanTask


def run(task):
    """Run the task's work in this thread, collecting what it emits."""
    out = {"progress": [], "started": [], "finished": None, "failed": None}
    task.progress.connect(lambda *a: out["progress"].append(a))
    task.started.connect(lambda *a: out["started"].append(a[0]))
    task.finished_ok.connect(lambda *a: out.update(finished=a))
    task.failed.connect(lambda message: out.update(failed=message))
    task.run()
    QCoreApplication.processEvents()  # progress from the fetchers' threads arrives queued
    return out


def test_scan(qapp, library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    out = run(ScanTask(store, library))

    catalog, report = out["finished"]
    assert len(catalog.tracks) == 3 and len(report.added) == 3
    assert len(out["progress"]) == 3
    assert store.load().tracks.keys() == catalog.tracks.keys()
    assert store.settings_path.is_file()  # a new catalog gets the default settings

    # Later scans default to the catalog's library.
    _, report = run(ScanTask(store))["finished"]
    assert report.unchanged == 3


def test_scan_without_library_fails(qapp, tmp_path):
    out = run(ScanTask(MetadataStore(tmp_path / "out")))
    assert "library folder is needed" in out["failed"]


def test_enrich_works_on_a_copy_and_saves(qapp, library, add_genres, tmp_path, fake_fetchers, enable):
    store = MetadataStore(tmp_path / "out")
    catalog = add_genres(store.scan(library)[0])
    enable(store, "chosic-genres")

    task = EnrichTask(store, catalog)
    assert task.steps == ["chosic-genres"]
    out = run(task)

    enriched, report = out["finished"]
    assert report.embedded is None  # no embeddings chosen
    # The Light has a chosic result with the same key (done); the other two are looked up.
    assert report.enriched.looked_up["chosic-genres"][SourceStatus.FOUND] == 1
    assert report.enriched.looked_up["chosic-genres"][SourceStatus.NOT_FOUND] == 1
    assert enriched.tracks["Abba - Waterloo.flac"].genre_names("chosic-genres") == ["pop"]
    assert "chosic-genres" not in catalog.tracks["Abba - Waterloo.flac"].genres
    assert store.load().tracks["Abba - Waterloo.flac"].genre_names("chosic-genres") == ["pop"]


def test_unreadable_settings_fail(qapp, library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    catalog, _ = store.scan(library)
    store.settings_path.write_text("fetchers: [nope\n")
    task = EnrichTask(store, catalog)
    assert task.steps == [] and "invalid YAML" in run(task)["failed"]


def test_enrich_then_embeddings(qapp, library, add_genres, tmp_path, fake_fetchers, enable):
    store = MetadataStore(tmp_path / "out")
    catalog = add_genres(store.scan(library)[0])
    enable(store, "discogs-effnet", "chosic-genres")

    out = run(EnrichTask(store, catalog))
    _, report = out["finished"]
    assert report.enriched.looked_up["chosic-genres"][SourceStatus.FOUND] == 1
    assert report.embedded.computed == {"discogs-effnet": 2}  # TEMPO fails
    assert out["started"] == ["chosic-genres", "discogs-effnet"]
    assert {name for name, *_ in out["progress"]} == {"chosic-genres", "discogs-effnet"}
    assert len(store.load_embeddings("discogs-effnet")) == 2

    task = EnrichTask(store, catalog, ["discogs-effnet"])
    task.stop()  # stopped before anything ran
    assert run(task)["finished"][1].embedded is None
