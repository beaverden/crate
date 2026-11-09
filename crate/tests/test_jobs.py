from libcrate import EmbedReport, EnrichReport, MetadataStore, RunReport, ScanReport
from PySide6.QtCore import QCoreApplication

from crate.jobs import Job, JobsPage
from crate.jobs.page import MAX_JOBS, details
from crate.tasks import EnrichTask, ScanTask


def run(task):
    task.run()
    QCoreApplication.processEvents()  # progress from the fetchers' threads arrives queued


# -- the model ---------------------------------------------------------------------

def test_enrich_job():
    job = Job.enrich("music-metadata", ["chosic", "llm", "discogs-effnet"])
    assert job.title == "Enrich  ·  music-metadata" and list(job.steps) == ["chosic", "llm", "discogs-effnet"]
    assert details(job.steps["chosic"]) == "waiting"

    job.progress("chosic", 1, 2, "found")
    assert job.steps["chosic"].state == "running" and details(job.steps["chosic"]) == "1 found"
    job.progress("chosic", 2, 2, "error")
    assert job.steps["chosic"].state == "done" and details(job.steps["chosic"]) == "1 found  ·  1 error"

    enriched = EnrichReport(skipped={"chosic": 5, "llm": 7}, gave_up={"llm": "no API key"})
    job.finish_enrich(RunReport(enriched, EmbedReport(skipped={"discogs-effnet": 7})))
    assert job.state == "done" and job.finished is not None
    assert details(job.steps["chosic"]) == "1 found  ·  1 error  ·  5 already done"
    assert details(job.steps["llm"]) == "gave up: no API key  ·  7 already done"
    assert details(job.steps["discogs-effnet"]) == "nothing to do  ·  7 already done"


def test_steps_with_nothing_to_do_are_done_when_they_start():
    job = Job.enrich("c", ["chosic", "essentia", "discogs-effnet"])
    job.start("chosic", 0, 2068)
    job.start("essentia", 840, 1228)
    assert details(job.steps["chosic"]) == "nothing to do  ·  2,068 already done"
    assert job.steps["chosic"].state == "done" and job.steps["essentia"].state == "running"
    assert details(job.steps["essentia"]) == "starting  ·  1,228 already done"
    assert details(job.steps["discogs-effnet"]) == "waiting"  # the models start after the fetchers


def test_problems():
    job = Job.enrich("c", ["chosic", "llm"])
    job.progress("chosic", 1, 1, "not found")
    job.finish_enrich(RunReport(EnrichReport(skipped={"chosic": 0, "llm": 0})))
    assert not job.has_problems  # not found isn't a problem

    job = Job.enrich("c", ["chosic"])
    job.progress("chosic", 1, 1, "error")
    assert job.has_problems
    job = Job.enrich("c", ["llm"])
    job.finish_enrich(RunReport(EnrichReport(skipped={"llm": 0}, gave_up={"llm": "no API key"})))
    assert job.has_problems
    job = Job.enrich("c", ["llm"])
    job.fail("unknown fetcher")
    assert job.has_problems


def test_stopped_enrich_job():
    job = Job.enrich("c", ["chosic", "discogs-effnet"])
    job.progress("chosic", 1, 3, "found")
    job.finish_enrich(RunReport(EnrichReport(skipped={"chosic": 0}, interrupted=True)))
    assert job.state == "stopped"
    assert details(job.steps["chosic"]) == "stopped  ·  1 found"
    assert details(job.steps["discogs-effnet"]) == "not run: stopped before it"


def test_scan_and_failed_jobs():
    job = Job.scan("c", creating=True)
    assert job.title == "New catalog  ·  c"
    job.progress("files", 3, 3)
    job.finish_scan(ScanReport(added=["a", "b"], unchanged=10))
    assert details(job.steps["files"]) == "2 added  ·  10 unchanged"

    job = Job.enrich("c", ["chosic"])
    job.fail("unknown fetcher")
    assert job.state == "failed" and job.error == "unknown fetcher" and details(job.steps["chosic"]) == "failed"


# -- the page ----------------------------------------------------------------------

def test_page_follows_an_enrich(qapp, library, add_genres, tmp_path, fake_fetchers, enable):
    store = MetadataStore(tmp_path / "out")
    catalog = add_genres(store.scan(library)[0])
    enable(store, "chosic-genres", "discogs-effnet")
    page = JobsPage()
    assert not page.empty.isHidden()

    task = EnrichTask(store, catalog)
    job = page.watch(task)
    card = page.cards[0]
    assert page.empty.isHidden() and card.title.text() == "Enrich  ·  out"
    assert card.rows["chosic-genres"].details.text() == "waiting"
    run(task)

    assert job.state == "done" and card.state.text().startswith("Done  ·  ")
    chosic, fake = card.rows["chosic-genres"], card.rows["discogs-effnet"]
    assert chosic.count.text() == "2 / 2"  # The Light had a result already
    assert chosic.details.text() == "1 found  ·  1 not found  ·  1 already done"
    assert fake.count.text() == "3 / 3" and fake.details.text() == "2 computed  ·  1 error"
    assert chosic.bar.value() == 2 and not chosic.bar.isError()
    assert not page.grab().isNull()


def test_steps_downloads_first_then_in_the_order_they_run(library, tmp_path, fake_fetchers, enable, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    store = MetadataStore(tmp_path / "out")
    catalog, _ = store.scan(library)
    enable(store, "description-bge", "llm-descriptions", "discogs-effnet", "chosic-genres")
    store.settings_path.write_text(store.settings_path.read_text() + f"models:\n  folder: {tmp_path / 'models'}\n")
    assert EnrichTask(store, catalog).steps == ["download bge-base", "chosic-genres", "discogs-effnet",
                                                "llm-descriptions", "description-bge"]
    job = Job.enrich("out", ["download bge-base", "chosic-genres"])
    job.progress("download bge-base", 1, 1, "downloaded")
    assert details(job.steps["download bge-base"]) == "1 downloaded"


def test_page_follows_a_scan_and_a_failure(qapp, library, tmp_path):
    page = JobsPage()
    store = MetadataStore(tmp_path / "out")
    task = ScanTask(store, library)
    page.watch(task)
    run(task)
    assert page.cards[0].title.text() == "New catalog  ·  out"
    assert page.cards[0].rows["files"].details.text() == "3 added"

    catalog = store.load()
    task = EnrichTask(store, catalog, ["nope"])
    page.watch(task)
    run(task)
    card = page.cards[0]  # newest first
    assert card.job.state == "failed" and "unknown fetcher 'nope'" in card.state.text()


def test_page_keeps_the_latest_jobs(qapp, tmp_path):
    page = JobsPage()
    store = MetadataStore(tmp_path / "out")
    for _ in range(MAX_JOBS + 3):
        page.watch(ScanTask(store))
    assert len(page.cards) == MAX_JOBS


def test_stop_from_the_card(qapp, library, tmp_path, add_genres, fake_fetchers, enable):
    store = MetadataStore(tmp_path / "out")
    catalog = add_genres(store.scan(library)[0])
    enable(store, "chosic-genres")
    page = JobsPage()
    task = EnrichTask(store, catalog)
    page.watch(task)
    card = page.cards[0]
    assert not card.stop_button.isHidden() and card.stop_button.isEnabled()

    card.stop_button.click()
    assert task._stop.is_set()
    assert not card.stop_button.isEnabled() and card.state.text().startswith("Stopping after the lookups in flight")
    run(task)
    assert card.job.state == "stopped" and card.stop_button.isHidden()


def test_scans_cant_be_stopped(qapp, tmp_path):
    page = JobsPage()
    page.watch(ScanTask(MetadataStore(tmp_path / "out")))
    assert page.cards[0].stop_button.isHidden()
