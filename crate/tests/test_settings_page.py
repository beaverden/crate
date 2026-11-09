from pathlib import Path

import pytest
from libcrate import MetadataStore, models
from libcrate.models.essentia.models import EMBEDDING_MODEL, GENRE_MODEL
from PySide6.QtCore import QCoreApplication

import crate.settings.models_list as models_list_module
from crate.settings import SettingsPage
from crate.settings.editors import ChoiceEditor
from crate.settings.fetcher_section import size_text, status_lines

MB = ("fetchers", "genres", "musicbrainz-genres")
LLM = ("fetchers", "genres", "llm-genres")
CHOSIC = ("services", "chosic")


@pytest.fixture
def store(tmp_path):
    return MetadataStore(tmp_path / "catalog")


@pytest.fixture
def page(qapp, store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    page = SettingsPage()
    page.set_store(store)
    yield page
    page.shutdown()
    page.deleteLater()


def write(store, text: str) -> None:
    store.folder.mkdir(parents=True, exist_ok=True)
    store.settings_path.write_text(text)


def test_without_a_catalog_nothing_can_be_edited(qapp):
    page = SettingsPage()
    assert page.path is None and "Open a catalog" in page.location.text()
    assert not page.stack.isEnabled()
    assert page.editors[(*LLM, "workers")].value() == 8  # the defaults, for a look
    assert not page.save_button.isEnabled() and not page.revert_button.isEnabled()
    assert all(not row.button.isEnabled() for row in page.models_list.rows.values())  # no folder: no downloads


def test_follows_the_open_catalog(page, store, tmp_path):
    assert page.location.text() == str(store.settings_path) and page.stack.isEnabled()
    other = MetadataStore(tmp_path / "other")
    write(other, "fetchers:\n  genres:\n    llm-genres:\n      workers: 3\n")
    page.set_store(other)
    assert page.path == other.settings_path and page.editors[(*LLM, "workers")].value() == 3


def test_shows_the_defaults_without_a_file(page, store):
    assert not store.settings_path.exists()
    assert page.editors[("services", "llm", "model")].value() == "anthropic:claude-sonnet-5-5"
    assert page.editors[("services", "llm", "options")].value() == {"effort": "low"}
    assert page.editors[("keys", "anthropic")].value() is None
    assert page.editors[(*MB, "use_artist")].value() is True and page.editors[(*MB, "enabled")].value() is False
    assert not page.save_button.isEnabled() and not page.revert_button.isEnabled()


def test_edit_save_and_revert(page, store):
    page.editors[(*LLM, "workers")].set_value(3)
    page.editors[(*LLM, "enabled")].set_value(True)
    page.editors[("keys", "anthropic")].set_value("sk-test")
    page.editors[("services", "llm", "options")].widget.setPlainText("effort: high\ntemperature: 0.2")
    page.editors[(*MB, "use_artist")].set_value(False)
    page.editors[("models", "folder")].set_value(Path("/tmp/models"))
    assert page.save_button.isEnabled()

    page.save()
    settings = store.settings()
    llm = settings.fetchers.get("llm-genres")
    assert llm.workers == 3 and llm.enabled and settings.keys.anthropic.get_secret_value() == "sk-test"
    assert settings.services.llm.options == {"effort": "high", "temperature": 0.2}
    assert settings.fetchers.get("musicbrainz-genres").use_artist is False
    assert settings.models.folder == Path("/tmp/models")
    assert "# requests in parallel" in store.settings_path.read_text()  # still commented
    assert not page.save_button.isEnabled()

    page.editors[(*CHOSIC, "timeout")].set_value(5)
    assert page.revert_button.isEnabled()
    page.load()  # Revert
    assert page.editors[(*CHOSIC, "timeout")].value() == 20 and not page.revert_button.isEnabled()


def test_bad_options_are_not_saved(page, store):
    page.editors[("services", "llm", "options")].widget.setPlainText("effort: [unclosed")
    page.save()
    assert not store.settings_path.exists()
    assert page.save_button.isEnabled()  # still to save


def test_unusable_file_shows_the_problem(qapp, store):
    write(store, "fetchers:\n  genres:\n    llm-genres:\n      workers: lots\n")
    page = SettingsPage()
    page.set_store(store)
    assert not page.problem.isHidden() and "fetchers.genres.llm-genres.workers" in page.problem.text()
    assert page.save_button.isEnabled()  # saving replaces it with what's shown
    page.save()
    assert store.settings().fetchers.get("llm-genres").workers == 8


def test_decimals_read_as_in_the_file(page):
    timeout = page.editors[(*CHOSIC, "timeout")].widget
    assert timeout.text() == "20" and page.editors[(*CHOSIC, "min_interval")].widget.text() == "1.5"
    timeout.lineEdit().setText("2,5")  # a comma works too
    timeout.interpretText()
    assert page.editors[(*CHOSIC, "timeout")].value() == 2.5


def test_tabs_switch_in_any_order(page):
    for name in ("services", "fetchers", "models", "describe", "fetchers", "radio"):
        page.tabs.items[name].click()
        assert page.stack.currentWidget() is page.pages[name], name


def test_api_key_eye_toggles_on_click(page):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    key = page.editors[("keys", "anthropic")]
    key.set_value("sk-test")
    field = key.widget
    assert not field.isPasswordVisible()

    QTest.mousePress(field.viewButton, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert not field.isPasswordVisible()  # not just while held
    QTest.mouseRelease(field.viewButton, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert field.isPasswordVisible()  # stays shown after letting go
    QTest.mouseClick(field.viewButton, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert not field.isPasswordVisible()

    QTest.mouseClick(field.viewButton, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    page.load()  # e.g. Revert
    assert not field.isPasswordVisible()


# -- what a fetcher says --------------------------------------------------------------

def test_a_fetcher_says_what_it_does_and_why_it_cant_run(page, tmp_path):
    page.editors[("models", "folder")].set_value(tmp_path / "models")  # nothing downloaded, whatever's on this machine
    llm = page.sections["llm-genres"]
    assert llm.about.text().startswith("The Discogs styles a language model knows")
    assert llm.problems.text() == "Can't run: no API key for anthropic: set keys.anthropic, or ANTHROPIC_API_KEY."
    page.editors[("keys", "anthropic")].set_value("sk-test")  # follows the edit, before it's saved
    assert llm.problems.isHidden()

    bge = page.sections["description-bge"]
    page.editors[("fetchers", "embeddings", "description-bge", "enabled")].set_value(True)
    assert bge.problems.text() == "Can't run: needs llm-descriptions, which is off."
    assert "Downloads bge-base (440 MB) on the next Enrich." in bge.status.text()
    assert "Used by describe (words)." in bge.status.text()


def test_status_lines():
    from libcrate import FetcherCheck, FetcherProgress
    check = FetcherCheck(notes=["uses chosic-genres when it's on"], downloads=[EMBEDDING_MODEL, GENRE_MODEL])
    counts = FetcherProgress(total=2068, done=1800, redo=243, errors=12, cant=13)
    assert status_lines(check, ["llm-descriptions"], counts, enabled=True) == [
        f"Downloads {EMBEDDING_MODEL}, {GENRE_MODEL} (20 MB) on the next Enrich.",
        "Uses chosic-genres when it's on.",
        "Used by llm-descriptions.",
        "2,068 tracks: 1,800 done  ·  243 to redo  ·  12 failed  ·  13 with nothing to go on. The next Enrich does 255.",
    ]
    assert status_lines(check, [], counts, enabled=False)[-1].endswith("with nothing to go on.")  # off: does nothing
    assert size_text(2_670_000_000) == "2.7 GB" and size_text(85_000_000) == "85 MB"


def test_progress_on_the_catalog(qapp, page, store, library, add_genres):
    catalog = add_genres(store.scan(library)[0])
    page.set_catalog(catalog)
    page.progress_task.wait()
    QCoreApplication.processEvents()
    assert page.progress["essentia-genres"].redo == 1  # The Light's, from other settings
    assert "3 tracks: 0 done  ·  1 to redo  ·  2 never done." in page.sections["essentia-genres"].status.text()


# -- features ----------------------------------------------------------------------

def test_embedding_settings_are_dropdowns_of_what_fits(page, store):
    similar, sound = page.editors[("similar", "embeddings")], page.editors[("describe", "sound")]
    words = page.editors[("describe", "words")]
    assert isinstance(similar, ChoiceEditor) and similar.choices == (
        "discogs-effnet", "discogs-track", "muq-mulan", "laion-clap-music")
    assert sound.choices == ("muq-mulan", "laion-clap-music") and words.choices == ("description-bge",)
    words.set_value(None)
    assert words.widget.currentText() == "none"
    similar.widget.setCurrentText("laion-clap-music")
    page.save()
    settings = store.settings()
    assert settings.similar.embeddings == "laion-clap-music" and settings.describe.words is None


def test_a_feature_says_when_its_fetcher_is_off(page):
    note = page.slot_problems["similar"]
    assert note.text() == ("Enrich doesn't keep the discogs-effnet embeddings up to date while it's off: turn it on "
                           "in Fetchers.")
    page.editors[("fetchers", "embeddings", "discogs-effnet", "enabled")].set_value(True)
    assert note.isHidden()


# -- models ------------------------------------------------------------------------

def test_models_follow_the_folder_and_what_uses_them(page, tmp_path):
    page.editors[("models", "folder")].set_value(tmp_path / "models")
    row = page.models_list.rows[GENRE_MODEL]
    assert page.models_list.folder == tmp_path / "models" and row.button.text() == "Download"
    assert "Not downloaded, 2 MB" in row.contentLabel.text() and "not used by any fetcher that's on" in row.contentLabel.text()
    page.editors[("fetchers", "genres", "essentia-genres", "enabled")].set_value(True)
    assert "used by essentia-genres" in row.contentLabel.text()
    assert "used by essentia-genres" in page.models_list.rows[EMBEDDING_MODEL].contentLabel.text()  # it runs on it


def test_download_and_delete_a_model(page, tmp_path, monkeypatch):
    folder = tmp_path / "models"
    page.editors[("models", "folder")].set_value(folder)

    def download(target, names, on_model=None):
        for name in models.needed(names):
            on_model(name, "downloading")
            for file in models.MODELS[name].files:
                (target / name / file).parent.mkdir(parents=True, exist_ok=True)
                (target / name / file).write_text("model")
            on_model(name, "downloaded")

    monkeypatch.setattr(models, "download", download)
    done = []
    page.models_list.downloaded.connect(done.append)
    row = page.models_list.rows[GENRE_MODEL]
    row.button.click()
    assert not row.button.isEnabled()  # while it runs
    page.models_list.task.wait()
    QCoreApplication.processEvents()
    assert done == [[GENRE_MODEL]] and page.models_list.task is None
    assert models.installed(folder, EMBEDDING_MODEL) and models.installed(folder, GENRE_MODEL)
    assert row.button.text() == "Delete" and row.contentLabel.text().split("\n")[1].startswith("Downloaded")

    monkeypatch.setattr(models_list_module.MessageBox, "exec", lambda self: True)
    row.button.click()
    assert not models.installed(folder, GENRE_MODEL) and row.button.text() == "Download"


def test_a_failed_download_says_so(page, tmp_path, monkeypatch):
    page.editors[("models", "folder")].set_value(tmp_path / "models")

    def failing(target, names, on_model=None):
        raise models.ModelError("could not download x: HTTP 503")

    monkeypatch.setattr(models, "download", failing)
    failures = []
    page.models_list.failed.connect(failures.append)
    page.models_list.rows["bge-base"].button.click()
    page.models_list.task.wait()
    QCoreApplication.processEvents()
    assert failures == ["could not download x: HTTP 503"] and page.models_list.rows["bge-base"].button.isEnabled()
