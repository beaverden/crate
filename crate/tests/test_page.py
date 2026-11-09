from pathlib import Path

from libcrate import MetadataStore
from PySide6.QtWidgets import QWidget

from crate.library import LibraryPage
from crate.library.catalog_menu import CatalogMenu, _short
from crate.library.new_catalog_dialog import NewCatalogDialog
from crate.library.model import TRACK_ROLE


def wait(qapp, page):
    while page.task is not None:
        qapp.processEvents()


def test_opens_folder_searches_and_shows_details(qapp, library, add_genres, tmp_path, qsettings):
    store = MetadataStore(tmp_path / "out")
    store.save(add_genres(store.scan(library)[0]))
    page = LibraryPage(qsettings)
    page.open_catalog(tmp_path / "out")

    assert page.proxy.rowCount() == 3
    assert page.summary.text().startswith("3 tracks")
    assert page.enrich_button.isEnabled()

    page.search.setText("chillwave")
    assert page.proxy.rowCount() == 1
    page.search.setText("metronomy")  # an artist, drawn under the title
    assert page.proxy.rowCount() == 1
    page.table.selectRow(0)
    assert "The Light" in page.details.text()
    assert not page.grab().isNull()  # paints the rows: covers, titles and artists
    page.shutdown()

    # Reopened on the next start.
    assert LibraryPage(qsettings).proxy.rowCount() == 3


def test_scan_keeps_selection(qapp, library, tmp_path, qsettings):
    MetadataStore(tmp_path / "out").scan(library)
    page = LibraryPage(qsettings)
    page.open_catalog(tmp_path / "out")
    page.table.selectRow(1)
    selected = page._selected_track().path

    page.scan()
    assert not page.scan_button.isEnabled()
    wait(qapp, page)

    assert page.scan_button.isEnabled()
    assert page.table.selectionModel().selectedRows()[0].data(TRACK_ROLE).path == selected


def test_open_needs_a_catalog(qapp, library, tmp_path, qsettings):
    page = LibraryPage(qsettings)
    notices = []
    page._notify = lambda kind, title, content, duration=0: notices.append(title)
    assert not page.welcome.isHidden() and page.body.isHidden()  # New catalog and Open catalog instead of a table
    assert not page.scan_button.isEnabled() and not page.enrich_button.isEnabled()

    assert not page.open_catalog(tmp_path / "nothing")
    assert notices == ["No catalog there"] and page.store is None

    MetadataStore(tmp_path / "out").init(library)
    assert page.open_catalog(tmp_path / "out")  # the music folder comes from the catalog
    assert page.catalog.root == str(library.resolve()) and page.scan_button.isEnabled()
    assert page.welcome.isHidden() and not page.body.isHidden()
    page.shutdown()


def test_recent_catalogs_and_the_menu(qapp, library, tmp_path, qsettings):
    for name in ("a", "b", "c"):
        MetadataStore(tmp_path / name).init(library)
    page = LibraryPage(qsettings)
    for name in ("a", "b", "c", "a"):
        page.open_catalog(tmp_path / name)
    assert page.recent_catalogs() == [tmp_path / "a", tmp_path / "c", tmp_path / "b"]  # most recent first, once each

    (tmp_path / "b" / "catalog.json").unlink()
    assert page.recent_catalogs() == [tmp_path / "a", tmp_path / "c"]  # gone: not listed

    menu = CatalogMenu(page)
    assert list(menu.catalog_actions) == [tmp_path / "a", tmp_path / "c"]
    assert menu.catalog_actions[tmp_path / "a"].text().startswith("a   ·   ")
    assert not menu.catalog_actions[tmp_path / "a"].isEnabled()  # the open one
    menu.catalog_actions[tmp_path / "c"].trigger()
    assert page.store.folder == tmp_path / "c"

    page.scan()  # busy: nothing can be switched
    menu = CatalogMenu(page)
    assert not any(a.isEnabled() for a in [*menu.catalog_actions.values(), menu.new_action, menu.open_action])
    wait(qapp, page)
    page.shutdown()


def test_double_click_asks_to_play_in_table_order(qapp, library, tmp_path, qsettings):
    MetadataStore(tmp_path / "out").scan(library)
    page = LibraryPage(qsettings)
    page.open_catalog(tmp_path / "out")
    requests = []
    page.play_requested.connect(lambda tracks, row, root: requests.append((tracks, row, root)))

    page.table.doubleClicked.emit(page.proxy.index(1, 0))

    tracks, row, root = requests[0]
    assert [t.tags.artists[0] for t in tracks] == ["ABBA", "Metronomy", "NASAYA"]  # sorted by artist
    assert row == 1 and root == page.catalog.root
    page.set_playing(tracks[1])
    assert page.table.itemDelegate().playing == tracks[1].path
    page.shutdown()


def test_playing_track_is_selected_and_shown(qapp, library, tmp_path, qsettings):
    MetadataStore(tmp_path / "out").scan(library)
    page = LibraryPage(qsettings)
    page.open_catalog(tmp_path / "out")
    page.table.selectRow(0)
    tracks = [page.proxy.index(r, 0).data(TRACK_ROLE) for r in range(3)]

    page.set_playing(tracks[2])  # e.g. the queue moved on
    assert page._selected_track() is tracks[2]
    assert tracks[2].tags.title in page.details.text()

    page.search.setText("waterloo")  # hides the playing track's row
    page.set_playing(tracks[1])
    assert page._selected_track() is None
    assert tracks[1].tags.title in page.details.text()

    page.set_playing(None)  # stopped: the panel stays as it is
    assert tracks[1].tags.title in page.details.text()
    page.shutdown()


def test_create_catalog(qapp, library, tmp_path, qsettings):
    page = LibraryPage(qsettings)
    opened, notices = [], []
    page.store_changed.connect(opened.append)
    page._notify = lambda kind, title, content, duration=0: notices.append((title, content))
    page.create_catalog(library, tmp_path / "new")
    assert page.task is not None  # scanning straight away
    assert [store.folder for store in opened] == [tmp_path / "new"]  # e.g. for the settings page
    wait(qapp, page)
    assert page.proxy.rowCount() == 3 and page.catalog.root == str(library.resolve())
    assert (tmp_path / "new" / "catalog.json").is_file()
    assert (tmp_path / "new" / "settings.yaml").is_file()  # as MetadataStore.init writes it
    title, content = notices[-1]
    assert title == "Catalog created" and "Settings › Fetchers" in content
    # Reopened on the next start; the music folder is suggested for the next new catalog.
    assert (qsettings.value("output_folder"), qsettings.value("library_folder")) == \
        (str(tmp_path / "new"), str(library.resolve()))
    page.shutdown()


def test_create_catalog_refuses_an_existing_one(qapp, library, tmp_path, qsettings):
    MetadataStore(tmp_path / "out").init(library)
    page = LibraryPage(qsettings)
    notices = []
    page._notify = lambda kind, title, content, duration=0: notices.append(title)
    page.create_catalog(library, tmp_path / "out")
    assert notices == ["There's a catalog there already"]
    assert page.task is None and page.store is None


def test_new_catalog_dialog(qapp, library, tmp_path):
    window = QWidget()
    window.resize(800, 600)
    dialog = NewCatalogDialog(library, tmp_path / "new", parent=window)
    assert dialog.yesButton.isEnabled()
    dialog._library.setText(str(tmp_path / "missing"))
    assert not dialog.yesButton.isEnabled()  # the music folder must exist

    dialog._library.setText(str(library))
    MetadataStore(tmp_path / "out").init(library)
    dialog._catalog.setText(str(tmp_path / "out"))
    assert not dialog.yesButton.isEnabled() and "holds a catalog already: Open it instead" in dialog.hint.text()

    dialog._catalog.setText(str(tmp_path / "new"))
    dialog._settings_from.setText(str(tmp_path))  # a folder without settings
    assert not dialog.yesButton.isEnabled() and "No settings to copy" in dialog.hint.text()
    dialog._settings_from.setText(str(tmp_path / "out"))  # a catalog
    assert dialog.yesButton.isEnabled() and dialog.settings_from() == tmp_path / "out"
    dialog._settings_from.setText(str(tmp_path / "out" / "settings.yaml"))  # or its file
    assert dialog.yesButton.isEnabled()
    dialog._settings_from.clear()
    assert dialog.settings_from() is None


def test_create_catalog_with_settings_from_another(qapp, library, tmp_path, qsettings):
    first = MetadataStore(tmp_path / "first")
    first.folder.mkdir()
    first.settings_path.write_text("fetchers:\n  genres:\n    chosic-genres: {enabled: true}\n")
    page = LibraryPage(qsettings)
    page.create_catalog(library, tmp_path / "new", tmp_path / "first")
    wait(qapp, page)
    assert page.store.settings().fetchers.get("chosic-genres").enabled
    page.shutdown()


def test_failed_create_goes_back_to_the_welcome(qapp, library, tmp_path, qsettings):
    page = LibraryPage(qsettings)
    opened = []
    page.store_changed.connect(opened.append)
    page._notify = lambda *a, **k: None
    page.create_catalog(tmp_path / "missing", tmp_path / "new")
    wait(qapp, page)
    assert page.store is None and opened[-1] is None and not page.welcome.isHidden()


def test_create_catalog_reports_unreadable_settings(qapp, library, tmp_path, qsettings):
    (tmp_path / "bad.yaml").write_text("services:\n  llm:\n    modle: x\n")
    page = LibraryPage(qsettings)
    notices = []
    page._notify = lambda kind, title, content, duration=0: notices.append((title, content))
    page.create_catalog(library, tmp_path / "new", tmp_path / "bad.yaml")
    assert notices[0][0] == "Can't create the catalog" and "services.llm.modle" in notices[0][1]
    assert page.task is None and page.store is None and not (tmp_path / "new").exists()


def test_enrich_runs_what_the_settings_choose(qapp, library, tmp_path, qsettings, monkeypatch, enable):
    store = MetadataStore(tmp_path / "out")
    store.init(library)
    page = LibraryPage(qsettings)
    page.open_catalog(store.folder)
    notices, started = [], []
    page._notify = lambda kind, title, content, duration=0: notices.append(title)
    monkeypatch.setattr(page, "_start", started.append)

    page.enrich()  # nothing chosen yet
    assert notices == ["Nothing chosen to run"] and started == []

    enable(store, "discogs-effnet", "chosic-genres")
    page.enrich()
    assert started[0].names is None and started[0].steps[-2:] == ["chosic-genres", "discogs-effnet"]

    store.settings_path.write_text("fetchers: chosic\n")
    page.enrich()
    assert notices[-1] == "Can't read the settings" and len(started) == 1
    page.shutdown()


def test_long_paths_are_shortened_in_the_menu():
    assert _short(Path.home() / "music-metadata") == "~/music-metadata"
    long = _short(Path("/Volumes/KINGSTON/a/very/long/path/to/somewhere/deep/music-metadata"))
    assert len(long) == 47 and long.startswith("/Volumes/KINGSTON") and long.endswith("deep/music-metadata")
