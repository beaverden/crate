import numpy as np
import pytest
from libcrate import Embedding, EmbeddingTable, MetadataStore, make_fetcher
from PySide6.QtCore import QSettings, Qt

from crate.library import LibraryPage
from crate.library.covers import CoverCache
from crate.library.model import TRACK_ROLE
from crate.similar import SimilarPage

VECTORS = {"The Light": [1.0, 0.0, 0.0], "TEMPO": [0.8, 0.6, 0.0], "Waterloo": [0.0, 0.2, 1.0]}

# What discogs-effnet makes with the default settings: a feature uses only rows made so.
KEY = make_fetcher("discogs-effnet").key(None)


@pytest.fixture
def store(library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    store.scan(library)
    return store


def save_embeddings(store, titles):
    table = EmbeddingTable()
    for track in store.load().tracks.values():
        if track.tags.title in titles:
            table.put(track.path, KEY, Embedding(np.array(VECTORS[track.tags.title], dtype=np.float32)))
    path = store.embeddings_path("discogs-effnet")
    path.parent.mkdir(parents=True)
    table.save(path)


def track(catalog, title):
    return next(t for t in catalog.tracks.values() if t.tags.title == title)


@pytest.fixture
def page(qapp):
    page = SimilarPage(CoverCache(40, 4))
    notices = page.notices = []
    page._notify = lambda kind, title, content, duration=0: notices.append(title)
    return page


def test_similar_to_a_track(page, store):
    save_embeddings(store, VECTORS)
    catalog = store.load()
    assert not page.play_button.isEnabled()

    assert page.find(store, catalog, track(catalog, "The Light"))

    rows = [page.model.index(r, 0).data(TRACK_ROLE).tags.title for r in range(page.model.rowCount())]
    assert rows == ["The Light", "TEMPO", "Waterloo"]  # best match first
    assert list(page.model.COLUMNS)[:2] == ["Track", "Match"]
    matches = [page.model.index(r, 1).data() for r in range(3)]
    assert matches[0] == "100%" and all(m.endswith("%") for m in matches)
    assert page.model.index(1, 1).data(Qt.ItemDataRole.TextAlignmentRole) & Qt.AlignmentFlag.AlignRight
    assert page.summary.text() == "Like Metronomy – The Light  ·  2 tracks  ·  discogs-effnet"
    assert page.play_button.isEnabled()
    assert not page.grab().isNull()

    requests = []
    page.play_requested.connect(lambda tracks, row, root: requests.append((tracks, row, root)))
    page.play_button.click()
    page.table.doubleClicked.emit(page.model.index(2, 0))
    assert [(len(t), row, root) for t, row, root in requests] == [(3, 0, catalog.root), (3, 2, catalog.root)]


def test_similar_follows_the_settings(page, store):
    save_embeddings(store, VECTORS)
    store.settings_path.write_text("similar:\n  count: 1\n")
    catalog = store.load()
    assert page.find(store, catalog, track(catalog, "The Light"))
    assert page.model.rowCount() == 2  # the chosen track and one more


def test_without_embeddings(page, store):
    catalog = store.load()
    assert not page.find(store, catalog, track(catalog, "The Light"))
    assert page.notices == ["No discogs-effnet embeddings yet"]
    assert page.model.rowCount() == 0


def test_seed_without_embedding(page, store):
    save_embeddings(store, {"TEMPO", "Waterloo"})
    catalog = store.load()
    assert not page.find(store, catalog, track(catalog, "The Light"))
    assert page.notices == ["This track has no embedding yet"]


def test_library_asks_for_similar_tracks_from_the_selected_track(qapp, store, tmp_path):
    page = LibraryPage(QSettings(str(tmp_path / "crate.ini"), QSettings.Format.IniFormat))
    page.open_catalog(store.folder)
    requests = []
    page.similar_requested.connect(requests.append)
    assert not page.similar_button.isEnabled()  # nothing selected

    page.table.selectRow(1)
    assert page.similar_button.isEnabled()
    page.similar_button.click()

    assert [t.tags.title for t in requests] == ["The Light"]  # sorted by artist: ABBA, Metronomy, NASAYA
    page.shutdown()


def menu_texts(menu):
    return [action.text() for action in menu.actions()]


def test_right_click_menus_offer_similar_tracks_and_the_map(qapp, page, store, tmp_path):
    save_embeddings(store, VECTORS)
    catalog = store.load()
    page.find(store, catalog, track(catalog, "The Light"))
    library = LibraryPage(QSettings(str(tmp_path / "crate.ini"), QSettings.Format.IniFormat))
    library.open_catalog(store.folder)

    for source, index in ((page, page.model.index(1, 0)), (library, library.proxy.index(1, 0))):
        similar, shown, radio = [], [], []
        source.similar_requested.connect(similar.append)
        source.map_requested.connect(shown.append)
        source.radio_requested.connect(lambda seeds, name: radio.append(([t.tags.title for t in seeds], name)))
        menu = source.track_menu(index)
        assert menu_texts(menu) == ["Play", "Similar tracks", "Show in map", "Start radio"]
        for action in menu.actions()[1:]:
            action.trigger()
        title = index.data(TRACK_ROLE).tags.title
        assert [t.tags.title for t in similar] == [title] and [t.tags.title for t in shown] == [title]
        assert radio == [([title], f"{index.data(TRACK_ROLE).tags.artists[0]} – {title}")]
    library.shutdown()
