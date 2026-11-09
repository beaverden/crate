import numpy as np
import pytest
from libcrate import Embedding, EmbeddingTable, MetadataStore, make_fetcher
from PySide6.QtCore import QCoreApplication

from crate.map import MapPage
from crate.map.view import DIMMED_OPACITY

TRACKS = 12  # tracks 0-5 sound alike, and so do 6-11

# What discogs-effnet makes with the default settings: a feature uses only rows made so.
KEY = make_fetcher("discogs-effnet").key(None)


@pytest.fixture
def store(make_library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    store.scan(make_library(TRACKS))
    return store


def save_embeddings(store, count=TRACKS):
    rng = np.random.default_rng(0)
    table = EmbeddingTable()
    for i, path in enumerate(sorted(store.load().tracks)[:count]):
        centre = np.eye(8)[0 if i < TRACKS // 2 else 1] * 10
        table.put(path, KEY, Embedding((centre + rng.normal(size=8)).astype(np.float32)))
    path = store.embeddings_path("discogs-effnet")
    path.parent.mkdir(parents=True)
    table.save(path)
    store.settings_path.write_text("map:\n  clusters: 2\n")


def refresh(page, store, catalog=None):
    page.refresh(store, catalog or store.load())
    while page.task is not None:
        page.task.wait()
        QCoreApplication.processEvents()


@pytest.fixture
def page(qapp):
    return MapPage()


def test_map_of_the_catalog(page, store):
    save_embeddings(store)
    refresh(page, store)

    assert len(page.tracks) == TRACKS and len(page.view.dots) == TRACKS
    assert page.summary.text() == f"{TRACKS} tracks  ·  2 clusters  ·  discogs-effnet"
    by_title = {t.tags.title: c for t, c in zip(page.tracks, page.clusters)}
    assert len({by_title[f"Title {i:03d}"] for i in range(6)}) == 1
    assert by_title["Title 000"] != by_title["Title 011"]
    assert (store.folder / "embeddings" / "discogs-effnet" / "map.parquet").is_file()

    # Opening the page again keeps what's drawn: same map, same catalog.
    dots = page.view.dots
    refresh(page, store, page.catalog)
    assert page.view.dots is dots


def test_select_and_play_its_cluster(page, store):
    save_embeddings(store)
    refresh(page, store)
    assert not page.play_button.isEnabled() and not page.similar_button.isEnabled()
    played, similar = [], []
    page.play_requested.connect(lambda tracks, row, root: played.append((tracks, row, root)))
    page.similar_requested.connect(similar.append)

    index = next(i for i, t in enumerate(page.tracks) if t.tags.title == "Title 003")
    page.view.selected.emit(index)
    assert page.play_button.isEnabled() and page.view.ring is not None
    assert page.track_label.text().startswith("Artist 003 – Title 003   ·   cluster ")
    assert page.track_label.text().endswith(", 6 tracks")

    page.play_button.click()
    tracks, row, root = played[0]
    assert row == 0 and tracks[0].tags.title == "Title 003" and root == page.catalog.root
    assert sorted(t.tags.title for t in tracks) == [f"Title {i:03d}" for i in range(6)]
    page.similar_button.click()
    assert similar[0].tags.title == "Title 003"

    page.view.selected.emit(-1)  # empty space
    assert not page.play_button.isEnabled() and page.view.ring is None


def test_search_fades_the_others_and_return_selects(page, store):
    save_embeddings(store)
    refresh(page, store)
    page.search.setText("title 007")
    faded = [dot.opacity() == DIMMED_OPACITY for dot in page.view.dots]
    assert faded.count(False) == 1 and page.tracks[faded.index(False)].tags.title == "Title 007"
    page.search.returnPressed.emit()
    assert page.tracks[page.current].tags.title == "Title 007"
    page.search.clear()
    assert all(dot.opacity() == 1 for dot in page.view.dots)


def test_show_track_selects_it_once_the_map_is_made(page, store, monkeypatch):
    save_embeddings(store, TRACKS - 1)  # the last track has no embedding
    catalog = store.load()
    by_title = {t.tags.title: t for t in catalog.tracks.values()}
    warnings = []
    monkeypatch.setattr("crate.map.page.InfoBar.warning", lambda title, *a, **k: warnings.append(title))

    page.refresh(store, catalog)  # being made: shown when it's done
    page.show_track(by_title["Title 004"])
    refresh(page, store, catalog)
    assert page.tracks[page.current].tags.title == "Title 004"

    page.show_track(by_title["Title 002"])  # made already: at once
    assert page.tracks[page.current].tags.title == "Title 002" and warnings == []
    page.show_track(by_title[f"Title {TRACKS - 1:03d}"])
    assert warnings == ["This track isn't on the map"] and page.tracks[page.current].tags.title == "Title 002"


def test_without_embeddings(page, store):
    refresh(page, store)
    assert page.view.dots == [] and page.summary.text() == (
        "No map: no discogs-effnet embeddings yet. Turn on discogs-effnet in Settings › Fetchers, then press Enrich "
        "in the library to make them.")

    page.refresh(None, None)
    assert page.summary.text() == "Open a catalog to see its map."


def test_the_view_shows_every_track(page, store):
    save_embeddings(store)
    page.resize(900, 700)
    page.show()
    refresh(page, store)
    view = page.view
    inside = [view.viewport().rect().contains(view.mapFromScene(dot.pos())) for dot in view.dots]
    assert all(inside) and view.transform().m11() != 1
    page.hide()
