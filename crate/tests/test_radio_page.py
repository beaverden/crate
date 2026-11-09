import numpy as np
import pytest
from libcrate import Embedding, EmbeddingTable, MetadataStore, make_fetcher
from PySide6.QtCore import QObject, Signal

from crate.library.covers import CoverCache
from crate.library.model import TRACK_ROLE
from crate.radio import RadioPage
from crate.radio.page import AHEAD, SKIP_MS

TRACKS = 16  # tracks 0-7 sound alike, and so do 8-15

# What discogs-effnet makes with the default settings: a feature uses only rows made so.
KEY = make_fetcher("discogs-effnet").key(None)


class FakePlayer(QObject):
    """The Player as the radio page uses it, without libvlc: next() skips at `elapsed`, end() plays to the end."""

    track_changed = Signal(object)
    skipped = Signal(object, int, int)
    finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.queue, self.position, self.root, self.shuffle = [], -1, None, True

    @property
    def track(self):
        return self.queue[self.position] if 0 <= self.position < len(self.queue) else None

    def play_queue(self, tracks, index, root):
        self.queue, self.position, self.root = list(tracks), index, root
        self.track_changed.emit(self.track)

    def set_upcoming(self, tracks):
        self.queue = self.queue[:self.position + 1] + list(tracks)

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def next(self, elapsed=0):
        self.skipped.emit(self.track, elapsed, 200_000)
        self._advance()

    def end(self):
        self.finished.emit(self.track)
        self._advance()

    def stop(self):
        self.position = -1
        self.track_changed.emit(None)

    def _advance(self):
        if self.position + 1 < len(self.queue):
            self.position += 1
            self.track_changed.emit(self.track)
        else:
            self.stop()


@pytest.fixture
def store(make_library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    store.scan(make_library(TRACKS))
    return store


def save_embeddings(store):
    rng = np.random.default_rng(0)
    table = EmbeddingTable()
    for i, path in enumerate(sorted(store.load().tracks)):
        centre = np.eye(8)[0 if i < TRACKS // 2 else 1] * 10
        table.put(path, KEY, Embedding((centre + rng.normal(size=8)).astype(np.float32)))
    path = store.embeddings_path("discogs-effnet")
    path.parent.mkdir(parents=True)
    table.save(path)
    store.settings_path.write_text("radio:\n  drift: 0.2\n  choices: 2\n  duplicate_similarity: 1.1\n")


def by_title(catalog, title):
    return next(t for t in catalog.tracks.values() if t.tags.title == title)


@pytest.fixture
def player(qapp):
    return FakePlayer()


@pytest.fixture
def page(qapp, player):
    page = RadioPage(CoverCache(40, 4), player)
    notices = page.notices = []
    page._notify = lambda kind, title, content, duration=0: notices.append(title)
    return page


def start(page, store, title="Title 002"):
    catalog = store.load()
    assert page.start(store, catalog, [by_title(catalog, title)], f"Artist – {title}")
    return catalog


def listed(page):
    return [page.model.index(r, 0).data(TRACK_ROLE).path for r in range(page.model.rowCount())]


def test_a_station_plays_and_keeps_tracks_queued(page, player, store):
    save_embeddings(store)
    start(page, store)
    assert page.on_air and not player.shuffle
    assert player.track.tags.title == "Title 002" and len(player.queue) == 1 + AHEAD
    assert listed(page) == [t.path for t in player.queue]
    assert page.drift.value() == 20 and page.stop_button.isEnabled()
    assert page.summary.text() == "Artist – Title 002  ·  1 played"

    for _ in range(3):
        player.end()
    assert len(player.queue) == 4 + AHEAD and len({t.path for t in player.queue}) == len(player.queue)
    assert all(int(t.tags.title[-3:]) < TRACKS // 2 for t in player.queue)  # the seed's scene
    assert listed(page) == [t.path for t in player.queue]
    assert page.summary.text().endswith("4 played")


def test_an_early_skip_picks_the_next_tracks_again(page, player, store, monkeypatch):
    save_embeddings(store)
    start(page, store)
    skipped = player.queue[1]
    player.end()
    picked = []
    monkeypatch.setattr(page.radio, "drop_after", lambda path, drop=page.radio.drop_after: (picked.append(path), drop(path)))
    player.next(elapsed=SKIP_MS - 1)
    assert skipped.path in page.skipped and page.table.itemDelegate().dimmed is page.skipped
    assert picked == [player.track.path] and len(player.queue) == 3 + AHEAD
    assert page.summary.text().endswith("2 played, 1 skipped")

    player.next(elapsed=100_000)  # half of it: not a skip
    assert len(page.skipped) == 1 and len(picked) == 1


def test_double_click_goes_on_from_that_track(page, player, store):
    save_embeddings(store)
    start(page, store)
    third = player.queue[2]
    page.table.doubleClicked.emit(page.model.index(2, 0))
    assert player.track is third and len(player.queue) == 3 + AHEAD and page.on_air


def test_the_station_ends_when_something_else_plays_or_stop(page, player, store):
    save_embeddings(store)
    catalog = start(page, store)
    player.play_queue([by_title(catalog, "Title 010")], 0, catalog.root)
    assert not page.on_air and page.radio is None and page.summary.text().endswith("ended  ·  1 played")
    assert not page.stop_button.isEnabled()
    played = []
    page.play_requested.connect(lambda tracks, row, root: played.append(row))
    page.table.doubleClicked.emit(page.model.index(1, 0))  # an ended station's list plays like any other
    assert played == [1]

    start(page, store)
    page.stop_button.click()
    assert player.track is None and page.radio is None


def test_save_the_tracks_played_through(page, player, store, tmp_path):
    save_embeddings(store)
    catalog = start(page, store)
    assert page.save_button.isEnabled()  # the current track counts
    player.next(elapsed=0)
    player.end()
    kept = [t for t in page.played if t.path not in page.skipped]
    assert page.save_button.isEnabled() and len(kept) == 2

    path = tmp_path / "station.m3u"
    page.save_playlist(path)
    lines = path.read_text().splitlines()
    assert lines[0] == "#EXTM3U" and len(lines) == 1 + 2 * len(kept)
    assert lines[1].startswith("#EXTINF:") and lines[1].endswith(f",Artist {kept[0].tags.title[-3:]} – {kept[0].tags.title}")
    assert lines[2] == f"{catalog.root}/{kept[0].path}"


def test_without_embeddings(page, store):
    catalog = store.load()
    assert not page.start(store, catalog, [by_title(catalog, "Title 001")], "x")
    assert page.notices == ["No discogs-effnet embeddings yet"] and page.radio is None
