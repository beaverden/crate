from pathlib import Path

from libcrate import Catalog, MetadataStore, Radio, SettingsError, TrackRecord
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout
from qfluentwidgets import CaptionLabel, FluentIcon, InfoBar, PushButton, Slider

from crate.library.covers import CoverCache
from crate.library.feature import Unavailable, slot_embeddings
from crate.library.match_page import MatchPage
from crate.library.model import TRACK_ROLE
from crate.library.track_menu import track_title
from crate.player import Player

__all__ = ["RadioPage", "write_playlist"]

EMPTY = ("Start a station: right-click a track and choose Start radio, or describe what you'd like to hear "
         "on the Describe page and press Start radio.")
# How many tracks the player has queued after the current one.
AHEAD = 3
# A track left before this many ms, or before this fraction of it, counts as skipped.
SKIP_MS = 30_000
SKIP_FRACTION = 0.4


def write_playlist(path: Path, tracks: list[TrackRecord], root: str | Path) -> None:
    """An .m3u playlist of `tracks` (paths relative to `root`, the library), with their lengths and titles."""
    lines = ["#EXTM3U"]
    for track in tracks:
        lines.append(f"#EXTINF:{round(track.audio.duration or 0)},{track_title(track)}")
        lines.append(str(Path(root) / track.path))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class RadioPage(MatchPage):
    """A radio station (libcrate.Radio, the radio section of the settings), played through the player.

    start(store, catalog, seeds, name) starts one from tracks: a track, or the best matches for a description.
    The player always has AHEAD tracks queued after the current one, picked as it moves on. A track played to
    its end pulls the station towards it; one skipped early (next before SKIP_MS or SKIP_FRACTION of it) steers
    it away from tracks like it, and the tracks queued after the current one are picked again. The drift slider
    changes how far the station wanders, for the tracks picked from then on.

    The list shows the station: what played (skipped tracks faded), the current track and the next ones, each
    with its match to the station. A double-click goes on from that track. The station ends with Stop, or when
    something else is played. Save as playlist writes the tracks played through to an .m3u file.
    """

    def __init__(self, covers: CoverCache, player: Player, parent=None):
        super().__init__("radio", "Radio", EMPTY, covers, parent)
        self.player = player
        self.radio: Radio | None = None
        self.name = ""
        self.played: list[TrackRecord] = []  # in order, skipped ones included
        self.skipped: set[str] = set()
        self._queue: list | None = None  # the player's queue while it's the station's
        self._replan = False  # pick the queued tracks again when the player moves on
        self._starting = False  # handing the player a new queue

        self.play_button.hide()  # the station plays as it's started
        self.drift = Slider(Qt.Orientation.Horizontal)
        self.drift.setRange(0, 100)
        self.drift.setFixedWidth(140)
        self.drift.setToolTip("How far the station wanders from where it started")
        self.drift.valueChanged.connect(self._on_drift)
        self.stop_button = PushButton(FluentIcon.CLOSE, "Stop")
        self.stop_button.clicked.connect(self.stop)
        self.save_button = PushButton(FluentIcon.SAVE, "Save as playlist")
        self.save_button.setToolTip("The tracks played through, as an .m3u playlist")
        self.save_button.clicked.connect(self._choose_playlist)
        drift = QHBoxLayout()
        drift.setSpacing(6)
        drift.addWidget(CaptionLabel("Close"))
        drift.addWidget(self.drift)
        drift.addWidget(CaptionLabel("Wander"))
        for n, item in enumerate((drift, self.stop_button, self.save_button), start=1):
            if isinstance(item, QHBoxLayout):
                self.header.insertLayout(n, item)
            else:
                self.header.insertWidget(n, item)

        player.track_changed.connect(self._on_track)
        player.skipped.connect(self._on_skipped)
        player.finished.connect(self._on_finished)
        self._update_buttons()

    @property
    def on_air(self) -> bool:
        """A station is playing: the player's queue is still the one it gave it."""
        return self.radio is not None and self.player.queue is self._queue

    def start(self, store: MetadataStore, catalog: Catalog, seeds: list[TrackRecord], name: str) -> bool:
        """Start a station from `seeds` (the first one with an embedding plays first) and play it; False (with a
        notification) when it can't: unreadable settings, no embeddings yet, or none for the seeds."""
        try:
            settings = store.settings()
        except SettingsError as e:
            self._notify(InfoBar.error, "Can't read the settings", str(e))
            return False
        try:
            _, embeddings = slot_embeddings(store, catalog, settings, "radio")
        except Unavailable as e:
            self._notify(InfoBar.warning, e.title, e.message)
            return False
        except OSError as e:
            self._notify(InfoBar.error, "Can't read the embeddings", str(e))
            return False
        try:
            radio = Radio(catalog, embeddings, [t.path for t in seeds], **settings.radio.options())
        except KeyError:
            self._notify(InfoBar.warning, "No embedding yet for " + ("this track" if len(seeds) == 1 else "these tracks"),
                         "Press Enrich in the library to make them.")
            return False
        self.radio, self.name, self.root = radio, name, catalog.root
        self.played, self.skipped, self._replan = [], set(), False
        self.table.itemDelegate().dimmed = self.skipped
        self.drift.blockSignals(True)
        self.drift.setValue(round(settings.radio.drift * 100))
        self.drift.blockSignals(False)
        for _ in range(AHEAD):
            radio.next()
        self.player.set_shuffle(False)  # the station's order is the point
        self._play_from(0)
        return True

    def stop(self) -> None:
        """End the station and the playback."""
        if self.on_air:
            self.player.stop()
        self._end()

    def save_playlist(self, path: Path) -> None:
        write_playlist(path, [t for t in self.played if t.path not in self.skipped], self.root)

    def _play_from(self, row: int) -> None:
        """Play the station from its track at `row`, picking what comes after it again."""
        tracks = [item.track for item in self.radio.queue[:row + 1]]
        self.radio.drop_after(tracks[-1].path)
        self._starting = True
        try:
            self.player.play_queue(tracks, row, self.root)
        finally:
            self._starting = False
        if self.player.track is None:  # couldn't play it (the player says why)
            self._end()
            return
        self._queue = self.player.queue
        self._top_up(self.player.track)

    def _play(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        if self.radio is None:
            super()._play(index)  # the station has ended: the list plays like any other
        else:
            self._play_from(index.row())

    def _on_track(self, track: TrackRecord | None) -> None:
        self.set_playing(track)
        if self.radio is None or self._starting:
            return
        if not self.on_air or track is None:
            self._end()
            return
        self._top_up(track)

    def _top_up(self, track: TrackRecord | None) -> None:
        """Queue AHEAD tracks after `track` (the current one), after picking them again if asked to."""
        paths = [item.track.path for item in self.radio.queue]
        if track is None or track.path not in paths:
            return
        if not self.played or self.played[-1].path != track.path:
            self.played.append(track)
        at = paths.index(track.path)
        if self._replan:
            self._replan = False
            self.radio.drop_after(track.path)
        while len(self.radio.queue) - 1 - at < AHEAD and self.radio.next() is not None:
            pass
        self.player.set_upcoming([item.track for item in self.radio.queue[at + 1:]])
        self._queue = self.player.queue
        self._show()

    def _on_skipped(self, track: TrackRecord, elapsed: int, length: int) -> None:
        if self.on_air and (elapsed < SKIP_MS or (length and elapsed < SKIP_FRACTION * length)):
            self.radio.skipped(track.path)
            self.skipped.add(track.path)
            self._replan = True

    def _on_finished(self, track: TrackRecord) -> None:
        if self.on_air:
            self.radio.listened(track.path)

    def _on_drift(self, value: int) -> None:
        if self.on_air:
            self.radio.drift = value / 100
            self._replan = True
            self._top_up(self.player.track)  # the queued tracks again, with the new drift

    def _end(self) -> None:
        if self.radio is None:
            return
        self.radio, self._queue = None, None
        self.summary.setText(f"{self.name}  ·  ended  ·  {self._counts()}")
        self._update_buttons()

    def _counts(self) -> str:
        played = sum(t.path not in self.skipped for t in self.played)
        return f"{played} played" + (f", {len(self.skipped)} skipped" if self.skipped else "")

    def _show(self) -> None:
        self.show_matches(self.radio.queue, self.root, f"{self.name}  ·  {self._counts()}")
        if (row := self._row_of(self.player.track)) is not None:
            self.table.scrollTo(self.model.index(row, 0))
        self._update_buttons()

    def _row_of(self, track: TrackRecord | None) -> int | None:
        for row in range(self.model.rowCount()):
            if track is not None and self.model.index(row, 0).data(TRACK_ROLE).path == track.path:
                return row
        return None

    def _choose_playlist(self) -> None:
        suggested = Path.home() / f"{''.join(c for c in self.name if c not in '/\\:*?\"<>|“”').strip()}.m3u"
        path, _ = QFileDialog.getSaveFileName(self, "Save as playlist", str(suggested), "Playlists (*.m3u)")
        if path:
            self.save_playlist(Path(path))

    def _update_buttons(self) -> None:
        self.stop_button.setEnabled(self.radio is not None)
        self.drift.setEnabled(self.radio is not None)
        self.save_button.setEnabled(any(t.path not in self.skipped for t in self.played))
