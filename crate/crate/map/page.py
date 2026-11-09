import numpy as np
from libcrate import Catalog, MetadataStore, MusicMap, TrackRecord
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, FluentIcon, InfoBar, InfoBarPosition, PrimaryPushButton, PushButton,
    SearchLineEdit, TitleLabel,
)

from crate.icons import Icon
from crate.library.track_menu import track_title
from crate.map.task import MapTask
from crate.map.view import MapView, cluster_colors

__all__ = ["MapPage"]

EMPTY = "Open a catalog to see its map."
HINT = "Click a track to select it, double-click to play it; scroll to zoom, drag to move, double-click empty space to see everything."


class MapPage(QWidget):
    """The catalog's tracks laid out by similarity (libcrate.MusicMap, the map section of the settings),
    a colour per cluster. refresh(store, catalog) brings it up to date, in the background: quick when the
    embeddings haven't changed since the map was made, several seconds when they have.

    A selected track can be played (with the rest of its cluster after it, nearest first), or lead to similar
    tracks, or start a radio station: play_requested(tracks, row, root), similar_requested(track) and
    radio_requested([track], name), as the library page does.
    The search box fades the tracks whose artist, title or album don't match. show_track(track) selects a
    track and brings it into view, once the map is made.
    """

    play_requested = Signal(list, int, str)
    similar_requested = Signal(object)
    radio_requested = Signal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("map")
        self.store: MetadataStore | None = None
        self.catalog: Catalog | None = None
        self.music_map: MusicMap | None = None
        self.tracks: list[TrackRecord] = []  # the dots' tracks, in the view's order
        self.xy = np.empty((0, 2), dtype=np.float32)
        self.clusters = np.empty(0, dtype=np.int32)
        self.current = -1  # the selected dot
        self.task: MapTask | None = None
        self._stale = False  # refresh() was called while a task ran
        self._wanted: str | None = None  # show_track()'s track, shown when the map being made is done
        # What's drawn: the map's fingerprint and the catalog its tracks come from.
        self._shown_fingerprint: str | None = None
        self._shown_catalog: Catalog | None = None

        self.title = TitleLabel("Map")
        self.summary = CaptionLabel(EMPTY)
        self.summary.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        self.search = SearchLineEdit()
        self.search.setPlaceholderText("Find tracks")
        self.search.setFixedWidth(260)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._select_first_match)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.title)
        titles.addWidget(self.summary)
        header = QHBoxLayout()
        header.addLayout(titles)
        header.addStretch(1)
        header.addWidget(self.search)

        self.view = MapView()
        self.view.selected.connect(self._select)
        self.view.activated.connect(self._play_from)
        card = CardWidget()
        frame = QVBoxLayout(card)
        frame.setContentsMargins(4, 4, 4, 4)
        frame.addWidget(self.view)

        self.track_label = BodyLabel(HINT)
        self.track_label.setWordWrap(True)
        self.play_button = PrimaryPushButton(FluentIcon.PLAY, "Play cluster")
        self.play_button.setToolTip("Play the track, then the rest of its cluster, nearest first")
        self.play_button.clicked.connect(lambda: self._play_from(self.current))
        self.similar_button = PushButton(Icon.SIMILAR, "Similar")
        self.similar_button.setToolTip("Similar-sounding tracks, from this one")
        self.similar_button.clicked.connect(lambda: self.similar_requested.emit(self.tracks[self.current]))
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self.track_label, 1)
        footer.addWidget(self.similar_button)
        self.radio_button = PushButton(Icon.RADIO, "Radio")
        self.radio_button.setToolTip("An endless station, from this track")
        self.radio_button.clicked.connect(
            lambda: self.radio_requested.emit([self.tracks[self.current]], track_title(self.tracks[self.current])))
        footer.addWidget(self.radio_button)
        footer.addWidget(self.play_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(card, 1)
        layout.addLayout(footer)
        self._update_actions()

    def refresh(self, store: MetadataStore | None, catalog: Catalog | None) -> None:
        """Show `catalog`'s map, made in the background unless the saved one is up to date."""
        self.store, self.catalog = store, catalog
        if store is None or catalog is None:
            self._show(None)
            self.summary.setText(EMPTY)
            return
        if self.task is not None:
            self._stale = True  # again when this one ends: the catalog or its settings may have changed
            return
        if self.music_map is None:
            self.summary.setText("Making the map…")
        self.task = MapTask(store, catalog, self)
        self.task.finished_ok.connect(self._on_done)
        self.task.failed.connect(self._on_failed)
        self.task.finished.connect(self._on_task_finished)
        self.task.start()

    def _on_done(self, music_map: MusicMap, model: str) -> None:
        if self._shown_fingerprint != music_map.fingerprint or self._shown_catalog is not self.catalog:
            self._show(music_map)  # else keep the zoom and the selection
        if self._wanted is not None:
            self._show_wanted()
        clusters = int(self.clusters.max()) + 1 if len(self.clusters) else 0
        self.summary.setText(f"{len(self.tracks):,} tracks  ·  {clusters} clusters  ·  {model}")

    def _on_failed(self, message: str) -> None:
        self._wanted = None  # the summary says why there's no map
        self._show(None)
        self.summary.setText(f"No map: {message}")

    def _on_task_finished(self) -> None:
        self.task = None
        if self._stale:
            self._stale = False
            self.refresh(self.store, self.catalog)

    def _show(self, music_map: MusicMap | None) -> None:
        """Draw `music_map`'s tracks that are in the catalog, keeping the selected track selected."""
        selected = self.tracks[self.current].path if self.current >= 0 else None
        self.music_map = music_map
        self._shown_fingerprint = music_map.fingerprint if music_map else None
        self._shown_catalog = self.catalog
        keep = [i for i, path in enumerate(music_map.paths) if path in self.catalog.tracks] if music_map else []
        self.tracks = [self.catalog.tracks[music_map.paths[i]] for i in keep]
        self.xy = music_map.xy[keep] if music_map else np.empty((0, 2), dtype=np.float32)
        self.clusters = music_map.clusters[keep] if music_map else np.empty(0, dtype=np.int32)
        colors = cluster_colors(int(self.clusters.max()) + 1 if len(self.clusters) else 0)
        self.view.set_points(self.xy, [colors[c] for c in self.clusters], [track_title(t) for t in self.tracks])
        self.current = -1
        self._filter(self.search.text())
        if selected is not None:
            self._select(next((i for i, t in enumerate(self.tracks) if t.path == selected), -1))
        else:
            self._select(-1)

    def show_track(self, track: TrackRecord) -> None:
        """Select `track` and bring it into view: now, or when the map being made is done."""
        self._wanted = track.path
        if self.task is None:
            self._show_wanted()

    def _show_wanted(self) -> None:
        index = next((i for i, t in enumerate(self.tracks) if t.path == self._wanted), -1)
        self._wanted = None
        if index < 0:
            InfoBar.warning("This track isn't on the map", "It has no embedding for the map yet: press Enrich in "
                            "the library to make it.",
                            orient=Qt.Orientation.Vertical, duration=6000, position=InfoBarPosition.TOP_RIGHT,
                            parent=self.window())
            return
        self._select(index)
        self.view.center_on(index)

    def _select(self, index: int) -> None:
        self.current = index
        self.view.select(index)
        if index < 0:
            self.track_label.setText(HINT)
        else:
            size = int((self.clusters == self.clusters[index]).sum())
            self.track_label.setText(f"{track_title(self.tracks[index])}   ·   cluster {self.clusters[index] + 1}, "
                                     f"{size} tracks")
        self._update_actions()

    def _cluster_queue(self, index: int) -> list[int]:
        """The track at `index`, then the rest of its cluster, nearest on the map first."""
        members = np.flatnonzero(self.clusters == self.clusters[index])
        distances = np.linalg.norm(self.xy[members] - self.xy[index], axis=1)
        return [int(i) for i in members[np.argsort(distances, kind="stable")]]

    def _play_from(self, index: int) -> None:
        if index >= 0:
            self._select(index)
            queue = self._cluster_queue(index)
            self.play_requested.emit([self.tracks[i] for i in queue], 0, self.catalog.root)

    def _matches(self, text: str) -> np.ndarray:
        words = text.lower().split()
        haystacks = [" ".join([*t.tags.artists, t.tags.title or t.path, t.tags.album or ""]).lower()
                     for t in self.tracks]
        return np.array([all(w in h for w in words) for h in haystacks], dtype=bool)

    def _filter(self, text: str) -> None:
        self.view.set_dimmed(~self._matches(text) if text.strip() else np.zeros(len(self.tracks), dtype=bool))

    def _select_first_match(self) -> None:
        """Return in the search box: select the first track it matches and bring it into view."""
        if self.search.text().strip() and (matches := np.flatnonzero(self._matches(self.search.text()))).size:
            self._select(int(matches[0]))
            self.view.center_on(int(matches[0]))

    def _update_actions(self) -> None:
        self.play_button.setEnabled(self.current >= 0)
        self.similar_button.setEnabled(self.current >= 0)
        self.radio_button.setEnabled(self.current >= 0)

    def shutdown(self) -> None:
        """Wait for a map being made (it can't be stopped halfway)."""
        if self.task is not None:
            self.task.wait()
