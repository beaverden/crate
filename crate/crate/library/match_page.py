from PySide6.QtCore import QModelIndex, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel, FluentIcon, InfoBarPosition, PrimaryPushButton, RoundMenu, TableView, TitleLabel, qconfig,
)

from crate.library.covers import CoverCache
from crate.library.delegate import ROW_HEIGHT, TrackDelegate
from crate.library.model import RIGHT_ALIGNED, TRACK_ROLE, MatchTableModel
from crate.library.scrolling import calm_scrolling
from crate.library.track_menu import track_menu

__all__ = ["MatchPage"]

NOTICE_MS = 6000


class MatchPage(QWidget):
    """A page listing tracks with how well they match something (MatchTableModel): a title, a summary under it,
    Play, and the table. show_matches() fills it. Play, or a double-click on a track, asks for the list to be
    played: play_requested(tracks, row, root), as the library page does. A right-click on a track also offers
    similar tracks, its place on the map and a radio station from it (track_menu). Subclasses add how they're
    asked.
    """

    play_requested = Signal(list, int, str)
    similar_requested = Signal(object)
    map_requested = Signal(object)
    radio_requested = Signal(list, str)

    def __init__(self, name: str, title: str, empty: str, covers: CoverCache, parent=None):
        super().__init__(parent)
        self.setObjectName(name)
        self.root: str | None = None

        self.title = TitleLabel(title)
        self.summary = CaptionLabel(empty)
        self.summary.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        self.summary.setWordWrap(True)
        self.play_button = PrimaryPushButton(FluentIcon.PLAY, "Play")
        self.play_button.clicked.connect(lambda: self._play(self.model.index(0, 0)))

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.title)
        titles.addWidget(self.summary)
        self.header = QHBoxLayout()  # subclasses add their controls before Play
        self.header.setSpacing(8)
        self.header.addLayout(titles, 1)
        self.header.addWidget(self.play_button)

        self.model = MatchTableModel(self)
        self.table = TableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(TrackDelegate(self.table, covers))
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        calm_scrolling(self.table)
        covers.loaded.connect(self.table.viewport().update)
        qconfig.themeChanged.connect(lambda theme: self.table.viewport().update())
        self.table.setBorderVisible(True)
        self.table.setBorderRadius(8)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        for column, name in enumerate(self.model.COLUMNS):
            mode = QHeaderView.ResizeMode.ResizeToContents if name in RIGHT_ALIGNED else QHeaderView.ResizeMode.Stretch
            self.table.horizontalHeader().setSectionResizeMode(column, mode)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.doubleClicked.connect(self._play)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_menu)
        play = QShortcut(QKeySequence(Qt.Key.Key_Return), self.table, context=Qt.ShortcutContext.WidgetShortcut)
        play.activated.connect(lambda: self._play(self.table.currentIndex()))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(14)
        layout.addLayout(self.header)
        layout.addWidget(self.table, 1)
        self._update_actions()

    def show_matches(self, items, root: str, summary: str) -> None:
        """Show `items` (TrackMatch, best first) of the library at `root`."""
        self.root = root
        self.model.set_matches(items)
        self.table.scrollToTop()
        self.summary.setText(summary)
        self._update_actions()

    def set_playing(self, track) -> None:
        """Draw the playing track's title in the theme colour, if it's in the list."""
        self.table.itemDelegate().playing = track.path if track is not None else None
        self.table.viewport().update()

    def _play(self, index: QModelIndex) -> None:
        if index.isValid():
            tracks = [self.model.index(row, 0).data(TRACK_ROLE) for row in range(self.model.rowCount())]
            self.play_requested.emit(tracks, index.row(), self.root)

    def _show_menu(self, pos: QPoint) -> None:
        index = self.table.indexAt(pos)
        if index.isValid():
            self.table.selectRow(index.row())
            self.track_menu(index).exec(self.table.viewport().mapToGlobal(pos))

    def track_menu(self, index: QModelIndex) -> RoundMenu:
        return track_menu(self, index, self._play)

    def _update_actions(self) -> None:
        self.play_button.setEnabled(self.model.rowCount() > 0)

    def _notify(self, kind, title: str, content: str, duration: int = NOTICE_MS) -> None:
        kind(title, content, orient=Qt.Orientation.Vertical, duration=duration,
             position=InfoBarPosition.TOP_RIGHT, parent=self.window())
