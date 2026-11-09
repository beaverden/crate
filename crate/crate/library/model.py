from libcrate import Catalog, TrackMatch, TrackRecord
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt

__all__ = ["RIGHT_ALIGNED", "SEARCH_ROLE", "SORT_ROLE", "TRACK_COLUMN", "TRACK_ROLE", "MatchTableModel", "TrackTableModel"]

# What a column sorts by (numbers as numbers, text case-insensitively), the row's TrackRecord,
# and the text the search box matches.
SORT_ROLE = Qt.ItemDataRole.UserRole
TRACK_ROLE = Qt.ItemDataRole.UserRole + 1
SEARCH_ROLE = Qt.ItemDataRole.UserRole + 2
# Numbers line up on the right.
RIGHT_ALIGNED = {"Match", "Year", "Length"}
# Drawn by TrackDelegate: cover, title, and artists underneath.
TRACK_COLUMN = 0


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def _artists(track: TrackRecord) -> str:
    return ", ".join(track.tags.artists)


def _title(track: TrackRecord) -> str:
    return track.tags.title or track.path


class TrackTableModel(QAbstractTableModel):
    """One row per track of a Catalog (or of a given list), in that order; sort and filter through a
    QSortFilterProxyModel.

    Each row's texts and sort values are worked out once, in set_catalog: the view asks for them
    many times per cell on every repaint, so scrolling stays smooth.
    """

    # header -> (displayed text, sort value)
    COLUMNS = {
        # Sorted by artist, like the library's file names.
        "Track": (_title, lambda t: (_artists(t).casefold(), _title(t).casefold())),
        "Album": (lambda t: t.tags.album or "", lambda t: (t.tags.album or "").casefold()),
        "Year": (lambda t: str(t.tags.year or ""), lambda t: t.tags.year or 0),
        "Length": (lambda t: _duration(t.audio.duration), lambda t: t.audio.duration),
        "Genres": (lambda t: ", ".join(t.genre_names()), lambda t: ", ".join(t.genre_names()).casefold()),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: list[TrackRecord] = []
        # Per row: the columns' displayed texts, sort values and searched texts.
        self._display: list[list[str]] = []
        self._sort: list[list] = []
        self._search: list[list[str]] = []
        self._align_columns()

    def _align_columns(self) -> None:
        """Each column's alignment, from its name; again after a subclass changes COLUMNS."""
        self._alignment = [Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter if name in RIGHT_ALIGNED
                           else None for name in self.COLUMNS]

    def set_catalog(self, catalog: Catalog | None) -> None:
        self.set_tracks(list(catalog.tracks.values()) if catalog else [])

    def set_tracks(self, tracks: list[TrackRecord]) -> None:
        """Show `tracks`, in this order."""
        self.beginResetModel()
        self._tracks = list(tracks)
        self._display = [[display(t) for display, _ in self.COLUMNS.values()] for t in self._tracks]
        self._sort = [[sort(t) for _, sort in self.COLUMNS.values()] for t in self._tracks]
        self._search = [[f"{row[TRACK_COLUMN]} {_artists(t)}", *row[1:]] for t, row in zip(self._tracks, self._display)]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._tracks)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index: QModelIndex | QPersistentModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display[index.row()][index.column()]
        if role == SORT_ROLE:
            return self._sort[index.row()][index.column()]
        if role == TRACK_ROLE:
            return self._tracks[index.row()]
        if role == SEARCH_ROLE:
            return self._search[index.row()][index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return self._alignment[index.column()]
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return list(self.COLUMNS)[section]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return self._alignment[section]
        return None


class MatchTableModel(TrackTableModel):
    """The library's columns, plus how well each track matches what was asked (Match, after Track): a chosen
    track for similar tracks, a description for Describe."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.matches: dict[str, float] = {}  # path -> similarity
        match = (lambda t: f"{self.matches[t.path]:.0%}", lambda t: self.matches[t.path])
        columns = list(TrackTableModel.COLUMNS.items())
        self.COLUMNS = dict([columns[0], ("Match", match), *columns[1:]])
        self._align_columns()

    def set_matches(self, items: list[TrackMatch]) -> None:
        self.matches = {item.track.path: item.similarity for item in items}
        self.set_tracks([item.track for item in items])
