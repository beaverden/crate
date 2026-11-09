from PySide6.QtCore import QModelIndex, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem
from qfluentwidgets import FluentIcon, TableItemDelegate, getFont, isDarkTheme, themeColor

from crate.library.covers import CoverCache, thumbnail
from crate.library.model import TRACK_COLUMN, TRACK_ROLE

__all__ = ["COVER", "COVER_RADIUS", "ROW_HEIGHT", "TrackDelegate"]

COVER = 40
COVER_RADIUS = 4
ROW_HEIGHT = 56
# Left of the cover: room for the selected row's indicator.
INSET = 16
GAP = 12
# How opaque a dimmed track's cover and title are.
DIMMED_OPACITY = 0.4


class TrackDelegate(TableItemDelegate):
    """The Fluent table's rows, with the track column drawn like Spotify's: the cover, then the title
    with the artists underneath in a muted colour.

    It runs for every visible cell on every frame while scrolling, so fonts, metrics and the
    placeholder cover are made once rather than per cell.
    """

    def __init__(self, table, covers: CoverCache):
        super().__init__(table)
        self.covers = covers
        self.font = getFont(13)
        self.title_metrics = QFontMetrics(self.font)
        self.artist_font = getFont(12)
        self.artist_metrics = QFontMetrics(self.artist_font)
        self._placeholders: dict[bool, QPixmap] = {}  # per theme: dark?
        # Path of the track being played, whose title is drawn in the theme colour.
        self.playing: str | None = None
        # Paths of tracks drawn faded, e.g. the ones skipped on the radio.
        self.dimmed: set[str] = set()

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        # What TableItemDelegate.initStyleOption does, without building a font for every cell.
        QStyledItemDelegate.initStyleOption(self, option, index)
        option.font = self.font
        text = QColor(255, 255, 255) if isDarkTheme() else QColor(0, 0, 0)
        option.palette.setColor(QPalette.ColorRole.Text, text)
        option.palette.setColor(QPalette.ColorRole.HighlightedText, text)
        if index.column() == TRACK_COLUMN:
            option.text = ""  # drawn in paint()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().paint(painter, option, index)  # row background, hover, selection
        if index.column() != TRACK_COLUMN:
            return
        track = index.data(TRACK_ROLE)
        rect = option.rect
        painter.save()
        if track.path in self.dimmed:
            painter.setOpacity(DIMMED_OPACITY)
        cover_x, cover_y = rect.x() + INSET, rect.center().y() - COVER // 2 + 1
        painter.drawPixmap(cover_x, cover_y, self.covers.get(track) or self._placeholder())

        text_x = cover_x + COVER + GAP
        width = rect.right() - text_x - 8
        dark = isDarkTheme()
        painter.setFont(self.font)
        if track.path == self.playing:
            painter.setPen(themeColor())
        else:
            painter.setPen(QColor(255, 255, 255) if dark else QColor(0, 0, 0))
        title = self.title_metrics.elidedText(index.data(), Qt.TextElideMode.ElideRight, width)
        painter.drawText(QRect(text_x, rect.y(), width, rect.height() // 2), Qt.AlignmentFlag.AlignBottom, title)

        painter.setFont(self.artist_font)
        painter.setPen(QColor(160, 160, 160) if dark else QColor(100, 100, 100))
        artists = self.artist_metrics.elidedText(", ".join(track.tags.artists), Qt.TextElideMode.ElideRight, width)
        painter.drawText(QRect(text_x, rect.center().y() + 3, width, rect.height() // 2), Qt.AlignmentFlag.AlignTop,
                         artists)
        painter.restore()

    def _placeholder(self) -> QPixmap:
        """A music note on a faint square, for tracks without a cover (or while it loads); rendered once per theme."""
        dark = isDarkTheme()
        if dark not in self._placeholders:
            image = QImage(COVER * 2, COVER * 2, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(QColor(255, 255, 255, 18) if dark else QColor(0, 0, 0, 12))
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            FluentIcon.MUSIC.render(painter, QRectF(24, 24, COVER * 2 - 48, COVER * 2 - 48))
            painter.end()
            self._placeholders[dark] = QPixmap.fromImage(thumbnail(image, COVER, COVER_RADIUS))
        return self._placeholders[dark]
