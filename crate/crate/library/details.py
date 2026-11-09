from pathlib import Path

from libcrate import DescriptionSource, Genre, GenreSource, SourceStatus, TrackRecord
from PySide6.QtCore import QMargins, QSize, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, FlowLayout, ImageLabel, ScrollArea, StrongBodyLabel, SubtitleLabel, TitleLabel,
    getFont, isDarkTheme, themeColor,
)

from crate.library.covers import read_cover

__all__ = ["TrackDetails"]

COVER_SIZE = 280
STATUS_LABEL = {SourceStatus.FOUND: "found", SourceStatus.NOT_FOUND: "not found", SourceStatus.ERROR: "error"}


def _genre_label(genre: Genre) -> str:
    """'deep house 41%' (essentia confidence), 'pop · 5' (musicbrainz votes), or just the name."""
    if genre.confidence is not None:
        return f"{genre.name}  {genre.confidence:.0%}"
    if genre.votes is not None:
        return f"{genre.name}  ·  {genre.votes}"
    return genre.name


class GenreChip(QWidget):
    """A genre as a rounded chip tinted with the theme colour."""

    PADDING = QMargins(10, 3, 10, 3)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.text = text
        self.setFont(getFont(12))

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self.text), metrics.height()).grownBy(self.PADDING)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        tint = QColor(themeColor())
        tint.setAlphaF(0.28 if isDarkTheme() else 0.18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tint)
        radius = self.height() / 2
        painter.drawRoundedRect(self.rect(), radius, radius)
        painter.setPen(QColor(255, 255, 255) if isDarkTheme() else QColor(0, 0, 0))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text)


class TrackDetails(ScrollArea):
    """Everything the catalog knows about one track: cover, tags, its descriptions, genres per source, audio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("QScrollArea {background: transparent; border: none}")
        self.show_track(None)

    def text(self) -> str:
        """All the text shown, for tests."""
        page = self.widget()
        return "\n".join([label.text() for label in page.findChildren(QLabel)]
                         + [chip.text for chip in page.findChildren(GenreChip)])

    def show_track(self, track: TrackRecord | None, root: str | None = None) -> None:
        """Show `track`; `root` is the library folder, to read the cover from the file."""
        page = QWidget()
        page.setObjectName("trackDetails")
        page.setStyleSheet("#trackDetails {background: transparent}")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(6)
        if track is None:
            hint = BodyLabel("Select a track to see its details.")
            hint.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
            layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignCenter)
        else:
            self._fill(layout, track, root)
        layout.addStretch(1)
        self.setWidget(page)  # deletes the previous page

    def _fill(self, layout: QVBoxLayout, track: TrackRecord, root: str | None) -> None:
        tags, audio = track.tags, track.audio
        cover = read_cover(Path(root) / track.path) if root and track.cover else None
        if cover is not None:
            image = ImageLabel(cover)
            image.scaledToWidth(COVER_SIZE)
            image.setBorderRadius(8, 8, 8, 8)
            layout.addWidget(image, 0, Qt.AlignmentFlag.AlignHCenter)
            layout.addSpacing(12)

        title = TitleLabel(tags.title or track.path)
        title.setWordWrap(True)
        layout.addWidget(title)
        if tags.artists:
            artist = SubtitleLabel(", ".join(tags.artists))
            artist.setWordWrap(True)
            layout.addWidget(artist)
        album = " · ".join(str(part) for part in (tags.album, tags.year) if part)
        if album:
            album_label = BodyLabel(album)
            album_label.setWordWrap(True)
            album_label.setTextColor(QColor(96, 96, 96), QColor(170, 170, 170))
            layout.addWidget(album_label)

        if track.descriptions:
            layout.addSpacing(16)
            layout.addWidget(StrongBodyLabel("Description"))
            for name, result in track.descriptions.items():
                layout.addWidget(self._description(name, result))

        layout.addSpacing(16)
        layout.addWidget(StrongBodyLabel("Genres"))
        if not track.genres:
            layout.addWidget(CaptionLabel("None yet: run Enrich."))
        for name, result in track.genres.items():
            layout.addWidget(self._source(name, result))

        layout.addSpacing(16)
        layout.addWidget(StrongBodyLabel("Details"))
        rows = [
            ("Album artist", ", ".join(tags.album_artists)),
            ("Date", tags.date),
            ("Label", tags.label),
            ("ISRC", tags.isrc),
            ("Spotify", tags.spotify_track_id),
            ("BPM", f"{tags.bpm:g}" if tags.bpm else None),
            ("Audio", f"{audio.format.upper()} · {(audio.sample_rate or 0) / 1000:g} kHz · "
                      f"{audio.bits_per_sample or '?'} bit · {audio.channels or '?'} ch"),
            ("File", track.path),
        ]
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        for row, (name, value) in enumerate((n, v) for n, v in rows if v):
            key = CaptionLabel(name)
            key.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
            text = CaptionLabel(str(value))
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(key, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(text, row, 1)
        layout.addLayout(grid)

    def _description(self, name: str, result: DescriptionSource) -> QWidget:
        """One source's description: the text, then its keywords as chips, and where it came from."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 6, 0, 2)
        layout.setSpacing(6)
        described = result.description if result.status == SourceStatus.FOUND else None
        if described is not None:
            text = BodyLabel(described.text)
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(text)
            chips = QWidget()
            flow = FlowLayout(chips)
            flow.setContentsMargins(0, 0, 0, 0)
            flow.setHorizontalSpacing(6)
            flow.setVerticalSpacing(6)
            for keyword in described.keywords:
                flow.addWidget(GenreChip(keyword))
            layout.addWidget(chips)
            note = [name, described.language, "knew the track" if described.recognised else "from the facts"]
        else:
            note = [name, STATUS_LABEL[result.status], result.error]
        caption = CaptionLabel(" · ".join(part for part in note if part))
        caption.setWordWrap(True)
        caption.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        layout.addWidget(caption)
        return box

    def _source(self, name: str, result: GenreSource) -> QWidget:
        """One source's result: its name and status, then its genres as chips."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 6, 0, 2)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(BodyLabel(name))
        if result.status != SourceStatus.FOUND:
            caption = CaptionLabel(STATUS_LABEL[result.status])
            caption.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
            header.addWidget(caption)
        header.addStretch(1)
        layout.addLayout(header)

        if result.status == SourceStatus.FOUND:
            chips = QWidget()
            flow = FlowLayout(chips)
            flow.setContentsMargins(0, 0, 0, 0)
            flow.setHorizontalSpacing(6)
            flow.setVerticalSpacing(6)
            for genre in result.genres:
                flow.addWidget(GenreChip(_genre_label(genre)))
            layout.addWidget(chips)
        elif result.error:
            error = CaptionLabel(result.error)
            error.setWordWrap(True)
            error.setTextColor(QColor(140, 140, 140), QColor(130, 130, 130))
            layout.addWidget(error)
        return box
