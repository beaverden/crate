from libcrate import TrackRecord
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, FluentIcon, ImageLabel, Slider, TransparentToggleToolButton, TransparentToolButton,
    isDarkTheme, qconfig,
)

from crate.library.covers import CoverCache
from crate.library.delegate import COVER
from crate.player.icons import PlayerIcon
from crate.player.player import Player

__all__ = ["PlayerBar"]

HEIGHT = 76
# Shuffle, previous, play/pause and next: one size, so their glyphs line up.
BUTTON = 36
ICON = 16
# The sides (now playing, and the volume on the right) share what the controls leave.
SIDE_WIDTH = 320
VOLUME_WIDTH = 110


def _time(ms: int) -> str:
    minutes, seconds = divmod(round(ms / 1000), 60)  # rounded like the table's Length
    return f"{minutes}:{seconds:02d}"


class SeekSlider(Slider):
    """The seek bar: emits seek_requested(position) once the user lets go, whether they dragged the
    handle, clicked the groove or dragged along it; `dragging` is set meanwhile, so playback doesn't
    move it under the mouse.

    Fluent's Slider only reports releases of drags that start on the handle, and never sets
    isSliderDown(), so clicks on the groove were undone by the next position update.
    """

    seek_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.dragging = False
        self.sliderPressed.connect(self._start)  # on the handle
        self.sliderReleased.connect(self._finish)

    def mousePressEvent(self, e: QMouseEvent) -> None:  # on the groove
        self._start()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        super().mouseReleaseEvent(e)
        self._finish()

    def _start(self) -> None:
        self.dragging = True

    def _finish(self) -> None:
        if self.dragging:
            self.dragging = False
            self.seek_requested.emit(self.value())


class PlayerBar(QWidget):
    """Spotify-like bar at the bottom of the window: what's playing (cover, title, artists), shuffle /
    previous / play-pause / next, a seek slider with the elapsed and total time, and the volume."""

    def __init__(self, player: Player, covers: CoverCache, parent=None):
        super().__init__(parent)
        self.player = player
        self.covers = covers
        self.setFixedHeight(HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)

        # --- now playing
        self.cover = ImageLabel()
        self.cover.setFixedSize(COVER, COVER)
        self.title = BodyLabel()
        self.artists = CaptionLabel()
        self.artists.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addStretch(1)
        names.addWidget(self.title)
        names.addWidget(self.artists)
        names.addStretch(1)
        now = QWidget()
        now.setFixedWidth(SIDE_WIDTH)
        now_layout = QHBoxLayout(now)
        now_layout.setContentsMargins(0, 0, 0, 0)
        now_layout.setSpacing(12)
        now_layout.addWidget(self.cover)
        now_layout.addLayout(names, 1)

        # --- controls
        self.shuffle_button = TransparentToggleToolButton(PlayerIcon.SHUFFLE)
        self.shuffle_button.setToolTip("Shuffle")
        self.shuffle_button.setChecked(player.shuffle)
        self.shuffle_button.toggled.connect(player.set_shuffle)
        self.previous_button = TransparentToolButton(PlayerIcon.PREVIOUS)
        self.play_button = TransparentToolButton(FluentIcon.PLAY_SOLID)
        self.next_button = TransparentToolButton(PlayerIcon.NEXT)
        for button in (self.shuffle_button, self.previous_button, self.play_button, self.next_button):
            button.setFixedSize(BUTTON, BUTTON)
            button.setIconSize(QSize(ICON, ICON))
        self.previous_button.clicked.connect(player.previous)
        self.play_button.clicked.connect(player.toggle)
        self.next_button.clicked.connect(player.next)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        for button in (self.shuffle_button, self.previous_button, self.play_button, self.next_button):
            buttons.addWidget(button)
        buttons.addSpacing(BUTTON + buttons.spacing())  # balances shuffle, keeping play/pause in the middle
        buttons.addStretch(1)

        self.elapsed = CaptionLabel("0:00")
        self.length = CaptionLabel("0:00")
        self.slider = SeekSlider()
        self.slider.seek_requested.connect(player.seek)
        self.slider.sliderMoved.connect(lambda value: self.elapsed.setText(_time(value)))
        seek = QHBoxLayout()
        seek.setSpacing(10)
        seek.addWidget(self.elapsed)
        seek.addWidget(self.slider, 1)
        seek.addWidget(self.length)

        controls = QVBoxLayout()
        controls.setSpacing(2)
        controls.addLayout(buttons)
        controls.addLayout(seek)

        # --- volume
        self.mute_button = TransparentToolButton(FluentIcon.VOLUME)
        self.mute_button.setToolTip("Mute")
        self.mute_button.clicked.connect(lambda: player.set_muted(not player.muted))
        self.volume = Slider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setFixedWidth(VOLUME_WIDTH)
        self.volume.setValue(player.volume)
        self.volume.valueChanged.connect(player.set_volume)
        sound = QWidget()
        sound.setFixedWidth(SIDE_WIDTH)
        sound_layout = QHBoxLayout(sound)
        sound_layout.setContentsMargins(0, 0, 0, 0)
        sound_layout.setSpacing(6)
        sound_layout.addStretch(1)
        sound_layout.addWidget(self.mute_button)
        sound_layout.addWidget(self.volume)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(24)
        layout.addWidget(now)
        layout.addLayout(controls, 1)
        layout.addWidget(sound)

        player.track_changed.connect(self._on_track)
        player.playing_changed.connect(self._on_playing)
        player.position_changed.connect(self._on_position)
        player.shuffle_changed.connect(self.shuffle_button.setChecked)
        player.volume_changed.connect(self.volume.setValue)
        player.muted_changed.connect(self._on_muted)
        covers.loaded.connect(lambda: self._show_cover(player.track))
        qconfig.themeChanged.connect(lambda theme: self._update_style())
        self._update_style()
        self._on_track(None)

    def _update_style(self) -> None:
        line = "rgba(255, 255, 255, 0.08)" if isDarkTheme() else "rgba(0, 0, 0, 0.08)"
        background = "#202020" if isDarkTheme() else "#f7f7f7"
        self.setStyleSheet(f"PlayerBar {{background: {background}; border-top: 1px solid {line}}}")

    def _on_track(self, track: TrackRecord | None) -> None:
        for widget in (self.previous_button, self.play_button, self.next_button, self.slider):
            widget.setEnabled(track is not None)
        if track is None:
            self.title.setText("Nothing playing")
            self.artists.setText("Double-click a track to play it")
            self._on_position(0, 0)
        else:
            # Cut with "…" to fit next to the cover; the full text is in the tooltip.
            width = SIDE_WIDTH - COVER - 12
            for label, text in ((self.title, track.tags.title or track.path), (self.artists, ", ".join(track.tags.artists))):
                label.setText(label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, width))
                label.setToolTip(text)
        self._show_cover(track)

    def _show_cover(self, track: TrackRecord | None) -> None:
        thumbnail = self.covers.get(track) if track is not None else None
        self.cover.setVisible(thumbnail is not None)
        if thumbnail is not None:
            self.cover.setImage(thumbnail)
            self.cover.setFixedSize(COVER, COVER)

    def _on_muted(self, muted: bool) -> None:
        self.mute_button.setIcon(FluentIcon.MUTE if muted else FluentIcon.VOLUME)
        self.mute_button.setToolTip("Unmute" if muted else "Mute")

    def _on_playing(self, playing: bool) -> None:
        self.play_button.setIcon(FluentIcon.PAUSE_BOLD if playing else FluentIcon.PLAY_SOLID)

    def _on_position(self, elapsed: int, length: int) -> None:
        if self.slider.dragging:  # don't move it under the mouse
            return
        self.slider.setRange(0, max(length, 1))
        self.slider.setValue(elapsed)
        self.elapsed.setText(_time(elapsed))
        self.length.setText(_time(length))
