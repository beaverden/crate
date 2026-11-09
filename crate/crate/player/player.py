import random
import time
from pathlib import Path

import vlc
from libcrate import TrackRecord
from PySide6.QtCore import QObject, QTimer, Signal

__all__ = ["Player"]

# How often the position is read from libvlc, in ms.
POLL_MS = 200
# After a seek, libvlc keeps reporting the old position for a moment; until it reaches the target
# (within SEEK_TOLERANCE_MS) or SEEK_WAIT_S passes, the target is reported instead, so the seek bar
# doesn't jump back.
SEEK_TOLERANCE_MS = 1500
SEEK_WAIT_S = 1.0
# Previous restarts the track when it's further in than this, in ms; otherwise it goes to the previous track.
RESTART_AFTER_MS = 3000


class Player(QObject):
    """Plays tracks with libvlc (VLC must be installed), from a queue: the table's order when a track
    was started, so next/previous follow what was on screen. A finished track moves on to the next.

    Shuffle plays the rest of the queue in a random order: `order` lists queue positions in the
    order they play, and `position` is where in it playback is. Turning it on keeps the current
    track and shuffles the others after it; turning it off goes on from the current track in queue
    order.

    A track played to its end emits finished(track); next() before the end emits skipped(track, elapsed, length),
    in ms. set_upcoming() replaces what plays after the current track (a radio station tops it up).

    The volume (0-100) and mute are kept here and re-applied while playing: libvlc only takes them
    once its audio output exists, which is a moment after each track starts.

    libvlc runs on its own threads; its state is polled from the Qt event loop rather than read
    through its callbacks, so every signal here comes from the UI thread.
    """

    track_changed = Signal(object)  # TrackRecord, or None when stopped
    playing_changed = Signal(bool)
    position_changed = Signal(int, int)  # elapsed, length (ms)
    failed = Signal(str)
    finished = Signal(object)  # TrackRecord played to its end
    skipped = Signal(object, int, int)  # TrackRecord, elapsed, length (ms): next() before its end
    shuffle_changed = Signal(bool)
    volume_changed = Signal(int)
    muted_changed = Signal(bool)

    def __init__(self, instance: vlc.Instance | None = None, parent=None):
        super().__init__(parent)
        self._vlc = instance or vlc.Instance("--no-video", "--quiet")
        if self._vlc is None:
            raise RuntimeError("libvlc not found: install VLC (https://www.videolan.org/vlc/)")
        self._player = self._vlc.media_player_new()
        self.queue: list[TrackRecord] = []
        self.order: list[int] = []
        self.position = -1
        self.root: Path | None = None
        self.shuffle = False
        self.volume = 100
        self.muted = False
        self._playing = False
        self._seek: tuple[int, float] | None = None  # target ms, monotonic deadline
        self._timer = QTimer(self, interval=POLL_MS, timeout=self._poll)

    @property
    def track(self) -> TrackRecord | None:
        return self.queue[self.order[self.position]] if 0 <= self.position < len(self.order) else None

    @property
    def playing(self) -> bool:
        return self._playing

    def play_queue(self, tracks: list[TrackRecord], index: int, root: str | Path) -> None:
        """Play tracks[index], then the ones after it (or the others, shuffled); paths are relative
        to `root` (the library)."""
        self.queue, self.root = list(tracks), Path(root)
        self.order = self._order(index)
        self._play(self.order.index(index))

    def set_upcoming(self, tracks: list[TrackRecord]) -> None:
        """Play `tracks` after the current one, in order, instead of what was queued after it."""
        played = [self.queue[i] for i in self.order[:self.position + 1]]
        self.queue = played + list(tracks)
        self.order = list(range(len(self.queue)))

    def set_shuffle(self, shuffle: bool) -> None:
        if shuffle == self.shuffle:
            return
        self.shuffle = shuffle
        if self.track is not None:
            current = self.order[self.position]
            self.order = self._order(current)
            self.position = self.order.index(current)
        self.shuffle_changed.emit(shuffle)

    def set_volume(self, volume: int) -> None:
        volume = max(0, min(100, volume))
        if volume != self.volume:
            self.volume = volume
            self._player.audio_set_volume(volume)
            self.volume_changed.emit(volume)

    def set_muted(self, muted: bool) -> None:
        if muted != self.muted:
            self.muted = muted
            self._player.audio_set_mute(muted)
            self.muted_changed.emit(muted)

    def toggle(self) -> None:
        """Pause or resume; with nothing loaded, nothing happens."""
        if self.track is None:
            return
        if self._player.get_state() in (vlc.State.Ended, vlc.State.Stopped):
            self._play(self.position)
        else:
            self._player.pause()
            self._set_playing(not self._playing)

    def next(self) -> None:
        if self.track is not None:
            self.skipped.emit(self.track, max(0, self._player.get_time()), max(0, self._player.get_length()))
        self._advance()

    def _advance(self) -> None:
        if self.position + 1 < len(self.order):
            self._play(self.position + 1)
        else:
            self.stop()

    def previous(self) -> None:
        if self._player.get_time() > RESTART_AFTER_MS or self.position == 0:
            self.seek(0)
        elif self.position > 0:
            self._play(self.position - 1)

    def seek(self, ms: int) -> None:
        ms = max(0, ms)
        self._player.set_time(ms)
        self._seek = (ms, time.monotonic() + SEEK_WAIT_S)
        self.position_changed.emit(ms, max(0, self._player.get_length()))

    def stop(self) -> None:
        self._player.stop()
        self._timer.stop()
        self.position = -1
        self._set_playing(False)
        self.track_changed.emit(None)

    def release(self) -> None:
        """Stop and free libvlc's resources; for closing."""
        self._timer.stop()
        self._player.stop()
        self._player.release()
        self._vlc.release()

    def _order(self, current: int) -> list[int]:
        """The play order of the queue: in order, or `current` first and the others shuffled."""
        if not self.shuffle:
            return list(range(len(self.queue)))
        others = [i for i in range(len(self.queue)) if i != current]
        random.shuffle(others)
        return [current, *others]

    def _play(self, position: int) -> None:
        path = self.root / self.queue[self.order[position]].path
        # libvlc reports a missing file as played to the end, which would race through the queue.
        if not path.is_file():
            self.stop()
            self.failed.emit(f"Can't play {path.name}: the file isn't there. Is the library drive connected?")
            return
        self.position = position
        self._seek = None
        self._player.set_media(self._vlc.media_new_path(str(path)))
        self._player.play()
        self._timer.start()
        self._set_playing(True)
        self.track_changed.emit(self.track)

    def _set_playing(self, playing: bool) -> None:
        if playing != self._playing:
            self._playing = playing
            self.playing_changed.emit(playing)

    def _poll(self) -> None:
        state = self._player.get_state()
        if state == vlc.State.Ended:
            self.finished.emit(self.track)
            self._advance()
        elif state == vlc.State.Error:
            track = self.track
            self.stop()
            self.failed.emit(f"Can't play {track.path}: VLC couldn't read it.")
        else:
            # -1: no audio output yet.
            if self._player.audio_get_volume() not in (-1, self.volume):
                self._player.audio_set_volume(self.volume)
            if self._player.audio_get_mute() not in (-1, int(self.muted)):
                self._player.audio_set_mute(self.muted)
            elapsed = max(0, self._player.get_time())
            if self._seek is not None:
                target, deadline = self._seek
                if abs(elapsed - target) > SEEK_TOLERANCE_MS and time.monotonic() < deadline:
                    elapsed = target
                else:
                    self._seek = None
            self.position_changed.emit(elapsed, max(0, self._player.get_length()))
