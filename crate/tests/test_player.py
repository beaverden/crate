import time
from pathlib import Path

import pytest
from libcrate import Catalog

TEST_MUSIC = Path(__file__).resolve().parents[2] / "test_music"

vlc = pytest.importorskip("vlc", exc_type=OSError)  # python-vlc raises OSError when libvlc is missing


@pytest.fixture
def player(qapp):
    from crate.player import Player
    instance = vlc.Instance("--no-video", "--quiet", "--aout=dummy")
    if instance is None:
        pytest.skip("libvlc not available")
    player = Player(instance)
    yield player
    player.release()


@pytest.fixture(scope="module")
def music():
    if not TEST_MUSIC.is_dir():
        pytest.skip("test_music/ missing")
    catalog, _ = Catalog.scan(TEST_MUSIC)
    return catalog


def wait(qapp, condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        qapp.processEvents()
        time.sleep(0.01)


def test_plays_and_moves_through_the_queue(qapp, player, music):
    tracks = list(music.tracks.values())[:3]
    changed, positions = [], []
    player.track_changed.connect(changed.append)
    player.position_changed.connect(lambda elapsed, length: positions.append((elapsed, length)))

    player.play_queue(tracks, 1, music.root)
    assert player.track is tracks[1] and player.playing
    wait(qapp, lambda: positions and positions[-1][0] > 0)
    assert positions[-1][1] == pytest.approx(tracks[1].audio.duration * 1000, abs=1000)

    player.toggle()
    assert not player.playing
    player.toggle()
    assert player.playing

    player.next()
    assert player.track is tracks[2]
    player.previous()  # just started: the previous track
    assert player.track is tracks[1]
    player.next()
    player.next()  # past the end: stops
    assert player.track is None and not player.playing
    assert changed[-1] is None


def test_previous_restarts_a_track_that_is_well_underway(qapp, player, music):
    tracks = list(music.tracks.values())[:2]
    player.play_queue(tracks, 1, music.root)
    wait(qapp, lambda: player._player.get_time() > 0)
    player.seek(10_000)
    wait(qapp, lambda: player._player.get_time() >= 10_000)
    player.previous()
    assert player.track is tracks[1]


def test_unplayable_file_reports_and_stops(qapp, player, music, tmp_path):
    failed = []
    player.failed.connect(failed.append)
    player.play_queue(list(music.tracks.values())[:1], 0, tmp_path)  # not where the files are
    wait(qapp, lambda: failed)
    assert "Can't play" in failed[0] and player.track is None


def test_seek_does_not_jump_back(qapp, player, music):
    track = max(music.tracks.values(), key=lambda t: t.audio.duration)
    player.play_queue([track], 0, music.root)
    wait(qapp, lambda: player._player.get_time() > 0)
    positions = []
    player.position_changed.connect(lambda elapsed, length: positions.append(elapsed))

    player.seek(30_000)
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert positions and min(positions) >= 29_000  # never back to where it was


def test_shuffle_plays_every_other_track_once(qapp, player, music):
    tracks = list(music.tracks.values())
    player.set_shuffle(True)
    player.play_queue(tracks, 2, music.root)
    assert player.track is tracks[2]  # the chosen track first
    played = [player.track]
    for _ in tracks[1:]:
        player.next()
        played.append(player.track)
    assert sorted(t.path for t in played) == sorted(t.path for t in tracks)
    player.previous()  # back through the shuffled order
    assert player.track is played[-2]


def test_shuffle_toggle_keeps_the_current_track(qapp, player, music):
    tracks = list(music.tracks.values())
    player.play_queue(tracks, 3, music.root)
    player.set_shuffle(True)
    assert player.track is tracks[3] and player.order[0] == 3
    player.set_shuffle(False)
    assert player.track is tracks[3]
    player.next()
    assert player.track is tracks[4]  # on in queue order


def test_upcoming_skips_and_finished_tracks(qapp, player, music):
    tracks = list(music.tracks.values())
    skipped, finished = [], []
    player.skipped.connect(lambda track, elapsed, length: skipped.append((track, elapsed, length)))
    player.finished.connect(finished.append)
    player.set_shuffle(True)
    player.play_queue(tracks[:3], 1, music.root)
    player.next()
    first, second = player.order[0], player.order[1]

    player.set_upcoming([tracks[5], tracks[6]])  # after the current track, instead of the rest
    assert player.queue == [tracks[first], tracks[second], tracks[5], tracks[6]] and player.track is tracks[second]
    player.next()
    assert player.track is tracks[5]
    player.previous()
    assert player.track is tracks[second]  # what played stays playable
    assert [s[0] for s in skipped] == [tracks[first], tracks[second]]

    wait(qapp, lambda: player._player.get_length() > 0)
    player.seek(player._player.get_length() - 500)
    wait(qapp, lambda: finished)
    assert finished == [tracks[second]] and player.track is tracks[5] and len(skipped) == 2


class FakeOutput:
    """libvlc's media player as far as volume goes: -1 until its audio output exists, as on a real start.
    (The dummy output the other tests use has no volume at all.)"""

    def __init__(self):
        self.ready, self.volume, self.mute = False, 100, 0

    def audio_get_volume(self):
        return self.volume if self.ready else -1

    def audio_set_volume(self, volume):
        if self.ready:
            self.volume = volume

    def audio_get_mute(self):
        return self.mute if self.ready else -1

    def audio_set_mute(self, mute):
        if self.ready:
            self.mute = int(mute)

    def get_state(self):
        return vlc.State.Playing

    def get_time(self):
        return 0

    def get_length(self):
        return 0


def test_volume_and_mute_are_applied_once_the_output_exists(qapp, player):
    changes = []
    player.volume_changed.connect(changes.append)
    player.set_volume(40)
    player.set_volume(140)  # clamped
    assert changes == [40, 100]

    real, player._player = player._player, FakeOutput()
    try:
        player.set_volume(35)
        player.set_muted(True)
        assert player._player.audio_get_volume() == -1  # not taken yet
        player._player.ready = True
        player._poll()
        assert (player._player.volume, player._player.mute) == (35, 1)
    finally:
        player._player = real
