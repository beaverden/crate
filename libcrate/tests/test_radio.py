"""Radio stations over embeddings (radio/radio.py), with hand-made vectors."""

import numpy as np
import pytest

from libcrate.catalog import Catalog
from libcrate.embeddings import Embedding, EmbeddingTable
from libcrate.radio import Radio
from libcrate.settings import Settings

OPTIONS = {"drift": 0.0, "choices": 1, "artist_gap": 0, "duplicate_similarity": 1.1}  # the best, every time


def scan(root, make_flac, tracks: dict[str, tuple[str, list[float]]]):
    """A library of `tracks` (title -> (artist, vector)) and its embedding table."""
    for title, (artist, _) in tracks.items():
        make_flac(root / f"{artist} - {title}.flac", {"artist": artist, "title": title})
    catalog, _ = Catalog.scan(root)
    table = EmbeddingTable()
    for track in catalog.tracks.values():
        table.put(track.path, "k", Embedding(np.array(tracks[track.tags.title][1], dtype=np.float32)))
    return catalog, table


def path_of(catalog, title):
    return next(t.path for t in catalog.tracks.values() if t.tags.title == title)


def titles(radio):
    return [item.track.tags.title for item in radio.queue]


@pytest.fixture
def scenes(tmp_path, make_flac):
    """Two scenes far apart: warm (four artists, two tracks each) and cold (eight artists)."""
    rng = np.random.default_rng(0)
    tracks = {}
    for n in range(8):
        tracks[f"Warm {n}"] = (f"W{n // 2}", list(np.eye(4)[0] + rng.normal(scale=0.2, size=4)))
        tracks[f"Cold {n}"] = (f"C{n}", list(np.eye(4)[1] + rng.normal(scale=0.2, size=4)))
    return scan(tmp_path / "scenes", make_flac, tracks)


def test_the_seed_first_then_never_a_track_twice(scenes):
    catalog, table = scenes
    radio = Radio(catalog, table, [path_of(catalog, "Warm 0")], **{**OPTIONS, "choices": 4},
                  rng=np.random.default_rng(1))
    assert len(radio) == 16 and titles(radio) == ["Warm 0"]
    assert radio.queue[0].similarity == pytest.approx(1.0)

    while radio.next() is not None:
        pass
    assert len(titles(radio)) == 16 and len(set(titles(radio))) == 16  # every track once, then nothing left


def test_low_drift_stays_in_the_seeds_scene(scenes):
    catalog, table = scenes
    radio = Radio(catalog, table, [path_of(catalog, "Cold 3")], **{**OPTIONS, "drift": 0.1, "choices": 3},
                  rng=np.random.default_rng(2))
    for _ in range(5):  # while there are at least `choices` cold tracks left
        radio.next()
    assert all(title.startswith("Cold") for title in titles(radio))


def test_artist_gap(scenes):
    catalog, table = scenes
    radio = Radio(catalog, table, [path_of(catalog, "Warm 0")], **{**OPTIONS, "artist_gap": 3})
    for _ in range(3):
        radio.next()
    artists = [item.track.tags.artists[0] for item in radio.queue]
    assert all(len(set(artists[i:i + 3])) == len(artists[i:i + 3]) for i in range(len(artists)))
    assert "Warm 1" not in titles(radio)[:3]  # by W0, as the seed

    # When only the same artists are left, they play anyway rather than nothing.
    radio = Radio(catalog, table, [path_of(catalog, "Warm 0")], **{**OPTIONS, "artist_gap": 100})
    while radio.next() is not None:
        pass
    assert len(radio.queue) == 16


def test_no_other_version_of_a_song_and_no_duplicates(tmp_path, make_flac):
    catalog, table = scan(tmp_path / "lib", make_flac, {
        "Seed": ("A", [1, 0, 0]), "Seed (Remastered)": ("A feat. X", [1, 0.05, 0]),
        "Copy": ("B", [1, 0.01, 0]), "Other": ("C", [0.6, 0.8, 0]), "Far": ("D", [0, 0, 1]),
    })
    radio = Radio(catalog, table, [path_of(catalog, "Seed")], **{**OPTIONS, "duplicate_similarity": 0.99})
    while radio.next() is not None:
        pass
    assert titles(radio) == ["Seed", "Other", "Far"]


@pytest.fixture
def crossroads(tmp_path, make_flac):
    """From the seed, two ways: Left (two tracks) a little closer than Right; Away is elsewhere."""
    return scan(tmp_path / "crossroads", make_flac, {
        "Seed": ("S", [1, 0, 0, 0]),
        "Left": ("L", [1, 0.9, 0, 0]), "Left too": ("L2", [1, 1.0, 0, 0]),
        "Right": ("R", [1, 0, 1.1, 0]), "Right too": ("R2", [0.6, 0, 1.4, 0]),
        "Away": ("X", [0, 0, 0, 1]), "Away too": ("Y", [0, 0, 0.1, 1]),
    })


def test_skips_steer_away_from_tracks_like_them(crossroads):
    catalog, table = crossroads
    seed = [path_of(catalog, "Seed")]
    radio = Radio(catalog, table, seed, **OPTIONS)
    assert radio.next().track.tags.title == "Left"

    radio = Radio(catalog, table, seed, **OPTIONS)
    radio.skipped(path_of(catalog, "Left too"))
    assert radio.next().track.tags.title == "Right"


def test_played_through_tracks_pull_the_station(crossroads):
    catalog, table = crossroads
    radio = Radio(catalog, table, [path_of(catalog, "Seed")], **OPTIONS)
    radio.listened(path_of(catalog, "Right too"))
    radio.listened(path_of(catalog, "Right too"))
    assert radio.next().track.tags.title == "Right"


def test_high_drift_follows_the_last_track(tmp_path, make_flac):
    """A chain of tracks, each a step further from the seed: drift 1 walks along it, drift 0 stays by the seed."""
    steps = {f"Step {n}": (f"A{n}", [np.cos(n / 4), np.sin(n / 4), 0]) for n in range(6)}
    catalog, table = scan(tmp_path / "chain", make_flac, {
        **steps, "Seed's twin": ("T", [1, -0.5, 0]), "Elsewhere": ("E", [0, 0, 1]),
    })
    seed = [path_of(catalog, "Step 0")]
    radio = Radio(catalog, table, seed, **{**OPTIONS, "drift": 1.0})
    for _ in range(5):
        radio.next()
    assert titles(radio)[:3] == ["Step 0", "Step 1", "Step 2"]
    radio = Radio(catalog, table, seed, **OPTIONS)
    assert radio.next().track.tags.title == "Step 1" and radio.next().track.tags.title == "Seed's twin"


def test_drop_after_forgets_what_came_after(crossroads):
    catalog, table = crossroads
    radio = Radio(catalog, table, [path_of(catalog, "Seed")], **OPTIONS)
    radio.next(), radio.next(), radio.next()
    radio.drop_after(path_of(catalog, "Left"))
    assert titles(radio) == ["Seed", "Left"]
    assert radio.next().track.tags.title == "Left too"  # picked again


def test_several_seeds(crossroads):
    catalog, table = crossroads
    paths = ["missing.flac", path_of(catalog, "Right"), path_of(catalog, "Left")]
    radio = Radio(catalog, table, paths, **OPTIONS)
    assert titles(radio) == ["Right"]  # the first one with an embedding plays first
    assert radio.queue[0].similarity < 1  # the anchor is between the seeds

    with pytest.raises(KeyError):
        Radio(catalog, table, ["missing.flac"], **OPTIONS)


def test_settings_options():
    options = Settings().radio.options()
    assert set(options) == {"drift", "choices", "artist_gap", "duplicate_similarity"}
