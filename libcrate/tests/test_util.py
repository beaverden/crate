import pytest

from libcrate.util import isrc_from_filename, normalize_isrc, number_pair, split_label, spotify_track_id, to_float, to_int

ID = "64R2n9O7Aif7YP3ivkOWYS"


@pytest.mark.parametrize("value, expected", [
    (f"https://open.spotify.com/track/{ID}", ID),
    (f"http://open.spotify.com/track/{ID}", ID),
    (f"https://open.spotify.com/track/{ID}?si=abc123", ID),
    (f"https://open.spotify.com/intl-de/track/{ID}", ID),
    (f"  https://open.spotify.com/track/{ID}/  ", ID),
    (ID, ID),  # bare ID, as a SPOTIFY_TRACK_ID tag would hold it
    (f"https://open.spotify.com/album/{ID}", None),  # not a track
    (f"https://open.spotify.com/track/{ID}/extra", None),
    (f"https://open.spotify.com/track/{ID[:-1]}", None),  # too short
    (f"https://evil.example/open.spotify.com/track/{ID}", None),  # wrong host
    (f"Ripped from https://open.spotify.com/track/{ID}", None),  # not a URL
    (f"spotify:track:{ID}", None),
    ("Great song!", None),
    ("", None),
    (None, None),
])
def test_spotify_track_id(value, expected):
    assert spotify_track_id(value) == expected


@pytest.mark.parametrize("value, expected", [("3", 3), (" 12 ", 12), ("A1", None), ("", None), (None, None)])
def test_to_int(value, expected):
    assert to_int(value) == expected


@pytest.mark.parametrize("value, expected", [("120", 120.0), ("98.5", 98.5), ("fast", None), (None, None)])
def test_to_float(value, expected):
    assert to_float(value) == expected


@pytest.mark.parametrize("number, total, expected", [
    ("3/12", None, (3, 12)),
    ("3", "12", (3, 12)),
    ("3/12", "14", (3, 14)),  # explicit total wins
    ("3", None, (3, None)),
    (None, None, (None, None)),
])
def test_number_pair(number, total, expected):
    assert number_pair(number, total) == expected


def test_normalize_isrc():
    assert normalize_isrc(" us-ee1-02-51052 ") == "USEE10251052"


@pytest.mark.parametrize("name, expected", [
    ("Óperentzia - 60 Pengő [QZZ792501624].flac", "QZZ792501624"),
    ("Mr. Mojo (DE) - Senses - Kyotto Extended Remix [US83Z2546339].flac", "US83Z2546339"),
    ("Artist - Title [qz-zq7-25-01624].flac", "QZZQ72501624"),  # normalised
    ("Artist - Title [QZZ792501624]", "QZZ792501624"),  # no extension
    ("Artist - Title [Live].flac", None),
    ("Artist - Title [2019 Remaster].flac", None),
    ("Artist - Title [QZZ79250162].flac", None),  # one digit short
    ("Artist - [QZZ792501624] Title.flac", None),  # not at the end
    ("Artist - Title.flac", None),
])
def test_isrc_from_filename(name, expected):
    assert isrc_from_filename(name) == expected


@pytest.mark.parametrize("label, expected", [
    ("Electronic---Deep House", ("Electronic", "Deep House")),
    ("Folk, World, & Country---Fado", ("Folk, World, & Country", "Fado")),
    ("Ambient", (None, "Ambient")),
])
def test_split_label(label, expected):
    assert split_label(label) == expected
