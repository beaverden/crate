"""Qt runs offscreen; FLAC files are built on the fly (headers only, which is all mutagen needs)."""

import os
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml
from libcrate import (
    FETCHERS, Embedding, EmbeddingFetcher, Genre, GenreFetcher, GenreSource, LookupNotFound, SourceStatus,
)
from mutagen.flac import FLAC, Picture
from PySide6.QtCore import QBuffer, QIODevice, QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"

SAMPLE_RATE = 44100
LIGHT = "Metronomy - The Light [GBMVH1900150].flac"


def _add_genres(catalog):
    """Give the library fixture's The Light some fetched genres: essentia found chillwave, chosic nothing."""
    track = catalog.tracks[LIGHT]
    track.genres["chosic-genres"] = GenreSource(status=SourceStatus.NOT_FOUND, key="title:The Light")
    track.genres["essentia-genres"] = GenreSource(genres=[Genre(name="chillwave", confidence=0.2583)], key="model")
    return catalog


def _png() -> bytes:
    """A small valid PNG, for cover art."""
    image = QImage(4, 4, QImage.Format.Format_RGB32)
    image.fill(0x8B5CF6)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def _flac_bytes(seconds: float) -> bytes:
    packed = (SAMPLE_RATE << 44) | (1 << 41) | (15 << 36) | int(seconds * SAMPLE_RATE)  # stereo, 16 bit
    streaminfo = struct.pack(">HH", 4096, 4096) + b"\0" * 6 + packed.to_bytes(8, "big") + b"\0" * 16
    return b"fLaC" + bytes([0x80]) + len(streaminfo).to_bytes(3, "big") + streaminfo


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qsettings(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "crate.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def make_library(tmp_path):
    """A library of `count` plain tagged tracks, for tests that need many rows."""
    def make(count: int) -> Path:
        root = tmp_path / "many"
        root.mkdir()
        for i in range(count):
            path = root / f"Artist {i:03d} - Title {i:03d}.flac"
            path.write_bytes(_flac_bytes(120.0))
            f = FLAC(path)
            f.add_tags()
            f["artist"], f["title"] = f"Artist {i:03d}", f"Title {i:03d}"
            f.save()
        return root
    return make


class FakeGenres(GenreFetcher):
    """chosic-genres, without the network: pop for every track but TEMPO, which it doesn't know."""

    name = "chosic-genres"

    def key(self, record):
        return f"title:{record.tags.title}"

    def fetch(self, record):
        if record.tags.title == "TEMPO":
            raise LookupNotFound("nothing")
        return [Genre(name="pop")]


class FakeEmbeddings(EmbeddingFetcher):
    """discogs-effnet, without a model to download or run: ones for every track but TEMPO, which fails."""

    name, space = "discogs-effnet", "audio"

    def key(self, record) -> str:
        return "v1"

    def fetch(self, record):
        if record.tags.title == "TEMPO":
            raise RuntimeError("cannot embed")
        return Embedding(np.ones(4, dtype=np.float32))


@pytest.fixture
def fake_fetchers(monkeypatch):
    """chosic-genres and discogs-effnet are FakeGenres and FakeEmbeddings for the test."""
    for fake in (FakeGenres, FakeEmbeddings):
        monkeypatch.setitem(FETCHERS, fake.name, fake)


def _enable(store, *names) -> None:
    data = {"fetchers": {}}
    for name in names:
        data["fetchers"].setdefault(FETCHERS[name].kind, {})[name] = {"enabled": True}
    store.folder.mkdir(parents=True, exist_ok=True)
    store.settings_path.write_text(yaml.safe_dump(data))


@pytest.fixture
def enable():
    """enable(store, *names): turn `names` on in the catalog's settings file, the rest off."""
    return _enable


@pytest.fixture
def add_genres():
    """add_genres(catalog): The Light gets genres, as an enrich would have given it; returns the catalog."""
    return _add_genres


@pytest.fixture
def library(tmp_path) -> Path:
    """Three tagged tracks; The Light has a cover."""
    root = tmp_path / "library"
    root.mkdir()
    tracks = {
        "Metronomy - The Light [GBMVH1900150].flac": (
            {"artist": "Metronomy", "title": "The Light", "album": "Metronomy Forever", "date": "2019"}, 233.0),
        "NASAYA - TEMPO [GBCFB2100500].flac": ({"artist": "NASAYA", "title": "TEMPO", "date": "2021"}, 180.0),
        "Abba - Waterloo.flac": ({"artist": "ABBA", "title": "Waterloo", "date": "1974"}, 167.4),
    }
    for name, (tags, seconds) in tracks.items():
        path = root / name
        path.write_bytes(_flac_bytes(seconds))
        f = FLAC(path)
        f.add_tags()
        for key, value in tags.items():
            f[key] = value
        if name.startswith("Metronomy"):
            picture = Picture()
            picture.type, picture.mime, picture.data = 3, "image/png", _png()
            f.add_picture(picture)
        f.save()
    return root
