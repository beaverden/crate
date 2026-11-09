"""Similar tracks over embeddings (similar/similar.py), with hand-made vectors."""

import numpy as np
import pytest

from libcrate.catalog import Catalog
from libcrate.embeddings import Embedding, EmbeddingTable
from libcrate.similar import Similar
from libcrate.settings import Settings

OPTIONS = {"count": 10}  # everything, in order


@pytest.fixture
def library(tmp_path, make_flac):
    """Tracks along a line of sound: each is closest to its neighbours in the list."""
    root = tmp_path / "library"
    tracks = [
        ("A", "Seed"), ("B", "Near"), ("C", "Nearish"), ("A", "Same Artist"), ("D", "Far"),
        ("E", "Seed - Radio Edit"),          # left without an embedding
        ("A feat. X", "Seed (Remastered)"),  # the seed's song again, by the same main artist
    ]
    for artist, title in tracks:
        make_flac(root / f"{artist} - {title}.flac", {"artist": artist, "title": title})
    return root


def table_for(catalog: Catalog, vectors: dict[str, list[float]]) -> EmbeddingTable:
    table = EmbeddingTable()
    for track in catalog.tracks.values():
        if track.tags.title in vectors:
            table.put(track.path, "k", Embedding(np.array(vectors[track.tags.title], dtype=np.float32)))
    return table


def path_of(catalog, title):
    return next(t.path for t in catalog.tracks.values() if t.tags.title == title)


VECTORS = {
    "Seed": [1.0, 0.0, 0.0],
    "Near": [0.9, 0.3, 0.0],
    "Nearish": [0.7, 0.6, 0.1],
    "Same Artist": [0.95, 0.1, 0.0],
    "Far": [-0.5, 0.2, 0.9],
    "Seed (Remastered)": [0.99, 0.0, 0.05],
}


def titles(items):
    return [item.track.tags.title for item in items]


def test_closest_first(library):
    catalog, _ = Catalog.scan(library)
    similar = Similar(catalog, table_for(catalog, VECTORS))
    assert len(similar) == 6  # "Seed - Radio Edit" has no embedding

    items = similar.find(path_of(catalog, "Seed"), **OPTIONS)

    # Nothing skipped: the seed's other version and its artist's other track are listed too.
    assert titles(items) == ["Seed", "Seed (Remastered)", "Same Artist", "Near", "Nearish", "Far"]
    assert items[0].similarity == pytest.approx(1.0)
    assert [i.similarity for i in items] == sorted((i.similarity for i in items), reverse=True)
    assert similar.find(path_of(catalog, "Seed"), **OPTIONS) == items  # the same every time


def test_count(library):
    catalog, _ = Catalog.scan(library)
    similar = Similar(catalog, table_for(catalog, VECTORS))
    assert titles(similar.find(path_of(catalog, "Seed"), count=2)) == ["Seed", "Seed (Remastered)", "Same Artist"]


def test_tracks_without_embedding_or_gone_from_the_catalog(library):
    catalog, _ = Catalog.scan(library)
    table = table_for(catalog, VECTORS)
    table.put("deleted.flac", "k", Embedding(np.ones(3, dtype=np.float32)))
    similar = Similar(catalog, table)
    assert "deleted.flac" not in similar and len(similar) == 6
    with pytest.raises(KeyError, match="has no embedding"):
        similar.find(path_of(catalog, "Seed - Radio Edit"), **OPTIONS)


def test_empty_table(library):
    catalog, _ = Catalog.scan(library)
    assert len(Similar(catalog, EmbeddingTable())) == 0


def test_settings_options():
    assert Settings().similar.options() == {"count": 30}
