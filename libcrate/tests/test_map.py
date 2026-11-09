"""The music map: laying tracks out, saving it, and MetadataStore.music_map's cache."""

import numpy as np
import pytest

from libcrate.embeddings import Embedding, EmbeddingTable
from libcrate.map import MIN_TRACKS, MusicMap
from libcrate.store import MetadataStore


def table(groups=3, per_group=10, dims=16, key="k1") -> EmbeddingTable:
    """`groups` well-separated groups of similar vectors."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(groups, dims)) * 10
    t = EmbeddingTable()
    for g in range(groups):
        for i in range(per_group):
            t.put(f"g{g}/t{i:02}.flac", key, Embedding((centres[g] + rng.normal(size=dims)).astype(np.float32)))
    return t


def test_compute_lays_out_and_clusters():
    music_map = MusicMap.compute(table(), clusters=3)
    assert len(music_map) == 30 and music_map.paths == sorted(music_map.paths)
    assert music_map.xy.shape == (30, 2) and music_map.xy.dtype == np.float32
    # Each group is one cluster, and its tracks sit closer to each other than to the other groups.
    groups = np.array([int(p[1]) for p in music_map.paths])
    for g in range(3):
        assert len(set(music_map.clusters[groups == g])) == 1
    centres = np.array([music_map.xy[groups == g].mean(axis=0) for g in range(3)])
    spread = max(np.linalg.norm(music_map.xy[groups == g] - centres[g], axis=1).mean() for g in range(3))
    assert spread < min(np.linalg.norm(centres[a] - centres[b]) for a in range(3) for b in range(a + 1, 3))


def test_compute_needs_enough_tracks():
    with pytest.raises(ValueError, match=f"at least {MIN_TRACKS} tracks"):
        MusicMap.compute(table(groups=1, per_group=MIN_TRACKS - 1), clusters=3)
    assert len(MusicMap.compute(table(groups=1, per_group=MIN_TRACKS), clusters=12).clusters) == MIN_TRACKS


def test_save_and_load(tmp_path):
    music_map = MusicMap.compute(table(), clusters=3)
    music_map.save(tmp_path / "map.parquet")
    loaded = MusicMap.load(tmp_path / "map.parquet")
    assert loaded.paths == music_map.paths and loaded.fingerprint == music_map.fingerprint
    assert np.array_equal(loaded.xy, music_map.xy) and np.array_equal(loaded.clusters, music_map.clusters)
    assert MusicMap.load(tmp_path / "none.parquet") is None


def test_fingerprint_follows_embeddings_and_settings():
    base = MusicMap.fingerprint_of(table(), 3)
    assert MusicMap.fingerprint_of(table(), 3) == base
    assert MusicMap.fingerprint_of(table(), 4) != base
    assert MusicMap.fingerprint_of(table(key="k2"), 3) != base
    assert MusicMap.fingerprint_of(table(per_group=11), 3) != base


def test_store_reuses_the_saved_map(tmp_path, monkeypatch):
    store = MetadataStore(tmp_path)
    store._save_embeddings({"m": table()})
    computed = []
    real = MusicMap.compute.__func__
    monkeypatch.setattr(MusicMap, "compute", classmethod(lambda cls, *a, **kw: computed.append(1) or real(cls, *a, **kw)))

    first = store.music_map("m", clusters=3)
    assert (tmp_path / "embeddings" / "m" / "map.parquet").is_file()
    assert store.music_map("m", clusters=3).paths == first.paths and len(computed) == 1
    store.music_map("m", clusters=4)  # other settings
    assert len(computed) == 2
    store._save_embeddings({"m": table(per_group=11)})  # new tracks
    assert len(store.music_map("m", clusters=4)) == 33 and len(computed) == 3
