"""Radio: an endless station from one track, or from several (the best matches for a description), each next
track picked as it's needed and steered by what's played through and what's skipped.

    settings = store.settings()
    table = store.current_embeddings(catalog, settings, settings.radio.embeddings)
    radio = Radio(catalog, table, [seed_path], **settings.radio.options())
    radio.queue[0].track                # the seed plays first
    item = radio.next()                 # TrackMatch, or None when nothing's left
    radio.listened(item.track.path)     # played through: the station leans towards it
    radio.skipped(item.track.path)      # skipped early: tracks like it come less
    radio.drop_after(path)              # forget the tracks picked after `path`, to pick them again

The station walks the similarity space (mean-centred, as for similar tracks). Each candidate scores

    (1 - drift) * its similarity to the anchor + drift * its similarity to the last track picked
    - SKIP_PENALTY * its similarity to the most alike skipped track

where the anchor starts as the seeds' mean and moves a little towards every track played through. Low
drift stays close to the seeds; high drift lets one track lead to the next, wherever that goes. The next
track is drawn at random from the `choices` best candidates, so a station from the same seed differs
every time.

Never picked: a track already picked, another version of a song already picked (same main artist and
title, as for similar tracks), a track nearly identical to one picked (duplicate_similarity), or an artist
picked within the last `artist_gap` tracks (unless nothing else is left). Tracks without an embedding
in the table are left out.
"""

import numpy as np

from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable, TrackMatch, similarity_space
from libcrate.similar import main_artist, song_name

__all__ = ["Radio"]

# How far a track played through moves the anchor towards it (the anchor has length 1, so does the track).
LISTENED_PULL = 0.25
# How much a skipped track's similarity counts against the candidates like it.
SKIP_PENALTY = 0.5


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-12)


class Radio:
    """A station over one embedding model's vectors of a catalog's tracks. `queue` lists the tracks picked so
    far, in order, each with its similarity to the anchor when it was picked."""

    def __init__(self, catalog: Catalog, embeddings: EmbeddingTable, seeds: list[str], *, drift: float,
                 choices: int, artist_gap: int, duplicate_similarity: float, rng: np.random.Generator | None = None):
        """`seeds`: paths of the tracks to start from; the first one with an embedding plays first. Raises
        KeyError if none has an embedding."""
        paths, vectors = embeddings.matrix()
        keep = [i for i, path in enumerate(paths) if path in catalog.tracks]
        self.tracks = [catalog.tracks[paths[i]] for i in keep]
        self._index = {track.path: i for i, track in enumerate(self.tracks)}
        self._x = similarity_space(vectors[keep])
        self._artists = np.array([main_artist(t) for t in self.tracks], dtype=object)
        self._songs = [(a, song_name(t)) for a, t in zip(self._artists, self.tracks)]
        self.drift, self.choices, self.artist_gap = drift, choices, artist_gap
        self.duplicate_similarity = duplicate_similarity
        self._rng = rng or np.random.default_rng()

        found = [self._index[path] for path in seeds if path in self._index]
        if not found:
            raise KeyError("none of the seeds has an embedding")
        self._anchor = _unit(self._x[found].mean(axis=0))
        self._picked: list[int] = []
        self._skipped: list[int] = []
        self.queue: list[TrackMatch] = []
        self._pick(found[0])

    def __len__(self) -> int:
        """How many tracks the station can choose from."""
        return len(self.tracks)

    def next(self) -> TrackMatch | None:
        """Pick the next track and add it to the queue; None when every track left is ruled out."""
        x = self._x
        score = (1 - self.drift) * (x @ self._anchor) + self.drift * (x @ x[self._picked[-1]])
        if self._skipped:
            score -= SKIP_PENALTY * (x @ x[self._skipped].T).max(axis=1)
        songs = set(self._songs[i] for i in self._picked)
        ruled_out = np.array([s in songs for s in self._songs], dtype=bool)
        ruled_out[self._picked] = True
        ruled_out |= (x @ x[self._picked].T).max(axis=1) > self.duplicate_similarity
        recent = self._artists[self._picked[-(self.artist_gap - 1):]] if self.artist_gap > 1 else []
        too_soon = np.isin(self._artists, [a for a in recent if a])
        candidates = np.flatnonzero(~ruled_out & ~too_soon)
        if not len(candidates):
            candidates = np.flatnonzero(~ruled_out)
        if not len(candidates):
            return None
        best = candidates[np.argsort(-score[candidates], kind="stable")[:self.choices]]
        return self._pick(int(self._rng.choice(best)))

    def listened(self, path: str) -> None:
        """`path` was played through: move the anchor towards it."""
        if path in self._index:
            self._anchor = _unit(self._anchor + LISTENED_PULL * self._x[self._index[path]])

    def skipped(self, path: str) -> None:
        """`path` was skipped early: count it against the candidates like it."""
        if path in self._index:
            self._skipped.append(self._index[path])

    def drop_after(self, path: str) -> None:
        """Forget the tracks picked after `path` (in the queue), e.g. to pick again after a skip."""
        at = next(n for n, item in enumerate(self.queue) if item.track.path == path) + 1
        del self.queue[at:], self._picked[at:]

    def _pick(self, i: int) -> TrackMatch:
        item = TrackMatch(self.tracks[i], float(self._x[i] @ self._anchor))
        self._picked.append(i)
        self.queue.append(item)
        return item
