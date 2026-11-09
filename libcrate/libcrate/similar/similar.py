"""Similar tracks: the tracks of a catalog that sound most like a seed track, best match first.

    settings = store.settings()
    similar = Similar(catalog, store.current_embeddings(catalog, settings, settings.similar.embeddings))
    for item in similar.find("Metronomy - The Light [GBMVH1900150].flac", **settings.similar.options()):
        item.track.tags.title, item.similarity

Similarity is cosine similarity on mean-centred, L2-normalised embeddings
(centring stops a few "average" tracks from being everyone's neighbour). The
list is the same every time: the closest tracks in order, skipping other
versions of a song already listed (same main artist and title), tracks nearly
identical to one listed, and artists listed `per_artist` times already.

Tracks without an embedding in the table are left out.
"""

import re

import numpy as np

from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable, TrackMatch, similarity_space
from libcrate.track import TrackRecord

__all__ = ["Similar", "main_artist", "song_name"]

# "(Explicit)", "[Remastered]", "(feat. X)"
_BRACKETED = re.compile(r"\s*[\[(][^\])]*[\])]")
# Versions of the same song; remixes count as different songs.
_VERSION = re.compile(r"\s+-\s+(radio edit|radio mix|original mix|extended mix|extended|edit|"
                      r"remaster(ed)?.*|\d{4} remaster.*|single version|album version|live.*)$", re.I)
# "A & B", "A feat. B", "A, B", "A x B" -> "a"
_MORE_ARTISTS = re.compile(r"\s*(?:,|&| x | feat\.? | ft\.? )\s*")


def main_artist(track: TrackRecord) -> str:
    """The first artist, without the ones featured or credited with it, lowercased."""
    artist = track.tags.artists[0] if track.tags.artists else ""
    return _MORE_ARTISTS.split(artist.lower())[0].strip()


def song_name(track: TrackRecord) -> str:
    """The title without versions and bracketed extras, to spot other versions of a song."""
    title = _BRACKETED.sub("", track.tags.title or track.path).strip()
    return re.sub(r"\W+", " ", _VERSION.sub("", title).lower()).strip()


class Similar:
    """Tracks ranked by similarity to a seed, over one embedding model's vectors of a catalog's tracks."""

    def __init__(self, catalog: Catalog, embeddings: EmbeddingTable):
        paths, vectors = embeddings.matrix()
        keep = [i for i, path in enumerate(paths) if path in catalog.tracks]
        self.tracks = [catalog.tracks[paths[i]] for i in keep]
        self._index = {track.path: i for i, track in enumerate(self.tracks)}
        self._x = similarity_space(vectors[keep])
        self._artists = [main_artist(t) for t in self.tracks]
        self._songs = [song_name(t) for t in self.tracks]

    def __len__(self) -> int:
        return len(self.tracks)

    def __contains__(self, path: str) -> bool:
        return path in self._index

    def find(self, seed: str, *, count: int, per_artist: int, duplicate_similarity: float) -> list[TrackMatch]:
        """The track at path `seed`, then up to `count` tracks most like it, best first; the settings'
        similar section describes the rest. Raises KeyError if the seed has no embedding."""
        if seed not in self._index:
            raise KeyError(f"{seed} has no embedding")
        x, start = self._x, self._index[seed]
        similarity = x @ x[start]
        listed = [start]
        songs = {(self._artists[start], self._songs[start])}
        per = {self._artists[start]: 1}
        for i in np.argsort(-similarity, kind="stable"):
            if len(listed) > count:
                break
            artist, song = self._artists[i], (self._artists[i], self._songs[i])
            if i == start or song in songs or (per_artist and artist and per.get(artist, 0) >= per_artist):
                continue
            if (x[listed] @ x[i]).max() > duplicate_similarity:  # the same recording, or nearly
                continue
            listed.append(i)
            songs.add(song)
            per[artist] = per.get(artist, 0) + 1
        return [TrackMatch(self.tracks[i], float(similarity[i])) for i in listed]
