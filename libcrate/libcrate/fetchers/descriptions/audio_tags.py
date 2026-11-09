"""Audio tags: the phrases of PHRASES (sound, mood, use; not genres, which other sources give) that a CLAP model says
fit a track unusually well, given to a language model as facts about tracks it doesn't know.

    tags = AudioTags(store.load_embeddings("muq-mulan"), ClapClient(models_dir, "muq-mulan", processes=1), count=8)
    tags.tags("song.flac")   # ["acoustic guitar", "melancholic", "female vocals", ...]

A phrase fits a track as well as their vectors' dot product says, but some phrases score high on every track, so
each phrase's scores are standardised over the library (z-scores) and a track gets the phrases that fit it best
relative to the others. The phrases' vectors are computed on the first request (loading the model's text side, some
seconds), and the model's process is stopped right after. The tags are often partly wrong; the prompt says so.
"""

import threading

import numpy as np

from libcrate.models.clap import ClapClient
from libcrate.embeddings import EmbeddingTable

__all__ = ["PHRASES", "AudioTags"]

VOCABULARY = {
    "vocals": ["male vocals", "female vocals", "male and female vocals", "choir", "rap vocals", "spoken word",
               "instrumental, no vocals", "auto-tuned vocals", "vocoder vocals", "falsetto vocals", "raspy vocals",
               "soft breathy vocals", "powerful belting vocals", "screamed vocals", "chanting", "vocal chops",
               "children singing", "operatic vocals"],
    "instruments": ["acoustic guitar", "distorted electric guitar", "clean electric guitar", "piano", "electric piano",
                    "synthesizer pads", "arpeggiated synthesizer", "acid bassline", "808 bass", "slap bass",
                    "string section", "violin", "cello", "brass section", "saxophone", "trumpet", "flute", "accordion",
                    "organ", "harp", "sitar", "oud", "bouzouki", "pan flute", "bells", "handclaps", "congas and bongos",
                    "steel drums", "orchestra", "banjo", "harmonica", "ukulele", "marimba", "kalimba"],
    "rhythm": ["slow tempo", "mid tempo", "fast tempo", "four-on-the-floor kick", "breakbeat drums", "trap hi-hats",
               "swing rhythm", "reggae offbeat", "latin percussion", "no drums", "heavy drums", "drum machine",
               "live drums", "shuffle groove", "half-time beat", "pounding kick drum", "polyrhythmic percussion"],
    "mood": ["melancholic", "sad", "happy", "uplifting", "euphoric", "dark", "aggressive", "angry", "romantic",
             "sensual", "dreamy", "nostalgic", "mysterious", "eerie", "tense", "peaceful", "calm", "hopeful",
             "epic", "playful", "funny", "triumphant", "lonely", "hypnotic", "psychedelic", "bittersweet",
             "anxious", "warm", "cold", "spiritual"],
    "energy": ["high energy", "low energy", "laid-back", "groovy", "driving", "building up", "explosive drop",
               "gentle", "intense", "bouncy", "danceable", "not danceable"],
    "production": ["lo-fi", "polished modern production", "live recording", "reverb-drenched", "minimal",
                   "dense wall of sound", "vintage 1960s sound", "1970s sound", "1980s sound", "1990s sound",
                   "2000s sound", "modern 2020s sound", "vinyl crackle", "heavy sidechain compression", "distorted",
                   "clean and sparse", "atmospheric", "cinematic", "glitchy", "field recordings"],
    "use": ["party", "workout", "studying", "sleep", "road trip", "late night", "summer", "winter", "club", "festival",
            "lounge", "dinner", "meditation", "rainy day", "sunrise", "wedding", "movie soundtrack"],
}
PHRASES = [phrase for group in VOCABULARY.values() for phrase in group]


class AudioTags:
    def __init__(self, embeddings: EmbeddingTable, client: ClapClient, *, count: int):
        """
        embeddings -> the CLAP model's audio embeddings of the library's tracks.
        client     -> that model, for the phrases' vectors; closed once they're computed.
        count      -> how many phrases a track gets.
        """
        paths, x = embeddings.matrix()
        self._index = {path: i for i, path in enumerate(paths)}
        self._x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) if len(x) else x
        self._client, self.count = client, count
        self._z: np.ndarray | None = None  # tracks x phrases
        self._lock = threading.Lock()

    def tags(self, path: str) -> list[str]:
        """The `count` phrases that fit the track at `path` best; [] if it has no embedding."""
        if path not in self._index or self.count <= 0:
            return []
        with self._lock:
            if self._z is None:
                phrases = self._client.text_embeddings(PHRASES)
                self._client.close()
                scores = self._x @ phrases.T
                self._z = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-9)
        return [PHRASES[j] for j in np.argsort(-self._z[self._index[path]], kind="stable")[:self.count]]

    def close(self) -> None:
        self._client.close()
