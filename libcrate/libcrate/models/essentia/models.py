"""The Essentia models: their files, and running them.

Each model is a folder in the models folder, MODELS_DIR/<model>/ (libcrate.models downloads them):

  discogs-effnet-bs64-1                    Discogs-EffNet, the base network: embeddings every head scores
  genre_discogs400-discogs-effnet-1        the genre head (400 Discogs styles), on Discogs-EffNet's embeddings
  discogs_track_embeddings-effnet-bs64-1   Discogs-EffNet trained for track similarity

Importing this is cheap: essentia (and with it TensorFlow, about 1 GB once it has
run) is only imported when Models is created, which only the child process
(worker.py) does. That keeps TensorFlow out of the calling process entirely.
"""

import json
from pathlib import Path

__all__ = [
    "EMBEDDING_MODEL", "GENRE_MODEL", "MODEL_FILES", "RESAMPLE_QUALITY", "TRACK_EMBEDDING_MODEL", "EssentiaError",
    "Models", "check_requirements",
]

EMBEDDING_MODEL = "discogs-effnet-bs64-1"
GENRE_MODEL = "genre_discogs400-discogs-effnet-1"
# The same network trained to put tracks of the same release close together, for similarity.
TRACK_EMBEDDING_MODEL = "discogs_track_embeddings-effnet-bs64-1"
SAMPLE_RATE = 16000
# MonoLoader's resampling to SAMPLE_RATE: 4 is the fastest, and what Essentia's own examples for
# these models use; its scores are as close to the best quality (0) as quality 1's.
RESAMPLE_QUALITY = 4

_ZOO = "https://essentia.upf.edu/models"
# Each model's files in its folder, and where they come from.
MODEL_FILES = {
    EMBEDDING_MODEL: {f"{EMBEDDING_MODEL}.pb": f"{_ZOO}/feature-extractors/discogs-effnet/{EMBEDDING_MODEL}.pb"},
    GENRE_MODEL: {
        f"{GENRE_MODEL}.pb": f"{_ZOO}/classification-heads/genre_discogs400/{GENRE_MODEL}.pb",
        f"{GENRE_MODEL}.json": f"{_ZOO}/classification-heads/genre_discogs400/{GENRE_MODEL}.json",
    },
    TRACK_EMBEDDING_MODEL: {
        f"{TRACK_EMBEDDING_MODEL}.pb": f"{_ZOO}/feature-extractors/discogs-effnet/{TRACK_EMBEDDING_MODEL}.pb",
    },
}


class EssentiaError(Exception):
    """The model files are missing or broken, or a file can't be analysed."""


def check_requirements(models_dir: Path, models: tuple[str, ...]) -> None:
    """Raises EssentiaError if a file of one of `models` is missing from its folder.

    Cheap: doesn't import TensorFlow, so it can run before deciding to analyse anything.
    """
    models_dir = Path(models_dir).expanduser()
    missing = [model for model in models if not all((models_dir / model / f).is_file() for f in MODEL_FILES[model])]
    if missing:
        raise EssentiaError(f"missing Essentia models in {models_dir}: {', '.join(missing)} (Settings › Models)")


def _file(models_dir: Path, model: str, suffix: str = ".pb") -> Path:
    return models_dir / model / f"{model}{suffix}"


class Models:
    """The loaded models. Imports TensorFlow, so only the child process (worker.py) uses this."""

    # The methods the client may call through the child process (EssentiaClient._request).
    REQUESTS = ("genres", "embeddings", "track_embeddings")

    def __init__(self, models_dir: Path, max_seconds: float):
        """Each model loads on its first use: a process that only embeds never loads the genre head."""
        # Here rather than at the top: importing essentia loads TensorFlow (see the module docstring).
        import essentia
        from essentia.standard import MonoLoader

        essentia.log.warningActive = False
        essentia.log.infoActive = False
        self.models_dir = Path(models_dir).expanduser()
        self.max_seconds = max_seconds
        self._load = MonoLoader
        self._loaded: dict[str, object] = {}

    def genres(self, path: str) -> list[tuple[str, float]]:
        """[(Discogs style label, score)], highest first, averaged over the track."""
        labels, head = self._model(GENRE_MODEL)
        averaged = head(self.embeddings(path)).mean(axis=0)
        return sorted(zip(labels, map(float, averaged)), key=lambda ls: -ls[1])

    def embeddings(self, path: str):
        """Discogs-EffNet embeddings of the start of the track, one row per frame; every head scores these."""
        return self._model(EMBEDDING_MODEL)(self._audio(path))

    def track_embeddings(self, path: str):
        """Discogs-EffNet track-similarity embeddings of the start of the track, one row per frame."""
        return self._model(TRACK_EMBEDDING_MODEL)(self._audio(path))

    def _model(self, name: str):
        if name not in self._loaded:
            from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

            check_requirements(self.models_dir, (name,))
            if name == GENRE_MODEL:
                labels = json.loads(_file(self.models_dir, name, ".json").read_text())["classes"]
                head = TensorflowPredict2D(graphFilename=str(_file(self.models_dir, name)),
                                           input="serving_default_model_Placeholder", output="PartitionedCall")
                self._loaded[name] = (labels, head)
            else:
                self._loaded[name] = TensorflowPredictEffnetDiscogs(graphFilename=str(_file(self.models_dir, name)),
                                                                    output="PartitionedCall:1")
        return self._loaded[name]

    def _audio(self, path: str):
        """The first max_seconds of the track, mono at SAMPLE_RATE."""
        try:
            audio = self._load(filename=path, sampleRate=SAMPLE_RATE, resampleQuality=RESAMPLE_QUALITY)()
        except RuntimeError as e:
            raise EssentiaError(f"could not decode {path}: {e}") from e
        audio = audio[: int(self.max_seconds * SAMPLE_RATE)]
        if len(audio) == 0:
            raise EssentiaError(f"no audio in {path}")
        return audio
