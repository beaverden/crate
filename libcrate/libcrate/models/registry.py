"""Every model libcrate runs locally, in one folder (the models section of the settings): each in its own subfolder,
FOLDER/<name>/, with its files, its kind and size, and the models it needs.

    missing(folder, ["genre_discogs400-discogs-effnet-1"])   # with "discogs-effnet-bs64-1", the base model it needs
    download(folder, ["genre_discogs400-discogs-effnet-1"])  # fetches what's missing, needed models included
    installed(folder, "muq-mulan")           # whether its folder is complete

Kinds:

  essentia   Essentia's TensorFlow models: Discogs-EffNet, the base network whose embeddings the classification
             heads (genres) score, and the same network trained for track similarity
  clap       audio and text in one space: a track's sound and a description of it (models.clap)
  text       text only: for searching what's been written about tracks (models.clap TEXT_MODELS)

A model is downloaded into FOLDER/<name>.part and only renamed once complete, so a stopped download never leaves
a model that looks installed. Nothing else in libcrate downloads; the clients that run them (models.essentia, models.clap, each in child
processes through models.workers) raise when their model is missing.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from curl_cffi import requests  # brings its own CA bundle, unlike some macOS Pythons

from libcrate.models.clap import models as clap
from libcrate.models.essentia import models as essentia

__all__ = ["MODELS", "Model", "ModelError", "delete", "download", "folder_size", "installed", "missing", "needed"]


class ModelError(Exception):
    """A model is unknown, or can't be downloaded."""


@dataclass(frozen=True)
class Model:
    name: str
    kind: str              # "essentia", "clap" or "text"
    about: str             # what it is, for the Models page
    size: int              # bytes on disk once installed, roughly
    files: tuple[str, ...]  # in its folder; all there means installed
    fetch: Callable[[Path], None] = field(repr=False)  # fills an empty folder with the model
    requires: tuple[str, ...] = ()  # other models it runs on


def _urls(urls: dict[str, str]) -> Callable[[Path], None]:
    """Download each file of `urls` (name -> URL) into the folder, checking its size against the server's."""

    def fetch(folder: Path) -> None:
        for name, url in urls.items():
            try:
                resp = requests.get(url, stream=True, timeout=60)
                try:
                    if resp.status_code != 200:
                        raise ModelError(f"could not download {url}: HTTP {resp.status_code}")
                    expected = int(resp.headers.get("Content-Length") or 0)
                    with open(folder / name, "wb") as f:
                        for chunk in resp.iter_content():
                            f.write(chunk)
                finally:
                    resp.close()
            except requests.RequestsError as e:
                raise ModelError(f"could not download {url}: {e}") from e
            size = (folder / name).stat().st_size
            if expected and size != expected:
                raise ModelError(f"incomplete download of {name}: {size} of {expected} bytes")

    return fetch


def _essentia(name: str, about: str, size: int, requires: tuple[str, ...] = ()) -> Model:
    urls = essentia.MODEL_FILES[name]
    return Model(name, "essentia", about, size, tuple(urls), _urls(urls), requires)


def _clap(spec, kind: str, about: str, size: int) -> Model:
    return Model(spec.name, kind, about, size, spec.files, spec.download)


MODELS: dict[str, Model] = {model.name: model for model in (
    _essentia(essentia.EMBEDDING_MODEL, "Essentia's Discogs-EffNet: the base network the genre model runs on, and "
              "the discogs-effnet embeddings", 18_400_000),
    _essentia(essentia.GENRE_MODEL, "Essentia's genre model: a score for each of 400 Discogs styles",
              2_100_000, requires=(essentia.EMBEDDING_MODEL,)),
    _essentia(essentia.TRACK_EMBEDDING_MODEL, "Discogs-EffNet trained for track similarity: tracks of the same "
              "release close together", 19_000_000),
    _clap(clap.MODELS["muq-mulan"], "clap", "Tencent's MuQ-MuLan: music and descriptions of it in one space; best "
          "at telling genres apart (CC BY-NC 4.0: not for commercial use)", 2_670_000_000),
    _clap(clap.MODELS["laion-clap-music"], "clap", "LAION's CLAP trained on music, converted from the original "
          "release (a 2.35 GB download)", 780_000_000),
    _clap(clap.TEXT_MODELS["bge-base"], "text", "BAAI's bge-base-en-v1.5: English text, for searching the tracks' "
          "descriptions", 440_000_000),
)}


def _model(name: str) -> Model:
    if name not in MODELS:
        raise ModelError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    return MODELS[name]


def needed(names: list[str]) -> list[str]:
    """`names` and the models they require, each once, a required model before the ones needing it."""
    out: list[str] = []

    def add(name: str) -> None:
        for required in _model(name).requires:
            add(required)
        if name not in out:
            out.append(name)

    for name in names:
        add(name)
    return out


def installed(folder: Path, name: str) -> bool:
    model = _model(name)
    return all((Path(folder).expanduser() / name / file).is_file() for file in model.files)


def missing(folder: Path, names: list[str]) -> list[str]:
    """The models of `names`, and those they require, that aren't installed in `folder`."""
    return [name for name in needed(names) if not installed(folder, name)]


def folder_size(folder: Path, name: str) -> int:
    """Bytes the model takes in `folder` (0 when it isn't there)."""
    path = Path(folder).expanduser() / name
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def download(folder: Path, names: list[str], *, on_model: Callable[[str, str], None] | None = None,
             stop: Callable[[], bool] | None = None) -> list[str]:
    """Install the models of `names` (and those they require) missing from `folder`; the ones downloaded.

    on_model(name, status) as each starts ("downloading") and ends ("downloaded"). stop() is asked between models:
    True leaves the rest. Raises ModelError (or ClapError for a failed conversion); what was downloaded so far stays.
    """
    folder = Path(folder).expanduser()
    done = []
    for name in missing(folder, names):
        if stop is not None and stop():
            break
        if on_model:
            on_model(name, "downloading")
        part = folder / f"{name}.part"
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        try:
            MODELS[name].fetch(part)
        except Exception as e:  # network, disk, Hugging Face errors, a failed conversion
            shutil.rmtree(part, ignore_errors=True)
            if isinstance(e, ModelError | clap.ClapError):
                raise
            raise ModelError(f"could not download {name}: {type(e).__name__}: {e}") from e
        shutil.rmtree(folder / name, ignore_errors=True)
        part.rename(folder / name)
        done.append(name)
        if on_model:
            on_model(name, "downloaded")
    return done


def delete(folder: Path, name: str) -> None:
    """Remove the model's folder."""
    _model(name)
    shutil.rmtree(Path(folder).expanduser() / name, ignore_errors=True)
