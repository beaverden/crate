"""A catalog's settings: FOLDER/settings.yaml.

    settings = MetadataStore(folder).settings()   # load_settings(FOLDER/settings.yaml)
    settings.services.llm.model, settings.models.folder
    settings.fetchers.get("chosic-genres").enabled

Sections:

  keys       API keys, by provider
  services   the online services: a language model, MusicBrainz, chosic
  models     the folder of the models that run locally (libcrate.models)
  fetchers   one entry per fetcher, by what it fetches (genres, embeddings, descriptions): whether Enrich runs it,
             and its own options; a fetcher names the service or the other fetchers it uses
  similar, map, describe, radio
             the features, each naming the fetchers whose results it uses

Each catalog has its own: MetadataStore.init writes a commented default file next
to the catalog, and save_settings() writes given values the same way (the app's
settings page). Each setting's comment is its field description below, and each
section's its docstring: for a fetcher, what it does. Unknown keys are errors, so a
typo doesn't go unnoticed. Without a file the built-in defaults apply.

The defaults live only here: the clients and fetchers take every setting as an
argument, and the fetchers (libcrate.fetchers) build their clients from these.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

__all__ = ["SETTINGS_FILE", "FetcherSettings", "about", "LLMService", "Settings", "SettingsError", "load_settings",
           "save_settings", "settings_from", "settings_yaml"]


# In the catalog folder, next to catalog.json.
SETTINGS_FILE = "settings.yaml"


class SettingsError(Exception):
    def __init__(self, message: str, path: Path | None):
        super().__init__(message)
        self.path = path


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


def about(section: type[BaseModel]) -> str:
    """What a section is for (its docstring), on one line: a fetcher's says what it does."""
    return " ".join((section.__doc__ or "").split())


def _expand(path: Path) -> Path:
    return path.expanduser()


# --- keys and services ------------------------------------------------------------

# Each provider's environment variable, used when its key isn't set here.
KEY_VARIABLES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY"}


class KeysSettings(_Section):
    """API keys, by provider: each is used by the services of that provider (empty: its environment variable)"""

    anthropic: SecretStr | None = Field(None, description="for anthropic:... models (else ANTHROPIC_API_KEY)")
    openai: SecretStr | None = Field(None, description="for openai:... models (else OPENAI_API_KEY)")
    google_genai: SecretStr | None = Field(None, description="for google_genai:... models (else GOOGLE_API_KEY)")

    def key(self, provider: str) -> str | None:
        """The provider's key: this file's, else its environment variable's; None without either."""
        value = getattr(self, provider, None) if provider in KEY_VARIABLES else None
        if value is not None:
            return value.get_secret_value()
        return os.environ.get(KEY_VARIABLES.get(provider, "")) or None


class LLMService(_Section):
    """a language model, through LangChain (paid API calls)"""

    # Chosen from the pilots: knows clearly more of a mostly-electronic library than
    # Claude Haiku 5.5, at ~$3 per 2,000 tracks with effort=low.
    model: str = Field("anthropic:claude-sonnet-5-5", description="LangChain model, e.g. openai:..., google_genai:...")
    options: dict[str, Any] = Field(default_factory=lambda: {"effort": "low"},
                                    description="passed to the model, e.g. effort (reasoning effort: low, medium, high)")

    @property
    def provider(self) -> str:
        return self.model.split(":", 1)[0]

    def model_kwargs(self, keys: KeysSettings, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Keyword arguments for init_chat_model: `options` (the LLM pilot's) replace these options when given; the
        provider's key from `keys` is added when set there (otherwise LangChain reads its environment variable)."""
        kwargs = dict(self.options if options is None else options)
        key = getattr(keys, self.provider, None) if self.provider in KEY_VARIABLES else None
        if key is not None:
            kwargs.setdefault("api_key", key.get_secret_value())
        return kwargs


class MusicBrainzService(_Section):
    """MusicBrainz, the open music encyclopedia (one request a second)"""

    timeout: float = Field(20, description="seconds to wait for each request")


class ChosicService(_Section):
    """chosic.com, which knows the Spotify genres of artists"""

    timeout: float = Field(20, description="seconds to wait for each request")
    min_interval: float = Field(1.5, description="seconds between requests; lower risks being blocked as a bot")


class ServicesSettings(_Section):
    """the online services the fetchers use"""

    llm: LLMService = Field(default_factory=LLMService)
    musicbrainz: MusicBrainzService = Field(default_factory=MusicBrainzService)
    chosic: ChosicService = Field(default_factory=ChosicService)


class ModelsSettings(_Section):
    """the models that run on this computer, a subfolder each; Enrich downloads those its fetchers need"""

    folder: Path = Field(Path("~/crate_models").expanduser(), description="model folder")

    _expand_folder = field_validator("folder", mode="after")(_expand)


# --- fetchers ---------------------------------------------------------------------

class FetcherSettings(_Section):
    """A fetcher's settings: whether Enrich runs it, then its own options. Its docstring says what it does."""

    enabled: bool = Field(False, description="Enrich runs it on the tracks that need it; off, its results stay as they are")


class MusicBrainzGenres(FetcherSettings):
    """community-voted genres from MusicBrainz, looked up by the track's recording ID or ISRC"""

    max_genres: int = Field(5, description="keep at most this many genres")
    min_count: int = Field(1, description="drop genres with fewer votes")
    min_ratio: float = Field(0.0, description="drop genres with less than this fraction of the top genre's votes (0-1)")
    use_artist: bool = Field(True, description="add the artist's genre votes")
    use_release_group: bool = Field(True, description="add the album's genre votes")


class ChosicGenres(FetcherSettings):
    """the Spotify genres of the track's artist, from chosic.com, looked up by the track's Spotify ID"""

    max_genres: int = Field(5, description="keep at most this many genres")


class EssentiaGenres(FetcherSettings):
    """genres predicted from the audio by Essentia's model of 400 Discogs styles, on this computer"""

    min_confidence: float = Field(0.15, description="drop genres scored below this (0-1)")
    max_genres: int = Field(3, description="keep at most this many genres")
    max_seconds: float = Field(300, description="analyse only the start of each track, in seconds")
    processes: int = Field(4, description="tracks analysed at once, each in its own process holding about 1.2 GB")


class LLMGenres(FetcherSettings):
    """the Discogs styles a language model knows the track or its artist by, from the tags (paid API calls)"""

    service: str = Field("llm", description="the language model to ask (services)")
    min_confidence: float = Field(0.5, description="drop genres rated below this (0-1)")  # the pilots showed 0-20% filler
    max_genres: int = Field(3, description="keep at most this many genres")
    workers: int = Field(8, description="requests in parallel")


class GenreFetchers(_Section):
    """genres, from several sources: each keeps its own"""

    musicbrainz: MusicBrainzGenres = Field(default_factory=MusicBrainzGenres, alias="musicbrainz-genres")
    chosic: ChosicGenres = Field(default_factory=ChosicGenres, alias="chosic-genres")
    essentia: EssentiaGenres = Field(default_factory=EssentiaGenres, alias="essentia-genres")
    llm: LLMGenres = Field(default_factory=LLMGenres, alias="llm-genres")


class EssentiaEmbeddings(FetcherSettings):
    max_seconds: float = Field(300, description="embed only the start of each track, in seconds")
    processes: int = Field(4, description="tracks embedded at once, each in its own process holding about 1.2 GB")


class DiscogsEffnetEmbeddings(EssentiaEmbeddings):
    """Essentia's Discogs-EffNet embeddings of the audio (the ones its genre model scores): what sounds alike, for
    Similar, Map and Radio"""


class DiscogsTrackEmbeddings(EssentiaEmbeddings):
    """the same network trained for track similarity, so tracks of the same release end up close together"""


class ClapEmbeddings(FetcherSettings):
    processes: int = Field(4, description="tracks embedded at once, each in its own process holding the model (1-3 GB)")


class MuqMulanEmbeddings(ClapEmbeddings):
    """MuQ-MuLan's embeddings of three 10 s clips of each track, in the space descriptions are embedded in: for
    Describe, and the audio tags of descriptions"""


class LaionClapMusicEmbeddings(ClapEmbeddings):
    """LAION's CLAP (trained on music) embeddings of three 10 s clips of each track, in the space descriptions are
    embedded in: for Describe"""


class DescriptionBgeEmbeddings(FetcherSettings):
    """bge-base's embeddings of the tracks' descriptions (llm-descriptions), for searching what they say in Describe"""


class EmbeddingFetchers(_Section):
    """embeddings: a vector per track, so tracks can be compared with each other or with a description"""

    discogs_effnet: DiscogsEffnetEmbeddings = Field(default_factory=DiscogsEffnetEmbeddings, alias="discogs-effnet")
    discogs_track: DiscogsTrackEmbeddings = Field(default_factory=DiscogsTrackEmbeddings, alias="discogs-track")
    muq_mulan: MuqMulanEmbeddings = Field(default_factory=MuqMulanEmbeddings, alias="muq-mulan")
    laion_clap_music: LaionClapMusicEmbeddings = Field(default_factory=LaionClapMusicEmbeddings,
                                                       alias="laion-clap-music")
    description_bge: DescriptionBgeEmbeddings = Field(default_factory=DescriptionBgeEmbeddings,
                                                      alias="description-bge")


class LLMDescriptions(FetcherSettings):
    """a language model's description of how each track sounds and feels, for Describe: from the tags, the genres
    and audio tags (phrases a CLAP model says fit the track), plus what the model knows (paid API calls)"""

    service: str = Field("llm", description="the language model to ask (services)")
    audio_tags_from: str = Field("muq-mulan", description="the CLAP embeddings whose best-fitting phrases "
                                                          "(instruments, mood, ...) go to the model, once computed")
    audio_tags: int = Field(8, description="how many of those phrases per track (0: none)")
    workers: int = Field(8, description="requests in parallel")


class DescriptionFetchers(_Section):
    """descriptions: how each track sounds and feels, in words"""

    llm: LLMDescriptions = Field(default_factory=LLMDescriptions, alias="llm-descriptions")


class FetchersSettings(_Section):
    """what Enrich runs, by what it fetches"""

    genres: GenreFetchers = Field(default_factory=GenreFetchers)
    embeddings: EmbeddingFetchers = Field(default_factory=EmbeddingFetchers)
    descriptions: DescriptionFetchers = Field(default_factory=DescriptionFetchers)

    def get(self, name: str) -> FetcherSettings:
        """The settings of the fetcher called `name` ("chosic-genres"); KeyError for an unknown one."""
        for kind in type(self).model_fields:
            group = getattr(self, kind)
            for field_name, field in type(group).model_fields.items():
                if field.alias == name:
                    return getattr(group, field_name)
        raise KeyError(name)


# --- features ---------------------------------------------------------------------

class SimilarSettings(_Section):
    """similar tracks: the ones that sound most like a chosen track, best match first"""

    embeddings: str = Field("discogs-effnet", description="the embeddings to compare tracks by")
    count: int = Field(30, description="how many similar tracks to list")

    def options(self) -> dict[str, Any]:
        """Keyword arguments for Similar.find: every setting but the embeddings."""
        return self.model_dump(exclude={"embeddings"})


class MapSettings(_Section):
    """the music map: tracks laid out by similarity, coloured by cluster"""

    embeddings: str = Field("discogs-effnet", description="the embeddings to lay tracks out by")
    clusters: int = Field(12, description="how many groups to colour the tracks in")


class DescribeSettings(_Section):
    """describe: the tracks that sound like a description, best match first"""

    sound: str = Field("muq-mulan", description="the CLAP embeddings to match the description against the sound")
    words: str | None = Field("description-bge", description="the embeddings of the tracks' descriptions to match "
                                                             "it against too (empty: the sound only)")
    count: int = Field(50, description="how many tracks to list")


class RadioSettings(_Section):
    """radio: an endless station from a track or a description, steered by what's played through and skipped"""

    embeddings: str = Field("discogs-effnet", description="the embeddings to pick the tracks by")
    drift: float = Field(0.3, description="0: stay close to where the station started; 1: let each track lead to "
                                          "the next, wherever that goes")
    choices: int = Field(8, description="the next track is drawn from this many best candidates (higher: more "
                                        "variety, less alike)")
    artist_gap: int = Field(10, description="an artist plays at most once in this many tracks")
    duplicate_similarity: float = Field(0.97, description="tracks more similar than this count as the same song")

    def options(self) -> dict[str, Any]:
        """Keyword arguments for Radio: every setting but the embeddings."""
        return self.model_dump(exclude={"embeddings"})


class Settings(_Section):
    """Settings() are the built-in defaults; load_settings() reads a catalog's file."""

    keys: KeysSettings = Field(default_factory=KeysSettings)
    services: ServicesSettings = Field(default_factory=ServicesSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    fetchers: FetchersSettings = Field(default_factory=FetchersSettings)
    similar: SimilarSettings = Field(default_factory=SimilarSettings)
    map: MapSettings = Field(default_factory=MapSettings)
    describe: DescribeSettings = Field(default_factory=DescribeSettings)
    radio: RadioSettings = Field(default_factory=RadioSettings)


# --- the file ---------------------------------------------------------------------

_HEADER = "# This catalog's settings, edited on crate's Settings page.\n"


def _yaml_value(value: Any) -> str:
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, float) and value.is_integer():
        return f"{value:g}"
    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).removesuffix("\n...\n").strip()


def _section_yaml(section: BaseModel, indent: str) -> list[str]:
    """Every key of `section`, each with its description (a nested section: its docstring) as a comment above it."""
    out = []
    for name, field in type(section).model_fields.items():
        key, value = field.alias or name, getattr(section, name)
        if isinstance(value, BaseModel):
            out.append(f"{indent}# {about(type(value))}\n{indent}{key}:\n")
            out.extend(_section_yaml(value, indent + "  "))
            continue
        out.append(f"{indent}# {field.description}\n")
        if isinstance(value, dict) and value:
            out.append(f"{indent}{key}:\n")
            out.extend(f"{indent}  {k}: {_yaml_value(v)}\n" for k, v in value.items())
        else:
            out.append(f"{indent}{key}: {_yaml_value(value)}\n")
    return out


def settings_yaml(settings: Settings | None = None) -> str:
    """The settings file for `settings` (default: the built-in defaults): every section and key, each with its
    description as a comment on the line above, a blank line between top-level sections. API keys are written as
    they are."""
    settings = settings or Settings()
    out = [_HEADER]
    for name in Settings.model_fields:
        section = getattr(settings, name)
        out.append(f"\n# {about(type(section))}\n{name}:\n")
        out.extend(_section_yaml(section, "  "))
    return "".join(out)


def _read_file(path: Path) -> dict:
    """The file's settings as a mapping ({} when it's missing or empty)."""
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SettingsError(f"{path}: invalid YAML: {e}", path) from e
    if not isinstance(data, dict | None):
        raise SettingsError(f"{path}: expected a mapping of settings at the top level", path)
    return data or {}


def _empty_sections_as_defaults(model: type[BaseModel], data: Any) -> Any:
    """`section:` with nothing under it (None) means its defaults, at every level of `model`."""
    if not isinstance(data, dict):
        return data
    fields = {field.alias or name: field.annotation for name, field in model.model_fields.items()}
    out = {}
    for key, value in data.items():
        section = fields.get(key)
        if isinstance(section, type) and issubclass(section, BaseModel):
            value = _empty_sections_as_defaults(section, {} if value is None else value)
        out[key] = value
    return out


def load_settings(path: Path | None = None) -> Settings:
    """What the file at `path` says, over the built-in defaults (no path, or no file: the defaults)."""
    data = _read_file(path) if path is not None else {}
    if unknown := set(data) - set(Settings.model_fields):
        raise SettingsError(f"{path}: unknown sections: {', '.join(sorted(unknown))}", path)
    try:
        return Settings.model_validate(_empty_sections_as_defaults(Settings, data))
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise SettingsError(f"{path}: {problems}", path) from e


def settings_from(path: Path) -> Settings:
    """What another catalog's settings file says (`path` is the catalog folder or the file itself),
    over the built-in defaults: the values to copy."""
    path = path.expanduser()
    file = path / SETTINGS_FILE if path.is_dir() else path
    if not file.is_file():
        raise SettingsError(f"no settings file at {file}", file)
    return load_settings(file)


def save_settings(settings: Settings, path: Path) -> Path:
    """Write `settings` to `path` as settings_yaml() does, readable only by the user, replacing the
    file (and any comments added to it by hand)."""
    _write_private(path, settings_yaml(settings))
    return path


def _write_private(path: Path, text: str) -> None:
    """Write `text` to `path` atomically, readable only by the user (it may hold an API key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    tmp.chmod(0o600)  # O_CREAT's mode doesn't apply to an existing file
    tmp.replace(path)
