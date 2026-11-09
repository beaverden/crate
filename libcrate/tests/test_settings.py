"""A catalog's settings file (FOLDER/settings.yaml); every test gets its own folder."""

import stat
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from libcrate.catalog import Catalog
from libcrate.fetchers import FETCHERS, make_fetcher
from libcrate.settings import (
    FetcherSettings, Settings, SettingsError, about, load_settings, save_settings, settings_yaml,
)
from libcrate.store import MetadataStore
from libcrate.track import Genre, GenreSource, SourceStatus


DEFAULT_SETTINGS = settings_yaml()


@pytest.fixture
def config(tmp_path) -> Path:
    """The settings file of the catalog in tmp_path/out (not written yet)."""
    return tmp_path / "out" / "settings.yaml"


def write_config(path: Path, text: str, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# -- loading -------------------------------------------------------------------

def test_defaults_without_a_file(config):
    for settings in (load_settings(), load_settings(config)):  # no file, or a missing one
        assert settings == Settings()
        assert settings.services.llm.model == "anthropic:claude-sonnet-5-5"
        assert settings.services.llm.options == {"effort": "low"}
        assert settings.keys.anthropic is None
        assert not any(settings.fetchers.get(name).enabled for name in FETCHERS)  # nothing runs until chosen


def test_default_file_matches_the_defaults():
    assert Settings.model_validate(yaml.safe_load(DEFAULT_SETTINGS)) == Settings()


def test_custom_file(config, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_config(config, """
keys:
  anthropic: sk-ant-secret
services:
  llm:
    model: anthropic:claude-haiku-5-5
    options: {effort: medium}
models:
  folder: ~/models
fetchers:
  genres:
    llm-genres:
      enabled: true
      min_confidence: 0.3
      workers: 2
""")
    s = load_settings(config)
    assert s.models.folder == tmp_path / "models"  # ~ expanded
    assert (s.services.llm.model, s.services.llm.options) == ("anthropic:claude-haiku-5-5", {"effort": "medium"})
    llm = s.fetchers.get("llm-genres")
    assert (llm.enabled, llm.min_confidence, llm.workers, llm.max_genres) == (True, 0.3, 2, 3)
    assert s.keys.anthropic.get_secret_value() == "sk-ant-secret"
    assert "sk-ant-secret" not in repr(s) and "sk-ant-secret" not in s.model_dump_json()


def test_partial_file_keeps_other_defaults(config):
    write_config(config, "fetchers:\n  genres:\n    chosic-genres:\n      enabled: true\n  embeddings:\n")
    s = load_settings(config)
    assert s.fetchers.get("chosic-genres").enabled and s.fetchers.get("chosic-genres").max_genres == 5
    assert s.services.llm.model == "anthropic:claude-sonnet-5-5"


def test_empty_file_means_defaults(config):
    write_config(config, "# nothing yet\n")
    assert load_settings(config) == Settings()


@pytest.mark.parametrize("text, match", [
    ("services:\n  llm:\n    modle: x\n", r"services\.llm\.modle: Extra inputs are not permitted"),  # typos are caught
    ("fetchers:\n  genres:\n    chosic:\n      enabled: true\n", r"fetchers\.genres\.chosic: Extra inputs"),
    ("library: /music\n", r"unknown sections: library"),
    ("fetchers:\n  genres:\n    llm-genres:\n      workers: many\n", r"fetchers\.genres\.llm-genres\.workers"),
    ("services: [unclosed\n", "invalid YAML"),
    ("- a list\n", "expected a mapping"),
])
def test_invalid_files(config, text, match):
    write_config(config, text)
    with pytest.raises(SettingsError, match=match):
        load_settings(config)


def test_fetcher_settings_by_name():
    s = Settings()
    assert [s.fetchers.get(name).enabled for name in FETCHERS] == [False] * len(FETCHERS)  # every fetcher has some
    with pytest.raises(KeyError):
        s.fetchers.get("chosic")


def test_keys_by_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    s = Settings.model_validate({"keys": {"openai": "sk-1"}})
    assert (s.keys.key("openai"), s.keys.key("anthropic"), s.keys.key("google_genai")) == ("sk-1", "sk-env", None)

    llm = Settings.model_validate({"services": {"llm": {"model": "openai:gpt-6"}}}).services.llm
    assert llm.provider == "openai"
    assert llm.model_kwargs(s.keys) == {"effort": "low", "api_key": "sk-1"}
    assert llm.model_kwargs(s.keys, {"temperature": 0}) == {"temperature": 0, "api_key": "sk-1"}  # options replaced
    assert llm.model_kwargs(s.keys, {"api_key": "sk-2"}) == {"api_key": "sk-2"}  # an explicit key wins
    assert Settings().services.llm.model_kwargs(Settings().keys) == {"effort": "low"}  # LangChain reads the env var


# -- a catalog's settings ----------------------------------------------------------

def test_store_init_writes_the_defaults(library, tmp_path):
    store = MetadataStore(tmp_path / "out")
    assert store.settings() == Settings()  # no file yet

    store.init(library)
    assert store.settings_path == tmp_path / "out/settings.yaml"
    assert store.settings_path.read_text() == DEFAULT_SETTINGS
    assert mode_of(store.settings_path) == 0o600


def test_store_init_keeps_existing_settings(library, config):
    text = "fetchers:\n  genres:\n    llm-genres:\n      workers: 4\n"
    write_config(config, text, mode=0o644)
    store = MetadataStore(config.parent)
    store.init(library)
    assert config.read_text() == text
    assert store.settings().fetchers.get("llm-genres").workers == 4


# -- fetchers ----------------------------------------------------------------------

def test_llm_fetcher_takes_the_file_settings(config, monkeypatch):
    import libcrate.fetchers.genres.llm as llm_mod

    made = {}
    monkeypatch.setattr(llm_mod, "LLMClient", lambda model, **kwargs: made.update(model=model, **kwargs))
    write_config(config, """
keys:
  anthropic: sk-ant-secret
services:
  llm:
    model: anthropic:claude-haiku-5-5
    options: {effort: medium}
fetchers:
  genres:
    llm-genres: {min_confidence: 0.3, max_genres: 4, workers: 2}
""")
    fetcher = make_fetcher("llm-genres", load_settings(config)).open()
    assert made == {"model": "anthropic:claude-haiku-5-5", "effort": "medium", "api_key": "sk-ant-secret"}
    assert (fetcher.options.max_genres, fetcher.options.min_confidence, fetcher.workers) == (4, 0.3, 2)


def test_pilot_uses_settings(config, llm_pilot, library, tmp_path, monkeypatch, capsys):
    pilot_mod = llm_pilot
    catalog, _ = Catalog.scan(library)
    catalog.tracks["Artist A - One [AAA111].flac"].genres["chosic-genres"] = GenreSource(
        status=SourceStatus.FOUND, genres=[Genre(name="pop")])
    MetadataStore(config.parent).save(catalog)
    write_config(config, "keys:\n  anthropic: sk-ant-secret\nservices:\n  llm:\n    model: anthropic:claude-haiku-5-5\n")
    seen = {}

    class Recorded(Exception):
        pass

    def fake(model, **options):
        seen.update(model=model, **options)
        raise Recorded

    monkeypatch.setattr(pilot_mod, "LLMClient", fake)
    with pytest.raises(Recorded):
        pilot_mod.main([str(config.parent)])  # model and key from the catalog's settings
    assert seen == {"model": "anthropic:claude-haiku-5-5", "effort": "low", "api_key": "sk-ant-secret"}


# -- editing (the app's settings page) -----------------------------------------

def _sections(model: type[BaseModel], path: str = ""):
    """Every section under `model`, with its path ("fetchers.genres.chosic-genres")."""
    for name, field in model.model_fields.items():
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            here = f"{path}{field.alias or name}"
            yield here, field.annotation
            yield from _sections(field.annotation, here + ".")


def test_every_section_and_setting_has_a_comment():
    for path, section in _sections(Settings):
        assert about(section), path
        assert f"# {about(section)}\n" in DEFAULT_SETTINGS
        for key, field in section.model_fields.items():
            if not (isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)):
                assert field.description, f"{path}.{key}"
                assert f"# {field.description}\n" in DEFAULT_SETTINGS


def test_each_fetcher_says_what_it_does():
    for name in FETCHERS:
        section = type(Settings().fetchers.get(name))
        assert section is not FetcherSettings and about(section) != about(FetcherSettings), name


def test_save_settings_round_trip(config):
    settings = load_settings(config)
    settings.fetchers.get("llm-genres").workers = 2
    settings.fetchers.get("chosic-genres").enabled = True
    settings.services.llm.options = {"effort": "high", "temperature": 0.2}
    settings.keys.anthropic = "sk-test: with # odd chars"
    settings.fetchers.get("musicbrainz-genres").use_artist = False
    settings.models.folder = Path("/tmp/models dir")
    settings.describe.words = None
    path = save_settings(settings, config)

    assert mode_of(path) == 0o600
    assert path.read_text() == settings_yaml(settings)
    assert "      # requests in parallel\n      workers: 2\n" in path.read_text()  # comments kept
    again = load_settings(config)
    assert again.fetchers.get("llm-genres").workers == 2 and again.fetchers.get("chosic-genres").enabled
    assert again.services.llm.options == {"effort": "high", "temperature": 0.2}
    assert again.keys.anthropic.get_secret_value() == "sk-test: with # odd chars"
    assert again.fetchers.get("musicbrainz-genres").use_artist is False
    assert again.models.folder == Path("/tmp/models dir") and again.describe.words is None


def test_save_settings_tightens_a_readable_file(config):
    write_config(config, "services:\n  chosic:\n    timeout: 4\n", mode=0o644)
    save_settings(Settings(), config)
    assert mode_of(config) == 0o600


def test_save_settings_with_empty_options_and_no_key(config):
    settings = load_settings(config)
    settings.services.llm.options = {}
    save_settings(settings, config)
    again = load_settings(config)
    assert again.services.llm.options == {} and again.keys.anthropic is None
