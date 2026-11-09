import threading

import pytest

import libcrate
from libcrate import Catalog, MetadataStore, Settings, fetcher_names, make_fetcher
from libcrate.models.essentia.models import MODEL_FILES
from libcrate.fetchers.genres import ChosicGenreFetcher, GenreFetcher, LLMGenreFetcher, MusicBrainzGenreFetcher
from libcrate.track import Genre


def test_scan_creates_the_folder_and_updates_incrementally(library, tmp_path, make_flac):
    store = MetadataStore(tmp_path / "out" / "nested")
    assert store.load() is None

    catalog, report = store.scan(library)
    assert len(report.added) == 3
    assert Catalog.load(store.catalog_path).tracks.keys() == catalog.tracks.keys()

    make_flac(library / "new.flac", {"title": "New"})
    catalog, report = store.scan()  # the library folder comes from the saved catalog
    assert (report.added, report.unchanged) == (["new.flac"], 3)
    assert "new.flac" in store.load().tracks


def test_scan_without_catalog_needs_the_library(tmp_path):
    with pytest.raises(ValueError, match="library folder is needed"):
        MetadataStore(tmp_path).scan()


def test_output_must_be_a_folder(tmp_path):
    old = tmp_path / "library.json"
    old.write_text("{}")
    with pytest.raises(NotADirectoryError, match="is a file, not a folder"):
        MetadataStore(old)


class OneGenre(GenreFetcher):
    name = "fake"

    def key(self, record):
        return "k"

    def fetch(self, record):
        return [Genre(name="pop")]


def test_enrich_saves(library, tmp_path):
    store = MetadataStore(tmp_path)
    catalog, _ = store.scan(library)

    report = store.enrich(catalog, [OneGenre()], stop=threading.Event())

    assert report.looked_up["fake"]["found"] == 3
    assert all(t.genre_names("fake") == ["pop"] for t in store.load().tracks.values())


def test_make_fetcher(tmp_path):
    models = tmp_path / "models"
    for model, files in MODEL_FILES.items():
        (models / model).mkdir(parents=True)
        for name in files:
            (models / model / name).write_text("{}")
    settings = Settings.model_validate({
        "keys": {"anthropic": "sk-test"},
        "services": {"llm": {"model": "anthropic:claude-haiku-5-5"}, "chosic": {"timeout": 5, "min_interval": 3}},
        "models": {"folder": str(models)},
        "fetchers": {"genres": {
            "musicbrainz-genres": {"max_genres": 4, "min_count": 2, "min_ratio": 0.5, "use_artist": False},
            "chosic-genres": {"max_genres": 2},
            "essentia-genres": {"min_confidence": 0.3, "max_genres": 2, "max_seconds": 60, "processes": 3},
            "llm-genres": {"min_confidence": 0.7, "workers": 2},
        }},
    })
    with make_fetcher("musicbrainz-genres", settings) as mb, make_fetcher("chosic-genres", settings) as ch:
        assert isinstance(mb, MusicBrainzGenreFetcher) and isinstance(ch, ChosicGenreFetcher)
        assert (ch.client.timeout, ch.client.max_genres, ch.client.min_interval) == (5, 2, 3)
        c = mb.client
        assert (c.max_genres, c.min_count, c.min_ratio, c.use_artist, c.use_release_group) == (4, 2, 0.5, False, True)
    essentia = make_fetcher("essentia-genres", settings, tmp_path)
    assert essentia.client is None and essentia.workers == 3  # nothing started before open()
    with essentia as es:  # checks the model files, loads nothing
        assert (es.client.models_dir, es.client.max_seconds) == (models, 60)
    llm = make_fetcher("llm-genres", settings).open()  # builds the client, makes no call
    assert isinstance(llm, LLMGenreFetcher)
    assert (llm.model, llm.options.min_confidence, llm.workers) == ("anthropic:claude-haiku-5-5", 0.7, 2)
    assert fetcher_names("genres") == ["musicbrainz-genres", "chosic-genres", "essentia-genres", "llm-genres"]
    with pytest.raises(ValueError, match="unknown fetcher 'spotify'"):
        make_fetcher("spotify", settings)


def test_public_api_is_exported():
    assert all(hasattr(libcrate, name) for name in libcrate.__all__)
