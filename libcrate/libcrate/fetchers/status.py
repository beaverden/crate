"""Whether a fetcher can run, and how far its results are: for the settings page, and for Enrich to skip what can't
run.

    check(settings, "llm-descriptions")            # problems (no API key, a required fetcher that's off), notes
                                                   # (an optional one that's off), models to download first
    progress(catalog, settings, "chosic-genres")   # done, to redo, not done yet, errors, can't (no ID)

Checks are local and free: they look at the settings, the environment and the model folder, never at the services
themselves (whether an API key works costs a request).
"""

from dataclasses import dataclass, field

from libcrate import models
from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable
from libcrate.fetchers.base import MetadataFetcher
from libcrate.fetchers.registry import FETCHERS, make_fetcher
from libcrate.settings import KEY_VARIABLES, LLMService, Settings
from libcrate.track import SourceStatus

__all__ = ["FetcherCheck", "FetcherProgress", "check", "progress"]


@dataclass
class FetcherCheck:
    # Why it can't run: a service that isn't set up, a required fetcher that's off. Empty: it can.
    problems: list[str] = field(default_factory=list)
    # What it does without: an optional fetcher that's off.
    notes: list[str] = field(default_factory=list)
    # The models it needs that aren't downloaded yet, with the ones they need: Enrich downloads them first.
    downloads: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.problems

    @property
    def download_size(self) -> int:
        return sum(models.MODELS[name].size for name in self.downloads)


def _service_problem(settings: Settings, name: str) -> str | None:
    service = getattr(settings.services, name, None)
    if service is None:
        return f"there's no service called {name!r} (services)"
    if isinstance(service, LLMService):
        provider = service.provider
        if provider in KEY_VARIABLES and settings.keys.key(provider) is None:
            return f"no API key for {provider}: set keys.{provider}, or {KEY_VARIABLES[provider]}"
    return None


def check(settings: Settings, name: str, _checking: tuple[str, ...] = ()) -> FetcherCheck:
    """What stands between the fetcher called `name` and running it; raises ValueError for an unknown name. A
    required input that's on but can't run is a problem too."""
    fetcher = make_fetcher(name, settings)
    result = FetcherCheck()
    for service in fetcher.services():
        if problem := _service_problem(settings, service):
            result.problems.append(problem)
    for put in fetcher.inputs():
        if put.fetcher not in FETCHERS:
            result.problems.append(f"there's no fetcher called {put.fetcher!r}")
        elif not settings.fetchers.get(put.fetcher).enabled:
            if put.required:
                result.problems.append(f"needs {put.fetcher}, which is off")
            else:
                result.notes.append(f"uses {put.fetcher} when it's on")
        elif put.required and put.fetcher not in _checking and not check(settings, put.fetcher,
                                                                          (*_checking, name)).ready:
            result.problems.append(f"needs {put.fetcher}, which can't run")
    try:
        result.downloads = models.missing(settings.models.folder, fetcher.models())
    except models.ModelError as e:
        result.problems.append(str(e))
    return result


@dataclass
class FetcherProgress:
    total: int = 0      # tracks in the catalog
    done: int = 0       # with a result for the current settings (found, or nothing to find)
    redo: int = 0       # with a result for other settings (another model, threshold, ID or description)
    todo: int = 0       # never done
    errors: int = 0     # failed last time: tried again
    cant: int = 0       # nothing to go on (no ID, no description)

    @property
    def to_do(self) -> int:
        """What the next Enrich does."""
        return self.redo + self.todo + self.errors


def progress(catalog: Catalog, settings: Settings, name: str, table: EmbeddingTable | None = None) -> FetcherProgress:
    """How far the fetcher called `name` is on `catalog`'s tracks with the current settings; an embedding fetcher's
    needs its `table` (none: nothing done yet). Computing the keys takes a moment on a large catalog (the
    description fetcher hashes each track's facts): run it off the UI thread."""
    fetcher = make_fetcher(name, settings, catalog.root)
    counts = FetcherProgress(total=len(catalog.tracks))
    for record in catalog.tracks.values():
        key = fetcher.key(record)
        if key is None:
            counts.cant += 1
        elif isinstance(fetcher, MetadataFetcher):
            result = getattr(record, fetcher.kind).get(name)
            if result is None:
                counts.todo += 1
            elif result.status == SourceStatus.ERROR:
                counts.errors += 1
            elif result.key != key:
                counts.redo += 1
            else:
                counts.done += 1
        else:
            stored = table.key(record.path) if table is not None else None
            if stored is None:
                counts.todo += 1
            elif stored != key:
                counts.redo += 1
            else:
                counts.done += 1
    return counts
