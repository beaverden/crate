"""Enrich: run the chosen fetchers over a catalog, in the order their inputs need, after downloading their models.

    report = run(store, catalog, settings)    # the enabled fetchers; or names=[...]
    plan(settings)                            # their order: ["chosic-genres", "muq-mulan", "llm-descriptions", ...]

A fetcher runs after the fetchers whose results it uses (Fetcher.inputs, required or not), so description-bge
comes after llm-descriptions, which comes after the genres and muq-mulan. At the same depth, genres come before
embeddings, and embeddings before descriptions (KINDS). Consecutive fetchers of one engine run together as a
stage: metadata fetchers (genres, descriptions) in parallel, by enrich(); embedding fetchers one after another,
by embed().

A fetcher with a problem (status.check: a service that isn't set up, a required fetcher that's off) doesn't run;
the report says why. The models the others need are downloaded first, each a step of its own ("download <model>").
"""

import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field, fields

from libcrate import models
from libcrate.catalog import Catalog
from libcrate.fetchers.base import KINDS, MetadataFetcher
from libcrate.fetchers.embeddings.embed import EmbedReport
from libcrate.fetchers.enrich import EnrichReport
from libcrate.fetchers.registry import FETCHERS, make_fetcher
from libcrate.fetchers.status import check
from libcrate.settings import Settings
from libcrate.track import SourceStatus

__all__ = ["DOWNLOAD", "OUTCOMES", "RunReport", "plan", "run", "steps"]

# A model's download as a step, next to the fetchers: "download muq-mulan".
DOWNLOAD = "download {}"
# What a track's result came to (a metadata fetcher's: found, not found, error; an embedding fetcher's: computed,
# error), or a download ("downloaded"), in the order they're listed.
OUTCOMES = ("found", "not found", "computed", "downloaded", "error")
_STATUS_OUTCOMES = {SourceStatus.FOUND: "found", SourceStatus.NOT_FOUND: "not found", SourceStatus.ERROR: "error"}


@dataclass
class RunReport:
    enriched: EnrichReport | None = None   # the metadata fetchers' steps together; None if none ran
    embedded: EmbedReport | None = None    # the embedding fetchers'
    downloaded: list[str] = field(default_factory=list)
    not_run: dict[str, str] = field(default_factory=dict)  # fetcher -> why it couldn't run
    stopped: bool = False  # stop was set: what was left didn't run

    @property
    def interrupted(self) -> bool:
        return self.stopped or any(r is not None and r.interrupted for r in (self.enriched, self.embedded))


def plan(settings: Settings, names: list[str] | None = None) -> list[str]:
    """`names` (default: the enabled fetchers) in the order they run."""
    if names is None:
        names = [name for name in FETCHERS if settings.fetchers.get(name).enabled]
    chosen = {name: make_fetcher(name, settings) for name in names}
    depth: dict[str, int] = {}

    def depth_of(name: str, path: tuple[str, ...] = ()) -> int:
        if name not in depth:
            inputs = [put.fetcher for put in chosen[name].inputs() if put.fetcher in chosen and put.fetcher not in path]
            depth[name] = 1 + max((depth_of(other, (*path, name)) for other in inputs), default=-1)
        return depth[name]

    order = list(FETCHERS)
    return sorted(names, key=lambda name: (depth_of(name), KINDS.index(chosen[name].kind), order.index(name)))


def steps(settings: Settings, names: list[str] | None = None) -> list[str]:
    """What run() goes through, as it names the steps to on_start: the downloads, then the fetchers that can run."""
    order = [name for name in plan(settings, names) if check(settings, name).ready]
    needed = [model for name in order for model in make_fetcher(name, settings).models()]
    return [DOWNLOAD.format(m) for m in models.missing(settings.models.folder, needed)] + order


def _stages(fetchers: list) -> list[list]:
    """Consecutive fetchers of one engine (metadata or embeddings), together."""
    stages: list[list] = []
    for fetcher in fetchers:
        if stages and isinstance(stages[-1][0], MetadataFetcher) == isinstance(fetcher, MetadataFetcher):
            stages[-1].append(fetcher)
        else:
            stages.append([fetcher])
    return stages


def _merged(first, second):
    """Two reports of the same kind (EnrichReport, EmbedReport) as one: their per-name dicts together."""
    if first is None or second is None:
        return first or second
    for f in fields(first):
        value = getattr(first, f.name)
        if isinstance(value, dict):
            value.update(getattr(second, f.name))
        else:  # interrupted
            setattr(first, f.name, value or getattr(second, f.name))
    return first


def run(store, catalog: Catalog, settings: Settings, names: list[str] | None = None, *,
        on_start: Callable[[str, int, int], None] | None = None,
        on_progress: Callable[[str, int, int, str, str], None] | None = None,
        stop: threading.Event | None = None) -> RunReport:
    """Run `names` (default: the enabled fetchers) over `catalog` (updated in place, saved by `store`, a
    MetadataStore, as results come in), downloading their models first.

    on_start(step, total, skipped) as each step starts (a download: total 1); on_progress(step, done, total,
    rel_path, outcome) after each track (a download: once, rel_path ""), the outcome one of OUTCOMES. Setting
    `stop` finishes what's in flight and starts nothing more.

    Raises the error of a fetcher that can't be set up (ModelError for a failed download, EssentiaError, ClapError,
    LLMError) and OSError if the catalog can't be saved; what was saved stays.
    """
    stop = stop or threading.Event()
    report = RunReport()
    order = []
    for name in plan(settings, names):
        problems = check(settings, name).problems
        if problems:
            report.not_run[name] = "; ".join(problems)
        else:
            order.append(name)
    fetchers = [make_fetcher(name, settings, catalog.root) for name in order]

    folder = settings.models.folder
    for model in models.missing(folder, [m for f in fetchers for m in f.models()]):
        if stop.is_set():
            break
        step = DOWNLOAD.format(model)
        if on_start:
            on_start(step, 1, 0)
        models.download(folder, [model])
        report.downloaded.append(model)
        if on_progress:
            on_progress(step, 1, 1, "", "downloaded")

    for stage in _stages(fetchers):
        if stop.is_set():
            break
        with ExitStack() as stack:
            ready = []
            for fetcher in stage:
                ready.append(fetcher.open(store.load_embeddings))
                stack.callback(fetcher.close)
            if isinstance(stage[0], MetadataFetcher):
                done = store.enrich(catalog, ready, stop=stop, on_start=on_start, on_progress=None if on_progress is None
                                    else lambda name, d, t, rel, result: on_progress(
                                        name, d, t, rel, _STATUS_OUTCOMES[result.status]))
                report.enriched = _merged(report.enriched, done)
            else:
                done = store.embed(catalog, ready, stop=stop, on_start=on_start, on_progress=None if on_progress is None
                                   else lambda name, d, t, rel, error: on_progress(
                                       name, d, t, rel, "computed" if error is None else "error"))
                report.embedded = _merged(report.embedded, done)
    report.stopped = stop.is_set()
    return report
