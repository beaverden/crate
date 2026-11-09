"""Compute each model's embedding of every track in a catalog, into one EmbeddingTable per model.

    from libcrate.fetchers import make_fetcher
    from libcrate.fetchers.embeddings import embed

    with make_fetcher("discogs-effnet", settings, catalog.root) as effnet:
        tables = {effnet.name: EmbeddingTable.load(path)}
        report = embed(catalog, [effnet], tables, on_checkpoint=save_tables)

A track is computed when its model's table has no row for it, or the row was
made with another key (another model version or settings, or another description
for models that embed descriptions). A track the model has nothing to embed for
yet (no description) is skipped. Rows of tracks no
longer in the catalog are dropped. Failures aren't stored: they're in the
report, and the track is tried again on the next run.

Models run one after another, so only one model's processes are loaded at a
time; within a model, `fetcher.workers` tracks are computed at once (Essentia
models: one per model process). A model whose first GIVE_UP_AFTER tracks all
fail is stopped, as genre sources are.
"""

import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable
from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.fetchers.enrich import GIVE_UP_AFTER

__all__ = ["EmbedReport", "embed"]


@dataclass
class EmbedReport:
    # model -> tracks computed in this run
    computed: dict[str, int] = field(default_factory=dict)
    # model -> tracks that already had an embedding
    skipped: dict[str, int] = field(default_factory=dict)
    # model -> [(rel_path, error)] for the tracks that failed
    failed: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # model -> the error it was stopped on, for models whose first tracks all failed
    gave_up: dict[str, str] = field(default_factory=dict)
    interrupted: bool = False

    @property
    def errors(self) -> int:
        return sum(map(len, self.failed.values()))


def embed(
    catalog: Catalog,
    fetchers: list[EmbeddingFetcher],
    tables: dict[str, EmbeddingTable],
    *,
    on_progress: Callable[[str, int, int, str, str | None], None] | None = None,
    on_start: Callable[[str, int, int], None] | None = None,
    on_checkpoint: Callable[[dict[str, EmbeddingTable]], None] | None = None,
    checkpoint_every: int = 25,
    stop: threading.Event | None = None,
) -> EmbedReport:
    """Run every model on the tracks that need it, updating `tables` (model name -> table) in place.

    on_progress(model, done, total, rel_path, error) is called after each track; error is None on success.
    on_start(model, total, skipped) is called as each model starts (one after another), with the
    tracks it will compute and those it has an embedding for already; also when total is 0.
    on_checkpoint(tables) is called every `checkpoint_every` tracks (across all models) and once
    at the end. Setting `stop` (or Ctrl+C in the calling thread) finishes the tracks in flight,
    checkpoints, and returns with `report.interrupted = True`.
    """
    stop = stop or threading.Event()
    report = EmbedReport()
    since_checkpoint = 0

    def compute(fetcher: EmbeddingFetcher, record) -> tuple:
        try:
            return record, fetcher.fetch(record), None
        except Exception as e:  # one bad track must not stop the run; it's retried next time
            return record, None, f"{type(e).__name__}: {e}"

    def run(fetcher: EmbeddingFetcher) -> None:
        nonlocal since_checkpoint
        name, table = fetcher.name, tables[fetcher.name]
        table.keep_only(catalog.tracks)
        keys = {t.path: fetcher.key(t) for t in catalog.tracks.values()}
        records = [t for t in catalog.tracks.values() if keys[t.path] is not None and table.key(t.path) != keys[t.path]]
        total, todo = len(records), iter(records)
        report.computed[name] = 0
        report.skipped[name] = len(catalog.tracks) - total
        report.failed[name] = failed = []
        if on_start:
            on_start(name, total, report.skipped[name])
        workers = max(1, fetcher.workers)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()
            while True:
                # Keep `workers` tracks in flight; submit lazily so stopping doesn't wait for a queue.
                while len(pending) < workers and not (stop.is_set() or name in report.gave_up):
                    record = next(todo, None)
                    if record is None:
                        break
                    pending.add(pool.submit(compute, fetcher, record))
                if not pending:
                    return
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Results are stored here, in one thread, so tables and checkpoints need no lock.
                for future in finished:
                    record, embedding, error = future.result()
                    done += 1
                    if error is None:
                        table.put(record.path, keys[record.path], embedding)
                        report.computed[name] += 1
                    else:
                        failed.append((record.path, error))
                    if done == GIVE_UP_AFTER == len(failed):
                        report.gave_up[name] = error
                    since_checkpoint += 1
                    if on_checkpoint and since_checkpoint >= checkpoint_every:
                        since_checkpoint = 0
                        on_checkpoint(tables)
                    if on_progress:
                        on_progress(name, done, total, record.path, error)

    def run_all() -> None:
        for fetcher in fetchers:
            if stop.is_set():
                return
            run(fetcher)

    # In a thread, so Ctrl+C in the calling thread can stop it.
    thread = threading.Thread(target=run_all, name="embed", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(timeout=0.2)
    except KeyboardInterrupt:
        stop.set()
        thread.join()
    report.interrupted = stop.is_set()

    if on_checkpoint:
        on_checkpoint(tables)
    return report
