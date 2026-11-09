"""Get metadata from each source (online lookups, audio analysis, a language model) and store it in the catalog.

    from libcrate.fetchers import make_fetcher
    from libcrate.fetchers.enrich import enrich

    with make_fetcher("musicbrainz-genres", settings) as mb, make_fetcher("chosic-genres", settings) as chosic:
        report = enrich(catalog, [mb, chosic], on_checkpoint=lambda c: c.save("catalog.json"))

Each metadata fetcher gets one kind of result for one track at a time: genres
(libcrate.fetchers.genres: MusicBrainz and chosic look them up online by an ID
from the tags, Essentia analyses the audio file, the llm fetcher asks a language
model) or a description (libcrate.fetchers.descriptions). The result is stored in
the record's field for that kind, `TrackRecord.genres[fetcher.name]` or
`TrackRecord.descriptions[fetcher.name]` (its kind), with a status:

  found      genres were found
  not_found  the source has nothing (or the track has no usable ID); not retried
  error      it failed (network, rate limit, undecodable audio, ...); retried on the next run

A track is processed when it has no result for the source yet, its last attempt
failed, or the fetcher's key for it changed (a new ID, or new Essentia
settings).

Fetchers run in parallel, one thread each, since they don't compete: each
online source has its own rate limit, and Essentia uses the CPU, not the
network. Within a fetcher, `fetcher.workers` tracks are processed at a time
(1 for rate-limited sources, more for ones that take concurrent requests).

A fetcher whose first GIVE_UP_AFTER results are all errors is stopped: a
missing API key or a dead network fails every track, so going on would only
fill the catalog with errors (they're retried next run anyway).
"""

import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone

from libcrate.catalog import Catalog
from libcrate.fetchers.base import LookupNotFound, MetadataFetcher
from libcrate.track import DescriptionSource, FetchedSource, GenreSource, SourceStatus, TrackRecord

__all__ = ["GIVE_UP_AFTER", "RESULTS", "EnrichReport", "enrich", "lookup", "needs_lookup"]

GIVE_UP_AFTER = 3
# For each kind of result a MetadataFetcher fetches (its kind, a TrackRecord field): its type, and the field of it
# that holds what the fetcher returns.
RESULTS: dict[str, tuple[type[FetchedSource], str]] = {
    "genres": (GenreSource, "genres"),
    "descriptions": (DescriptionSource, "description"),
}


@dataclass
class EnrichReport:
    # source -> Counter of SourceStatus values for the lookups done in this run
    looked_up: dict[str, Counter] = field(default_factory=dict)
    # source -> tracks that already had a result
    skipped: dict[str, int] = field(default_factory=dict)
    # source -> the error it was stopped on, for fetchers whose first results all failed
    gave_up: dict[str, str] = field(default_factory=dict)
    interrupted: bool = False

    @property
    def errors(self) -> int:
        return sum(c[SourceStatus.ERROR] for c in self.looked_up.values())


def needs_lookup(record: TrackRecord, fetcher: MetadataFetcher) -> bool:
    current = getattr(record, fetcher.kind).get(fetcher.name)
    if current is None or current.status == SourceStatus.ERROR:
        return True
    return current.key != fetcher.key(record)


def lookup(record: TrackRecord, fetcher: MetadataFetcher) -> FetchedSource:
    """Run one fetcher on one track and wrap the outcome as its kind of result, e.g. a GenreSource (never raises)."""
    key = fetcher.key(record)
    source, content = RESULTS[fetcher.kind]
    result = source(key=key, fetched_at=datetime.now(timezone.utc))
    if key is None:
        result.status = SourceStatus.NOT_FOUND
        result.error = f"track has no ID {fetcher.name} can look it up by"
        return result
    try:
        setattr(result, content, fetcher.fetch(record))
        result.status = SourceStatus.FOUND
    except LookupNotFound as e:
        result.status = SourceStatus.NOT_FOUND
        result.error = str(e)
    except Exception as e:  # one bad lookup must not stop the run; it's retried next time
        result.status = SourceStatus.ERROR
        result.error = f"{type(e).__name__}: {e}"
    return result


def enrich(
    catalog: Catalog,
    fetchers: list[MetadataFetcher],
    *,
    on_progress: Callable[[str, int, int, str, FetchedSource], None] | None = None,
    on_start: Callable[[str, int, int], None] | None = None,
    on_checkpoint: Callable[[Catalog], None] | None = None,
    checkpoint_every: int = 25,
    stop: threading.Event | None = None,
) -> EnrichReport:
    """Run every fetcher on the tracks that need it, updating `catalog` in place.

    on_progress(source, done, total, rel_path, result) is called after each lookup.
    on_start(source, total, skipped) is called as each fetcher starts (they all run at once),
    with the tracks it will look up and those it has a result for already; also when total is 0.
    on_checkpoint(catalog) is called every `checkpoint_every` lookups (across all
    fetchers) and once at the end, so a long run can be saved as it goes.
    Setting `stop` (or Ctrl+C in the calling thread) finishes the lookups in
    flight, checkpoints, and returns with `report.interrupted = True`.
    """
    stop = stop or threading.Event()
    lock = threading.Lock()
    report = EnrichReport()
    since_checkpoint = 0

    def run(fetcher: MetadataFetcher) -> None:
        nonlocal since_checkpoint
        records = [t for t in catalog.tracks.values() if needs_lookup(t, fetcher)]
        total, todo = len(records), iter(records)
        counts: Counter = Counter()
        with lock:
            report.looked_up[fetcher.name] = counts
            report.skipped[fetcher.name] = len(catalog.tracks) - total
            if on_start:
                on_start(fetcher.name, total, report.skipped[fetcher.name])
        done, give_up = 0, False
        with ThreadPoolExecutor(max_workers=max(1, fetcher.workers)) as pool:
            pending = set()
            while True:
                # Keep `workers` lookups in flight; submit lazily so stopping doesn't wait for a queue.
                while len(pending) < max(1, fetcher.workers) and not (stop.is_set() or give_up):
                    record = next(todo, None)
                    if record is None:
                        break
                    pending.add(pool.submit(lambda r=record: (r, lookup(r, fetcher))))
                if not pending:
                    return
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    record, result = future.result()
                    done += 1
                    with lock:
                        results = {**getattr(record, fetcher.kind), fetcher.name: result}
                        setattr(record, fetcher.kind, dict(sorted(results.items())))
                        counts[result.status] += 1
                        if done == GIVE_UP_AFTER == counts[SourceStatus.ERROR]:
                            give_up = True
                            report.gave_up[fetcher.name] = result.error or "unknown error"
                        since_checkpoint += 1
                        if on_checkpoint and since_checkpoint >= checkpoint_every:
                            since_checkpoint = 0
                            catalog.updated_at = datetime.now(timezone.utc)
                            on_checkpoint(catalog)
                    if on_progress:
                        on_progress(fetcher.name, done, total, record.path, result)

    threads = [threading.Thread(target=run, args=(f,), name=f"enrich-{f.name}", daemon=True) for f in fetchers]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join()
    report.interrupted = stop.is_set()

    catalog.updated_at = datetime.now(timezone.utc)
    if on_checkpoint:
        on_checkpoint(catalog)
    return report
