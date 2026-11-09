"""The catalog: the JSON-serialisable snapshot of a music library, and how a scan builds it.

One `TrackRecord` (libcrate.track) per audio file, keyed by its path relative
to the library root (so the catalog survives the drive being mounted somewhere else).

    catalog, report = Catalog.scan(Path("/Volumes/KINGSTON/Spoti/All"))   # first time
    catalog, report = Catalog.scan(Path(catalog.root), catalog)           # incremental
    catalog.save("catalog.json")

    catalog = Catalog.load("catalog.json")
    for rel_path, track in catalog.tracks.items():
        track.tags.title, track.genre_names("chosic"), track.audio.duration

A file is re-extracted only when it is new, its size or mtime changed (its tags
were edited), its record has an outdated shape (TrackRecord.is_stale), or it
failed on the previous scan. Records of files that no longer exist are dropped.
Re-extraction keeps the fetchers' results (fetchers/enrich.py decides whether
those need redoing).
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from libcrate.track import ExtractError, SourceStatus, TrackRecord

__all__ = ["Catalog", "ScanError", "ScanReport"]


class ScanError(BaseModel):
    path: str
    error: str


@dataclass
class ScanReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: int = 0
    removed: list[str] = field(default_factory=list)
    failed: list[ScanError] = field(default_factory=list)


class Catalog(BaseModel):
    root: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tracks: dict[str, TrackRecord] = Field(default_factory=dict)
    # Files that couldn't be read on the last scan; retried on the next one.
    errors: list[ScanError] = Field(default_factory=list)

    # --- files ----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        return cls.model_validate_json(Path(path).read_bytes())

    def save(self, path: str | Path) -> None:
        """Write atomically, so an interrupted save never leaves a truncated catalog."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    # --- scanning -------------------------------------------------------------

    @staticmethod
    def find_audio_files(root: Path) -> list[Path]:
        return sorted(p for p in root.rglob("*") if TrackRecord.is_audio_file(p) and p.is_file())

    @classmethod
    def scan(
        cls,
        root: Path,
        previous: "Catalog | None" = None,
        *,
        workers: int = 8,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple["Catalog", ScanReport]:
        """Read the library at `root` into a new catalog, reusing `previous`'s records of unchanged files.

        Returns the new catalog (`previous` is not modified) and what changed.
        on_progress(done, total, rel_path) is called after each extracted file.
        """
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"library path {root} is not a directory")

        old = dict(previous.tracks) if previous else {}
        report = ScanReport()
        tracks: dict[str, TrackRecord] = {}
        todo: list[Path] = []

        for path in cls.find_audio_files(root):
            rel = path.relative_to(root).as_posix()
            prev = old.get(rel)
            if prev is not None and not prev.is_stale(path):
                tracks[rel] = prev
                report.unchanged += 1
            else:
                todo.append(path)

        def work(path: Path) -> TrackRecord | ScanError:
            try:
                return TrackRecord.from_file(path, root)
            except (ExtractError, OSError) as e:
                return ScanError(path=path.relative_to(root).as_posix(), error=str(e))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for done, result in enumerate(pool.map(work, todo), start=1):
                if isinstance(result, ScanError):
                    report.failed.append(result)
                    # Keep the last good record rather than losing the track over a transient error.
                    if result.path in old:
                        tracks[result.path] = old[result.path]
                    rel = result.path
                else:
                    rel = result.path
                    if rel in old:
                        result.keep_results(old[rel])
                        report.updated.append(rel)
                    else:
                        report.added.append(rel)
                    tracks[rel] = result
                if on_progress:
                    on_progress(done, len(todo), rel)

        report.removed = sorted(set(old) - set(tracks))
        catalog = cls(
            root=str(root),
            updated_at=datetime.now(timezone.utc),
            tracks=dict(sorted(tracks.items())),
            errors=report.failed,
        )
        return catalog, report

    # --- queries --------------------------------------------------------------

    def missing_source(self, source: str) -> list[TrackRecord]:
        """Tracks with no result for `source`, or whose last attempt failed."""
        return [t for t in self.tracks.values()
                if source not in t.genres or t.genres[source].status == SourceStatus.ERROR]
