"""One audio file of the library: `TrackRecord` and its parts, and how they're read from the file.

    root = Path("/Volumes/KINGSTON/Spoti/All")
    record = TrackRecord.from_file(root / "song.flac", root)
    record.tags.title, record.audio.duration, record.genre_names(), record.description_text()
    record.is_stale(root / "song.flac")   # the file changed, or the record's shape is outdated

FLAC only. The Vorbis comments are first flattened into {lowercased key:
[values]}, then normalised into `Tags`. What the fetchers find (see
fetchers/enrich.py) goes to `TrackRecord.genres` and `TrackRecord.descriptions`,
one result per source; the file is never read for those.
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from mutagen import MutagenError
from mutagen.flac import FLAC
from pydantic import BaseModel, Field

from libcrate.util import isrc_from_filename, number_pair, spotify_track_id, to_float, to_int

__all__ = [
    "AUDIO_EXTENSIONS", "SCHEMA_HASH", "AudioInfo", "CoverInfo", "Description", "DescriptionSource", "ExtractError",
    "FetchedSource", "FileStat", "Genre", "GenreSource", "SourceStatus", "Tags", "TrackRecord",
]

AUDIO_EXTENSIONS = {".flac"}



class ExtractError(Exception):
    """The file can't be read as audio."""


# --- genres -------------------------------------------------------------------

class Genre(BaseModel):
    name: str
    # Classifier or model confidence in [0, 1] (essentia, llm).
    confidence: float | None = None
    # Community votes (musicbrainz), summed over recording, release group and artist.
    votes: int | None = None
    # Broader genre the name belongs to (essentia: "Electronic" for "deep house").
    parent: str | None = None


class SourceStatus(StrEnum):
    FOUND = "found"          # genres were found
    NOT_FOUND = "not_found"  # the source has nothing for this track; not retried
    ERROR = "error"          # failed (network, rate limit, undecodable audio...); retried next run


class FetchedSource(BaseModel):
    """What one source says about a track, and how it came to say it: the parts every kind of result
    (genres, descriptions) shares."""

    status: SourceStatus = SourceStatus.FOUND
    # What the result was based on: the track's ID ("isrc:...", "spotify:...") or,
    # for Essentia, the model and its settings. If it changes, the result is redone.
    key: str | None = None
    fetched_at: datetime | None = None
    error: str | None = None


class GenreSource(FetchedSource):
    """What one source (musicbrainz, chosic, essentia, ...) says about a track's genres.

    A source missing from `TrackRecord.genres` entirely was never asked.
    """

    genres: list[Genre] = Field(default_factory=list)


# --- descriptions -------------------------------------------------------------

class Description(BaseModel):
    """How a track sounds and feels, in words, for searching by text."""

    text: str
    keywords: list[str] = Field(default_factory=list)
    # The language of the lyrics; None for instrumentals, or when the source can't tell.
    language: str | None = None
    # Whether the source knew the track or its artist (an LLM), rather than describing it from the other facts alone.
    recognised: bool = False

    def search_text(self) -> str:
        """The text to embed for searching: the description, then the keywords."""
        return f"{self.text}\nKeywords: {', '.join(self.keywords)}" if self.keywords else self.text


class DescriptionSource(FetchedSource):
    """What one source (llm) says about a track, in words. A source missing from `TrackRecord.descriptions`
    was never asked."""

    description: Description | None = None


# --- what the file holds ------------------------------------------------------

class Tags(BaseModel):
    """Well-known tags, normalised. Multi-valued tags are lists."""

    title: str | None = None
    artists: list[str] = Field(default_factory=list)
    album: str | None = None
    album_artists: list[str] = Field(default_factory=list)
    composers: list[str] = Field(default_factory=list)
    date: str | None = None
    year: int | None = None
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    isrc: str | None = None
    upc: str | None = None
    label: str | None = None
    publisher: str | None = None
    copyright: str | None = None
    comment: str | None = None
    spotify_track_id: str | None = None
    musicbrainz_track_id: str | None = None
    bpm: float | None = None
    lyrics: str | None = None
    # The plain GENRE tag as written in the file (merged union of all sources).
    genre: list[str] = Field(default_factory=list)
    # Every other tag, raw: lowercased key -> values.
    extra: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, list[str]], filename: str) -> "Tags":
        """Normalise {lowercased key: [values]}; keys not mapped to a field end up in `extra`."""
        used: set[str] = set()

        def many(*keys) -> list[str]:
            for k in keys:
                used.add(k)
            for k in keys:
                vals = [v.strip() for v in raw.get(k, []) if v.strip()]
                if vals:
                    return vals
            return []

        def one(*keys) -> str | None:
            vals = many(*keys)
            return vals[0] if vals else None

        track_number, track_total = number_pair(one("tracknumber"), one("tracktotal", "totaltracks"))
        disc_number, disc_total = number_pair(one("discnumber"), one("disctotal", "totaldiscs"))
        date = one("date", "originaldate", "year")
        comment = one("comment", "description")
        # The library stores the Spotify track URL in COMMENT; a SPOTIFY_TRACK_ID tag is the fallback.
        spotify = spotify_track_id(comment) or spotify_track_id(one("spotify_track_id"))

        tags = cls(
            title=one("title"),
            artists=many("artist"),
            album=one("album"),
            album_artists=many("albumartist", "album artist"),
            composers=many("composer"),
            date=date,
            year=to_int(date[:4]) if date else None,
            track_number=track_number,
            track_total=track_total,
            disc_number=disc_number,
            disc_total=disc_total,
            # The library's filenames end in "[ISRC]"; used when the ISRC tag is missing.
            isrc=one("isrc") or isrc_from_filename(filename),
            upc=one("upc", "barcode"),
            label=one("label", "organization"),
            publisher=one("publisher"),
            copyright=one("copyright"),
            comment=comment,
            spotify_track_id=spotify,
            musicbrainz_track_id=one("musicbrainz_trackid", "musicbrainz track id"),
            bpm=to_float(one("bpm")),
            lyrics=one("lyrics", "unsyncedlyrics"),
            genre=many("genre"),
        )
        tags.extra = {k: v for k, v in sorted(raw.items()) if k not in used}
        return tags


class AudioInfo(BaseModel):
    format: str
    duration: float
    sample_rate: int | None = None
    channels: int | None = None
    bits_per_sample: int | None = None
    bitrate: int | None = None

    @classmethod
    def from_flac(cls, f: FLAC) -> "AudioInfo":
        info = f.info
        return cls(
            format="flac",
            duration=round(info.length, 3),
            sample_rate=info.sample_rate,
            channels=info.channels,
            bits_per_sample=info.bits_per_sample,
            bitrate=info.bitrate or None,
        )


class CoverInfo(BaseModel):
    mime: str
    size: int
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_flac(cls, f: FLAC) -> "CoverInfo | None":
        """The front cover, else the first picture; None without pictures."""
        pics = f.pictures
        if not pics:
            return None
        pic = next((p for p in pics if p.type == 3), pics[0])  # 3 = front cover
        return cls(
            mime=pic.mime or "application/octet-stream",
            size=len(pic.data),
            width=pic.width or None,
            height=pic.height or None,
        )


class FileStat(BaseModel):
    """What the incremental scan compares to decide whether to re-extract."""

    size: int
    mtime_ns: int


# --- the record ---------------------------------------------------------------

class TrackRecord(BaseModel):
    path: str  # relative to Catalog.root, POSIX separators
    stat: FileStat
    # SCHEMA_HASH when extracted; "" (never current) for records from before hashes.
    schema_hash: str = ""
    extracted_at: datetime
    tags: Tags
    audio: AudioInfo
    cover: CoverInfo | None = None
    genres: dict[str, GenreSource] = Field(default_factory=dict)
    descriptions: dict[str, DescriptionSource] = Field(default_factory=dict)

    @staticmethod
    def is_audio_file(path: Path) -> bool:
        # "._*" are macOS resource-fork files on non-HFS drives, not audio.
        return path.suffix.lower() in AUDIO_EXTENSIONS and not path.name.startswith(".")

    @classmethod
    def from_file(cls, path: Path, root: Path) -> "TrackRecord":
        """Read `path`; the record's path is relative to `root`. Raises ExtractError if it isn't FLAC."""
        st = path.stat()
        try:
            f = FLAC(path)
        except MutagenError as e:
            raise ExtractError(f"mutagen could not read file: {e}") from e

        raw: dict[str, list[str]] = {}
        for k, v in f.tags or []:
            raw.setdefault(k.lower(), []).append(v)

        return cls(
            path=path.relative_to(root).as_posix(),
            stat=FileStat(size=st.st_size, mtime_ns=st.st_mtime_ns),
            schema_hash=SCHEMA_HASH,
            extracted_at=datetime.now(timezone.utc),
            tags=Tags.from_raw(raw, path.name),
            audio=AudioInfo.from_flac(f),
            cover=CoverInfo.from_flac(f),
        )

    def is_stale(self, path: Path) -> bool:
        """Whether the file at `path` must be read again: its size or mtime changed, it's gone,
        or this record was written with another shape (SCHEMA_HASH)."""
        if self.schema_hash != SCHEMA_HASH:
            return True
        try:
            st = path.stat()
        except OSError:
            return True
        return (st.st_size, st.st_mtime_ns) != (self.stat.size, self.stat.mtime_ns)

    def keep_results(self, old: "TrackRecord") -> None:
        """Take over `old`'s results (genres, descriptions): the fetchers made them, the file can't give them again."""
        self.genres, self.descriptions = old.genres, old.descriptions

    def description_text(self) -> str:
        """Every found description's search text (Description.search_text), by source name; "" if none."""
        return "\n\n".join(result.description.search_text() for _, result in sorted(self.descriptions.items())
                           if result.status == SourceStatus.FOUND and result.description)

    def genre_names(self, source: str | None = None) -> list[str]:
        """Genre names from one source, or from all sources (deduplicated case-insensitively)."""
        if source is None:
            sources = list(self.genres.values())
        else:
            sources = [self.genres[source]] if source in self.genres else []
        out, seen = [], set()
        for src in sources:
            for g in src.genres:
                if g.name.lower() not in seen:
                    seen.add(g.name.lower())
                    out.append(g.name)
        return out


def _schema_hash(model: type[BaseModel]) -> str:
    """Short hash of the model's JSON schema: field names, types and defaults, not the docstrings."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if not (k == "description" and isinstance(v, str))}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    schema = json.dumps(strip(model.model_json_schema()), sort_keys=True)
    return hashlib.sha256(schema.encode()).hexdigest()[:12]


# Changes whenever TrackRecord's shape does (a field added, renamed or retyped, here or in
# a nested model): records with another hash are re-extracted on the next incremental scan.
# Changes to the extraction logic alone don't change it: delete catalog.json to redo everything.
SCHEMA_HASH = _schema_hash(TrackRecord)
