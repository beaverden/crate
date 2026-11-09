"""A track of the library: TrackRecord, its parts, and reading them from the file (track.py)."""

from libcrate.track.track import (
    AUDIO_EXTENSIONS, SCHEMA_HASH, AudioInfo, CoverInfo, Description, DescriptionSource, ExtractError, FetchedSource,
    FileStat, Genre, GenreSource, SourceStatus, Tags, TrackRecord,
)

__all__ = [
    "AUDIO_EXTENSIONS", "SCHEMA_HASH", "AudioInfo", "CoverInfo", "Description", "DescriptionSource", "ExtractError",
    "FetchedSource", "FileStat", "Genre", "GenreSource", "SourceStatus", "Tags", "TrackRecord",
]
