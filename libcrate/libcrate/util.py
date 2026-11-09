"""Small parsing helpers shared across libcrate."""

import re
from urllib.parse import urlparse

__all__ = ["isrc_from_filename", "normalize_isrc", "number_pair", "split_label", "spotify_track_id", "to_float", "to_int"]

# country (2 letters), registrant (3 alphanumerics), year (2 digits), designation (5 digits)
_ISRC = re.compile(r"[A-Z]{2}[A-Z0-9]{3}\d{7}")
_TRAILING_BRACKETS = re.compile(r"\[([^\[\]]+)\]\s*(?:\.\w+)?$")  # optional file extension
_SPOTIFY_ID = re.compile(r"[A-Za-z0-9]{22}")
_SPOTIFY_HOST = "open.spotify.com"


def to_int(s: str | None) -> int | None:
    try:
        return int(s.strip()) if s else None
    except ValueError:
        return None


def to_float(s: str | None) -> float | None:
    try:
        return float(s.strip()) if s else None
    except ValueError:
        return None


def number_pair(number: str | None, total: str | None) -> tuple[int | None, int | None]:
    """Track/disc number and total: ('3/12', None) -> (3, 12); ('3', '12') -> (3, 12)."""
    n, _, t = (number or "").partition("/")
    return to_int(n), to_int(total) or to_int(t)


def normalize_isrc(isrc: str) -> str:
    """' us-ee1-02-51052 ' -> 'USEE10251052'."""
    return isrc.strip().upper().replace("-", "")


def isrc_from_filename(name: str) -> str | None:
    """ISRC in trailing brackets: 'Artist - Title [QZZ792501624].flac' -> 'QZZ792501624'.

    The bracketed text must be a valid ISRC, so '[Live]' or '[2019 Remaster]' give None.
    """
    m = _TRAILING_BRACKETS.search(name)
    if not m:
        return None
    isrc = normalize_isrc(m.group(1))
    return isrc if _ISRC.fullmatch(isrc) else None


def spotify_track_id(value: str | None) -> str | None:
    """Track ID from a Spotify track URL, or a bare track ID; None for anything else.

    Accepts https://open.spotify.com/track/<id> with an optional /intl-xx/
    prefix and query string (?si=...).
    """
    value = (value or "").strip()
    if _SPOTIFY_ID.fullmatch(value):
        return value
    url = urlparse(value)
    if url.scheme not in ("http", "https") or url.netloc != _SPOTIFY_HOST:
        return None
    parts = [p for p in url.path.split("/") if p]
    if parts and parts[0].startswith("intl-"):
        parts = parts[1:]
    if len(parts) == 2 and parts[0] == "track" and _SPOTIFY_ID.fullmatch(parts[1]):
        return parts[1]
    return None


def split_label(label: str) -> tuple[str | None, str]:
    """Discogs style label: 'Electronic---Deep House' -> ('Electronic', 'Deep House')."""
    parent, sep, style = label.partition("---")
    return (parent, style) if sep else (None, label)
