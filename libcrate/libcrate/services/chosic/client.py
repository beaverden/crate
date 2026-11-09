"""Client for chosic.com's tools API: Spotify track and artist data, keyed by Spotify IDs.

    from libcrate.services.chosic import ChosicClient

    with ChosicClient(timeout=20, max_genres=5, min_interval=1.5) as chosic:
        chosic.get_genres("53iuhJlwXhSER5J2IYYv1W")

Available today:

  get_genres   the genres of the track's first artist (artist-level, so every
               track by the same artist gets the same list)

New lookups go through _get_json(path), which handles the session and the
request spacing.

The handshake runs once, lazily, on the first request and is reused for all
subsequent calls. If the session expires (401/403), the client re-handshakes
once and retries transparently. Artist lookups are cached, so tracks by the
same artist only cost one extra request. Requests are spaced at least
`min_interval` seconds apart to stay under chosic's bot detection.
"""

import threading
import time
from urllib.parse import quote

# curl_cffi impersonates Chrome's TLS/HTTP2 fingerprint, which is what
# Cloudflare checks before it even looks at your headers. Plain `requests`
# (OpenSSL fingerprint) gets a 403 from chosic.com.
from curl_cffi import requests


__all__ = ["ChosicClient", "ChosicError", "ChosicNotFound"]

CHOSIC_BASE_URL = "https://www.chosic.com"
CHOSIC_TOOLS_API = CHOSIC_BASE_URL + "/api/tools"
CHOSIC_FINDER_PAGE = CHOSIC_BASE_URL + "/music-genre-finder/"

CHOSIC_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": CHOSIC_BASE_URL,
    "Referer": CHOSIC_FINDER_PAGE,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

_SESSION_EXPIRED = {401, 403}


class ChosicError(Exception):
    """Raised for any chosic API failure (network, HTTP status, bad payload)."""


class ChosicNotFound(ChosicError):
    """Chosic has no genres for this track (unknown track, no artist, or artist without genres)."""


class ChosicClient:
    def __init__(self, *, timeout: float, max_genres: int, min_interval: float):
        """
        timeout      -> seconds to wait for each request.
        max_genres   -> keep at most this many of the artist's genres.
        min_interval -> seconds between requests, to stay under chosic's bot detection.

        Defaults: the chosic section of the settings (libcrate.settings).
        """
        self.timeout = timeout
        self.max_genres = max_genres
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._session = requests.Session(impersonate="chrome")
        self._session.headers.update(CHOSIC_HEADERS)
        self._handshaken = False
        self._artist_cache = {}  # artist_id -> [genres]

    # -- lifecycle -----------------------------------------------------------

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- public API ----------------------------------------------------------

    def get_genres(self, track_id):
        """Up to `max_genres` genres of the first artist on a Spotify track."""
        track_id = (track_id or "").strip()
        if not track_id:
            raise ChosicError("empty track ID")

        track = self._get_json("/tracks/" + quote(track_id, safe=""))
        artist_id = next(
            (
                (a.get("id") or "").strip()
                for a in track.get("artists") or []
                if (a.get("id") or "").strip()
            ),
            "",
        )
        if not artist_id:
            raise ChosicNotFound("chosic track response did not contain artist ID")

        return self._artist_genres(artist_id)

    # -- internals -----------------------------------------------------------

    def _artist_genres(self, artist_id):
        """Genres for an artist ID, deduped case-insensitively and capped. Cached."""
        if artist_id in self._artist_cache:
            return list(self._artist_cache[artist_id])

        artist = self._get_json("/artists/" + quote(artist_id, safe=""))
        genres, seen = [], set()
        for genre in artist.get("genres") or []:
            genre = (genre or "").strip()
            key = genre.lower()
            if not genre or key in seen:
                continue
            seen.add(key)
            genres.append(genre)
            if len(genres) >= self.max_genres:
                break

        if not genres:
            raise ChosicNotFound("chosic artist response did not contain genres")

        self._artist_cache[artist_id] = genres
        return list(genres)

    def _handshake(self):
        """Establish a session; runs before the first request and when the session expires."""
        self._throttle()
        try:
            resp = self._session.post(CHOSIC_TOOLS_API + "/handshake/", timeout=self.timeout)
        except requests.RequestsError as e:
            raise ChosicError(f"chosic handshake request failed: {e}") from e

        if resp.status_code != 200:
            raise ChosicError(f"chosic handshake failed: HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except Exception as e:
            raise ChosicError(f"failed to decode chosic handshake response: {e}") from e

        if not isinstance(payload, dict) or not payload.get("success"):
            raise ChosicError("chosic handshake returned unsuccessful response")

        self._handshaken = True

    def _get_json(self, path, _retry=True):
        if not self._handshaken:
            self._handshake()

        self._throttle()
        try:
            resp = self._session.get(CHOSIC_TOOLS_API + path, timeout=self.timeout)
        except requests.RequestsError as e:
            raise ChosicError(f"chosic request failed: {e}") from e

        if resp.status_code in _SESSION_EXPIRED and _retry:
            # Session cookie probably expired: re-handshake once and retry.
            self._handshaken = False
            return self._get_json(path, _retry=False)

        if resp.status_code == 404:
            raise ChosicNotFound(f"chosic has no entry for {path}")
        if resp.status_code != 200:
            body = resp.text[:512].strip()
            raise ChosicError(f"chosic request failed: HTTP {resp.status_code} ({body})")

        try:
            return resp.json()
        except Exception as e:
            raise ChosicError(f"failed to decode chosic response: {e}") from e

    def _throttle(self):
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
