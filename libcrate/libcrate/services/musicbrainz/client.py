"""Client for the MusicBrainz web service, by recording ID or ISRC.

    from libcrate.services.musicbrainz import MusicBrainzClient

    with MusicBrainzClient(timeout=20, max_genres=5, min_count=1, min_ratio=0.0,
                           use_artist=True, use_release_group=True) as mb:
        mb.recording_id_for_isrc("USEE10251052")
        mb.get_genres(recording_mbid="b1a9c0e9-...")  # by recording ID
        mb.get_genres(isrc="USEE10251052")            # or by ISRC
        mb.get_genres_detailed(isrc=...)              # {'recording': [...], 'release_group': [...], 'artist': [...]}

To fill a catalog, use libcrate.fetchers.genres.MusicBrainzGenreFetcher instead.

Available today:

  get_genres   community-voted genres: each has a `count` (net votes). The genres
               of the recording, its release group and its artist(s) are pooled,
               votes summed per genre (case-insensitively), and the ones with
               enough votes kept, most-voted first

New lookups go through _get(path, params), which handles the User-Agent, the
rate limit and the errors.

Requirements:
  - A User-Agent naming the app (MusicBrainz blocks anonymous clients):
    hardcoded as USER_AGENT.
  - Max 1 request/second — the client enforces this.
"""

import threading
import time
from urllib.parse import quote

from curl_cffi import requests

from libcrate.util import normalize_isrc

__all__ = ["MusicBrainzClient", "MusicBrainzError", "MusicBrainzNotFound"]

MB_API = "https://musicbrainz.org/ws/2"
MB_INC = "genres+artist-credits+release-groups"
USER_AGENT = "libcrate/0.1.0"


class MusicBrainzError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class MusicBrainzNotFound(MusicBrainzError):
    """No recording for this ID or ISRC."""


class MusicBrainzClient:
    def __init__(self, *, timeout: float, max_genres: int, min_count: int, min_ratio: float,
                 use_artist: bool, use_release_group: bool):
        """
        timeout             -> seconds per HTTP request.
        max_genres          -> cap on the returned list.
        min_count           -> drop genres with fewer votes than this.
        min_ratio           -> drop genres with < this fraction of the top genre's votes
                               (Picard's "minimal genre usage" default of 90% is 0.9;
                               0.0 keeps everything above min_count).
        use_artist / use_release_group -> whether to pool those genres in too.

        Defaults: the musicbrainz section of the settings (libcrate.settings).
        """
        self.timeout = timeout
        self.max_genres = max_genres
        self.min_count = min_count
        self.min_ratio = min_ratio
        self.use_artist = use_artist
        self.use_release_group = use_release_group

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        self._lock = threading.Lock()
        self._last_request = 0.0
        self._cache = {}

    # -- lifecycle -----------------------------------------------------------

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- public --------------------------------------------------------------

    def get_genres(self, recording_mbid=None, isrc=None):
        """Pooled genre names, most-voted first (see the constructor for the filters)."""
        detailed = self.get_genres_detailed(recording_mbid=recording_mbid, isrc=isrc)
        return [name for name, _ in self.pool(detailed)]

    def get_genres_detailed(self, recording_mbid=None, isrc=None):
        """{'recording': [(name, count)], 'release_group': [...], 'artist': [...]}"""
        if not recording_mbid and isrc:
            recording_mbid = self.recording_id_for_isrc(isrc)
        if not recording_mbid:
            raise MusicBrainzError("need recording_mbid or isrc")
        try:
            rec = self._get(f"/recording/{quote(recording_mbid)}", {"inc": MB_INC})
        except MusicBrainzError as e:
            if e.status_code == 404:
                raise MusicBrainzNotFound(
                    f"no MusicBrainz recording with ID {recording_mbid} "
                    "(release/album, artist and work IDs are not recording IDs)", 404) from e
            raise

        out = {"recording": _votes(rec), "release_group": [], "artist": []}

        # Release group: MusicBrainz lists one entry per release; pick the group
        # with the most votes (an album usually beats a single/compilation).
        rgs = {}
        for rel in rec.get("releases") or []:
            rg = rel.get("release-group") or {}
            if rg.get("id"):
                rgs[rg["id"]] = _votes(rg)
        if rgs:
            out["release_group"] = max(rgs.values(), key=lambda v: sum(c for _, c in v))

        seen = set()
        for credit in rec.get("artist-credit") or []:
            artist = credit.get("artist") or {}
            if artist.get("id") and artist["id"] not in seen:
                seen.add(artist["id"])
                out["artist"].extend(_votes(artist))
        return out

    def recording_id_for_isrc(self, isrc):
        """Resolve an ISRC to a recording MBID (the /isrc endpoint can't return genres itself)."""
        isrc = normalize_isrc(isrc)
        try:
            data = self._get(f"/isrc/{quote(isrc)}", {})
        except MusicBrainzError as e:
            if e.status_code == 404:
                raise MusicBrainzNotFound(f"no MusicBrainz recording for ISRC {isrc}", 404) from e
            raise
        recs = data.get("recordings") or []
        if not recs:
            raise MusicBrainzNotFound(f"no MusicBrainz recording for ISRC {isrc}", 404)
        return recs[0]["id"]

    def pool(self, detailed):
        """[(name, votes)] from get_genres_detailed() output, filtered and most-voted first."""
        totals = {}
        sources = [detailed["recording"]]
        if self.use_release_group:
            sources.append(detailed["release_group"])
        if self.use_artist:
            sources.append(detailed["artist"])
        for votes in sources:
            for name, count in votes:
                key = name.lower()
                totals[key] = (name, totals.get(key, (name, 0))[1] + count)

        if not totals:
            return []
        ranked = sorted(totals.values(), key=lambda nc: -nc[1])
        top = ranked[0][1]
        keep = [(n, c) for n, c in ranked if c >= self.min_count and c >= top * self.min_ratio]
        return keep[: self.max_genres]

    # -- internals -----------------------------------------------------------

    def _get(self, path, params):
        key = (path, tuple(sorted(params.items())))
        if key in self._cache:
            return self._cache[key]

        params = dict(params, fmt="json")
        for attempt in range(3):
            self._throttle()
            try:
                resp = self._session.get(MB_API + path, params=params, timeout=self.timeout)
            except requests.RequestsError as e:
                raise MusicBrainzError(f"musicbrainz request failed: {e}") from e

            if resp.status_code == 503 and attempt < 2:      # rate-limited: back off and retry
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 404:
                raise MusicBrainzNotFound(f"not found: {path}", 404)
            if resp.status_code != 200:
                raise MusicBrainzError(
                    f"musicbrainz request failed: HTTP {resp.status_code} ({resp.text[:300].strip()})",
                    resp.status_code,
                )
            try:
                data = resp.json()
            except ValueError as e:
                raise MusicBrainzError(f"bad JSON from musicbrainz: {e}") from e
            self._cache[key] = data
            return data
        raise MusicBrainzError("musicbrainz rate limit: gave up after retries", 503)

    def _throttle(self):
        with self._lock:
            wait = 1.0 - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()


def _votes(entity):
    """[(genre name, vote count)] for a recording/release-group/artist payload."""
    return [(g["name"], int(g.get("count") or 0)) for g in entity.get("genres") or [] if g.get("name")]
