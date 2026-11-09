from libcrate.services.musicbrainz.client import MusicBrainzClient, MusicBrainzNotFound
from libcrate.fetchers.genres.base import GenreFetcher, LookupNotFound
from libcrate.track import Genre, TrackRecord
from libcrate.util import normalize_isrc

__all__ = ["MusicBrainzGenreFetcher"]


class MusicBrainzGenreFetcher(GenreFetcher):
    """Community-voted genres, looked up by MusicBrainz recording ID, then ISRC."""

    name = "musicbrainz-genres"

    def services(self) -> list[str]:
        return ["musicbrainz"]

    def make_client(self) -> MusicBrainzClient:
        o = self.options
        return MusicBrainzClient(timeout=self.settings.services.musicbrainz.timeout, max_genres=o.max_genres,
                                 min_count=o.min_count, min_ratio=o.min_ratio, use_artist=o.use_artist,
                                 use_release_group=o.use_release_group)

    def key(self, record: TrackRecord) -> str | None:
        if record.tags.musicbrainz_track_id:
            return f"mbid:{record.tags.musicbrainz_track_id.lower()}"
        if record.tags.isrc:
            return f"isrc:{normalize_isrc(record.tags.isrc)}"
        return None

    def fetch(self, record: TrackRecord) -> list[Genre]:
        try:
            detailed = self.client.get_genres_detailed(
                recording_mbid=record.tags.musicbrainz_track_id, isrc=record.tags.isrc)
        except MusicBrainzNotFound as e:
            raise LookupNotFound(str(e)) from e
        pooled = self.client.pool(detailed)
        if not pooled:
            raise LookupNotFound("recording has no genre votes on MusicBrainz")
        return [Genre(name=name, votes=votes) for name, votes in pooled]
