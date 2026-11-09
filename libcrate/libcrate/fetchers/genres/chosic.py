from libcrate.services.chosic.client import ChosicClient, ChosicNotFound
from libcrate.fetchers.genres.base import GenreFetcher, LookupNotFound
from libcrate.track import Genre, TrackRecord

__all__ = ["ChosicGenreFetcher"]


class ChosicGenreFetcher(GenreFetcher):
    """Genres of the track's first artist, looked up by Spotify track ID."""

    name = "chosic-genres"

    def services(self) -> list[str]:
        return ["chosic"]

    def make_client(self) -> ChosicClient:
        service = self.settings.services.chosic
        return ChosicClient(timeout=service.timeout, max_genres=self.options.max_genres,
                            min_interval=service.min_interval)

    def key(self, record: TrackRecord) -> str | None:
        return f"spotify:{record.tags.spotify_track_id}" if record.tags.spotify_track_id else None

    def fetch(self, record: TrackRecord) -> list[Genre]:
        try:
            return [Genre(name=g) for g in self.client.get_genres(record.tags.spotify_track_id)]
        except ChosicNotFound as e:
            raise LookupNotFound(str(e)) from e
