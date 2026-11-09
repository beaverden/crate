from libcrate.models.essentia.client import GENRES, EssentiaClient
from libcrate.models.essentia.models import GENRE_MODEL, RESAMPLE_QUALITY
from libcrate.fetchers.genres.base import GenreFetcher, LookupNotFound
from libcrate.track import Genre, TrackRecord
from libcrate.util import split_label

__all__ = ["EssentiaGenreFetcher"]


class EssentiaGenreFetcher(GenreFetcher):
    """Genres predicted from the audio by Essentia's discogs400 model (runs locally, no network).

    Names are lowercased Discogs styles ("deep house") with the parent genre ("Electronic") and the model's
    confidence; genres under min_confidence are dropped, and at most max_genres kept, most confident first. As
    many tracks are analysed at once as the options' processes, each in its own process.
    """

    name = "essentia-genres"

    @property
    def workers(self) -> int:
        return max(1, self.options.processes)

    def models(self) -> list[str]:
        return list(GENRES)

    def make_client(self) -> EssentiaClient:
        return EssentiaClient(self.settings.models.folder, GENRES, max_seconds=self.options.max_seconds,
                              processes=self.options.processes)

    def key(self, record: TrackRecord) -> str | None:
        # The audio is always there; the key records the model and settings, so
        # changing either re-analyses the library.
        o = self.options
        return f"{GENRE_MODEL}:resample={RESAMPLE_QUALITY}:min={o.min_confidence}:top={o.max_genres}"

    def fetch(self, record: TrackRecord) -> list[Genre]:
        scores = self.client.genres(self.root / record.path)
        genres = []
        for label, score in scores[: self.options.max_genres]:
            if score < self.options.min_confidence:
                break
            parent, style = split_label(label)
            genres.append(Genre(name=style.lower(), parent=parent, confidence=round(score, 4)))
        if not genres:
            top_label, top_score = scores[0]
            raise LookupNotFound(f"no genre above {self.options.min_confidence:.0%} "
                                 f"(best: {top_label} {top_score:.0%})")
        return genres
