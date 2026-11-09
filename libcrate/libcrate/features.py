"""The features that use the fetchers' embeddings, and which ones they use: each has slots, filled in its settings
section with an embedding fetcher.

    similar    embeddings   any that compares tracks by their sound (required)
    map        embeddings   the same
    radio      embeddings   the same
    describe   sound        a CLAP model's, matched against the description by the same model (required)
               words        a text model's of the tracks' descriptions (optional)

    slots(settings)                                   # every slot and the fetcher in it
    problem(settings, slot)                           # why it can't be used: empty, or the wrong kind of fetcher
    note(settings, slot)                              # its fetcher is off: what it has isn't kept up to date
    used_by(settings, "description-bge")              # ["describe (words)"]
    current(table, catalog, settings, "muq-mulan")    # only the rows made with the fetcher's current settings

A feature only uses rows made with its fetcher's current settings: after a change (another model revision,
other clips), the old rows are another model's vectors, which a query embedded by the current model can't be
compared with. They count as missing until the next Enrich redoes them.
"""

from dataclasses import dataclass

from libcrate.catalog import Catalog
from libcrate.embeddings import EmbeddingTable
from libcrate.fetchers.embeddings.base import EmbeddingFetcher
from libcrate.fetchers.registry import FETCHERS, make_fetcher
from libcrate.settings import Settings

__all__ = ["FEATURES", "Slot", "current", "note", "problem", "slots", "used_by"]

# The embedding spaces each slot takes (EmbeddingFetcher.space).
_TRACKS = ("audio", "clap")
# feature -> its slots: (setting, the spaces it takes, required).
FEATURES: dict[str, tuple[tuple[str, tuple[str, ...], bool], ...]] = {
    "similar": (("embeddings", _TRACKS, True),),
    "map": (("embeddings", _TRACKS, True),),
    "describe": (("sound", ("clap",), True), ("words", ("text",), False)),
    "radio": (("embeddings", _TRACKS, True),),
}


@dataclass(frozen=True)
class Slot:
    feature: str
    name: str                 # its setting: "embeddings", "sound", "words"
    fetcher: str | None       # the embedding fetcher in it (None: empty)
    spaces: tuple[str, ...]   # what it takes
    required: bool

    @property
    def label(self) -> str:
        return self.feature if self.name == "embeddings" else f"{self.feature} ({self.name})"


def slots(settings: Settings, feature: str | None = None) -> list[Slot]:
    """Every feature's slots (or `feature`'s), with the fetchers the settings put in them."""
    return [Slot(name, setting, getattr(getattr(settings, name), setting), spaces, required)
            for name, feature_slots in FEATURES.items() if feature in (None, name)
            for setting, spaces, required in feature_slots]


def problem(settings: Settings, slot: Slot) -> str | None:
    """Why the slot's fetcher can't be used: none when the slot is optional and empty."""
    if slot.fetcher is None:
        return f"choose the {slot.name} for {slot.feature}" if slot.required else None
    cls = FETCHERS.get(slot.fetcher)
    if cls is None or not issubclass(cls, EmbeddingFetcher) or cls.space not in slot.spaces:
        return f"the {slot.name} of {slot.feature} can't be {slot.fetcher}"
    return None


def note(settings: Settings, slot: Slot) -> str | None:
    """The slot's fetcher is off: the embeddings it made can still be used, but new or changed tracks get none."""
    if slot.fetcher is None or problem(settings, slot) or settings.fetchers.get(slot.fetcher).enabled:
        return None
    return f"Enrich doesn't keep the {slot.fetcher} embeddings up to date while it's off: turn it on in Fetchers"


def used_by(settings: Settings, fetcher: str) -> list[str]:
    """The features and the other enabled fetchers that use `fetcher`'s results: "describe (words)", "llm-descriptions"."""
    users = [slot.label for slot in slots(settings) if slot.fetcher == fetcher]
    for name in FETCHERS:
        if settings.fetchers.get(name).enabled and any(
                put.fetcher == fetcher for put in make_fetcher(name, settings).inputs()):
            users.append(name)
    return users


def current(table: EmbeddingTable, catalog: Catalog, settings: Settings, fetcher: str) -> EmbeddingTable:
    """The rows of `table` (`fetcher`'s embeddings) for `catalog`'s tracks made with the fetcher's current settings."""
    made = make_fetcher(fetcher, settings, catalog.root)
    out = EmbeddingTable()
    for path, record in catalog.tracks.items():
        key = table.key(path)
        if key is not None and key == made.key(record):
            out.put(path, key, table.get(path))
    return out
