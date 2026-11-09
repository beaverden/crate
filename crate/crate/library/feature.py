from libcrate import Catalog, EmbeddingTable, MetadataStore, Settings, features

__all__ = ["Unavailable", "slot_embeddings"]


class Unavailable(Exception):
    """A feature's slot can't be used: `title` and `message` for a notification."""

    def __init__(self, title: str, message: str):
        super().__init__(f"{title}: {message}")
        self.title, self.message = title, message


def slot_embeddings(store: MetadataStore, catalog: Catalog, settings: Settings, feature: str,
                    slot: str = "embeddings") -> tuple[str, EmbeddingTable]:
    """The embedding fetcher in `feature`'s `slot` and its rows made with its current settings (libcrate.features),
    whether the fetcher is on or not; raises Unavailable when the slot is empty or wrong or there are no rows yet,
    and OSError when they can't be read."""
    [found] = [s for s in features.slots(settings, feature) if s.name == slot]
    if found.fetcher is None:
        raise Unavailable(f"No embeddings for {found.label}", f"Choose them in Settings › {feature.title()}.")
    if problem := features.problem(settings, found):
        raise Unavailable(f"No embeddings for {found.label}", f"{problem[0].upper()}{problem[1:]}.")
    table = store.current_embeddings(catalog, settings, found.fetcher)
    if len(table) == 0:
        off = not settings.fetchers.get(found.fetcher).enabled
        raise Unavailable(f"No {found.fetcher} embeddings yet",
                          f"{f'Turn on {found.fetcher} in Settings › Fetchers, then press' if off else 'Press'} "
                          "Enrich in the library to make them.")
    return found.fetcher, table
