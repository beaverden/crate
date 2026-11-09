from libcrate import Catalog, MetadataStore, SettingsError, TextSearch, fuse, make_fetcher
from libcrate.models.clap import ClapClient, ClapError
from qfluentwidgets import InfoBar, PushButton, SearchLineEdit

from crate.describe.task import DescribeTask
from crate.icons import Icon
from crate.library.covers import CoverCache
from crate.library.feature import Unavailable, slot_embeddings
from crate.library.match_page import MatchPage
from crate.library.model import TRACK_ROLE

__all__ = ["DescribePage"]

EMPTY = "Describe what you'd like to hear, and the tracks that sound like it are listed, best match first."
# A description's vector needs a model's text side only, so one process per model does.
PROCESSES = 1
# A station from a description starts from its best matches: the first plays first, all of them set the course.
RADIO_SEEDS = 10


class DescribePage(MatchPage):
    """The tracks that sound like a description ("chill late-night electronic"), best match first (libcrate.TextSearch,
    the describe section of the settings): matched against their sound by a CLAP embedding fetcher (describe.sound)
    and, once the tracks' own descriptions have embeddings (describe.words, optional), against those too, the two
    rankings fused (libcrate.fuse). The description is embedded by each fetcher's own model.

    set_catalog(store, catalog) picks the catalog; Return in the box describes. Each model runs in a child process,
    loaded on the first description (several seconds) and kept until shutdown() or the settings change. Start radio
    asks for a station from the best matches: radio_requested(tracks, name).
    """

    def __init__(self, covers: CoverCache, parent=None):
        super().__init__("describe", "Describe", EMPTY, covers, parent)
        self.store: MetadataStore | None = None
        self.catalog: Catalog | None = None
        self.clients: dict[str, ClapClient] = {}  # model -> its running client
        self.task: DescribeTask | None = None
        self._pending: tuple | None = None  # what the running task is for: (text, searches, their names, count)

        self.box = SearchLineEdit()
        self.box.setPlaceholderText("chill late-night electronic, with a warm bassline")
        self.box.setMinimumWidth(380)
        self.box.returnPressed.connect(self.describe)
        self.box.searchSignal.connect(lambda text: self.describe())
        self.header.insertWidget(1, self.box)
        self.description = ""  # what the list matches
        self.radio_button = PushButton(Icon.RADIO, "Start radio")
        self.radio_button.setToolTip("An endless station from the best matches")
        self.radio_button.clicked.connect(self._start_radio)
        self.header.insertWidget(2, self.radio_button)
        self._update_actions()

    def set_catalog(self, store: MetadataStore | None, catalog: Catalog | None) -> None:
        self.store, self.catalog = store, catalog

    def describe(self) -> bool:
        """List the tracks that sound like the box's text; False (with a notification) when it can't: no catalog,
        unreadable settings, no embeddings yet, or no model. The list comes when the description's vector does."""
        text = self.box.text().strip()
        if not text or self.task is not None:
            return False
        if self.store is None or self.catalog is None:
            self._notify(InfoBar.warning, "No catalog open", "Open a catalog in the library first.")
            return False
        try:
            settings = self.store.settings()
        except SettingsError as e:
            self._notify(InfoBar.error, "Can't read the settings", str(e))
            return False
        try:
            sound, table = slot_embeddings(self.store, self.catalog, settings, "describe", "sound")
            searches = {sound: TextSearch(self.catalog, table)}
            try:
                words, table = slot_embeddings(self.store, self.catalog, settings, "describe", "words")
                searches[words] = TextSearch(self.catalog, table)
            except Unavailable:
                pass  # the words are a bonus: without them, the sound alone
        except Unavailable as e:
            self._notify(InfoBar.warning, e.title, e.message)
            return False
        except OSError as e:
            self._notify(InfoBar.error, "Can't read the embeddings", str(e))
            return False
        try:
            clients = self._clients(settings.models.folder, [make_fetcher(n, settings).model for n in searches])
        except ClapError as e:
            self._notify(InfoBar.warning, "The model isn't there", f"{e}.")
            return False
        self._pending = (text, list(searches.values()), list(searches), settings.describe.count)
        self.summary.setText(f"Matching “{text}”…")
        self.box.setEnabled(False)
        self.task = DescribeTask(clients, text, self)
        self.task.finished_ok.connect(self._on_vectors)
        self.task.failed.connect(lambda message: self._notify(InfoBar.error, "Couldn't describe", message))
        self.task.finished.connect(self._on_task_finished)
        self.task.start()
        return True

    def _clients(self, models, names: list[str]) -> list[ClapClient]:
        """A client per model of `names`: the running ones, new ones for the others; the clients of models no longer
        used (or in another folder) are stopped."""
        for name, client in list(self.clients.items()):
            if name not in names or client.folder != models / name:
                client.close()
                del self.clients[name]
        for name in names:
            if name not in self.clients:
                self.clients[name] = ClapClient(models, name, processes=PROCESSES)
        return [self.clients[name] for name in names]

    def _on_vectors(self, vectors: list) -> None:
        text, searches, names, count = self._pending
        if len(searches) == 1:
            items = searches[0].find(vectors[0], count=count)
        else:
            items = fuse([search.find(v, count=len(search)) for search, v in zip(searches, vectors)], count=count)
        self.description = text
        self.show_matches(items, self.catalog.root, f"“{text}”  ·  {len(items)} tracks  ·  {' + '.join(names)}")

    def _start_radio(self) -> None:
        seeds = [self.model.index(row, 0).data(TRACK_ROLE) for row in range(min(RADIO_SEEDS, self.model.rowCount()))]
        if seeds:
            self.radio_requested.emit(seeds, f"“{self.description}”")

    def _update_actions(self) -> None:
        super()._update_actions()
        if hasattr(self, "radio_button"):  # MatchPage's __init__ calls this first
            self.radio_button.setEnabled(self.model.rowCount() > 0)

    def _on_task_finished(self) -> None:
        self.task = None
        self.box.setEnabled(True)
        if self.model.rowCount() == 0:
            self.summary.setText(EMPTY)

    def shutdown(self) -> None:
        """Wait for a description in progress and stop the models' processes; for closing."""
        if self.task is not None:
            self.task.wait()
        for client in self.clients.values():
            client.close()
