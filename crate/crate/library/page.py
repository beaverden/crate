from pathlib import Path

from libcrate import CATALOG_FILE, Catalog, MetadataStore, RunReport, ScanReport, SettingsError, SourceStatus, plan
from pydantic import ValidationError
from PySide6.QtCore import QItemSelection, QModelIndex, QPoint, QSettings, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel, FluentIcon, InfoBar, InfoBarPosition, PrimaryPushButton, PushButton, RoundMenu,
    SearchLineEdit, SimpleCardWidget, SubtitleLabel, TableView, TitleLabel,
    qconfig,
)

from crate.icons import Icon
from crate.library.covers import THUMBNAILS_DIR, CoverCache
from crate.library.delegate import COVER, COVER_RADIUS, ROW_HEIGHT, TrackDelegate
from crate.library.details import TrackDetails
from crate.library.model import RIGHT_ALIGNED, SEARCH_ROLE, SORT_ROLE, TRACK_ROLE, TrackTableModel
from crate.library.new_catalog_dialog import NewCatalogDialog
from crate.library.scrolling import calm_scrolling
from crate.library.track_menu import track_menu
from crate.tasks import EnrichTask, ScanTask

__all__ = ["LibraryPage"]

# QSettings keys: the catalog folder opened last (reopened on start), and the music folder of the last
# new catalog (suggested for the next).
OUTPUT_FOLDER_KEY = "output_folder"
LIBRARY_KEY = "library_folder"
# Suggested folder for a new catalog.
DEFAULT_CATALOG_FOLDER = Path.home() / "music-metadata"
EMPTY = "Make a new catalog from a music folder, or open one made before."
# QSettings key: catalog folders opened or created, most recent first (the catalog menu lists them).
RECENT_KEY = "recent_catalogs"
MAX_RECENT = 8
DETAILS_WIDTH = 340
# How long a finished task's notification stays up, in ms.
NOTICE_MS = 6000


class LibraryPage(QWidget):
    """The catalog of one catalog folder: a searchable track table, the selected track's details,
    and Scan / Enrich running in the background (their progress, and Stop, are on the jobs page). Open picks a catalog folder
    (the catalog knows its music folder); New catalog makes one from a music folder. Without a
    catalog, the page offers both; otherwise the window's catalog menu does (recent_catalogs() too).

    Double-click (or Return) on a track asks for it to be played: play_requested(tracks, row, root)
    with the tracks in the table's current order, so the player's queue follows it. Similar (or a
    right-click on a track) asks for tracks similar to the selected one: similar_requested(track); the
    right-click menu also offers its place on the map and a radio station from it (track_menu).
    store_changed(store) when another catalog folder is opened; task_started(task) when a scan or an
    enrich starts (the jobs page follows it).
    """

    play_requested = Signal(list, int, str)
    similar_requested = Signal(object)
    map_requested = Signal(object)
    radio_requested = Signal(list, str)
    store_changed = Signal(object)
    task_started = Signal(object)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setObjectName("library")
        self.settings = settings
        self.store: MetadataStore | None = None
        self.catalog: Catalog | None = None
        self.task: ScanTask | EnrichTask | None = None

        self.model = TrackTableModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(SORT_ROLE)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterRole(SEARCH_ROLE)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        # --- header: title, what's open, actions, search
        self.title = TitleLabel("Library")
        self.summary = CaptionLabel()
        self.summary.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        self.search = SearchLineEdit()
        self.search.setPlaceholderText("Search artist, title, album, genre")
        self.search.setFixedWidth(300)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.scan_button = PushButton(FluentIcon.SYNC, "Scan")
        self.scan_button.clicked.connect(self.scan)
        self.enrich_button = PrimaryPushButton(FluentIcon.TAG, "Enrich")
        self.enrich_button.clicked.connect(self.enrich)
        self.similar_button = PushButton(Icon.SIMILAR, "Similar")
        self.similar_button.setToolTip("Similar-sounding tracks, from the selected one")
        self.similar_button.clicked.connect(self._show_similar)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(self.title)
        titles.addWidget(self.summary)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addLayout(titles)
        header.addStretch(1)
        header.addWidget(self.search)
        header.addSpacing(8)
        header.addWidget(self.similar_button)
        header.addSpacing(8)
        for button in (self.scan_button, self.enrich_button):
            header.addWidget(button)


        # --- body: table and details
        self.covers = CoverCache(COVER, COVER_RADIUS, self)
        self.table = TableView()
        self.table.setModel(self.proxy)
        self.table.setItemDelegate(TrackDelegate(self.table, self.covers))
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        calm_scrolling(self.table)
        self.covers.loaded.connect(self.table.viewport().update)
        qconfig.themeChanged.connect(lambda theme: self.table.viewport().update())  # the rows are painted in the theme's colours
        self.table.setBorderVisible(True)
        self.table.setBorderRadius(8)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        # Text columns share the width; numbers take what they need.
        for column, name in enumerate(TrackTableModel.COLUMNS):
            mode = QHeaderView.ResizeMode.ResizeToContents if name in RIGHT_ALIGNED else QHeaderView.ResizeMode.Stretch
            self.table.horizontalHeader().setSectionResizeMode(column, mode)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.selectionModel().selectionChanged.connect(self._on_selection)
        self.table.doubleClicked.connect(self._play)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_menu)
        play = QShortcut(QKeySequence(Qt.Key.Key_Return), self.table, context=Qt.ShortcutContext.WidgetShortcut)
        play.activated.connect(lambda: self._play(self.table.currentIndex()))

        self.details = TrackDetails()
        details_card = SimpleCardWidget()
        details_card.setFixedWidth(DETAILS_WIDTH)
        card_layout = QVBoxLayout(details_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addWidget(self.details)

        self.body = QWidget()
        body = QHBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(16)
        body.addWidget(self.table, 1)
        body.addWidget(details_card)

        # --- without a catalog: what to do instead of an empty table
        self.welcome = QWidget()
        self.new_button = PrimaryPushButton(FluentIcon.ADD, "New catalog")
        self.new_button.setToolTip("A catalog of a music folder: its settings, then a first scan")
        self.new_button.clicked.connect(self.new_catalog)
        self.open_button = PushButton(FluentIcon.FOLDER, "Open catalog")
        self.open_button.setToolTip("A catalog folder made before; its music folder comes with it")
        self.open_button.clicked.connect(self.choose_catalog)
        about = CaptionLabel(EMPTY)
        about.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        buttons.addStretch(1)
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        welcome = QVBoxLayout(self.welcome)
        welcome.addStretch(1)
        welcome.addWidget(SubtitleLabel("No catalog open"), 0, Qt.AlignmentFlag.AlignHCenter)
        welcome.addWidget(about, 0, Qt.AlignmentFlag.AlignHCenter)
        welcome.addSpacing(12)
        welcome.addLayout(buttons)
        welcome.addStretch(2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(self.body, 1)
        layout.addWidget(self.welcome, 1)

        self._update_actions()
        folder = self.settings.value(OUTPUT_FOLDER_KEY)
        if folder and (Path(folder) / CATALOG_FILE).is_file():
            self.open_catalog(Path(folder))

    # --- the catalog ------------------------------------------------------------

    def choose_catalog(self) -> None:
        """Ask for a catalog folder and open it."""
        start = str(self.store.folder if self.store else self.settings.value(OUTPUT_FOLDER_KEY, Path.home()))
        folder = QFileDialog.getExistingDirectory(self.window(), "Open a catalog", start)
        if folder:
            self.open_catalog(Path(folder))

    def open_catalog(self, folder: Path) -> bool:
        """Show the catalog in `folder` (on start: the one used last); its music folder comes with it.
        False (with a notification) if there's none."""
        try:
            store = MetadataStore(folder)
            catalog = store.load()
        except (OSError, ValidationError) as e:
            self._notify(InfoBar.error, "Can't open the catalog", f"{folder}: {e}")
            return False
        if catalog is None:
            self._notify(InfoBar.error, "No catalog there",
                         f"{store.folder} has no {CATALOG_FILE}: New catalog makes one from a music folder.")
            return False
        self._use(store)
        self._show_catalog(catalog)
        return True

    def new_catalog(self) -> None:
        """Ask for a music folder and a folder for its catalog, and create it."""
        library = self.settings.value(LIBRARY_KEY)
        dialog = NewCatalogDialog(Path(library) if library else None, DEFAULT_CATALOG_FOLDER, parent=self.window())
        if dialog.exec():
            self.create_catalog(dialog.library(), dialog.catalog_folder(), dialog.settings_from())

    def create_catalog(self, library: Path, folder: Path, settings_from: Path | None = None) -> None:
        """Start a catalog of `library` in `folder`, as `MetadataStore.init` does: its settings (the
        defaults, or copied from `settings_from`, another catalog folder or settings file), then
        the first scan; shown once scanned."""
        try:
            store = MetadataStore(folder)
            exists = store.catalog_path.is_file()
        except OSError as e:
            self._notify(InfoBar.error, "Can't create the catalog", f"{folder}: {e}")
            return
        if exists:
            self._notify(InfoBar.error, "There's a catalog there already", f"{store.folder}: Open it instead.")
            return
        try:
            store.create_settings(settings_from)
        except (SettingsError, OSError) as e:
            self._notify(InfoBar.error, "Can't create the catalog", str(e))
            return
        library = library.expanduser().resolve()
        self.settings.setValue(LIBRARY_KEY, str(library))
        self._use(store)
        self._show_catalog(None)
        self._scan(library)

    def recent_catalogs(self) -> list[Path]:
        """Catalog folders opened or created, most recent first; those still holding a catalog."""
        folders = [Path(f) for f in self.settings.value(RECENT_KEY, [], type=list)]
        return [f for f in folders if (f / CATALOG_FILE).is_file()]

    def _use(self, store: MetadataStore) -> None:
        self.store = store
        self.store_changed.emit(store)
        self.settings.setValue(OUTPUT_FOLDER_KEY, str(store.folder))
        recent = [str(store.folder)] + [f for f in self.settings.value(RECENT_KEY, [], type=list)
                                        if f != str(store.folder)]
        self.settings.setValue(RECENT_KEY, recent[:MAX_RECENT])

    def _show_catalog(self, catalog: Catalog | None) -> None:
        """Show `catalog`, keeping the selected track selected."""
        selected = self._selected_track()
        self.catalog = catalog
        self.covers.set_library(catalog.root if catalog else None, self.store.folder / THUMBNAILS_DIR)
        if catalog is not None:
            self.covers.prune(catalog)
        self.model.set_catalog(catalog)
        if catalog is None:
            self.summary.setText(f"Creating the catalog in {self.store.folder}…")
        else:
            self.summary.setText(f"{len(catalog.tracks):,} tracks  ·  {catalog.root}")
        if selected is not None and not self._select(selected.path):
            self.details.show_track(None)
        self._update_actions()

    def _selected_track(self):
        rows = self.table.selectionModel().selectedRows()
        return rows[0].data(TRACK_ROLE) if rows else None

    def _select(self, path: str) -> bool:
        """Select the track at `path` and scroll it into view; False if the table doesn't show it."""
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            if index.data(TRACK_ROLE).path == path:
                self.table.selectRow(row)
                self.table.scrollTo(index)  # only scrolls when it's out of view
                return True
        return False

    def _play(self, index: QModelIndex) -> None:
        if index.isValid():
            tracks = [self.proxy.index(row, 0).data(TRACK_ROLE) for row in range(self.proxy.rowCount())]
            self.play_requested.emit(tracks, index.row(), self.catalog.root)

    def _show_similar(self) -> None:
        track = self._selected_track()
        if track is not None:
            self.similar_requested.emit(track)

    def _show_menu(self, pos: QPoint) -> None:
        index = self.table.indexAt(pos)
        if index.isValid():
            self.table.selectRow(index.row())
            self.track_menu(index).exec(self.table.viewport().mapToGlobal(pos))

    def track_menu(self, index: QModelIndex) -> RoundMenu:
        return track_menu(self, index, self._play)

    def set_playing(self, track) -> None:
        """Mark `track` (or nothing) as the one playing: its title in the theme colour, its row selected
        and scrolled into view, its details in the side panel (even when the search hides its row)."""
        self.table.itemDelegate().playing = track.path if track is not None else None
        self.table.viewport().update()
        if track is not None and not self._select(track.path):
            self.table.clearSelection()
            self.details.show_track(track, self.catalog.root if self.catalog else None)

    def _on_selection(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        self.details.show_track(self._selected_track(), self.catalog.root if self.catalog else None)
        self.similar_button.setEnabled(self._selected_track() is not None)

    # --- background tasks -----------------------------------------------------

    def scan(self) -> None:
        """Read new and changed files of the open catalog's library."""
        if self.catalog is not None:
            self._scan(None)

    def _scan(self, library: Path | None) -> None:
        """Scan the open catalog's library, or with `library`, create the catalog from it."""
        task = ScanTask(self.store, library, self)
        task.finished_ok.connect(lambda catalog, report: self._on_scan_done(catalog, report, created=library is not None))
        task.failed.connect(lambda message: self._on_scan_failed(library is not None))
        self._start(task)

    def _on_scan_done(self, catalog: Catalog, report: ScanReport, created: bool = False) -> None:
        self._show_catalog(catalog)
        parts = [f"{len(report.added)} added", f"{len(report.updated)} updated", f"{len(report.removed)} removed",
                 f"{report.unchanged} unchanged"]
        if report.failed:
            parts.append(f"{len(report.failed)} failed")
        kind = InfoBar.warning if report.failed else InfoBar.success
        if created:
            self._notify(kind, "Catalog created", f"{', '.join(parts)}.\nNext: turn on the fetchers you want in "
                         "Settings › Fetchers, then Enrich.", duration=-1)
        else:
            self._notify(kind, "Scan done", ", ".join(parts))

    def _on_scan_failed(self, creating: bool) -> None:
        if creating and self.catalog is None:  # back to the welcome; the failure says why
            self.store = None
            self.store_changed.emit(None)
            self.summary.setText("")

    def enrich(self) -> None:
        """Run the fetchers the catalog's settings turn on (Settings › Fetchers), downloading their models first."""
        try:
            chosen = plan(self.store.settings())
        except SettingsError as e:
            self._notify(InfoBar.error, "Can't read the settings", str(e), duration=-1)
            return
        if not chosen:
            self._notify(InfoBar.warning, "Nothing chosen to run", "Turn on the fetchers you want in Settings › Fetchers.")
            return
        task = EnrichTask(self.store, self.catalog, parent=self)
        task.finished_ok.connect(self._on_enrich_done)
        self._start(task)

    def _on_enrich_done(self, catalog: Catalog, report: RunReport) -> None:
        self._show_catalog(catalog)
        parts, gave_up = [f"downloaded {name}" for name in report.downloaded], {}
        enriched, embedded = report.enriched, report.embedded
        if enriched is not None:
            parts += [f"{source}: {counts[SourceStatus.FOUND]} found, {counts[SourceStatus.NOT_FOUND]} not found, "
                      f"{counts[SourceStatus.ERROR]} errors" for source, counts in enriched.looked_up.items()]
            gave_up |= enriched.gave_up
        if embedded is not None:
            parts += [f"{model}: {embedded.computed.get(model, 0)} computed, "
                      f"{len(embedded.failed.get(model, []))} errors" for model in embedded.computed]
            gave_up |= embedded.gave_up
        title = "Enrich stopped, progress saved" if report.interrupted else "Enrich done"
        self._notify(InfoBar.success, title, "\n".join(parts) or "Nothing to do")
        if report.not_run:
            self._notify(InfoBar.warning, "Not run: they aren't set up", "\n".join(
                f"{name}: {reason}" for name, reason in report.not_run.items()), duration=-1)
        if gave_up:
            self._notify(InfoBar.warning, "Stopped: their first tracks all failed",
                         "\n".join(f"{name}: {reason}" for name, reason in gave_up.items()), duration=-1)

    def shutdown(self) -> None:
        """Let a running task finish (an enrich stops after the lookups in flight and saves) and stop
        loading covers; for closing."""
        self.covers.shutdown()
        if self.task is not None:
            if isinstance(self.task, EnrichTask):
                self.task.stop()
            self.task.wait()

    def _start(self, task: ScanTask | EnrichTask) -> None:
        self.task = task
        task.failed.connect(lambda message: self._notify(InfoBar.error, "Failed", message, duration=-1))
        task.finished.connect(self._on_task_finished)
        self._update_actions()
        self.task_started.emit(task)
        task.start()


    def _on_task_finished(self) -> None:
        self.task.deleteLater()
        self.task = None
        self._update_actions()

    def _update_actions(self) -> None:
        idle = self.task is None
        # Nothing open (and nothing being created): the welcome instead of an empty table.
        welcome = idle and self.catalog is None
        self.welcome.setVisible(welcome)
        self.body.setVisible(not welcome)
        self.scan_button.setEnabled(idle and self.catalog is not None)
        self.enrich_button.setEnabled(idle and self.catalog is not None)
        self.similar_button.setEnabled(self._selected_track() is not None)

    def _notify(self, kind, title: str, content: str, duration: int = NOTICE_MS) -> None:
        kind(title, content, orient=Qt.Orientation.Vertical, duration=duration,
             position=InfoBarPosition.TOP_RIGHT, parent=self.window())
