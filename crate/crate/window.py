from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtGui import QCloseEvent, QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import QVBoxLayout
from qfluentwidgets import (
    DotInfoBadge, FluentIcon, InfoBar, InfoBarPosition, MSFluentWindow, NavigationItemPosition, Theme, isDarkTheme,
    setTheme, themeColor,
)

from crate.describe import DescribePage
from crate.icons import Icon
from crate.jobs import JobsPage
from crate.library import LibraryPage
from crate.library.catalog_menu import CatalogMenu
from crate.map import MapPage
from crate.player import Player, PlayerBar
from crate.radio import RadioPage
from crate.settings import SettingsPage
from crate.similar import SimilarPage
from crate.tasks import EnrichTask

__all__ = ["MainWindow"]

# QSettings keys: "light" or "dark" once the theme button was used (until then the system's mode);
# the player's shuffle and volume, kept between starts.
THEME_KEY = "theme"
SHUFFLE_KEY = "shuffle"
VOLUME_KEY = "volume"
# Room for the catalog's name under its icon in the navigation bar, in pixels.
CATALOG_NAME_WIDTH = 58
# The dots on the jobs item (a job running, one that had problems), and where they go on the item
# (the icon's top right), in pixels.
RUNNING_DOT = 8
RUNNING_DOT_AT = (38, 10)


class MainWindow(MSFluentWindow):
    """The app window: a navigation bar on the left, one page per section (the library, similar tracks, the map,
    describe, the radio, the jobs, the settings), the open catalog at the top of the bar (a click opens the catalog menu:
    recent catalogs, New, Open), a light/dark switch at the bottom of the bar, and the player bar under the pages."""

    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        setTheme({"light": Theme.LIGHT, "dark": Theme.DARK}.get(settings.value(THEME_KEY), Theme.AUTO))

        self.player = Player(parent=self)
        self.player.set_shuffle(settings.value(SHUFFLE_KEY, False, type=bool))
        self.player.set_volume(settings.value(VOLUME_KEY, 100, type=int))
        self.player.shuffle_changed.connect(lambda shuffle: settings.setValue(SHUFFLE_KEY, shuffle))
        self.player.volume_changed.connect(lambda volume: settings.setValue(VOLUME_KEY, volume))
        self.library = LibraryPage(settings, self)
        self.catalog_item = self.navigationInterface.addItem(
            "catalog", FluentIcon.FOLDER, "Catalog", self.show_catalog_menu, selectable=False,
            position=NavigationItemPosition.TOP)
        self.addSubInterface(self.library, FluentIcon.LIBRARY, "Library")
        self.similar = SimilarPage(self.library.covers, self)
        self.addSubInterface(self.similar, Icon.SIMILAR, "Similar")
        self.map = MapPage(self)
        self.addSubInterface(self.map, FluentIcon.GLOBE, "Map")
        self.describe = DescribePage(self.library.covers, self)
        self.addSubInterface(self.describe, FluentIcon.SEARCH, "Describe")
        self.radio = RadioPage(self.library.covers, self.player, self)
        self.addSubInterface(self.radio, Icon.RADIO, "Radio")
        self.jobs = JobsPage(self)
        self.jobs_item = self.addSubInterface(self.jobs, FluentIcon.HISTORY, "Jobs")
        # Dots at the top right of the jobs icon: while a scan or an enrich runs, and (red) after one
        # ended with problems, until the jobs page is opened.
        self.running_badge = DotInfoBadge.custom(themeColor(), themeColor(), self.jobs_item)
        self.problem_badge = DotInfoBadge.error(self.jobs_item)
        for badge in (self.running_badge, self.problem_badge):
            badge.setFixedSize(RUNNING_DOT, RUNNING_DOT)
            badge.move(*RUNNING_DOT_AT)
            badge.hide()
        self.stackedWidget.currentChanged.connect(self._on_page_changed)
        self.settings_page = SettingsPage(self)
        self.settings_page.set_store(self.library.store)  # the catalog reopened on start
        self.library.store_changed.connect(self.settings_page.set_store)
        self.library.store_changed.connect(self._show_store)
        self.addSubInterface(self.settings_page, FluentIcon.SETTING, "Settings", position=NavigationItemPosition.BOTTOM)
        self.navigationInterface.addItem("theme", FluentIcon.CONSTRACT, "Theme", self.toggle_theme,
                                         selectable=False, position=NavigationItemPosition.BOTTOM)

        self.player_bar = PlayerBar(self.player, self.library.covers)
        # The pages and the player bar under them, right of the navigation bar.
        self.hBoxLayout.removeWidget(self.stackedWidget)
        column = QVBoxLayout()
        column.setSpacing(0)
        column.addWidget(self.stackedWidget, 1)
        column.addWidget(self.player_bar)
        self.hBoxLayout.addLayout(column, 1)

        self.library.play_requested.connect(self.player.play_queue)
        self.library.similar_requested.connect(self.show_similar)
        self.library.map_requested.connect(self.show_in_map)
        self.library.radio_requested.connect(self.start_radio)
        self.library.task_started.connect(self._watch)
        for page in (self.similar, self.describe, self.radio):
            page.similar_requested.connect(self.show_similar)
            page.map_requested.connect(self.show_in_map)
            page.radio_requested.connect(self.start_radio)
        self.radio.play_requested.connect(self.player.play_queue)  # its list, once the station has ended
        self.similar.play_requested.connect(self.player.play_queue)
        self.map.play_requested.connect(self.player.play_queue)
        self.map.similar_requested.connect(self.show_similar)
        self.map.radio_requested.connect(self.start_radio)
        self.describe.play_requested.connect(self.player.play_queue)
        self.player.track_changed.connect(self.library.set_playing)
        self.player.track_changed.connect(self.similar.set_playing)
        self.player.track_changed.connect(self.describe.set_playing)
        self.player.failed.connect(lambda message: InfoBar.error(
            "Can't play", message, orient=Qt.Orientation.Vertical, duration=6000,
            position=InfoBarPosition.TOP_RIGHT, parent=self))
        # Space plays and pauses; text fields still get their spaces.
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.player.toggle)

        self.setWindowIcon(FluentIcon.MUSIC.icon())
        self.resize(1400, 900)
        self._show_store(self.library.store)

    def show_catalog_menu(self) -> None:
        """The catalog menu, beside the catalog's item in the navigation bar."""
        self.catalog_menu = CatalogMenu(self.library, self)
        self.catalog_menu.exec(self.catalog_item.mapToGlobal(QPoint(self.catalog_item.width() + 4, 0)))

    def _show_store(self, store) -> None:
        """The open catalog's name in the navigation bar and the window title."""
        name = store.folder.name if store is not None else ""
        metrics = QFontMetrics(self.catalog_item.font())
        self.catalog_item.setText(metrics.elidedText(name, Qt.TextElideMode.ElideRight, CATALOG_NAME_WIDTH)
                                  if name else "Catalog")
        self.catalog_item.setToolTip(str(store.folder) if store is not None else "No catalog open")
        self.setWindowTitle(f"Crate — {name}" if name else "Crate")

    def _watch(self, task) -> None:
        """Follow a task the library started on the jobs page, with a dot on its item while it runs;
        an enrich (long) takes the user there."""
        job = self.jobs.watch(task)
        self.problem_badge.hide()
        self.running_badge.show()
        self.jobs_item.setToolTip(f"{'Enrich' if isinstance(task, EnrichTask) else 'Scan'} running")
        task.finished.connect(lambda: self._job_done(job))
        if isinstance(task, EnrichTask):
            self.switchTo(self.jobs)

    def _job_done(self, job) -> None:
        self.running_badge.hide()
        self.jobs_item.setToolTip("")
        if job.has_problems and self.stackedWidget.currentWidget() is not self.jobs:
            self.problem_badge.show()
            self.jobs_item.setToolTip(f"{job.title}: {job.state}, with problems")

    def _on_page_changed(self, index: int) -> None:
        if self.stackedWidget.widget(index) is self.map:  # up to date with the catalog and its embeddings
            self.map.refresh(self.library.store, self.library.catalog)
        if self.stackedWidget.widget(index) is self.describe:
            self.describe.set_catalog(self.library.store, self.library.catalog)
        if self.stackedWidget.widget(index) is self.settings_page:  # how far each fetcher is on it
            self.settings_page.set_catalog(self.library.catalog)
        if self.stackedWidget.widget(index) is self.jobs:  # seen
            self.problem_badge.hide()
            if not self.running_badge.isVisible():
                self.jobs_item.setToolTip("")

    def show_similar(self, track) -> None:
        """Tracks similar to `track`, shown on the similar page."""
        if self.similar.find(self.library.store, self.library.catalog, track):
            self.switchTo(self.similar)

    def show_in_map(self, track) -> None:
        """`track` selected on the map page."""
        self.switchTo(self.map)  # brings the map up to date with the catalog
        self.map.show_track(track)

    def start_radio(self, seeds: list, name: str) -> None:
        """A radio station from `seeds`, played, and shown on the radio page."""
        if self.radio.start(self.library.store, self.library.catalog, seeds, name):
            self.switchTo(self.radio)

    def toggle_theme(self) -> None:
        dark = not isDarkTheme()
        setTheme(Theme.DARK if dark else Theme.LIGHT)
        self.settings.setValue(THEME_KEY, "dark" if dark else "light")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.player.release()
        self.library.shutdown()
        self.map.shutdown()
        self.settings_page.shutdown()
        self.describe.shutdown()
        super().closeEvent(event)
