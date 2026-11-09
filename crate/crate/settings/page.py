from pathlib import Path

from libcrate import (
    FETCHERS, KINDS, Catalog, EmbeddingFetcher, MetadataStore, Settings, SettingsError, about, check, features,
    fetcher_names, make_fetcher, models,
)
from libcrate.settings import LLMService, ServicesSettings, load_settings, save_settings
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel, FluentIcon, InfoBar, InfoBarPosition, Pivot, PrimaryPushButton, PushButton, ScrollArea,
    SettingCard, StrongBodyLabel, TitleLabel,
)

from crate.icons import Icon
from crate.settings.editors import Editor, editor_for
from crate.settings.fetcher_section import GREY, RED, FetcherSection
from crate.settings.models_list import ModelsList
from crate.settings.progress_task import ProgressTask

__all__ = ["SettingsPage"]

# The tabs, their titles and their cards' icons; a feature's tab is its section of the settings.
TABS = {
    "fetchers": ("Fetchers", FluentIcon.TAG),
    "services": ("Services", FluentIcon.GLOBE),
    "models": ("Models", FluentIcon.IOT),
    "similar": ("Similar", Icon.SIMILAR),
    "map": ("Map", FluentIcon.GLOBE),
    "describe": ("Describe", FluentIcon.SEARCH),
    "radio": ("Radio", Icon.RADIO),
}
KIND_TITLES = {"genres": "Genres", "embeddings": "Embeddings", "descriptions": "Descriptions"}
NOTICE_MS = 5000
# An editor's value that didn't parse, left out of the settings it's checked with.
_INVALID = object()


def _choices() -> dict[tuple[str, ...], tuple[str, ...]]:
    """The settings that name one of a few things (a dropdown each), by their path in the file."""
    llm = tuple(name for name, field in ServicesSettings.model_fields.items() if field.annotation is LLMService)

    def embeddings(spaces) -> tuple[str, ...]:
        return tuple(name for name, cls in FETCHERS.items() if issubclass(cls, EmbeddingFetcher) and cls.space in spaces)

    choices = {
        ("fetchers", "genres", "llm-genres", "service"): llm,
        ("fetchers", "descriptions", "llm-descriptions", "service"): llm,
        ("fetchers", "descriptions", "llm-descriptions", "audio_tags_from"): embeddings(("clap",)),
    }
    for feature, slots in features.FEATURES.items():
        for setting, spaces, _ in slots:
            choices[(feature, setting)] = embeddings(spaces)
    return choices


CHOICES = _choices()


def _title(key: str) -> str:
    """'max_genres' -> 'Max genres', 'audio_tags_from' -> 'Audio tags from', 'enabled' -> 'Run on Enrich'."""
    if key == "enabled":
        return "Run on Enrich"
    return key.replace("_", " ").capitalize()


def _is_section(field: FieldInfo) -> bool:
    return isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)


def _sentence(text: str) -> str:
    return f"{text[0].upper()}{text[1:]}"


def _column(name: str) -> tuple[ScrollArea, QVBoxLayout]:
    """A transparent scrollable column, and its layout to fill."""
    page = QWidget()
    page.setObjectName(f"settings-{name}")
    page.setStyleSheet(f"#settings-{name} {{background: transparent}}")
    column = QVBoxLayout(page)
    column.setContentsMargins(0, 8, 12, 8)
    column.setSpacing(4)
    scroll = ScrollArea()
    scroll.setWidget(page)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setStyleSheet("QScrollArea {background: transparent; border: none}")
    return scroll, column


def _caption(text: str, colors=GREY) -> CaptionLabel:
    label = CaptionLabel(text)
    label.setTextColor(*colors)
    label.setWordWrap(True)
    return label


class SettingsPage(QWidget):
    """The open catalog's settings file (settings.yaml), with each setting's description from libcrate:

      Fetchers   a sub-tab per kind (genres, embeddings, descriptions), a section per fetcher: what it does, whether
                 Enrich runs it, why it can't (no API key, a fetcher it needs that's off), what it downloads first,
                 what it does without, what uses it, how far it is on the catalog, then its options
      Services   the API keys, the language model, MusicBrainz and chosic
      Models     the models folder, and every model: downloaded or not, what uses it, Download and Delete
      Similar, Map, Describe, Radio
                 the features, each with the embeddings it uses and why it can't use them

    What a fetcher or feature says follows the edits as they're made; how far a fetcher is follows the saved
    settings and the catalog (set_catalog()), worked out in the background. set_store() picks the catalog; without
    one the defaults are shown and nothing can be edited. Saving rewrites the file with its comments; the next
    Enrich uses it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settings")
        self.store: MetadataStore | None = None
        self.path: Path | None = None  # the catalog's settings.yaml
        self.catalog: Catalog | None = None
        self.editors: dict[tuple[str, ...], Editor] = {}
        self._saved: dict[tuple[str, ...], object] = {}
        self._unusable = False  # the file couldn't be read: saving replaces it
        self.progress: dict = {}  # fetcher -> FetcherProgress, with the saved settings
        self.progress_task: ProgressTask | None = None
        self._progress_stale = False

        title = TitleLabel("Settings")
        self.location = _caption("")
        self.problem = _caption("", RED)
        self.problem.hide()
        self.revert_button = PushButton(FluentIcon.CANCEL, "Revert")
        self.revert_button.clicked.connect(self.load)
        self.save_button = PrimaryPushButton(FluentIcon.SAVE, "Save")
        self.save_button.clicked.connect(self.save)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(title)
        titles.addWidget(self.location)
        header = QHBoxLayout()
        header.addLayout(titles, 1)
        header.addWidget(self.revert_button)
        header.addWidget(self.save_button)

        self.sections: dict[str, FetcherSection] = {}
        self.slot_problems: dict[str, CaptionLabel] = {}
        self.models_list = ModelsList()
        self.models_list.downloaded.connect(lambda names: self._notify(
            InfoBar.success, "Downloaded", f"{', '.join(names)}, into {self.models_list.folder}."))
        self.models_list.failed.connect(lambda message: self._notify(InfoBar.error, "Download failed", message,
                                                                     duration=-1))
        self.models_list.deleted.connect(lambda name: self._refresh_status())
        self.models_list.downloaded.connect(lambda names: self._refresh_status())

        self.tabs = Pivot()
        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {
            "fetchers": self._fetchers_tab(), "services": self._services_tab(), "models": self._models_tab(),
            **{name: self._feature_tab(name) for name in features.FEATURES},
        }
        for name, page in self.pages.items():
            self.stack.addWidget(page)
            self.tabs.addItem(name, TABS[name][0])
        # By the tab's key: Pivot calls an item's onClick with a `checked` argument, which a
        # `lambda i=index` default would take for the index.
        self.tabs.currentItemChanged.connect(lambda name: self.stack.setCurrentWidget(self.pages[name]))
        self.tabs.setCurrentItem("fetchers")
        for editor in self.editors.values():
            editor.on_change(self._on_edit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self.problem)
        layout.addWidget(self.tabs, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.stack, 1)
        self.set_store(None)

    # --- building ---------------------------------------------------------------

    def _cards(self, path: tuple[str, ...], section: type[BaseModel], icon) -> list[SettingCard]:
        """A card per setting of `section` (at `path` in the file), its editor in self.editors."""
        cards = []
        for name, field in section.model_fields.items():
            if _is_section(field):
                continue
            key = path + (field.alias or name,)
            editor = editor_for(field, CHOICES.get(key))
            card = SettingCard(icon, _title(name), _sentence(field.description))
            card.hBoxLayout.addWidget(editor.widget, 0, Qt.AlignmentFlag.AlignRight)
            card.hBoxLayout.addSpacing(16)
            card.setFixedHeight(max(70, editor.widget.minimumHeight() + 24))  # room for a taller input
            self.editors[key] = editor
            cards.append(card)
        return cards

    def _fetchers_tab(self) -> QWidget:
        """A sub-tab per kind, a FetcherSection per fetcher."""
        tab = QWidget()
        kinds, stack = Pivot(), QStackedWidget()
        pages = {}
        groups = Settings().fetchers
        for kind in KINDS:
            scroll, column = _column(f"fetchers-{kind}")
            group = getattr(groups, kind)
            column.addWidget(_caption(_sentence(about(type(group))) + "."))
            for name in fetcher_names(kind):
                options = type(groups.get(name))
                section = FetcherSection(name, about(options), self._cards(("fetchers", kind, name), options,
                                                                           TABS["fetchers"][1]))
                self.sections[name] = section
                column.addWidget(section)
            column.addStretch(1)
            pages[kind] = scroll
            stack.addWidget(scroll)
            kinds.addItem(kind, KIND_TITLES[kind])
        kinds.currentItemChanged.connect(lambda kind: stack.setCurrentWidget(pages[kind]))
        kinds.setCurrentItem(KINDS[0])
        self.fetcher_tabs = kinds
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(kinds, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(stack, 1)
        return tab

    def _heading(self, column: QVBoxLayout, title: str, section: type[BaseModel]) -> None:
        column.addSpacing(10)
        column.addWidget(StrongBodyLabel(title))
        column.addWidget(_caption(_sentence(about(section)) + "."))
        column.addSpacing(4)

    def _services_tab(self) -> QWidget:
        scroll, column = _column("services")
        defaults = Settings()
        self._heading(column, "API keys", type(defaults.keys))
        for card in self._cards(("keys",), type(defaults.keys), FluentIcon.FINGERPRINT):
            column.addWidget(card)
        for name in ServicesSettings.model_fields:
            section = type(getattr(defaults.services, name))
            self._heading(column, name, section)
            for card in self._cards(("services", name), section, TABS["services"][1]):
                column.addWidget(card)
        column.addStretch(1)
        return scroll

    def _models_tab(self) -> QWidget:
        scroll, column = _column("models")
        section = type(Settings().models)
        column.addWidget(_caption(_sentence(about(section)) + ". Download one here beforehand, or delete one to "
                                  "free the space."))
        column.addSpacing(6)
        for card in self._cards(("models",), section, FluentIcon.FOLDER):
            column.addWidget(card)
        column.addSpacing(10)
        column.addWidget(self.models_list)
        column.addStretch(1)
        return scroll

    def _feature_tab(self, name: str) -> QWidget:
        scroll, column = _column(name)
        section = Settings.model_fields[name].annotation
        column.addWidget(_caption(_sentence(about(section)) + "."))
        self.slot_problems[name] = problem = _caption("", RED)
        column.addWidget(problem)
        column.addSpacing(6)
        for card in self._cards((name,), section, TABS[name][1]):
            column.addWidget(card)
        column.addStretch(1)
        return scroll

    # --- the catalog -------------------------------------------------------------

    def set_store(self, store: MetadataStore | None) -> None:
        """Edit the settings of the catalog in `store` (None: no catalog open)."""
        if store is not self.store:
            self.catalog, self.progress = None, {}
        self.store = store
        self.path = store.settings_path if store is not None else None
        self.location.setText(str(self.path) if self.path else "Open a catalog in the library to edit its settings.")
        self.stack.setEnabled(self.path is not None)
        self.load()

    def set_catalog(self, catalog: Catalog | None) -> None:
        """The open catalog, for how far each fetcher is on it (worked out again in the background)."""
        self.catalog = catalog
        self._update_progress()

    def _update_progress(self) -> None:
        if self.store is None or self.catalog is None:
            self.progress = {}
            self._refresh_status()
            return
        if self.progress_task is not None:
            self._progress_stale = True
            return
        self.progress_task = ProgressTask(self.store, self.catalog, self)
        self.progress_task.finished_ok.connect(self._on_progress)
        self.progress_task.finished.connect(self._on_progress_finished)
        self.progress_task.start()

    def _on_progress(self, found: dict) -> None:
        self.progress = found
        self._refresh_status()

    def _on_progress_finished(self) -> None:
        self.progress_task = None
        if self._progress_stale:
            self._progress_stale = False
            self._update_progress()

    # --- values ------------------------------------------------------------------

    def load(self) -> None:
        """Show what the file says (the defaults for what it leaves out, or without a catalog)."""
        self._unusable = False
        self.problem.hide()
        settings = Settings()
        if self.path is not None:
            try:
                settings = load_settings(self.path)
            except SettingsError as e:
                self.problem.setText(f"The settings file can't be used, so the defaults are shown; saving replaces it.\n{e}")
                self.problem.show()
                self._unusable = True
        data = settings.model_dump(by_alias=True)
        for path, editor in self.editors.items():
            value = data
            for key in path:
                value = value[key]
            editor.set_value(value)
        self._saved = self._values(strict=False)
        self._on_edit()

    def save(self) -> None:
        try:
            settings = Settings.model_validate(self._nested(self._values(strict=True)))
            path = save_settings(settings, self.path)
        except (ValueError, ValidationError, OSError) as e:
            self._notify(InfoBar.error, "Not saved", str(e), duration=-1)
            return
        self._saved = self._values(strict=False)
        self._unusable = False
        self.problem.hide()
        self._update_buttons()
        self._update_progress()  # the keys follow the settings
        self._notify(InfoBar.success, "Saved", f"{path}: the next Enrich uses it.")

    def _values(self, strict: bool) -> dict[tuple[str, ...], object]:
        """Every editor's value; text that doesn't parse raises ValueError (strict) or is kept as _INVALID."""
        values = {}
        for path, editor in self.editors.items():
            try:
                values[path] = editor.value()
            except ValueError as e:
                if strict:
                    raise ValueError(f"{'.'.join(path)}: {e}") from e
                values[path] = _INVALID
        return values

    @staticmethod
    def _nested(values: dict[tuple[str, ...], object]) -> dict:
        """{("services", "llm", "model"): v} as {"services": {"llm": {"model": v}}}, without what didn't parse."""
        data: dict = {}
        for path, value in values.items():
            if value is _INVALID:
                continue
            node = data
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value
        return data

    def current(self) -> Settings | None:
        """The settings as edited, saved or not; None while some of it isn't valid."""
        try:
            return Settings.model_validate(self._nested(self._values(strict=False)))
        except ValidationError:
            return None

    # --- what follows the edits ------------------------------------------------------

    def _on_edit(self) -> None:
        self._update_buttons()
        self._refresh_status()

    def _refresh_status(self) -> None:
        """Each fetcher's and feature's status, and the models' users and folder, with the settings as edited."""
        settings = self.current()
        if settings is None:
            return
        for name, section in self.sections.items():
            section.show_status(check(settings, name), features.used_by(settings, name), self.progress.get(name),
                                settings.fetchers.get(name).enabled)
        for name, label in self.slot_problems.items():
            slots = features.slots(settings, name)
            problems = [p for slot in slots if (p := features.problem(settings, slot) or features.note(settings, slot))]
            label.setText("\n".join(f"{_sentence(p)}." for p in problems))
            label.setVisible(bool(problems))
        users: dict[str, list[str]] = {}
        for name in FETCHERS:
            if settings.fetchers.get(name).enabled:
                for model in models.needed(make_fetcher(name, settings).models()):
                    users.setdefault(model, []).append(name)
        self.models_list.users = users
        self.models_list.set_folder(settings.models.folder if self.path is not None else None)

    def _update_buttons(self) -> None:
        changed = self.path is not None and self._values(strict=False) != self._saved
        self.save_button.setEnabled(changed or self._unusable)
        self.revert_button.setEnabled(changed)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_status()  # models may have arrived since

    def shutdown(self) -> None:
        """Let model downloads and the progress finish; for closing."""
        self.models_list.shutdown()
        if self.progress_task is not None:
            self.progress_task.wait()

    def _notify(self, kind, title: str, content: str, duration: int = NOTICE_MS) -> None:
        kind(title, content, orient=Qt.Orientation.Vertical, duration=duration,
             position=InfoBarPosition.TOP_RIGHT, parent=self.window())
