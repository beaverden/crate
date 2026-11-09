from pathlib import Path

from qfluentwidgets import Action, FluentIcon, RoundMenu

__all__ = ["CatalogMenu"]

# Longer paths lose their middle in the menu.
MAX_PATH = 48


class CatalogMenu(RoundMenu):
    """Switch catalogs: the library page's recent ones (the open one ticked), then New catalog…
    and Open catalog…. Everything's disabled while the page runs a scan or an enrich."""

    def __init__(self, library, parent=None):
        super().__init__(parent=parent)
        current = library.store.folder if library.store else None
        idle = library.task is None
        self.catalog_actions = {}  # folder -> its action
        for folder in library.recent_catalogs():
            icon = FluentIcon.ACCEPT if folder == current else FluentIcon.LIBRARY
            action = Action(icon, f"{folder.name}   ·   {_short(folder)}",
                            triggered=lambda checked=False, f=folder: library.open_catalog(f))
            action.setEnabled(idle and folder != current)
            self.addAction(action)
            self.catalog_actions[folder] = action
        if self.catalog_actions:
            self.addSeparator()
        self.new_action = Action(FluentIcon.ADD, "New catalog…", triggered=library.new_catalog)
        self.open_action = Action(FluentIcon.FOLDER, "Open catalog…", triggered=library.choose_catalog)
        for action in (self.new_action, self.open_action):
            action.setEnabled(idle)
            self.addAction(action)


def _short(folder: Path) -> str:
    """The folder with ~ for the home folder, at most MAX_PATH characters (the start and the end)."""
    try:
        text = f"~/{folder.relative_to(Path.home())}"
    except ValueError:
        text = str(folder)
    if len(text) <= MAX_PATH:
        return text
    keep = (MAX_PATH - 1) // 2
    return f"{text[:keep]}…{text[-keep:]}"
