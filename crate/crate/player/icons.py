from enum import Enum
from pathlib import Path

from qfluentwidgets import FluentIconBase, Theme, getIconColor

__all__ = ["PlayerIcon"]

ICONS = Path(__file__).parent / "icons"


class PlayerIcon(FluentIconBase, Enum):
    """Icons Fluent doesn't have (its SKIP_BACK/SKIP_FORWARD are 10 s / 30 s jumps, and there is no shuffle), drawn like its own:
    an SVG per theme colour in icons/."""

    PREVIOUS = "Previous"
    NEXT = "Next"
    SHUFFLE = "Shuffle"

    def path(self, theme=Theme.AUTO) -> str:
        return str(ICONS / f"{self.value}_{getIconColor(theme)}.svg")
