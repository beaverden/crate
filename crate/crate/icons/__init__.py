from enum import Enum
from pathlib import Path

from qfluentwidgets import FluentIconBase, Theme, getIconColor

__all__ = ["Icon"]

FOLDER = Path(__file__).parent


class Icon(FluentIconBase, Enum):
    """Icons Fluent has none for, drawn like Fluent's own: an SVG per theme colour in this folder.

    SIMILAR, two overlapping circles: alike. RADIO, a dot sending out waves: a station.
    """

    SIMILAR = "Similar"
    RADIO = "Radio"

    def path(self, theme=Theme.AUTO) -> str:
        return str(FOLDER / f"{self.value}_{getIconColor(theme)}.svg")
