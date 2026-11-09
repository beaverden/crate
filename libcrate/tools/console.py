"""Coloured terminal output for the scripts in tools/.

ANSI colours are used only when the stream is a terminal; NO_COLOR=1 turns
them off and FORCE_COLOR=1 turns them on regardless (https://no-color.org).
Plain blue is avoided: it is barely readable on dark terminal backgrounds.
"""

import os
import sys
from typing import TextIO

__all__ = ["Console"]

_CODES = {
    "bold": "1", "dim": "2",
    "red": "31", "green": "32", "yellow": "33", "cyan": "36",
    "bright_magenta": "95", "bright_cyan": "96",
}


def _wants_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class Console:
    def __init__(self, stream: TextIO | None = None, color: bool | None = None):
        self.stream = stream or sys.stderr
        self.color = _wants_color(self.stream) if color is None else color

    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        return f"\033[{';'.join(_CODES[s] for s in styles)}m{text}\033[0m"

    def print(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def heading(self, text: str) -> None:
        self.print(self.paint(f"▸ {text}", "bold"))
