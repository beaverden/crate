from libcrate import FetcherCheck, FetcherProgress
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, StrongBodyLabel

__all__ = ["FetcherSection", "size_text", "status_lines"]

GREY = (QColor(96, 96, 96), QColor(160, 160, 160))
RED = (QColor(196, 43, 28), QColor(255, 153, 164))


def size_text(size: int) -> str:
    """2_670_000_000 -> '2.7 GB', 85_000_000 -> '85 MB'."""
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{max(1, round(size / 1e6))} MB"


def status_lines(check: FetcherCheck, users: list[str], counts: FetcherProgress | None, enabled: bool) -> list[str]:
    """What a fetcher's section says under its description, but its problems: what it downloads, does without and
    is used by, and how far it is on the catalog (`counts`; None: not known yet)."""
    lines = []
    if check.downloads:
        when = "on the next Enrich" if enabled else "when it's on, on the next Enrich"
        lines.append(f"Downloads {', '.join(check.downloads)} ({size_text(check.download_size)}) {when}.")
    lines += [f"{note[0].upper()}{note[1:]}." for note in check.notes]
    if users:
        lines.append(f"Used by {', '.join(users)}.")
    if counts is not None and counts.total:
        parts = [f"{counts.done:,} done"]
        for value, label in ((counts.redo, "to redo"), (counts.todo, "never done"), (counts.errors, "failed"),
                             (counts.cant, "with nothing to go on")):
            if value:
                parts.append(f"{value:,} {label}")
        line = f"{counts.total:,} tracks: {'  ·  '.join(parts)}."
        if enabled and counts.to_do:
            line += f" The next Enrich does {counts.to_do:,}."
        lines.append(line)
    return lines


class FetcherSection(QWidget):
    """A fetcher on the settings page: its name and what it does, why it can't run (red), its status_lines(), then
    `cards`, its settings (Run on Enrich first)."""

    def __init__(self, name: str, about: str, cards: list[QWidget], parent=None):
        super().__init__(parent)
        self.name = name
        self.title = StrongBodyLabel(name)
        self.about = CaptionLabel(about[0].upper() + about[1:] + ".")
        self.about.setWordWrap(True)
        self.problems = CaptionLabel()
        self.problems.setTextColor(*RED)
        self.problems.setWordWrap(True)
        self.status = CaptionLabel()
        self.status.setTextColor(*GREY)
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 6)
        layout.setSpacing(3)
        for label in (self.title, self.about, self.problems, self.status):
            layout.addWidget(label)
        layout.addSpacing(4)
        for card in cards:
            layout.addWidget(card)

    def show_status(self, check: FetcherCheck, users: list[str], counts: FetcherProgress | None,
                    enabled: bool) -> None:
        problems = "; ".join(check.problems)
        self.problems.setText(f"Can't run: {problems}." if problems else "")
        self.problems.setVisible(bool(problems))
        lines = status_lines(check, users, counts, enabled)
        self.status.setText("\n".join(lines))
        self.status.setVisible(bool(lines))
