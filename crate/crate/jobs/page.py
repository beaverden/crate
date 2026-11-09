from libcrate import OUTCOMES
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, FluentIcon, ProgressBar, PushButton, ScrollArea, SimpleCardWidget, StrongBodyLabel,
    TitleLabel,
)

from crate.jobs.job import DONE, FAILED, GAVE_UP, NOT_RUN, RUNNING, STOPPED, WAITING, Job, Step
from crate.tasks import EnrichTask, ScanTask

__all__ = ["JobsPage"]

# Jobs kept on the page; older ones are dropped.
MAX_JOBS = 20
# What a step's tracks came to, in the order they're listed: an enrich's, then a scan's.
OUTCOME_ORDER = (*OUTCOMES, "added", "updated", "removed", "failed")
PLURALS = {"error": "errors"}
NAME_WIDTH = 220
COUNT_WIDTH = 110
# Wide enough for '1,234 found  ·  56 not found  ·  7 errors  ·  1,197 already done', so bars line up across cards.
DETAILS_WIDTH = 420


class JobsPage(QWidget):
    """This session's scans and enriches, newest first: a card per job with a row per fetcher or
    download (or the files of a scan), each with its progress and what its tracks came to so far, and
    Stop on a running enrich. watch(task) follows a task the library page started."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("jobs")
        self.cards: list[JobCard] = []

        about = CaptionLabel("This session's scans and enriches, newest first.")
        about.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        self.empty = CaptionLabel("No jobs yet: Scan or Enrich in the library, and they show up here.")
        self.empty.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))

        column_widget = QWidget()
        column_widget.setObjectName("jobs-column")
        column_widget.setStyleSheet("#jobs-column {background: transparent}")
        self.column = QVBoxLayout(column_widget)
        self.column.setContentsMargins(0, 0, 12, 0)
        self.column.setSpacing(12)
        self.column.addWidget(self.empty)
        self.column.addStretch(1)
        scroll = ScrollArea()
        scroll.setWidget(column_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea {background: transparent; border: none}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(4)
        layout.addWidget(TitleLabel("Jobs"))
        layout.addWidget(about)
        layout.addSpacing(10)
        layout.addWidget(scroll, 1)

    def watch(self, task: ScanTask | EnrichTask) -> Job:
        """Add a card for `task` and keep it up to date until the task ends."""
        catalog = task.store.folder.name
        if isinstance(task, EnrichTask):
            job = Job.enrich(catalog, task.steps)
            task.started.connect(lambda name, total, skipped: self._update(card, job.start, name, total, skipped))
            task.progress.connect(lambda name, done, total, rel, outcome: self._update(
                card, job.progress, name, done, total, outcome))
            task.finished_ok.connect(lambda catalog, report: self._update(card, job.finish_enrich, report))
        else:
            job = Job.scan(catalog, creating=task.library is not None)
            task.progress.connect(lambda done, total, rel: self._update(card, job.progress, "files", done, total))
            task.finished_ok.connect(lambda catalog, report: self._update(card, job.finish_scan, report))
        task.failed.connect(lambda message: self._update(card, job.fail, message))
        card = JobCard(job, task.stop if isinstance(task, EnrichTask) else None)
        self.cards.insert(0, card)
        self.column.insertWidget(0, card)
        self.empty.hide()
        for old in self.cards[MAX_JOBS:]:
            old.deleteLater()
        del self.cards[MAX_JOBS:]
        return job

    def _update(self, card: "JobCard", change, *args) -> None:
        change(*args)
        card.refresh()


class JobCard(SimpleCardWidget):
    """One job: its title, state and (while it runs, if it can be stopped) Stop, then a row per step."""

    def __init__(self, job: Job, stop=None, parent=None):
        super().__init__(parent)
        self.job = job
        self._stop = stop
        self.stopping = False
        self.title = StrongBodyLabel(job.title)
        self.stop_button = PushButton(FluentIcon.CANCEL, "Stop")
        self.stop_button.setToolTip("Finish the lookups in flight, save, and skip the rest")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setVisible(stop is not None)
        self.state = CaptionLabel()
        self.state.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        self.state.setWordWrap(True)
        self.rows: dict[str, StepRow] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        for i, step in enumerate(job.steps.values()):
            self.rows[step.name] = StepRow(step, grid, i)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 16)
        layout.setSpacing(4)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(self.title)
        titles.addWidget(self.state)
        header = QHBoxLayout()
        header.addLayout(titles, 1)
        header.addWidget(self.stop_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)
        layout.addSpacing(8)
        layout.addLayout(grid)
        self.refresh()

    def stop(self) -> None:
        if self._stop is not None and not self.stopping:
            self._stop()
            self.stopping = True
            self.refresh()

    def refresh(self) -> None:
        job = self.job
        started = f"{job.started:%H:%M}"
        self.stop_button.setVisible(self._stop is not None and job.finished is None)
        self.stop_button.setEnabled(not self.stopping)
        if job.finished is None and self.stopping:
            state = f"Stopping after the lookups in flight  ·  started {started}"
        elif job.finished is None:
            state = f"Running  ·  started {started}"
        else:
            minutes = max(1, round((job.finished - job.started).total_seconds() / 60))
            state = f"{job.state.capitalize()}  ·  {started} → {job.finished:%H:%M}, {minutes} min"
            if job.error:
                state += f"\n{job.error}"
        self.state.setText(state)
        for row in self.rows.values():
            row.refresh()


class StepRow:
    """A step's row in its card's grid: name, progress bar, done/total, and what its tracks came to."""

    def __init__(self, step: Step, grid: QGridLayout, row: int):
        self.step = step
        self.name = BodyLabel(step.name)
        self.name.setFixedWidth(NAME_WIDTH)
        self.bar = ProgressBar(useAni=False)
        self.count = CaptionLabel()
        self.count.setFixedWidth(COUNT_WIDTH)
        self.count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.details = CaptionLabel()
        self.details.setFixedWidth(DETAILS_WIDTH)
        self.details.setTextColor(QColor(96, 96, 96), QColor(160, 160, 160))
        grid.addWidget(self.name, row, 0)
        grid.addWidget(self.bar, row, 1)
        grid.addWidget(self.count, row, 2)
        grid.addWidget(self.details, row, 3)

    def refresh(self) -> None:
        step = self.step
        finished_empty = step.state == DONE and step.total == 0
        self.bar.setRange(0, max(step.total, 1))
        self.bar.setValue(1 if finished_empty else step.done)
        self.bar.setError(step.state in (GAVE_UP, FAILED))
        self.bar.setPaused(step.state in (STOPPED, NOT_RUN))
        self.count.setText(f"{step.done:,} / {step.total:,}" if step.total else "")
        self.details.setText(details(step))


def details(step: Step) -> str:
    """'98 found · 20 not found · 2 errors · 1,197 already done', with the state when it says more."""
    outcomes = [f"{step.outcomes[o]:,} {PLURALS.get(o, o) if step.outcomes[o] != 1 else o}"
                for o in OUTCOME_ORDER if step.outcomes[o]]
    lead = {WAITING: "waiting", STOPPED: "stopped", NOT_RUN: "not run: stopped before it",
            FAILED: "failed", GAVE_UP: f"gave up: {step.note}"}.get(step.state)
    if step.state == DONE and not step.total and not outcomes:
        lead = "nothing to do"
    if step.state == RUNNING and not outcomes:
        lead = "starting"
    parts = [lead] if lead else []
    parts += outcomes
    if step.skipped:
        parts.append(f"{step.skipped:,} {'unchanged' if step.name == 'files' else 'already done'}")
    return "  ·  ".join(parts)
