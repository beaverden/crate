from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from libcrate import RunReport, ScanReport

__all__ = ["Job", "Step"]

# A step's and a job's states.
WAITING, RUNNING, DONE, STOPPED, GAVE_UP, NOT_RUN, FAILED = (
    "waiting", "running", "done", "stopped", "gave up", "not run", "failed")


@dataclass
class Step:
    """One step of an enrich (a fetcher, or a model's download) or the files of a scan: its tracks to do, done, and
    what they came to."""

    name: str
    state: str = WAITING
    done: int = 0
    total: int = 0
    outcomes: Counter = field(default_factory=Counter)  # "found", "error", ... -> tracks
    skipped: int = 0  # tracks it had a result for already
    note: str = ""  # why it gave up


@dataclass
class Job:
    """A scan or an enrich, followed by the jobs page: progress() per track, then one of the finish_*()
    methods or fail(). Plain data, so it can be tested without Qt."""

    title: str
    steps: dict[str, Step]
    started: datetime = field(default_factory=datetime.now)
    finished: datetime | None = None
    state: str = RUNNING
    error: str = ""

    @classmethod
    def enrich(cls, catalog: str, steps: list[str]) -> "Job":
        """An enrich's job, its fetchers and models (`steps`) in the order they run."""
        return cls(f"Enrich  ·  {catalog}", {name: Step(name) for name in steps})

    @classmethod
    def scan(cls, catalog: str, creating: bool) -> "Job":
        return cls(f"{'New catalog' if creating else 'Scan'}  ·  {catalog}", {"files": Step("files")})

    @property
    def has_problems(self) -> bool:
        """It failed, a step gave up, or some tracks came to an error."""
        return self.state == FAILED or any(
            s.state == GAVE_UP or s.outcomes["error"] or s.outcomes["failed"] for s in self.steps.values())

    def start(self, name: str, total: int, skipped: int) -> None:
        """A step starts: done straight away when it has nothing to do."""
        step = self.steps[name]
        step.total, step.skipped = total, skipped
        step.state = RUNNING if total else DONE

    def progress(self, name: str, done: int, total: int, outcome: str | None = None) -> None:
        step = self.steps[name]
        step.state, step.done, step.total = RUNNING, done, total
        if outcome is not None:
            step.outcomes[outcome] += 1
        if done == total:
            step.state = DONE

    def finish_enrich(self, run: RunReport) -> None:
        for report in (run.enriched, run.embedded):
            if report is None:
                continue
            for name, skipped in report.skipped.items():
                step = self.steps[name]
                step.skipped = skipped
                if name in report.gave_up:
                    step.state, step.note = GAVE_UP, report.gave_up[name]
                elif step.done < step.total:
                    step.state = STOPPED
                else:
                    step.state = DONE
        for step in self.steps.values():
            if step.state in (WAITING, RUNNING):  # never reached: the steps after a stop
                step.state = NOT_RUN
        self._finish(STOPPED if run.interrupted else DONE)

    def finish_scan(self, report: ScanReport) -> None:
        step = self.steps["files"]
        step.state = DONE
        step.outcomes = Counter({"added": len(report.added), "updated": len(report.updated),
                                 "removed": len(report.removed), "failed": len(report.failed)})
        step.skipped = report.unchanged
        self._finish(DONE)

    def fail(self, message: str) -> None:
        self.error = message
        for step in self.steps.values():
            if step.state in (WAITING, RUNNING):
                step.state = FAILED
        self._finish(FAILED)

    def _finish(self, state: str) -> None:
        self.state = state
        self.finished = datetime.now()
