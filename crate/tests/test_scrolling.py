import time

import pytest
from libcrate import MetadataStore
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from crate.library import LibraryPage
from crate.library.scrolling import NOTCH_PX


@pytest.fixture
def table(qapp, make_library, tmp_path, qsettings):
    MetadataStore(tmp_path / "out").scan(make_library(60))
    page = LibraryPage(qsettings)
    page.open_catalog(tmp_path / "out")
    page.resize(1000, 500)
    page.show()
    qapp.processEvents()
    yield page.table
    page.shutdown()


def wheel(qapp, table, angle: int, pixel: int) -> None:
    viewport = table.viewport()
    event = QWheelEvent(QPointF(50, 50), QPointF(viewport.mapToGlobal(QPoint(50, 50))), QPoint(0, pixel),
                        QPoint(0, angle), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase if not pixel else Qt.ScrollPhase.ScrollUpdate, False)
    qapp.sendEvent(viewport, event)


def settle(qapp, table, seconds=0.5) -> list[int]:
    """Let any glide finish; the scroll positions seen meanwhile."""
    bar, seen = table.verticalScrollBar(), []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        seen.append(bar.value())
        time.sleep(0.005)
    return seen


def test_trackpad_scrolls_as_far_as_the_fingers(qapp, table):
    wheel(qapp, table, angle=-20, pixel=-10)
    assert table.verticalScrollBar().value() == 10


def test_mouse_wheel_notch_glides_a_row_and_a_half(qapp, table):
    wheel(qapp, table, angle=-120, pixel=0)
    seen = settle(qapp, table)
    assert seen[-1] == NOTCH_PX
    steps = [b - a for a, b in zip([0, *seen], seen)]
    assert len(set(seen)) > 5 and max(steps) < NOTCH_PX / 2  # glides, rather than jumping in lines


def test_quick_notches_add_up(qapp, table):
    wheel(qapp, table, angle=-120, pixel=0)
    wheel(qapp, table, angle=-120, pixel=0)
    assert settle(qapp, table)[-1] == 2 * NOTCH_PX
    wheel(qapp, table, angle=120, pixel=0)
    assert settle(qapp, table)[-1] == NOTCH_PX
