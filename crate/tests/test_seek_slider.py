from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from crate.player.bar import SeekSlider


def slider_at(qapp, value=0):
    slider = SeekSlider()
    slider.resize(400, 22)
    slider.setRange(0, 100_000)
    slider.setValue(value)
    slider.show()
    qapp.processEvents()
    seeks = []
    slider.seek_requested.connect(seeks.append)
    return slider, seeks


def test_click_on_the_groove_seeks(qapp):
    slider, seeks = slider_at(qapp)
    QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(300, 11))
    assert len(seeks) == 1 and 60_000 < seeks[0] < 90_000
    assert not slider.dragging


def test_drag_along_the_groove_seeks_once_on_release(qapp):
    slider, seeks = slider_at(qapp)
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(100, 11))
    assert slider.dragging and not seeks
    QTest.mouseMove(slider, QPoint(250, 11))
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(250, 11))
    assert len(seeks) == 1 and seeks[0] > 50_000


def test_handle_drag_seeks_once_on_release(qapp):
    slider, seeks = slider_at(qapp, 20_000)
    handle = slider.handle
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert slider.dragging
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert seeks == [20_000]
