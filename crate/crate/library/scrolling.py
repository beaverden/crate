from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation
from PySide6.QtGui import QWheelEvent
from qfluentwidgets import TableView

__all__ = ["calm_scrolling"]

# One mouse-wheel notch (120 units) scrolls this many pixels: about one and a half rows.
NOTCH_PX = 84
# How long the glide after a notch takes, in ms.
GLIDE_MS = 200


class CalmScroll(QObject):
    """Wheel handling for a table, installed on its viewport ahead of Fluent's smooth scrolling:

    - trackpads (and other pixel-precise devices): scroll by exactly the distance reported. Qt's
      scroll bar ignores pixelDelta and turns angleDelta into lines of `singleStep` pixels, so on a
      Mac trackpad the table moved about 2.7x as far as the fingers did.
    - mouse wheels: each notch moves the target NOTCH_PX further, and the view glides there pixel by
      pixel. Fluent animates a notch by sending small wheel events to the scroll bar, which only
      moves in whole lines (54 px here), so its scrolling went in jumps, and too far: about 4.5 rows
      per notch.
    """

    def __init__(self, view: TableView):
        super().__init__(view)
        self.bar = view.verticalScrollBar()
        self.glide = QPropertyAnimation(self.bar, b"value", self)
        self.glide.setDuration(GLIDE_MS)
        self.glide.setEasingCurve(QEasingCurve.Type.OutCubic)

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.Wheel or not isinstance(event, QWheelEvent):
            return False
        pixels, angle = event.pixelDelta().y(), event.angleDelta().y()
        if pixels:
            self.glide.stop()
            self.bar.setValue(self.bar.value() - pixels)
        elif angle:
            # Notches that come while gliding extend the glide rather than restart from where it is.
            start = self.glide.endValue() if self.glide.state() == QPropertyAnimation.State.Running \
                else self.bar.value()
            target = max(self.bar.minimum(), min(self.bar.maximum(), round(start - angle / 120 * NOTCH_PX)))
            self.glide.stop()
            self.glide.setStartValue(self.bar.value())
            self.glide.setEndValue(target)
            self.glide.start()
        else:
            return False  # horizontal: not ours
        event.accept()
        return True


def calm_scrolling(view: TableView) -> None:
    """Make `view` scroll by what the trackpad reports, and glide NOTCH_PX per mouse-wheel notch."""
    # Filters installed later run first, so this one sees wheel events before Fluent's.
    view.viewport().installEventFilter(CalmScroll(view))
