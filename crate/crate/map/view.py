import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QGraphicsEllipseItem, QGraphicsItem, QGraphicsScene, QGraphicsView
from qfluentwidgets import isDarkTheme, themeColor

__all__ = ["MapView", "cluster_colors"]

# The map's width in scene units; dots keep their size in pixels whatever the zoom.
SCENE_SIZE = 1000.0
DOT_RADIUS = 4.0
HOVER_SCALE = 1.8
RING_RADIUS = 9.0
DIMMED_OPACITY = 0.1
ZOOM_STEP = 1.15  # per wheel notch
MAX_ZOOM = 40.0   # times the fitted view


def cluster_colors(count: int) -> list[QColor]:
    """`count` distinct colours, hues spread around the wheel by the golden ratio."""
    return [QColor.fromHsvF((i * 0.618034) % 1.0, 0.65, 0.9) for i in range(count)]


class _Dot(QGraphicsEllipseItem):
    """One track: a dot of the same size at any zoom, larger under the mouse, its title as tooltip."""

    def __init__(self, index: int, position: QPointF, color: QColor, tooltip: str):
        super().__init__(-DOT_RADIUS, -DOT_RADIUS, 2 * DOT_RADIUS, 2 * DOT_RADIUS)
        self.index = index
        self.setPos(position)
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setToolTip(tooltip)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event) -> None:
        self.setScale(HOVER_SCALE)
        self.setZValue(1)

    def hoverLeaveEvent(self, event) -> None:
        self.setScale(1)
        self.setZValue(0)


class MapView(QGraphicsView):
    """Tracks as dots at their map positions, coloured by cluster. The wheel zooms around the mouse,
    dragging pans, a click selects a track (selected(index), -1 for empty space), a double-click
    activates it (activated(index)). Indexes are rows of what set_points() got."""

    selected = Signal(int)
    activated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: transparent")
        self.dots: list[_Dot] = []
        self.ring: QGraphicsEllipseItem | None = None
        self._fitted = True  # follows the window's size until the user zooms
        self._bounds = QRectF()  # around the dots, in scene units
        self._fit_scale = 1.0  # the zoom that shows every dot
        self._pressed_at: QPointF | None = None

    def set_points(self, xy: np.ndarray, colors: list[QColor], tooltips: list[str]) -> None:
        """Replace the dots: one per row of `xy` (any scale), with its colour and tooltip."""
        scene = self.scene()
        scene.clear()
        self.dots, self.ring = [], None
        if len(xy):
            low, high = xy.min(axis=0), xy.max(axis=0)
            scaled = (xy - low) * (SCENE_SIZE / max(float((high - low).max()), 1e-9))
            scaled[:, 1] *= -1  # y up, as in a plot
            for i, ((x, y), color, tooltip) in enumerate(zip(scaled, colors, tooltips)):
                dot = _Dot(i, QPointF(float(x), float(y)), color, tooltip)
                scene.addItem(dot)
                self.dots.append(dot)
            (left, top), (right, bottom) = scaled.min(axis=0), scaled.max(axis=0)
            self._bounds = QRectF(float(left), float(top), float(right - left), float(bottom - top))
            # Room around the dots, so that zooming out and panning have somewhere to go.
            b = self._bounds
            scene.setSceneRect(b.adjusted(-b.width(), -b.height(), b.width(), b.height()))
        self.fit()

    def fit(self) -> None:
        """Show every dot."""
        self._fitted = True
        if self.dots:
            margin = max(self._bounds.width(), self._bounds.height()) * 0.04
            self.fitInView(self._bounds.adjusted(-margin, -margin, margin, margin), Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_scale = self.transform().m11()

    def select(self, index: int) -> None:
        """Ring the dot at `index` (-1: none)."""
        if self.ring is not None:
            self.scene().removeItem(self.ring)
            self.ring = None
        if 0 <= index < len(self.dots):
            self.ring = QGraphicsEllipseItem(-RING_RADIUS, -RING_RADIUS, 2 * RING_RADIUS, 2 * RING_RADIUS)
            self.ring.setPen(QPen(QColor("white") if isDarkTheme() else themeColor(), 2.5))
            self.ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            self.ring.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.ring.setPos(self.dots[index].pos())
            self.ring.setZValue(2)
            self.scene().addItem(self.ring)

    def center_on(self, index: int) -> None:
        if 0 <= index < len(self.dots):
            self.centerOn(self.dots[index])

    def set_dimmed(self, dimmed: np.ndarray) -> None:
        """Fade the dots where `dimmed` is True, e.g. those a search doesn't match."""
        for dot, dim in zip(self.dots, dimmed):
            dot.setOpacity(DIMMED_OPACITY if dim else 1.0)

    def _dot_at(self, event: QMouseEvent) -> int:
        item = self.itemAt(event.position().toPoint())
        return item.index if isinstance(item, _Dot) else -1

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed_at = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        # A click, not the end of a drag.
        if event.button() == Qt.MouseButton.LeftButton and self._pressed_at is not None \
                and (event.position() - self._pressed_at).manhattanLength() < 4:
            self.selected.emit(self._dot_at(event))

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        index = self._dot_at(event)
        if index >= 0:
            self.activated.emit(index)
        else:
            self.fit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = ZOOM_STEP ** (event.angleDelta().y() / 120)
        scale = self.transform().m11() * factor
        if not self.dots or not self._fit_scale * 0.8 <= scale <= self._fit_scale * MAX_ZOOM:
            return
        self._fitted = False
        self.scale(factor, factor)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self.fit()
