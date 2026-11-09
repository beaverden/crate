import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from libcrate import Catalog, TrackRecord
from mutagen import MutagenError
from mutagen.flac import FLAC
from PySide6.QtCore import QBuffer, QByteArray, QObject, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QImageReader, QPainter, QPainterPath, QPixmap

__all__ = ["THUMBNAILS_DIR", "CoverCache", "read_cover", "square", "thumbnail"]

# In the output folder, next to catalog.json.
THUMBNAILS_DIR = "thumbnails"

# Thumbnails are made for 2x (HiDPI) screens.
RATIO = 2


def _picture(path: Path) -> bytes | None:
    """The file's front cover (else its first picture) as stored, or None without one or if the file can't be read."""
    try:
        pictures = FLAC(path).pictures
    except (MutagenError, OSError):
        return None
    picture = next((p for p in pictures if p.type == 3), pictures[0] if pictures else None)  # 3 = front cover
    return picture.data if picture is not None else None


def read_cover(path: Path, side: int | None = None) -> QImage | None:
    """The file's cover, or None without one or if it can't be read.

    side: decode at about this size (covering a side x side square) rather than in full. The
    library's covers are 1500px JPEGs; decoding one in full takes about 25 ms and holds the GIL,
    which froze the table while covers loaded. Decoding straight to a small size is about 3x
    faster and releases the GIL.
    """
    data = _picture(path)
    if data is None:
        return None
    buffer = QBuffer()
    buffer.setData(QByteArray(data))
    buffer.open(QBuffer.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    if side is not None and reader.size().isValid():
        reader.setScaledSize(reader.size().scaled(side, side, Qt.AspectRatioMode.KeepAspectRatioByExpanding))
    image = reader.read()
    return None if image.isNull() else image


def square(image: QImage, side: int) -> QImage:
    """`image` cropped to a centred square of `side` pixels."""
    scaled = image.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                          Qt.TransformationMode.SmoothTransformation)
    return scaled.copy((scaled.width() - side) // 2, (scaled.height() - side) // 2, side, side)


def thumbnail(image: QImage, size: int, radius: float) -> QImage:
    """`image` cropped to a centred square, scaled to `size` logical pixels at RATIO, corners rounded.

    Done once per cover (any thread), so drawing it is a plain copy: scaling and clipping
    to rounded corners on every repaint made scrolling the table drop frames.
    """
    side = size * RATIO
    out = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(0, 0, side, side), radius * RATIO, radius * RATIO)
    painter.setClipPath(clip)
    painter.drawImage(0, 0, square(image, side))
    painter.end()
    out.setDevicePixelRatio(RATIO)
    return out


class CoverCache(QObject):
    """Cover thumbnails for the track table, loaded in background threads and kept in memory.

    get(track) returns the thumbnail, or None while it loads (loaded() follows) and for tracks
    without a cover. Thumbnails are ready to draw: see thumbnail().

    Each one is also saved as a small JPEG in the `folder` given to set_library (the output folder's
    THUMBNAILS_DIR), so a cover is decoded from its file only once rather than on every start. A
    thumbnail's file name hashes the track's path, size and mtime, so a changed file gets a new one;
    prune() deletes those no track uses any more.
    """

    loaded = Signal()
    _read = Signal(str, object)  # from the worker threads: key, QImage | None

    def __init__(self, size: int, radius: float, parent=None):
        super().__init__(parent)
        self.size = size
        self.radius = radius
        self.root: Path | None = None
        self.folder: Path | None = None
        self._thumbnails: dict[str, QPixmap | None] = {}  # by key()
        self._pending: set[str] = set()
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="covers")
        self._read.connect(self._on_read)

    def key(self, track: TrackRecord) -> str:
        """The thumbnail's name: changes with the track's file and the thumbnail size."""
        stamp = f"{track.path}\0{track.stat.size}\0{track.stat.mtime_ns}\0{self.size * RATIO}"
        return hashlib.sha1(stamp.encode()).hexdigest()

    def set_library(self, root: str | None, folder: Path | None) -> None:
        """The library folder the tracks' paths are relative to, and the folder thumbnails are saved in."""
        root = Path(root) if root else None
        if (root, folder) != (self.root, self.folder):
            self.root, self.folder = root, folder
            self._thumbnails.clear()

    def get(self, track: TrackRecord) -> QPixmap | None:
        if track.cover is None or self.root is None:
            return None
        key = self.key(track)
        if key in self._thumbnails:
            return self._thumbnails[key]
        if key not in self._pending and not self._closed:
            self._pending.add(key)
            self._pool.submit(self._load, self.root / track.path, self.folder, key)
        return None

    def prune(self, catalog: Catalog) -> None:
        """Delete the saved thumbnails of tracks that changed or are gone (in the background)."""
        if self.folder is not None and not self._closed:
            keep = {f"{self.key(t)}.jpg" for t in catalog.tracks.values() if t.cover is not None}
            self._pool.submit(self._prune, self.folder, keep)

    def shutdown(self) -> None:
        """Drop the queued work and wait for what's in progress; for closing. Nothing is loaded or pruned after:
        an enrich still running at closing finishes (and shows its catalog) after this."""
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _load(self, path: Path, folder: Path | None, key: str) -> None:
        saved = folder / f"{key}.jpg" if folder else None
        image = QImageReader(str(saved)).read() if saved and saved.is_file() else QImage()
        if image.isNull():
            image = read_cover(path, self.size * RATIO)
            if image is not None and saved:
                image = square(image, self.size * RATIO)
                self._save(image, saved)
        self._read.emit(key, thumbnail(image, self.size, self.radius) if image is not None else None)

    @staticmethod
    def _save(image: QImage, path: Path) -> None:
        """Write atomically; a folder that can't be written to only means no saved thumbnails."""
        tmp = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if image.save(str(tmp), "JPG", 90):
                tmp.replace(path)
        except OSError:
            pass

    @staticmethod
    def _prune(folder: Path, keep: set[str]) -> None:
        try:
            for path in folder.iterdir():
                if path.name not in keep:
                    path.unlink(missing_ok=True)
        except OSError:
            pass

    def _on_read(self, key: str, image: QImage | None) -> None:
        self._pending.discard(key)
        self._thumbnails[key] = QPixmap.fromImage(image) if image is not None else None
        self.loaded.emit()
