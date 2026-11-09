from libcrate import Catalog
from PySide6.QtCore import Qt

from crate.library.model import SEARCH_ROLE, SORT_ROLE, TRACK_ROLE, TrackTableModel

DISPLAY = Qt.ItemDataRole.DisplayRole


def test_rows_and_columns(qapp, library, add_genres):
    catalog = add_genres(Catalog.scan(library)[0])
    model = TrackTableModel()
    model.set_catalog(catalog)

    assert model.rowCount() == 3
    assert [model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())] == \
        ["Track", "Album", "Year", "Length", "Genres"]
    rows = {model.index(r, 0).data(DISPLAY): r for r in range(3)}
    light = rows["The Light"]
    assert [model.index(light, c).data(DISPLAY) for c in range(5)] == \
        ["The Light", "Metronomy Forever", "2019", "3:53", "chillwave"]
    assert model.index(light, 0).data(TRACK_ROLE).path == "Metronomy - The Light [GBMVH1900150].flac"
    # The track column is searched by artist too.
    assert model.index(light, 0).data(SEARCH_ROLE) == "The Light Metronomy"


def test_sort_values_are_typed(qapp, library):
    catalog, _ = Catalog.scan(library)
    model = TrackTableModel()
    model.set_catalog(catalog)

    assert sorted(model.index(r, 0).data(SORT_ROLE) for r in range(3)) == \
        [("abba", "waterloo"), ("metronomy", "the light"), ("nasaya", "tempo")]
    assert sorted(model.index(r, 2).data(SORT_ROLE) for r in range(3)) == [1974, 2019, 2021]
    assert sorted(model.index(r, 3).data(SORT_ROLE) for r in range(3)) == [167.4, 180.0, 233.0]


def test_no_catalog(qapp):
    model = TrackTableModel()
    model.set_catalog(None)
    assert model.rowCount() == 0
