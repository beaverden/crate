from libcrate import TrackRecord
from PySide6.QtCore import QModelIndex
from qfluentwidgets import Action, FluentIcon, RoundMenu

from crate.icons import Icon
from crate.library.model import TRACK_ROLE

__all__ = ["track_menu", "track_title"]


def track_title(track: TrackRecord) -> str:
    """"Artists – Title", for a summary or a station's name."""
    return f"{', '.join(track.tags.artists) or 'Unknown artist'} – {track.tags.title or track.path}"


def track_menu(page, index: QModelIndex, play) -> RoundMenu:
    """The right-click menu of the track at `index` on a page that lists tracks: play it (play(index)), find
    similar ones, show it on the map, start a radio station from it. The last three are the page's signals
    similar_requested(track), map_requested(track) and radio_requested([track], name)."""
    track = index.data(TRACK_ROLE)
    menu = RoundMenu(parent=page)
    menu.addAction(Action(FluentIcon.PLAY, "Play", triggered=lambda: play(index)))
    menu.addAction(Action(Icon.SIMILAR, "Similar tracks", triggered=lambda: page.similar_requested.emit(track)))
    menu.addAction(Action(FluentIcon.GLOBE, "Show in map", triggered=lambda: page.map_requested.emit(track)))
    menu.addAction(Action(Icon.RADIO, "Start radio",
                          triggered=lambda: page.radio_requested.emit([track], track_title(track))))
    return menu
