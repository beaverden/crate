"""The MusicBrainz web service."""

from libcrate.services.musicbrainz.client import MusicBrainzClient, MusicBrainzError, MusicBrainzNotFound

__all__ = ["MusicBrainzClient", "MusicBrainzError", "MusicBrainzNotFound"]
