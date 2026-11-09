"""Clients for the online services libcrate gets data from, usable on their own (the local models are in
libcrate.models).

    from libcrate.services.chosic import ChosicClient
    from libcrate.services.llm import LLMClient
    from libcrate.services.musicbrainz import MusicBrainzClient

Each wraps one service and knows nothing about the catalog; what it can look up
grows method by method (genres today). The fetchers in libcrate.fetchers use
them to fill a catalog.
"""
