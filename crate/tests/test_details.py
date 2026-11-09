from libcrate import Catalog, Description, DescriptionSource, SourceStatus

from qfluentwidgets import ImageLabel

from crate.library.details import TrackDetails


def test_shows_tags_and_genres_per_source(qapp, library, add_genres):
    catalog = add_genres(Catalog.scan(library)[0])
    details = TrackDetails()
    details.show_track(catalog.tracks["Metronomy - The Light [GBMVH1900150].flac"])

    text = details.text()
    assert "The Light" in text
    assert "GBMVH1900150" in text
    assert "chillwave  26%" in text
    assert "not found" in text  # chosic's


def test_no_genres_and_no_track(qapp, library):
    catalog, _ = Catalog.scan(library)
    details = TrackDetails()
    details.show_track(catalog.tracks["Abba - Waterloo.flac"])
    assert "None yet" in details.text()

    details.show_track(None)
    assert "Select a track" in details.text()


def test_cover_from_the_file(qapp, library):
    catalog, _ = Catalog.scan(library)
    details = TrackDetails()
    details.show_track(catalog.tracks["Metronomy - The Light [GBMVH1900150].flac"], catalog.root)
    assert len(details.widget().findChildren(ImageLabel)) == 1

    # No cover, or no library to read it from (the drive isn't mounted): no image.
    details.show_track(catalog.tracks["Abba - Waterloo.flac"], catalog.root)
    assert not details.widget().findChildren(ImageLabel)
    details.show_track(catalog.tracks["Metronomy - The Light [GBMVH1900150].flac"], "/nonexistent")
    assert not details.widget().findChildren(ImageLabel)


def test_descriptions(qapp, library):
    catalog, _ = Catalog.scan(library)
    track = catalog.tracks["Metronomy - The Light [GBMVH1900150].flac"]
    details = TrackDetails()
    details.show_track(track)
    assert "Description" not in details.text()  # none asked for yet

    track.descriptions["llm"] = DescriptionSource(description=Description(
        text="Breezy synth-pop.", keywords=["synth-pop", "summer"], language="English", recognised=True))
    details.show_track(track)
    text = details.text()
    assert "Breezy synth-pop." in text and "summer" in text and "llm · English · knew the track" in text

    track.descriptions["llm"] = DescriptionSource(status=SourceStatus.ERROR, error="RateLimitError: slow down")
    details.show_track(track)
    assert "llm · error · RateLimitError: slow down" in details.text()
