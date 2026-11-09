# crate

A desktop app for a FLAC music library: browse it, enrich it with genres, descriptions and audio embeddings, and find music by how it sounds. You can list the tracks most like one, search with a description ("chill late-night electronic, with a warm bassline"), explore a map of the library, or play an endless radio station that steers by what you skip.

The repository is a [uv](https://docs.astral.sh/uv/) workspace with two packages:

- [`crate`](crate/README.md): the app, built with PySide6 and PySide6-Fluent-Widgets.
- [`libcrate`](libcrate/README.md): the backend the app uses in-process. It holds the catalog, the fetchers that fill it (MusicBrainz, chosic, Essentia, a language model, CLAP models), the embeddings, and the similar-tracks, describe, radio and map features.

## Features

### Library

Every track in the catalog, with its cover, artists, album, year and length, searchable by artist, title, album or genre and sortable by any column. The card on the right shows the selected track: its cover, its tags, a description of how it sounds and feels (written by a language model from the facts about it and what it knows of the track), and its genres from each source. **Scan** reads new and changed files; **Enrich** runs the fetchers you've turned on.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/library-dark.png">
  <img alt="Library" src="docs/screenshots/library-light.png">
</picture>

### Similar

The tracks that sound most like a chosen one, best match first, with how alike each is. It compares the tracks' audio embeddings and lists the closest ones as they are, other versions and the same artist included, so it's a straight answer to "what sounds like this?". **Play** plays them in that order.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/similar-dark.png">
  <img alt="Similar" src="docs/screenshots/similar-light.png">
</picture>

### Describe

Type what you want to hear, like "chill late-night electronic, with a warm bassline", and get the tracks that fit, best match first. It matches the description against how each track sounds (a CLAP model, which puts music and text in one space) and against the tracks' own written descriptions, and combines the two rankings. Play the results, or start a radio station from the best ones.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/describe-dark.png">
  <img alt="Describe" src="docs/screenshots/describe-light.png">
</picture>

### Map

The whole library laid out by how it sounds: similar-sounding tracks sit close together, coloured by cluster. Hover a dot to see the track, click it to select it, double-click to play it; zoom and drag to explore. **Play cluster** plays the selected track and the rest of its neighbourhood, and the search box fades everything that doesn't match.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/map-dark.png">
  <img alt="Map" src="docs/screenshots/map-light.png">
</picture>

### Radio

An endless station from any track (in the library, on the map) or from a description's best matches. Each next track is picked as you listen: a track played to its end pulls the station towards it, and a track skipped early steers it away from tracks like it (skipped tracks are faded in the list). The **Close / Wander** slider sets how far it drifts from where it started, and **Save as playlist** writes what you listened to as an .m3u file.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/radio-dark.png">
  <img alt="Radio" src="docs/screenshots/radio-light.png">
</picture>

### Settings

Each catalog keeps its own settings. Every fetcher (the genre sources, the audio models, the language model) can be turned on and tuned on its own, and says what it does, why it can't run (a missing API key, a fetcher it needs that's off), what it will download, what uses it, and how far it is on the catalog. The Models tab downloads or deletes the local models, and each feature picks the embeddings it uses.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/settings-dark.png">
  <img alt="Settings" src="docs/screenshots/settings-light.png">
</picture>

## Running it

```sh
uv sync
uv run crate
```

It needs Python 3.12 and [VLC](https://www.videolan.org/vlc/) for playback. It has only been used on macOS.

In the app, **New catalog** scans a music folder into a catalog folder. That folder holds `catalog.json`, `settings.yaml`, the embeddings and the cover thumbnails; the music files are only read. Then turn on the fetchers you want in **Settings › Fetchers** and press **Enrich**. Enrich downloads the models it needs into `~/crate_models` (the `models.folder` setting).

The language-model fetchers need an API key, set in **Settings › Services** or with the provider's environment variable (`ANTHROPIC_API_KEY`, ...). Each of their calls costs money: with the default model, about $3 for LLM genres and $10 for descriptions per 2,000 tracks.

## Tests

```sh
uv run --package libcrate pytest libcrate
uv run --package crate pytest crate
```

The tests make their own small FLAC files. The tests on real recordings (Essentia, the CLAP models, playback) use a `test_music/` folder of tagged FLACs at the repo root, which isn't in the repository, and are skipped without it.
