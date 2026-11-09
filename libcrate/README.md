# libcrate

The backend of [crate](../crate/README.md). It extracts everything known about the songs in a FLAC music library into a JSON catalog described by pydantic models: file tags, audio info, cover info, genres per source (looked up online from MusicBrainz and chosic, predicted from the audio by Essentia, known to a language model) and descriptions written by a language model, plus embeddings of the audio and of those descriptions. With them it finds similar tracks, tracks that sound like a description, an endless radio station and a map of the library. The catalog, the embeddings and the settings live in a catalog folder. The music files are only read, never modified.

It has no command line: [crate](../crate/README.md) uses it in-process. Nothing in it prints or exits; errors are raised.

A catalog is made by `MetadataStore.init`, which writes the default settings to the catalog folder and scans the library into it. Then it works in two passes:

- **scan** is fast: it reads what the files themselves hold (tags, audio info, cover info) from the whole library, with no network and no models.
- **enrich** is slow: it runs the fetchers the settings turn on to add what the files don't hold, looking tracks up online, running ML models on the audio and asking language models, from a fraction of a second to several seconds per track.

Both are incremental: a scan only re-reads new or changed files, and an enrich only processes tracks a fetcher has no result for, whose last attempt failed, or whose result was made with other settings. Online fetchers run in parallel, and an enrich saves as it goes, so stopping it loses nothing. The catalog knows its library, so later scans need only the catalog folder.

The catalog folder holds `catalog.json`, `settings.yaml`, `embeddings/<model>/mean.parquet` and, once made, the music map `embeddings/<model>/map.parquet`.

```python
from libcrate import MetadataStore, check, progress

store = MetadataStore("~/music-metadata")
catalog, report = store.init("/Volumes/KINGSTON/Spoti/All")   # default settings.yaml (unless there is one), first scan
store.init("/Volumes/USB/Music", settings_from="~/music-metadata")   # or another catalog's settings
catalog, report = store.scan()                                # later: new and changed files of the same library

settings = store.settings()                                   # the catalog's settings.yaml
report = store.run(catalog, settings, on_progress=..., stop=...)   # Enrich: the fetchers that are on, in order,
                                                              # their models downloaded first; saved as it goes
check(settings, "llm-descriptions").problems                  # what stops a fetcher: ["no API key for anthropic: ..."]
progress(catalog, settings, "chosic-genres").to_do            # what the next Enrich does for it

catalog = store.load()                                        # None before init
track = catalog.tracks["Metronomy - The Light [GBMVH1900150].flac"]
track.tags.title, track.tags.spotify_track_id, track.audio.duration
track.genres["essentia-genres"].genres     # [Genre(name='chillwave', parent='Electronic', confidence=0.2583), ...]
track.genres["musicbrainz-genres"].status  # SourceStatus.FOUND / NOT_FOUND / ERROR
track.genre_names()                        # deduplicated across all sources

table = store.current_embeddings(catalog, settings, "discogs-effnet")   # rows made with its current settings
paths, matrix = table.matrix()             # sorted catalog paths, float32 (tracks, 1280)
```

## Settings

Each catalog has its own settings, in `settings.yaml` in the catalog folder: `MetadataStore.init` writes a commented default file, readable only by you, or copies another catalog's (`settings_from`: its folder or its `settings.yaml`) into a catalog that has none yet. A catalog without one (made before settings existed) uses the defaults. crate edits the file on its Settings page.

The file has these sections, each key with a comment saying what it is (a fetcher's section: what the fetcher does):

- `keys`: API keys by provider (`anthropic`, `openai`, `google_genai`); empty, the provider's environment variable is used (`ANTHROPIC_API_KEY`, ...).
- `services`: the online services: `llm` (the language model and its options), `musicbrainz`, `chosic` (timeouts, chosic's request interval).
- `models`: the folder of the models that run locally (default `~/crate_models`), one subfolder per model.
- `fetchers`: one entry per fetcher, under what it fetches: `genres` (`musicbrainz-genres`, `chosic-genres`, `essentia-genres`, `llm-genres`), `embeddings` (`discogs-effnet`, `discogs-track`, `muq-mulan`, `laion-clap-music`, `description-bge`) and `descriptions` (`llm-descriptions`). Each has `enabled` (whether Enrich runs it; all off by default) and its own options; the LLM fetchers name the service they ask (`service: llm`).
- `similar`, `map`, `describe`, `radio`: the features, each naming the embedding fetchers it uses (`embeddings`; Describe's `sound` and `words`) and its own options.

```yaml
# what Enrich runs, by what it fetches
fetchers:
  # genres, from several sources: each keeps its own
  genres:
    # the Spotify genres of the track's artist, from chosic.com, looked up by the track's Spotify ID
    chosic-genres:
      # Enrich runs it on the tracks that need it; off, its results stay as they are
      enabled: true
      # keep at most this many genres
      max_genres: 5
```

Unknown keys are reported as errors, so a typo doesn't silently fall back to a default. In code, `store.settings()` (`load_settings(path)`) reads the file over the defaults, `Settings()` is the defaults, `settings.fetchers.get("chosic-genres")` is a fetcher's section, and `save_settings(settings, path)` writes a commented file for given values. Each comment is a field's description, or a section's docstring (`about(section)`), in `settings.py`. The clients never read the file: the fetchers build them from the settings. The pilot script in `tools/` reads the settings of the catalog it's given.

## Fetchers

A fetcher (`libcrate.fetchers`) gets one kind of result for one track at a time: genres or a description, kept in the track's record (`MetadataFetcher`s, run by `enrich()`), or an embedding, kept in a table of its own (`EmbeddingFetcher`s, run by `embed()`). `make_fetcher(name, settings, root)` builds one from the settings; that's cheap, since nothing starts until `open()` builds its client (a network session, a model in a child process). Each says what it needs:

- `services()`: the services it asks, which have to be set up (a language model needs its provider's key).
- `models()`: the local models it runs (`libcrate.models`), downloaded before it runs.
- `inputs()`: the other fetchers whose results it uses, required or not. `description-bge` requires `llm-descriptions`; `llm-descriptions` uses the genre fetchers and `muq-mulan` (for its audio tags) when they're on.

`store.run(catalog, settings)` (`fetchers.run`) is Enrich: the fetchers that are on, in the order their inputs need (`plan(settings)`: at the same depth, genres before embeddings before descriptions), after downloading the models they need (each a step, `download <model>`). Consecutive metadata fetchers run in parallel, embedding fetchers one after another. A fetcher with a problem doesn't run; the `RunReport` says why (`not_run`).

`check(settings, name)` (`fetchers.status`) says what stands between a fetcher and running it: `problems` (no API key, a required fetcher that's off or can't run), `notes` (an optional one that's off) and `downloads` (models to fetch first, with `download_size`). It's local and free: it never calls a service. `progress(catalog, settings, name, table)` counts the tracks done with the current settings, to redo (made with other settings), never done, failed, and with nothing to go on (no ID, no description).

Every result's key holds what it depends on, the model and options included, so changing a fetcher's settings redoes its results on the next Enrich. The LLM fetchers' keys hold a hash of their prompt and answer schema, so editing either redoes them too.

## Models

`libcrate.models` knows every model that runs locally: Essentia's Discogs-EffNet (`discogs-effnet-bs64-1`), its genre head (`genre_discogs400-discogs-effnet-1`, which runs on it) and its track-similarity version (`discogs_track_embeddings-effnet-bs64-1`), the CLAP models (`muq-mulan`, `laion-clap-music`) and the text model `bge-base`, each with its kind, size, files and the models it requires. Each lives in its own subfolder of the `models.folder` setting. `missing(folder, names)` lists what isn't installed (required models included), `download(folder, names)` fetches it (a model is built in `<name>.part` and only renamed once complete), and `delete(folder, name)` removes one. Nothing else downloads: clients raise when their model is missing.

## Tests

```sh
uv run --package libcrate pytest libcrate      # from the repo root
```

The tests build small header-only FLAC files on the fly, so they don't need the library drive. They never read your settings files and remove provider API keys from the environment, so they can't make paid calls. `test_real_files.py` also reads the tagged FLAC files in `test_music/` at the repo root (read-only) and is skipped if that folder is missing. The folder isn't in the repository (the tracks are commercial recordings): put a few tagged FLACs of your own there, though the real-model tests below check genres recorded for specific tracks.

Tests marked `essentia` run the real Essentia models on `test_music/` and check they give the genres recorded in the tests. They take about 20 seconds and are skipped when essentia or the models aren't installed; `-m "not essentia"` leaves them out. Tests marked `clap` run the real CLAP models on `test_music/` (about a minute), checking each against its makers' own code (Tencent's `muq` package, LAION's `laion_clap`); a model's are skipped until it's downloaded, and `-m "not clap"` leaves them out.

## Incremental updates

A file is re-extracted only when it is new, its size or mtime changed, or its record has a different shape: each record stores `SCHEMA_HASH`, a hash of the `TrackRecord` schema, so adding, renaming or retyping a field re-extracts everything without a version number to bump. Changes to the extraction logic alone don't change the hash; deleting `catalog.json` redoes everything then. Records for deleted files are dropped, and a file that fails to read keeps its last good record. Catalog keys are paths relative to the library root, so the catalog stays valid if the drive is mounted somewhere else.

Genre and description results are saved to the JSON every 25 results, so a long run can be stopped and picked up later without losing what was done. A source whose first 3 results all fail (a missing API key, the network down) is stopped for that run instead of recording an error for every track; the failed tracks are retried next time.

## Genres

`TrackRecord.genres[source]` holds one result per source, with:

- `status`: `found`; `not_found` (the source has nothing, or the track has no ID it can be looked up by; not retried); or `error` (network failure, rate limit, undecodable audio, ...; retried on the next run). A source missing from `genres` was never asked.
- `key`: what the result was based on (`isrc:...`, `mbid:...`, `spotify:...`; for Essentia, the model and its settings). If it changes, the track is redone.
- `genres`: names, plus `votes` for MusicBrainz (net community votes, summed over recording, release group and artist), and `confidence` and `parent` for Essentia and the LLM.

When a file is re-read, its results are kept.

## Genre sources

An enrich uses these through the genre fetchers in `libcrate.fetchers.genres`. The clients behind them aren't tied to genres: each wraps one online service (`libcrate.services`) or one kind of local model (`libcrate.models.essentia`, `libcrate.models.clap`), with genres as the first thing it looks up, so moods or tempo can be added as more methods. They can also be used on their own.

**MusicBrainz** (`libcrate.services.musicbrainz`): community-voted genres, looked up by recording ID or ISRC. The ISRC comes from the `ISRC` tag, or from the `[ISRC]` at the end of the filename when the tag is missing. It pools the votes of the recording, its release group and its artist(s), then keeps the most-voted. Requests are limited to one per second, as MusicBrainz requires.

A recording ID is a track's, not a release's (album). The `musicbrainz-genres` settings control the pooling.

**Chosic** (`libcrate.services.chosic`): genres by Spotify track ID, which the files carry in their `COMMENT` URL. These are the genres of the track's first artist, so every track by an artist gets the same list.

**Essentia** (`libcrate.models.essentia`): genres predicted from the audio by the Discogs-EffNet `genre_discogs400` model, locally. It keeps the top 3 Discogs styles scoring at least 15% (`max_genres`, `min_confidence` of `essentia-genres`). The audio is resampled to 16 kHz with `resampleQuality=4`, as in Essentia's own examples: about 4× faster than quality 1, and closer to the best quality (0). Expect a few seconds per track, plus a few seconds to load the models once per run. The models run in separate processes, `processes` of them (default 4), each analysing one track at a time; they start as tracks need them and stop when the fetcher is closed (the end of the `with` block), because TensorFlow holds on to about 1.2 GB per process until it exits. Fewer processes save memory; more only help while there are idle cores, since TensorFlow already uses a few per process. It needs Discogs-EffNet and the genre head (about 20 MB) in the models folder; Enrich downloads them.

**LLM** (`libcrate.services.llm`): genres a language model knows for the track, asked by artist, title, album, year and label. It goes through LangChain, so any provider works: the model is a LangChain string like `anthropic:claude-sonnet-5-5` or `openai:...`, with that provider's LangChain package installed. The API key comes from the `keys` section, or the provider's environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, ...): in fish, `set -gx ANTHROPIC_API_KEY sk-ant-...`. Every call costs money: about $3 per 2,000 tracks with the default model.

The default is Claude Sonnet 5.5 with `effort=low`, chosen from the pilots: it knew about 30% more tracks of a mostly-electronic library than Claude Haiku 5.5. `services.llm` sets the model and its `options`; in `llm-genres`, `min_confidence` (default 0.5) drops styles the model rates lower, and `workers` (default 8) sets how many requests are in flight; 2,000 tracks take about 10–15 minutes.

- The model answers in the same 400 Discogs styles as Essentia (`Parent---Style`); answers outside the list are dropped. The style list is a fixed system prompt that providers cache between requests (marked for caching on Anthropic; OpenAI and Gemini cache long repeated prefixes on their own).
- The model says whether it knows the track at all; an unknown track becomes `not_found` instead of a guess from the title's words. Each genre carries the model's confidence.
- Without an API key it doesn't run (`check()` says so); a key that's set but rejected stops it after its first 3 lookups fail, as any fetcher does, instead of recording an error for every track.

To compare models before changing the default, see the pilot script in [`tools/`](tools/README.md).

## Descriptions

The description fetchers (`libcrate.fetchers.descriptions`) write how a track sounds and feels, in words, into `TrackRecord.descriptions[source]`: a `DescriptionSource` with the status, key and error a genre result has, and a `Description` (`text`, `keywords`, the lyrics' `language`, and whether the source `recognised` the track). They're `MetadataFetcher`s like the genre fetchers, run by the same `enrich()` (`libcrate.fetchers.enrich`); each fetcher's `kind` says which part of the record it fills.

**llm-descriptions** asks its service's language model (paid API calls; about $10 per 2,000 tracks with the default model) to describe the track from facts: its tags, the musicbrainz, chosic and Essentia genres, and **audio tags**, the 8 phrases (instruments, vocals, mood, rhythm, production, use; `fetchers/descriptions/audio_tags.py`) a CLAP model's embedding fits unusually well (`audio_tags_from`, default muq-mulan, while that fetcher is on and once its embeddings exist; `audio_tags` sets how many). If the model knows the track or artist, it adds what it knows. So every track with an artist and a title gets a description, known to the model or not; in an experiment on a 2,000-track library, searching these found the right tracks more often than searching descriptions from the model's knowledge alone (precision at 10: 0.54 against 0.43), which also leave out the half of the library the model doesn't know. The key holds a hash of the facts but the audio tags (which shift a little as the library grows), so a track is described again when its tags or genres change.

Enrich runs it after the genre fetchers and muq-mulan, since it uses them:

```python
store.run(catalog, settings, ["llm-descriptions"])
catalog.tracks[path].descriptions["llm-descriptions"].description.text
```

## Embeddings

`MetadataStore.embed` computes one vector per track per model, for similarity: similar tracks, the map, later sets and transitions. Each model has its own folder in the catalog folder, `embeddings/<model>/`, so a model can hold more than one kind of embedding; today that's `mean.parquet`, with one row per track: `path` (the catalog key), `key` (the model and its settings), `n_frames` (how many frames the vector averages) and `embedding` (float32). They stay out of `catalog.json`, which would otherwise grow by about 7 KB per track and model.

- A track is computed when its model's file has no row for it, or the row's `key` differs (a new model version or settings). Rows of tracks no longer in the catalog are dropped.
- Failures aren't stored: they're reported for that run, and the track is tried again next time. A model whose first 3 tracks all fail is stopped, as genre sources are.
- Models run one after another, so only one model's processes are loaded at a time; within a model, several tracks are computed at once (`EmbeddingFetcher.workers`; for Essentia models, one per process).
- Progress is saved every 25 tracks.

`libcrate.embeddings.EmbeddingTable` reads and writes the files; `MetadataStore.load_embeddings(name)` loads one, and `current_embeddings(catalog, settings, name)` only its rows made with the fetcher's current settings: what the features use, since vectors from another model revision can't be compared with a query from the current one. Embedding fetchers are `EmbeddingFetcher`s in `libcrate.fetchers.embeddings` (a `name`, a `space` saying what their vectors compare, a `key()`, and `fetch(record)` returning an `Embedding`); adding one means a class there, an entry in `fetchers.registry.FETCHERS`, and its settings section.

**discogs-effnet**: Essentia's Discogs-EffNet embeddings (1280 dimensions), the ones its genre model scores, averaged over the first `max_seconds` of the track (about one frame per second), with `processes` model processes (default 4, about 1.2 GB each). With both essentia-genres and discogs-effnet on, the audio is analysed twice: genres first, then embeddings, each with its own processes.

**discogs-track**: the same network, trained by Essentia's authors for track similarity rather than genres (contrastively, on Discogs metadata: tracks of the same release end up close together), so possibly better for similar tracks; 1280 dimensions too, the same settings, and as fast. Each analyses the audio on its own. To compare, compute both and switch `similar.embeddings`; on a 2,000-track library the plain discogs-effnet kept closer to the seed's genres, so it stays the default.

**The CLAP models** put a track's sound and descriptions of music in one 512-dimension space, so their embeddings can be matched to text (see Describe below) as well as to each other. A track is embedded from three 10 s clips, centred a quarter, half and three quarters of the way through it, and their mean (L2-normalised) kept. They run in `processes` child processes (`models.clap`, through the same `models.workers.WorkerPool` as Essentia), 1.5–3 GB each and sharing the cores between them, so torch never loads in the caller. They live in the models folder, one subfolder each, downloaded by `libcrate.models`. Another model is one more entry in `models.clap.models.MODELS`, in `libcrate.models.MODELS`, and a fetcher.

**muq-mulan**: Tencent's MuQ-MuLan (large), trained on music and descriptions of it; Describe's default. In a comparison on a 2,000-track library it matched genre descriptions best of four models (precision at 20: 0.35, against 0.30 for laion-clap-music), and it is the one that tells techno from psytrance. The release's weights are used as they are (`OpenMuQ/MuQ-MuLan-large`, 2.65 GB, at a pinned revision, with XLM-RoBERTa's tokenizer; CC BY-NC 4.0, so not for commercial use), run by our own rebuild of the network (`models/clap/muq_mulan.py`, from Tencent's MIT-licensed code, whose package needs transformers 4); it gives the same numbers as Tencent's `muq` package. Each process builds only the half it needs: the audio one to embed tracks, the text one for Describe. About 3 s a track on the CPU.

**laion-clap-music**: LAION's CLAP trained on music. The copy on Hugging Face (`laion/larger_clap_music`) isn't LAION's checkpoint: it maps every text to nearly the same vector. So LAION's original release (`lukewys/laion_clap`, `music_audioset_epoch_15_esc_90.14.pt`, 2.35 GB, at a pinned revision) is downloaded and converted into transformers' CLAP format in a child process (`models/clap/convert.py`), using the Hugging Face copy's config and tokenizer files; the original is deleted after, leaving 780 MB. The conversion is checked against LAION's own `laion_clap` package: the same text embeddings (`test_clap.py` keeps the reference numbers) and the same search results on `test_music/`.

## Features

Similar, Map, Radio and Describe use embeddings, from the fetchers their settings name (`libcrate.features`): a slot each (Describe has two, `sound` and the optional `words`), each taking only fetchers whose vectors fit (`EmbeddingFetcher.space`: Similar, Map and Radio compare tracks by their sound; Describe's sound needs a CLAP model, its words a text model). `features.problem(settings, slot)` says when a slot is empty or holds the wrong kind; `features.note()` when its fetcher is off, so Enrich won't keep it up to date (its embeddings are still used). `features.used_by(settings, name)` lists the features and fetchers that use a fetcher's results.

## Similar tracks

`libcrate.Similar` lists the tracks that sound most like a chosen one, compared by one embedding fetcher's (the `similar.embeddings` setting): the chosen track first, then up to `similar.count` others, best match first. Similarity is cosine similarity on centred embeddings (the chosen track scores 1). Nothing is skipped, other versions of the same song and the same artist's tracks included, so the list is the same every time; leaving those out is Radio's job. Tracks without an embedding are left out.

```python
settings = store.settings()
similar = Similar(catalog, store.current_embeddings(catalog, settings, settings.similar.embeddings))
for item in similar.find("Metronomy - The Light [GBMVH1900150].flac", **settings.similar.options()):
    item.track, item.similarity      # 1 for the chosen track
```

## Describe

`libcrate.TextSearch` lists the tracks that sound like a description ("chill late-night electronic"), best match first: the cosine similarity of each track's CLAP embedding (the `describe.sound` setting) and the description's, embedded by the same fetcher's model. Tracks without an embedding are left out; `describe.count` sets how many are listed.

The same works over the tracks' own descriptions: `description-bge` embeds each track's descriptions (`TrackRecord.description_text()`, with their keywords) with a text model, bge-base-en-v1.5 (`models.clap.TEXT_MODELS`, 440 MB), and the query is embedded by it with `ClapClient.query_embeddings`. It runs after the description fetchers and only embeds tracks that have a description, again when it changes. `libcrate.fuse` combines rankings by reciprocal rank fusion; the app fuses the sound's and the descriptions' (`describe.words`, optional) once both exist, and a fused match is relative, 1 for a track first in both.

```python
from libcrate import TextSearch, make_fetcher
from libcrate.models.clap import ClapClient

settings = store.settings()
sound = make_fetcher(settings.describe.sound, settings)       # a CLAP embedding fetcher, and its model
search = TextSearch(catalog, store.current_embeddings(catalog, settings, sound.name))
with ClapClient(settings.models.folder, sound.model, processes=1) as clap:
    [vector] = clap.query_embeddings(["chill late-night electronic"])
for item in search.find(vector, count=settings.describe.count):
    item.track, item.similarity
```

## Radio

`libcrate.Radio` is an endless station from one track, or from several (the app uses a description's best matches), compared by one embedding fetcher's (`radio.embeddings`, default `discogs-effnet`). The first seed plays first; `next()` picks each track after it, as it's needed. Every candidate scores `(1 - drift)` times its similarity to the station's anchor plus `drift` times its similarity to the last track picked, less `SKIP_PENALTY` times its similarity to the most alike skipped track. The anchor starts as the seeds' mean and moves a little towards each track reported as played through (`listened()`); `skipped()` reports an early skip. The next track is drawn at random from the `radio.choices` best, so a station from the same seed differs every time. Never picked: a track already picked, another version of a song already picked, a near-identical track (`radio.duplicate_similarity`), or an artist picked within the last `radio.artist_gap` tracks (unless nothing else is left). `drop_after(path)` forgets the tracks picked after one, so they're picked again after a skip.

```python
settings = store.settings()
table = store.current_embeddings(catalog, settings, settings.radio.embeddings)
radio = Radio(catalog, table, [seed_path], **settings.radio.options())
item = radio.next()                  # TrackMatch: the track, and its similarity to the anchor
radio.listened(item.track.path)      # played through: the station leans towards it
radio.skipped(item.track.path)       # skipped early: tracks like it come less
```

## Music map

`libcrate.MusicMap` lays the tracks out in 2-D by one embedding fetcher's (the `map.embeddings` setting), for the app's map: similar-sounding tracks close together, each in one of `map.clusters` clusters for colouring. The embeddings are normalised and centred as for similar tracks, reduced to 50 dimensions by PCA, placed by t-SNE (cosine), which keeps each track's nearest neighbours close (distances between far-apart regions mean less), and grouped by k-means on the same vectors. It needs at least 10 tracks with embeddings.

```python
name = settings.map.embeddings
table = store.current_embeddings(catalog, settings, name)
music_map = store.music_map(name, clusters=settings.map.clusters, table=table)   # a few seconds the first time
music_map.paths, music_map.xy, music_map.clusters                         # sorted paths, (tracks, 2), (tracks,)
```

`MetadataStore.music_map` saves it as `embeddings/<model>/map.parquet`, with a fingerprint of each track's path and embedding key and the settings, and makes it again only when that changes. scikit-learn is imported only when a map is made, since it takes seconds to import.
