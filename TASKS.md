# Tasks

Goal: everything collected here feeds a future radio / set maker ("I want something chill, 100 songs"). Genres alone don't make a good set; the tasks below add what's missing: mood and energy to pick the songs, tempo and key to order them, and embeddings for similarity and transitions.

## Radio

Done 2026-10-15: `libcrate.Radio` and the Radio page (see the READMEs). A station walks the embedding space from a track or a description's best matches, steered by tracks played through and early skips (session only). Still open:

- Tempo and key for smoother transitions, once the Tempo and key task below gives them (none of test_catalog's tracks has a BPM today).
- Remembering skips across sessions.
- Tuning `drift`, `choices` and `SKIP_PENALTY` by listening; drift 0.3 stays within a scene, 0.8 crosses genres within a few tracks.

## LLM description experiment

Done 2026-10-15 on 200 tracks of test_catalog: 100 the LLM genre fetcher knew, 100 it didn't. Each got two descriptions from the settings' LLM (3-4 sentences plus keywords): **A** from the tags and the model's own knowledge only (empty when it doesn't recognise the track), **B** from facts (tags, MusicBrainz/chosic/Essentia genres, and the 8 MuQ-MuLan phrases that fit the track unusually well out of ~150 sound/mood/use phrases) plus knowledge. Descriptions were embedded with bge-base-en-v1.5. 24 queries mixing sound, mood, use, language and era ranked the 200 tracks four ways; an LLM judge, seeing tags and all genre sources but no descriptions, rated every pooled top 10.

| method | P@10 | unknown tracks in the top 10s | their P@10 |
|---|---|---|---|
| MuQ-MuLan (Describe today) | 0.43 | 104 | 0.48 |
| A: knowledge only | 0.43 | 32 | 0.47 |
| **B: facts + knowledge** | **0.54** | 128 | 0.49 |
| B and MuQ-MuLan, rank-fused | **0.54** | 116 | **0.54** |

- Text wins on language and scene ("Romanian rock" 0.4 vs 0.1, "Russian-language pop" 0.4 vs 0.2), on mood and use ("sad piano song" 0.6 vs 0.1, "calm ambient music for sleeping" 0.5 vs 0.1) and on "deep house for late at night" (0.9 vs 0.6); MuQ-MuLan stays better on groove and feel ("funky disco groove", "romantic slow song for a wedding", "jazzy hip hop").
- A is empty for 87 of the 100 unknown tracks, so knowledge alone can't carry search. B describes every track; for unknown ones it is a fluent summary of the genres and audio tags.
- Caveats: the judge sees the genres B was given (a bias towards B); a 200-track pool leaves some queries with only 2-3 fitting tracks (reggae, Spanish); "recognised" isn't stable (the genre fetcher and this run disagree on Stromae, Black); the model still sometimes reads the title ("French-style" for "Say Oh La La"). Release years can't flag made-up knowledge: most mismatches are compilation or reissue years in the tags, where the model was right.
- Language is the biggest gap: B gives no language for 96 of 200 tracks, mostly the unknown ones.
- Cost: B took about 190k input and 34k output tokens for 200 tracks, so roughly $10 for the whole library at Sonnet prices.

Scripts and data: `experiments/descriptions/` (see `experiments/README.md`).

Done: built as the `llm-descriptions` fetcher (approach B: facts plus knowledge, with MuQ-MuLan audio tags), stored in `TrackRecord.descriptions`, embedded as `description-bge` (bge-base-en-v1.5, local), and fused with MuQ-MuLan in Describe. Still open: tightening the prompt against reading titles, Essentia moods as another fact, detecting the sung language from the audio, a text-similarity term for Radio, and a better Match % for fused results (reciprocal rank fusion bunches near 100%: 98%, 97% for ranks 2 and 3).

## Mood and energy

What "chill", "dark" or "energetic" means for a track. Possible sources, best combined like the genre sources:

- Essentia's Discogs-EffNet classification heads, which reuse the embeddings the genre fetcher already computes, so they cost little extra per track. The mood/theme model (`mtg_jamendo_moodtheme-discogs-effnet-1`, not yet in `libcrate.models`) gives 56 moods/themes; check Essentia's model zoo for single-mood heads (relaxed, happy, aggressive, sad, party) and danceability.
- The LLM fetcher could return mood and energy in the same request as the genres, for a few extra output tokens (a new answer field changes the prompt hash, so the genres are redone).

For Essentia, a new head is a method on `Models` (`models/essentia/models.py`), its name in `Models.REQUESTS`, a method on `EssentiaClient`, its files in `MODEL_FILES` and an entry in `libcrate.models`; for the LLM, a prompt, an answer schema and a method that calls `LLMClient._ask()`.

To decide: how to store them in `TrackRecord` (a new section next to `genres`, with source, status and confidence like genres?), and the head's entry in `libcrate.models` (it runs on Discogs-EffNet, so `requires` it).

## Tempo and key

BPM and musical key, to order a set and keep transitions smooth (harmonic mixing). Essentia's standard extractors compute both locally from the audio (rhythm and key extraction); add them as a source like Essentia genres, decoding the audio once for genres, moods, tempo and key.

## LLM genre experiment with multiple models

Compare several models (and providers, since the fetcher goes through LangChain) on genre accuracy rather than agreement, using tracks from the Discogs-400 dataset as ground truth: the same 400 styles the Essentia model and the LLM fetcher answer in. Start from `libcrate/tools/llm_pilot.py`, which today only measures agreement with chosic/MusicBrainz/Essentia on the user's own library, and add scoring against known labels. The same ground truth can show how far to trust Essentia at its 15% threshold.

## CLAP model experiment

Done 2026-10-15 on test_catalog (2,068 tracks, the same 30 s per track: 10 s at 25%, 50% and 75% of it). Describe: 32 genre descriptions ("techno", "dark techno", "psytrance", "hip hop", ...), scored by how many of the top 20 tracks carry the genre (LLM, chosic, MusicBrainz). Similar: how many of each track's 10 nearest neighbours share an LLM genre.

| model | Describe, top 20 | top 50 | descriptions won | Similar |
|---|---|---|---|---|
| MuQ-MuLan large (Tencent) | **0.35** | **0.27** | **19** of 32 | **0.38** |
| laion-clap-music (ours) | 0.30 | 0.23 | 10 | 0.37 |
| larger_clap_general (LAION, HF) | 0.26 | 0.22 | 6 | 0.34 |
| Microsoft CLAP 2023 | 0.22 | 0.19 | 5 | 0.33 |

(discogs-effnet over whole tracks, for Similar: 0.39.) MuQ-MuLan is the only one that keeps techno and psytrance apart ("dark techno": 0.95 vs laion's 0.35, "psytrance": 0.50 vs 0.25) and also wins downtempo, disco, folk, metal; laion stays better at indie rock, funk, organic house and tech house. Listening to free descriptions agrees ("energetic female vocal pop": Kesha and OneRepublic vs laion's psytrance).

Scripts and results: `experiments/clap/`.

Done: MuQ-MuLan is in as `muq-mulan`, Describe's default (`models/clap/muq_mulan.py`, our own rebuild for transformers 5, matching Tencent's package); laion-clap-music stays as an option. CLAP tracks are now embedded from the same three 10 s clips as here.
