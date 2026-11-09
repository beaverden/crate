# tools

Experiment scripts that use `libcrate` but aren't part of the installed package. Run them from `libcrate/` with `uv run python tools/<script>.py`.

## llm_pilot.py

How much do a language model's genres agree with the other sources? It samples tracks from a catalog that already have chosic or MusicBrainz genres, asks the model about each (paid API calls; nothing is written to the catalog), and reports:

- how often the model says it knows the track;
- how often its styles agree with chosic/MusicBrainz, exactly or loosely ("house" vs "deep house");
- agreement by the model's confidence, to see whether the confidence means anything;
- how often it agrees with Essentia;
- token usage, cache reads included.

```sh
uv run python tools/llm_pilot.py ~/music-metadata -m anthropic:claude-sonnet-5-5 -o effort=low -n 50 --report sonnet.json
uv run python tools/llm_pilot.py ~/music-metadata -m anthropic:claude-haiku-5-5 -n 50 --report haiku.json
```

The default `--seed` picks the same tracks every run, so models can be compared on the same sample. The model, options and API key default to the ones in the catalog's `settings.yaml`.

This measures agreement, not accuracy: none of the sources is ground truth (chosic's genres are the artist's, Essentia keeps weak guesses down to 15%, MusicBrainz knows few tracks), so read the per-track lines where they disagree.

Results so far (2026-10-11, 50 tracks): Claude Sonnet 5.5 knew 38 tracks and Claude Haiku 5.5 29; on the 29 both knew, each agreed with chosic/MusicBrainz on 24. Sonnet became the fetcher's default.
