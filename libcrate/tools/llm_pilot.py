"""Pilot: how much do a language model's genres agree with the other sources?

A standalone experiment script, not part of the installed package. From libcrate/:

    uv run python tools/llm_pilot.py ~/music-metadata -m anthropic:claude-sonnet-5-5 -o effort=low -n 50
    uv run python tools/llm_pilot.py ~/music-metadata -m anthropic:claude-haiku-5-5 -n 50 --report haiku.json

Samples tracks that already have chosic or MusicBrainz genres, asks the model
about each (paid API calls; nothing is written to the catalog), and reports:

  known           how often the model says it knows the track at all
  agrees exactly  a guessed style equals a chosic/MusicBrainz genre (case and punctuation ignored)
  agrees loosely  ... or one contains the other ("house" vs "deep house")
  by confidence   exact agreement per confidence of the model's top style,
                  to see whether its confidence means anything

Essentia's genres, when the track has them, are compared separately: same
Discogs styles, but predicted from the audio.

This measures agreement, not accuracy: none of the sources is ground truth.
chosic's genres are the artist's, Essentia keeps weak guesses down to 15%, and
MusicBrainz knows few tracks. Read the per-track lines where they disagree.

Token usage, cache reads included, is reported so cost can be extrapolated.

The API key comes from the provider's environment variable (ANTHROPIC_API_KEY,
OPENAI_API_KEY, ...). An api_key=... option works too, but is left out of the report.
"""

import argparse
import json
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path

from libcrate.catalog import Catalog
from libcrate.services.llm.client import LLMClient, LLMError, Usage
from libcrate.settings import SettingsError
from libcrate.store import MetadataStore
from libcrate.track import SourceStatus, TrackRecord
from libcrate.util import split_label

from console import Console  # next to this script

__all__ = ["compare", "main", "normalize", "pick_tracks"]

REFERENCE_SOURCES = ("chosic-genres", "musicbrainz-genres")
CONFIDENCE_BUCKETS = ((0.8, "≥ 80%"), (0.5, "50–80%"), (0.0, "< 50%"))
# Stop if this many tracks in a row fail at the start: a missing API key or a wrong
# model name fails every call, so there's no point going through the whole sample.
GIVE_UP_AFTER = 3



def parse_options(pairs: list[str]) -> dict:
    """["effort=low", "temperature=0"] -> {"effort": "low", "temperature": 0}; values are JSON when they parse."""
    options = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options

def normalize(name: str) -> str:
    """'Synth-Pop' -> 'synthpop'; 'Drum & Bass' -> 'drumbass'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def reference_genres(record: TrackRecord, sources=REFERENCE_SOURCES) -> list[str]:
    names = []
    for source in sources:
        result = record.genres.get(source)
        if result and result.status == SourceStatus.FOUND:
            names += [g.name for g in result.genres]
    return names


def pick_tracks(catalog: Catalog, n: int, seed: int) -> list[TrackRecord]:
    candidates = [t for t in catalog.tracks.values() if reference_genres(t) and t.tags.artists and t.tags.title]
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def compare(guessed: list[str], reference: list[str]) -> tuple[bool, bool]:
    """(exact, loose) match between guessed style names and reference genre names."""
    g = {normalize(x) for x in guessed} - {""}
    r = {normalize(x) for x in reference} - {""}
    exact = bool(g & r)
    loose = exact or any(a in b or b in a for a in g for b in r)
    return exact, loose


def main(argv=None):
    out = Console()
    parser = argparse.ArgumentParser(description="Pilot an LLM genre fetcher against genres the catalog already has")
    parser.add_argument("output", type=Path, metavar="FOLDER", help="a catalog folder (crate's New catalog)")
    parser.add_argument("-m", "--model", help="LangChain model (default: services.llm.model of the catalog's settings)")
    parser.add_argument("-o", "--option", action="append", default=[], metavar="KEY=VALUE",
                        help="model option, e.g. effort=low (repeatable; replaces the options from the settings)")
    parser.add_argument("-n", "--sample", type=int, default=50, help="how many tracks (default 50)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, to compare models on the same tracks")
    parser.add_argument("--report", type=Path, help="write per-track results and the summary to this JSON file")
    args = parser.parse_args(argv)

    try:
        store = MetadataStore(args.output)
        catalog = store.load()
        settings = store.settings()
    except (NotADirectoryError, SettingsError) as e:
        out.print(f"  {out.paint('error', 'red', 'bold')}  {e}")
        return 2
    tracks = pick_tracks(catalog, args.sample, args.seed) if catalog else []
    if not tracks:
        out.print(f"  {out.paint('error', 'red', 'bold')}  no tracks with {' or '.join(REFERENCE_SOURCES)} genres in {args.output}")
        return 2
    service = settings.services.llm
    args.model = args.model or service.model
    max_genres = settings.fetchers.get("llm-genres").max_genres
    try:
        options = service.model_kwargs(settings.keys, parse_options(args.option) if args.option else None)
        client = LLMClient(args.model, **options)
    except (LLMError, ValueError) as e:
        out.print(f"  {out.paint('error', 'red', 'bold')}  {e}")
        return 2

    out.heading(f"Asking {args.model} about {len(tracks)} tracks")
    rows, usage, width = [], Usage(), len(str(len(tracks)))
    for i, record in enumerate(tracks, start=1):
        reference = reference_genres(record)
        essentia = reference_genres(record, ("essentia-genres",))
        row = {"path": record.path, "reference": reference, "essentia": essentia}
        try:
            guess = client.genres(record.tags, max_genres=max_genres)
        except LLMError as e:
            row["error"] = str(e)
            out.print(f"  {i:>{width}}  {out.paint('error', 'red')}  {record.path}  {out.paint(str(e), 'dim')}")
            rows.append(row)
            if len(rows) == GIVE_UP_AFTER and all("error" in r for r in rows):
                out.print(f"  {out.paint('error', 'red', 'bold')}  the first {GIVE_UP_AFTER} tracks all failed, stopping")
                return 1
            continue
        usage += guess.usage
        names = [split_label(label)[1] for label, _ in guess.styles]
        exact, loose = compare(names, reference)
        row.update(known=guess.known, styles=guess.styles, invalid_labels=guess.invalid_labels,
                   exact=exact, loose=loose, essentia_exact=compare(names, essentia)[0] if essentia else None)
        rows.append(row)

        if not guess.known:
            verdict = out.paint("unknown ", "yellow")
        elif exact:
            verdict = out.paint("agrees  ", "green")
        elif loose:
            verdict = out.paint("close   ", "cyan")
        else:
            verdict = out.paint("differs ", "red")
        said = ", ".join(f"{n.lower()} {c:.0%}" for n, (_, c) in zip(names, guess.styles)) or "-"
        out.print(f"  {i:>{width}}  {verdict} {record.path.removesuffix('.flac')}")
        out.print(out.paint(f"  {'':>{width}}           llm: {said}  |  chosic/mb: {', '.join(reference)}", "dim"))

    summary = _summary(rows, usage)
    _print_summary(out, summary)
    if args.report:
        options = {k: v for k, v in options.items() if not _is_secret(k)}
        args.report.write_text(json.dumps({"model": args.model, "options": options,
                                           "seed": args.seed, "summary": summary, "tracks": rows},
                                          indent=2, ensure_ascii=False))
        out.print(f"  {out.paint('saved', 'green')} {args.report}")
    return 0


def _is_secret(option: str) -> bool:
    return any(word in option.lower() for word in ("key", "token", "secret", "password"))


def _summary(rows: list[dict], usage: Usage) -> dict:
    answered = [r for r in rows if "error" not in r]
    known = [r for r in answered if r["known"] and r["styles"]]
    by_confidence = {label: {"tracks": 0, "exact": 0} for _, label in CONFIDENCE_BUCKETS}
    for r in known:
        top_confidence = r["styles"][0][1]
        label = next(label for floor, label in CONFIDENCE_BUCKETS if top_confidence >= floor)
        by_confidence[label]["tracks"] += 1
        by_confidence[label]["exact"] += r["exact"]
    with_essentia = [r for r in known if r["essentia"]]
    n = max(len(answered), 1)
    summary = {
        "tracks": len(rows),
        "errors": len(rows) - len(answered),
        "known": len(known),
        "unknown": len(answered) - len(known),
        "exact": sum(r["exact"] for r in known),
        "loose": sum(r["loose"] for r in known),
        "invalid_labels": sum(len(r["invalid_labels"]) for r in answered),
        "by_confidence": by_confidence,
        "essentia_compared": len(with_essentia),
        "essentia_exact": sum(r["essentia_exact"] for r in with_essentia),
        "usage": asdict(usage),
        "usage_per_track": {k: round(v / n) for k, v in asdict(usage).items()},
    }
    return summary


def _print_summary(out: Console, s: dict) -> None:
    def pct(a, b):
        return f"{a}/{b} ({a / b:.0%})" if b else "-"

    out.print()
    out.heading("Summary")
    out.print(f"  known        {pct(s['known'], s['tracks'] - s['errors'])}   errors: {s['errors']}")
    out.print("  agrees with chosic/MusicBrainz, of known tracks:")
    out.print(f"    exactly    {pct(s['exact'], s['known'])}")
    out.print(f"    loosely    {pct(s['loose'], s['known'])}")
    for label, b in s["by_confidence"].items():
        out.print(out.paint(f"    top style {label:>7}: {pct(b['exact'], b['tracks'])} exactly", "dim"))
    out.print(f"  agrees with essentia exactly  {pct(s['essentia_exact'], s['essentia_compared'])}")
    out.print(out.paint("  (agreement, not accuracy: the other sources make mistakes too)", "dim"))
    if s["invalid_labels"]:
        out.print(out.paint(f"  {s['invalid_labels']} answers used labels outside the Discogs list (dropped)", "yellow"))
    u, per = s["usage"], s["usage_per_track"]
    out.print(f"  tokens       input {u['input']:,} (cache read {u['cache_read']:,}, written {u['cache_creation']:,}), "
              f"output {u['output']:,}")
    out.print(out.paint(f"               per track: input {per['input']:,}, cached {per['cache_read']:,}, "
                        f"output {per['output']:,}", "dim"))


if __name__ == "__main__":
    sys.exit(main())
