"""Fixtures that build tiny, valid FLAC files on the fly.

The files contain container headers only (no real audio), which is all mutagen
needs to read and write tags and stream info.
"""

import importlib.util
import os
import struct
from pathlib import Path

import pytest
from mutagen.flac import FLAC, Picture

SAMPLE_RATE = 44100

# 1x1 PNG, enough for mutagen to store as cover art.
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def _flac_bytes(seconds: float, channels: int = 2, bits: int = 16) -> bytes:
    total = int(seconds * SAMPLE_RATE)
    packed = (SAMPLE_RATE << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | total
    streaminfo = struct.pack(">HH", 4096, 4096) + b"\0" * 6 + packed.to_bytes(8, "big") + b"\0" * 16
    # 0x80 = last metadata block, type 0 = STREAMINFO
    return b"fLaC" + bytes([0x80]) + len(streaminfo).to_bytes(3, "big") + streaminfo


@pytest.fixture(autouse=True)
def plain_console(monkeypatch):
    """Keep the LLM pilot's output uncoloured whatever the developer's shell sets."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


@pytest.fixture(autouse=True)
def no_paid_api_calls(monkeypatch):
    """Tests stub the LLM; without provider keys, a stub that's missed fails instead of spending money."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def llm_pilot(monkeypatch):
    """tools/llm_pilot.py, which isn't part of the installed package, loaded by path (with tools/ importable,
    as when it's run, for its console module)."""
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    path = tools / "llm_pilot.py"
    spec = importlib.util.spec_from_file_location("llm_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def png() -> bytes:
    return PNG_1PX


@pytest.fixture
def make_flac():
    def make(path: Path, tags: dict[str, list[str] | str] | None = None, *,
             seconds: float = 3.0, cover: bytes | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_flac_bytes(seconds))
        f = FLAC(path)
        f.add_tags()
        for key, value in (tags or {}).items():
            f[key] = [value] if isinstance(value, str) else value
        if cover:
            pic = Picture()
            pic.type, pic.mime, pic.width, pic.height, pic.data = 3, "image/png", 1, 1, cover
            f.add_picture(pic)
        f.save()
        return path
    return make


@pytest.fixture
def touch_later():
    """Bump a file's mtime so a rescan sees it as changed even if its size didn't move
    (FLAC padding often absorbs tag edits)."""
    def bump(path: Path) -> None:
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    return bump


@pytest.fixture
def retag(touch_later):
    def edit(path: Path, **tags: list[str] | str | None) -> None:
        f = FLAC(path)
        for key, value in tags.items():
            if value is None:
                f.pop(key, None)
            else:
                f[key] = [value] if isinstance(value, str) else value
        f.save()
        touch_later(path)
    return edit


@pytest.fixture
def library(tmp_path, make_flac) -> Path:
    """A small library: two tracks at the top level, one in a subfolder, plus junk."""
    root = tmp_path / "library"
    make_flac(root / "Artist A - One [AAA111].flac", {
        "title": "One", "artist": "Artist A",
        "comment": "https://open.spotify.com/track/1111111111111111111111",
    })
    make_flac(root / "Artist B - Two [BBB222].flac", {"title": "Two", "artist": "Artist B"})
    make_flac(root / "sub" / "Artist C - Three.flac", {"title": "Three", "artist": "Artist C"})
    (root / "._Artist A - One [AAA111].flac").write_bytes(b"\0\x05\x16\x07macOS resource fork")
    (root / "notes.txt").write_text("not audio")
    return root
