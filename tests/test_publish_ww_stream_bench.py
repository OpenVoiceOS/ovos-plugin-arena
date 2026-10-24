"""Tests for ``scripts/publish_ww_stream_bench.py``: no wake phrase reaches the speech."""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "publish_ww_stream_bench", ROOT / "scripts" / "publish_ww_stream_bench.py"
)
pwb = importlib.util.module_from_spec(_spec)
sys.modules["publish_ww_stream_bench"] = pwb
_spec.loader.exec_module(pwb)


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype="float32"), 16000, format="WAV")
    return buf.getvalue()


def _librispeech_parquet(path: Path, rows: dict[str, str]) -> Path:
    audio = [{"bytes": _wav_bytes(), "path": None} for _ in rows]
    table = pa.table({"id": list(rows), "text": list(rows.values()), "audio": audio})
    pq.write_table(table, path)
    return path


def test_every_set_and_fighter_phrase_is_filtered():
    phrases = pwb.wake_phrases()
    for phrase in ("stop", "sam", "wake up", "weather", "hey mycroft", "hey marvin"):
        assert phrase in phrases
    assert all(p == p.lower() and "_" not in p for p in phrases)


def test_an_utterance_that_says_a_phrase_is_left_out(tmp_path, monkeypatch):
    rows = {
        "1-1-0001": "AND THEN HE SAID STOP AT ONCE",
        "1-1-0002": "SAM WENT HOME",
        "1-1-0003": "WHEN WILL YOU WAKE UP",
        "1-1-0004": "THE STOPPER WAS LOOSE",
        "1-1-0005": "SAMUEL READ THE PAPER",
        "1-1-0006": "IT WAS A COLD DAY",
    }
    parquet = _librispeech_parquet(tmp_path / "rows.parquet", rows)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(parquet))
    out = tmp_path / "speech"
    count, dropped = pwb.export_parquet(
        "repo", "rev", ("rows.parquet",), out, pwb.phrase_pattern(["stop", "sam", "wake up"])
    )
    assert sorted(dropped) == ["1-1-0001", "1-1-0002", "1-1-0003"]
    assert count == 3
    assert sorted(p.stem for p in out.glob("*.wav")) == ["1-1-0004", "1-1-0005", "1-1-0006"]
