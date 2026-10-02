"""Tests for ``scripts/build_ww_stream_corpus.py`` on tiny synthetic audio."""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_ww_stream_corpus", ROOT / "scripts" / "build_ww_stream_corpus.py"
)
bwc = importlib.util.module_from_spec(_spec)
sys.modules["build_ww_stream_corpus"] = bwc
_spec.loader.exec_module(bwc)

SR = 16000


def _noise(seed: int, seconds: float, amp: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * amp).astype("float32")


@pytest.fixture
def inputs(tmp_path):
    pos = tmp_path / "pos"
    pos.mkdir()
    names = []
    for i in range(4):
        name = f"clip{i}.wav"
        sf.write(str(pos / name), _noise(100 + i, 0.4 + 0.1 * i, 0.3), SR)
        names.append(name)
    (pos / "metadata.csv").write_text(
        "file_name,transcript\n" + "".join(f"{n},hey\n" for n in names)
    )
    neg = tmp_path / "neg"
    neg.mkdir()
    for i in range(3):
        sf.write(str(neg / f"n{i}.wav"), _noise(i, 40.0, 0.05), SR)
    sf.write(str(neg / "n_8k.flac"), _noise(9, 20.0, 0.05)[: 8000 * 20], 8000)
    return pos, neg


def _build(inputs, out, **kw):
    pos, neg = inputs
    args = dict(negative_hours=0.05, seed=3, file_seconds=60.0,
                positives_per_file=2, min_gap_s=3.0)
    args.update(kw)
    return bwc.build_corpus(pos, [neg], out, **args)


def _clip(pos, name):
    return bwc.trim_silence(bwc.load_audio(pos / name))


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestOnsets:
    def test_onsets_are_sample_exact_and_name_their_clip(self, inputs, tmp_path):
        pos, _ = inputs
        rows = _build(inputs, tmp_path / "out")
        fade = int(bwc.FADE_S * SR)
        checked = 0
        for row in rows:
            rec, sr = sf.read(str(tmp_path / "out" / row["audio"]), dtype="float32")
            assert sr == SR and rec.ndim == 1
            assert len(rec) == round(row["duration_s"] * SR)
            for onset, samples, src in zip(
                row["onsets"], row["onset_samples"], row["sources"], strict=True
            ):
                assert onset == samples / SR
                clip = _clip(pos, src)
                ref = clip[fade: len(clip) - fade]
                lo, hi = samples + fade, samples + len(clip) - fade
                assert _corr(rec[lo:hi], ref) > 0.99
                assert _corr(rec[lo + 1:hi + 1], ref) < 0.5
                assert _corr(rec[lo - 1:hi - 1], ref) < 0.5
                checked += 1
        assert checked == sum(len(r["onsets"]) for r in rows) > 0

    def test_onsets_do_not_overlap_and_keep_the_gap(self, inputs, tmp_path):
        pos, _ = inputs
        for row in _build(inputs, tmp_path / "out"):
            ends = [
                s + len(_clip(pos, n))
                for s, n in zip(row["onset_samples"], row["sources"], strict=True)
            ]
            for end, nxt in zip(ends, row["onset_samples"][1:], strict=False):
                assert nxt - end >= 3 * SR
            assert row["onset_samples"][0] >= 3 * SR

    def test_manifest_matches_returned_rows(self, inputs, tmp_path):
        out = tmp_path / "out"
        rows = _build(inputs, out)
        lines = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
        assert lines == rows
        for field in ("audio", "onsets", "duration_s", "sources"):
            assert all(field in r for r in lines)
        assert all(len(r["sources"]) == len(r["onsets"]) for r in lines)


class TestDeterminism:
    def test_same_seed_is_byte_identical(self, inputs, tmp_path):
        _build(inputs, tmp_path / "a")
        _build(inputs, tmp_path / "b")
        root = tmp_path / "a"
        files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
        assert files
        for rel in files:
            assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()

    def test_different_seed_differs(self, inputs, tmp_path):
        a = _build(inputs, tmp_path / "a", seed=1)
        b = _build(inputs, tmp_path / "b", seed=2)
        assert [r["onset_samples"] for r in a] != [r["onset_samples"] for r in b]


class TestNegativeHours:
    def test_negative_hours_reached_exactly(self, inputs, tmp_path):
        pos, _ = inputs
        rows = _build(inputs, tmp_path / "out", negative_hours=0.05)
        total = sum(round(r["duration_s"] * SR) for r in rows)
        inserted = sum(len(_clip(pos, n)) for r in rows for n in r["sources"])
        assert total - inserted == round(0.05 * 3600 * SR)
        assert len(rows) == 3  # 180 s of negatives in 60 s recordings

    def test_small_pool_is_replayed_with_a_warning(self, inputs, tmp_path, caplog):
        pos, neg = inputs
        small = tmp_path / "small"
        small.mkdir()
        sf.write(str(small / "only.wav"), _noise(5, 30.0, 0.05), SR)
        with caplog.at_level(logging.WARNING, logger="build_ww_stream_corpus"):
            rows = bwc.build_corpus(
                pos, [small], tmp_path / "out", negative_hours=0.05, seed=0,
                file_seconds=60.0, positives_per_file=1,
            )
        inserted = sum(len(_clip(pos, n)) for r in rows for n in r["sources"])
        total = sum(round(r["duration_s"] * SR) for r in rows)
        assert total - inserted == round(0.05 * 3600 * SR)
        assert "replayed" in caplog.text

    def test_no_negative_audio_is_an_error(self, inputs, tmp_path):
        pos, _ = inputs
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            bwc.build_corpus(pos, [empty], tmp_path / "out", 0.01, 0)


class TestLevelPolicy:
    def test_snr_is_within_the_configured_range(self, inputs, tmp_path):
        rows = _build(inputs, tmp_path / "out", snr_db=(6.0, 6.0))
        for r in rows:
            assert all(abs(s - 6.0) < 0.5 for s in r["snr_db"])


def _padded_positive(tmp_path, name, speech_s, lead_s=0.5, trail_s=0.5, seed=7):
    pos = tmp_path / "pos2"
    pos.mkdir(exist_ok=True)
    sig = np.concatenate([
        np.zeros(int(lead_s * SR), "float32"),
        _noise(seed, speech_s, 0.3),
        np.zeros(int(trail_s * SR), "float32"),
    ])
    sf.write(str(pos / name), sig, SR)
    meta = pos / "metadata.csv"
    existing = meta.read_text() if meta.exists() else "file_name\n"
    meta.write_text(existing + f"{name}\n")
    return pos


class TestLevelFloor:
    def test_clips_next_to_silence_stay_audible(self, tmp_path):
        pos = _padded_positive(tmp_path, "a.wav", 0.6, lead_s=0, trail_s=0)
        for amp in (0.0, 1e-5):
            neg = tmp_path / f"neg{amp}"
            neg.mkdir()
            sf.write(str(neg / "q.wav"), _noise(1, 90.0, amp), SR)
            out = tmp_path / f"out{amp}"
            rows = bwc.build_corpus(
                pos, [neg], out, 0.025, 0, file_seconds=90.0, positives_per_file=3
            )
            for row in rows:
                rec, _ = sf.read(str(out / row["audio"]), dtype="float32")
                for o, e, lvl in zip(
                    row["onset_samples"], row["end_samples"], row["level_dbfs"],
                    strict=True,
                ):
                    seg = rec[o:e]
                    measured = 20 * np.log10(np.sqrt(np.mean(seg ** 2)))
                    assert measured > bwc.MIN_LEVEL_DBFS - 1.0
                    assert abs(measured - lvl) < 1.0


class TestTrimAndEnds:
    def test_silence_is_trimmed_and_ends_recorded(self, inputs, tmp_path):
        _, neg = inputs
        pos = _padded_positive(tmp_path, "pad.wav", 0.6)
        rows = bwc.build_corpus(
            pos, [neg], tmp_path / "out", 0.03, 0, file_seconds=60.0,
            positives_per_file=2,
        )
        for row in rows:
            for o, e, end_s, sp in zip(
                row["onset_samples"], row["end_samples"], row["ends"],
                row["spoken_s"], strict=True,
            ):
                assert e - o == pytest.approx(0.6 * SR, abs=0.01 * SR)
                assert end_s == e / SR
                assert sp == (e - o) / SR

    def test_long_positives_are_kept_and_warned_above_four_seconds(
        self, inputs, tmp_path, caplog
    ):
        _, neg = inputs
        pos = _padded_positive(tmp_path, "slow.wav", 2.5, seed=2)
        _padded_positive(tmp_path, "huge.wav", 4.5, seed=3)
        with caplog.at_level(logging.WARNING, logger="build_ww_stream_corpus"):
            rows = bwc.build_corpus(
                pos, [neg], tmp_path / "out", 0.03, 0, file_seconds=60.0,
                positives_per_file=2,
            )
        assert {n for r in rows for n in r["sources"]} == {"slow.wav", "huge.wav"}
        assert "huge.wav is" in caplog.text and "slow.wav is" not in caplog.text
