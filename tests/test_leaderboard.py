"""Tests for the auto-metric leaderboard builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.arena.leaderboard import (
    _wer,
    _f1,
    _aggregate_stt,
    _aggregate_intent,
    _aggregate_ww,
    build_leaderboard,
    write_leaderboard_json,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:
    def test_wer_perfect(self):
        assert _wer("hello world", "hello world") == 0.0

    def test_wer_all_wrong(self):
        assert _wer("a b c", "x y z") == 1.0

    def test_wer_partial(self):
        assert _wer("a b c d", "a b x d") == 0.25

    def test_wer_empty_reference(self):
        assert _wer("", "hello") == 0.0

    def test_f1_perfect(self):
        assert _f1(10, 0, 0) == 1.0

    def test_f1_zero(self):
        assert _f1(0, 5, 5) == 0.0

    def test_f1_balanced(self):
        # p=0.5, r=0.5 → f1=0.5
        assert _f1(1, 1, 1) == 0.5


# ---------------------------------------------------------------------------
# Per-modality aggregators
# ---------------------------------------------------------------------------


class TestAggregators:
    def _stt_rows(self, refs, preds):
        return [
            {
                "reference": ref,
                "prediction": pred,
            }
            for ref, pred in zip(refs, preds)
        ]

    def test_stt_perfect(self):
        rows = self._stt_rows(["hello world"] * 3, ["hello world"] * 3)
        m = _aggregate_stt(rows)
        assert m["wer_mean"] == 0.0
        assert m["samples"] == 3

    def test_stt_mixed(self):
        rows = self._stt_rows(["a b", "a b", "a b"], ["a b", "a x", "x x"])
        m = _aggregate_stt(rows)
        assert m["wer_mean"] > 0.0
        assert m["samples"] == 3

    def test_stt_precomputed_wer(self):
        rows = [{"wer": 0.1, "prediction": "", "reference": ""}] * 5
        m = _aggregate_stt(rows)
        assert m["wer_mean"] == 0.1

    def test_intent_perfect_accuracy(self):
        rows = [
            {"reference_intent": "weather", "prediction": "weather", "exact_match": True},
            {"reference_intent": "timer", "prediction": "timer", "exact_match": True},
        ]
        m = _aggregate_intent(rows)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_intent_zero_accuracy(self):
        rows = [
            {"reference_intent": "weather", "prediction": "timer", "exact_match": False},
        ]
        m = _aggregate_intent(rows)
        assert m["accuracy"] == 0.0

    def test_intent_partial(self):
        rows = [
            {"reference_intent": "weather", "prediction": "weather", "exact_match": True},
            {"reference_intent": "timer", "prediction": "alarm", "exact_match": False},
        ]
        m = _aggregate_intent(rows)
        assert m["accuracy"] == 0.5

    def test_ww_no_errors(self):
        rows = [
            {"label": True, "prediction": True},
            {"label": False, "prediction": False},
        ]
        m = _aggregate_ww(rows)
        assert m["far"] == 0.0
        assert m["frr"] == 0.0

    def test_ww_all_fa(self):
        rows = [{"label": False, "prediction": True}] * 4
        m = _aggregate_ww(rows)
        assert m["far"] == 1.0

    def test_ww_all_fr(self):
        rows = [{"label": True, "prediction": False}] * 4
        m = _aggregate_ww(rows)
        assert m["frr"] == 1.0


# ---------------------------------------------------------------------------
# build_leaderboard end-to-end
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestBuildLeaderboard:
    def _intent_rows(self, competitor_id, n_correct, n_total, dataset_id="ds1", lang="en-US"):
        rows = []
        for i in range(n_total):
            exact = i < n_correct
            rows.append({
                "competitor_id": competitor_id,
                "sample_id": f"s_{i:04d}",
                "dataset_id": dataset_id,
                "lang": lang,
                "plugin_id": f"plugin-{competitor_id}",
                "plugin_version": "1.0",
                "utterance": f"utterance {i}",
                "reference_intent": "weather" if i % 2 == 0 else "timer",
                "prediction": "weather" if exact else "alarm",
                "exact_match": exact,
                "entity_f1": 0.0,
                "runner_version": "0.1.0",
                "created_at": "2026-01-01T00:00:00",
            })
        return rows

    def _stt_rows(self, competitor_id, wer_val, n=5, dataset_id="ds2", lang="pt-PT"):
        return [
            {
                "competitor_id": competitor_id,
                "sample_id": f"s_{i:04d}",
                "dataset_id": dataset_id,
                "lang": lang,
                "plugin_id": f"plugin-{competitor_id}",
                "plugin_version": "1.0",
                "reference": "olá mundo",
                "prediction": "olá mundo" if wer_val == 0 else "xpto yyy",
                "wer": wer_val,
                "runner_version": "0.1.0",
                "created_at": "2026-01-01T00:00:00",
            }
            for i in range(n)
        ]

    def test_intent_leaderboard_ordering(self, tmp_path):
        pdir = tmp_path / "predictions"
        # comp-a: 8/10 correct; comp-b: 4/10 correct
        _write_jsonl(pdir / "comp-a.jsonl", self._intent_rows("comp-a", 8, 10))
        _write_jsonl(pdir / "comp-b.jsonl", self._intent_rows("comp-b", 4, 10))

        lb = build_leaderboard(pdir, modality="intent")
        assert len(lb) == 2
        # Rank 1 should be comp-a (higher accuracy)
        assert lb[0]["competitor_id"] == "comp-a"
        assert lb[0]["accuracy"] > lb[1]["accuracy"]
        assert lb[0]["rank"] == 1

    def test_stt_leaderboard_ordering(self, tmp_path):
        pdir = tmp_path / "predictions"
        _write_jsonl(pdir / "good-stt.jsonl", self._stt_rows("good-stt", wer_val=0.05))
        _write_jsonl(pdir / "bad-stt.jsonl", self._stt_rows("bad-stt", wer_val=0.50))

        lb = build_leaderboard(pdir, modality="stt")
        assert lb[0]["competitor_id"] == "good-stt"
        assert lb[0]["wer_mean"] < lb[1]["wer_mean"]

    def test_mixed_modalities_no_filter(self, tmp_path):
        pdir = tmp_path / "predictions"
        _write_jsonl(pdir / "intent-comp.jsonl", self._intent_rows("intent-comp", 5, 10))
        _write_jsonl(pdir / "stt-comp.jsonl", self._stt_rows("stt-comp", 0.10))

        lb = build_leaderboard(pdir)
        mods = {r["modality"] for r in lb}
        assert "intent" in mods
        assert "stt" in mods

    def test_empty_dir(self, tmp_path):
        pdir = tmp_path / "empty"
        pdir.mkdir()
        lb = build_leaderboard(pdir)
        assert lb == []

    def test_malformed_jsonl_skipped(self, tmp_path):
        pdir = tmp_path / "predictions"
        bad = pdir / "bad.jsonl"
        bad.parent.mkdir()
        bad.write_text("not json\n")
        lb = build_leaderboard(pdir)
        assert lb == []


class TestWriteLeaderboardJson:
    def test_writes_files(self, tmp_path):
        pdir = tmp_path / "predictions"
        odir = tmp_path / "output"
        rows = [
            {
                "competitor_id": "comp1",
                "sample_id": "s1",
                "dataset_id": "ds1",
                "lang": "en-US",
                "plugin_id": "plugin1",
                "plugin_version": "1.0",
                "utterance": "hello",
                "reference_intent": "greeting",
                "prediction": "greeting",
                "exact_match": True,
                "entity_f1": 1.0,
                "runner_version": "0.1.0",
                "created_at": "2026-01-01T00:00:00",
            }
        ]
        _write_jsonl(pdir / "comp1.jsonl", rows)
        written = write_leaderboard_json(pdir, odir, modality="intent")
        assert len(written) == 1
        fname = written[0].name
        assert fname == "leaderboard-intent-en-US.json"
        data = json.loads(written[0].read_text())
        assert data["modality"] == "intent"
        assert data["lang"] == "en-US"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["accuracy"] == 1.0
