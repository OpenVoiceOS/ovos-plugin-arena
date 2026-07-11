"""Tests for arena.patch_notes (leaderboard changelog, §A5.4)."""

import json
from pathlib import Path

from arena.models import EloBoard, EloEntry
from arena.patch_notes import build_patch_notes, diff_board, load_board


def _board(pairs):
    """pairs: list of (competitor_id, rank)."""
    return EloBoard(
        modality="stt", lang="en-US", generated_at="",
        entries=[EloEntry(competitor_id=c, rank=r, elo=1200.0) for c, r in pairs],
    )


class TestDiffBoard:
    def test_no_prior_board_all_new(self):
        notes = diff_board(None, _board([("a", 1), ("b", 2)]))
        assert {n["kind"] for n in notes} == {"new"}
        assert len(notes) == 2

    def test_rank_climb_and_drop(self):
        old = _board([("a", 1), ("b", 2)])
        new = _board([("b", 1), ("a", 2)])
        notes = {n["fighter"]: n for n in diff_board(old, new)}
        assert notes["b"]["kind"] == "up" and notes["b"]["delta"] == 1
        assert notes["a"]["kind"] == "down" and notes["a"]["delta"] == -1

    def test_unchanged_rank_is_not_reported(self):
        old = _board([("a", 1), ("b", 2)])
        assert diff_board(old, _board([("a", 1), ("b", 2)])) == []

    def test_new_entrant_alongside_existing(self):
        old = _board([("a", 1)])
        notes = {n["fighter"]: n for n in diff_board(old, _board([("a", 1), ("c", 2)]))}
        assert notes["c"]["kind"] == "new"
        assert "a" not in notes  # unchanged, not reported


class TestBuildPatchNotes:
    def test_upset_is_largest_climb(self):
        notes = [
            {"kind": "up", "delta": 1, "fighter": "a", "modality": "stt"},
            {"kind": "up", "delta": 3, "fighter": "b", "modality": "stt"},
            {"kind": "down", "delta": -5, "fighter": "c", "modality": "stt"},
        ]
        payload = build_patch_notes(notes, "T")
        assert payload["upset_of_the_day"]["fighter"] == "b"
        assert payload["count"] == 3

    def test_no_climbs_no_upset(self):
        payload = build_patch_notes([{"kind": "down", "delta": -1, "fighter": "x"}], "T")
        assert payload["upset_of_the_day"] is None

    def test_deterministic_tie_break(self):
        notes = [
            {"kind": "up", "delta": 2, "fighter": "z", "modality": "stt"},
            {"kind": "up", "delta": 2, "fighter": "a", "modality": "stt"},
        ]
        # ties break by (delta, fighter, modality) -> 'z' wins on max fighter name
        assert build_patch_notes(notes, "T")["upset_of_the_day"]["fighter"] == "z"


class TestLoadBoard:
    def test_roundtrip(self, tmp_path: Path):
        p = tmp_path / "leaderboard-stt-en-US.json"
        p.write_text(json.dumps(_board([("a", 1)]).model_dump(mode="json")))
        loaded = load_board(p)
        assert loaded is not None and loaded.entries[0].competitor_id == "a"

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_board(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert load_board(p) is None
