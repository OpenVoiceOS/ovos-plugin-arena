"""Tests for the version-blend guard (§G) — arena.metrics.build_benchmark_board."""
from __future__ import annotations

from arena.metrics import build_benchmark_board
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestVersionBlendGuard:
    def test_single_version_not_flagged(self):
        rows = {
            "comp": [
                _row(competitor_id="comp", wer=0.1, plugin_version="1.0.0"),
                _row(competitor_id="comp", wer=0.2, plugin_version="1.0.0"),
            ],
        }
        board = build_benchmark_board("stt", "d", "en-US", rows, "t")
        entry = board.entries[0]
        assert entry.version_blended is False
        assert entry.plugin_versions == ["1.0.0"]

    def test_multiple_versions_flagged(self):
        rows = {
            "comp": [
                _row(competitor_id="comp", wer=0.1, plugin_version="1.0.0"),
                _row(competitor_id="comp", wer=0.2, plugin_version="2.0.0"),
            ],
        }
        board = build_benchmark_board("stt", "d", "en-US", rows, "t")
        entry = board.entries[0]
        assert entry.version_blended is True
        assert entry.plugin_versions == ["1.0.0", "2.0.0"]

    def test_missing_version_does_not_falsely_flag(self):
        rows = {
            "comp": [
                _row(competitor_id="comp", wer=0.1, plugin_version=""),
                _row(competitor_id="comp", wer=0.2, plugin_version=""),
            ],
        }
        board = build_benchmark_board("stt", "d", "en-US", rows, "t")
        entry = board.entries[0]
        assert entry.version_blended is False
        assert entry.plugin_versions == []

    def test_multiple_competitors_independent(self):
        rows = {
            "single": [_row(competitor_id="single", wer=0.1, plugin_version="1.0.0")],
            "blended": [
                _row(competitor_id="blended", wer=0.1, plugin_version="1.0.0"),
                _row(competitor_id="blended", wer=0.1, plugin_version="1.1.0"),
            ],
        }
        board = build_benchmark_board("stt", "d", "en-US", rows, "t")
        by_id = {e.competitor_id: e for e in board.entries}
        assert by_id["single"].version_blended is False
        assert by_id["blended"].version_blended is True
