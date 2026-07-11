"""Tests for arena.badges (embeddable SVG rank badges, §A5.3)."""

import xml.etree.ElementTree as ET
from pathlib import Path

from arena.badges import emit_badges, render_rank_badge
from arena.models import EloBoard, EloEntry


class TestRenderRankBadge:
    def test_is_well_formed_svg(self):
        svg = render_rank_badge("ovos-stt-plugin-server", "stt", 3, 1240.4)
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")
        assert root.attrib["role"] == "img"

    def test_contains_rank_and_rounded_rating(self):
        svg = render_rank_badge("piper", "tts", 1, 1305.6)
        assert "#1" in svg
        assert "1306" in svg  # rounded
        assert "1305.6" not in svg

    def test_label_is_escaped(self):
        svg = render_rank_badge("a<b>&c", "stt", 2, 1200.0)
        assert "<b>" not in svg
        assert "&lt;b&gt;" in svg
        ET.fromstring(svg)  # still parses

    def test_provisional_marks_value(self):
        svg = render_rank_badge("x", "ww", 4, 1100.0, provisional=True)
        assert "#4?" in svg

    def test_deterministic_no_timestamp(self):
        """Byte-identical across calls — no dates/timestamps inside (churn-free)."""
        a = render_rank_badge("x", "stt", 1, 1200.0)
        b = render_rank_badge("x", "stt", 1, 1200.0)
        assert a == b

    def test_unknown_modality_uses_default_color(self):
        svg = render_rank_badge("x", "quantum", 1, 1200.0)
        assert "#57606a" in svg  # default color, no crash


class TestEmitBadges:
    def _board(self):
        return EloBoard(
            modality="stt", lang="en-US", generated_at="", provisional=False,
            entries=[
                EloEntry(rank=1, competitor_id="alpha", elo=1300.0, bt_rating=1310.0),
                EloEntry(rank=2, competitor_id="beta", elo=1200.0),  # no bt_rating
            ],
        )

    def test_writes_one_svg_per_entry(self, tmp_path: Path):
        n = emit_badges(self._board(), tmp_path)
        assert n == 2
        assert (tmp_path / "badges" / "stt" / "alpha.svg").is_file()
        assert (tmp_path / "badges" / "stt" / "beta.svg").is_file()

    def test_uses_bt_rating_when_present_else_elo(self, tmp_path: Path):
        emit_badges(self._board(), tmp_path)
        alpha = (tmp_path / "badges" / "stt" / "alpha.svg").read_text()
        beta = (tmp_path / "badges" / "stt" / "beta.svg").read_text()
        assert "1310" in alpha       # bt_rating
        assert "1200" in beta        # falls back to elo

    def test_rebuild_is_byte_identical(self, tmp_path: Path):
        emit_badges(self._board(), tmp_path)
        first = (tmp_path / "badges" / "stt" / "alpha.svg").read_bytes()
        emit_badges(self._board(), tmp_path)
        second = (tmp_path / "badges" / "stt" / "alpha.svg").read_bytes()
        assert first == second
