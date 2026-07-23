"""Tests for the canonical WER normalizer (§E) — arena.metrics.normalize_transcript."""
from __future__ import annotations

from arena.metrics import (
    WER_NORMALIZER_VERSION,
    build_benchmark_board,
    normalize_transcript,
    row_wer,
)
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestNormalizeTranscript:
    def test_trailing_punctuation_ignored(self):
        assert normalize_transcript("hello.") == normalize_transcript("hello")

    def test_case_insensitive(self):
        assert normalize_transcript("Hello World") == normalize_transcript("hello world")

    def test_unicode_nfkc(self):
        # full-width Latin "Ａ" (U+FF21) NFKC-normalizes to ASCII "A"
        assert normalize_transcript("ＡＢＣ") == normalize_transcript("ABC")

    def test_whitespace_collapsed(self):
        assert normalize_transcript("hello   world\t\n") == normalize_transcript("hello world")

    def test_numeral_policy_digit_by_digit(self):
        assert normalize_transcript("7") == "seven"
        assert normalize_transcript("123") == "one two three"

    def test_numerals_vs_spelled_out_equal(self):
        assert normalize_transcript("I have 7 apples") == normalize_transcript(
            "I have seven apples"
        )

    def test_apostrophe_stripped(self):
        assert normalize_transcript("don't") == "dont"

    def test_version_constant(self):
        assert WER_NORMALIZER_VERSION == 1


class TestRowWerUsesNormalizer:
    def test_punctuation_only_difference_scores_zero(self):
        row = _row(reference_text="hello world.", prediction="hello world")
        assert row_wer(row) == 0.0

    def test_case_only_difference_scores_zero(self):
        row = _row(reference_text="Hello World", prediction="hello world")
        assert row_wer(row) == 0.0

    def test_numeral_form_difference_scores_zero(self):
        row = _row(reference_text="set a timer for 7 minutes",
                    prediction="set a timer for seven minutes")
        assert row_wer(row) == 0.0

    def test_prefers_recomputation_over_stale_stored_wer(self):
        # A stored wer that disagrees with the canonical recomputation from
        # the raw text must be ignored in favor of recomputing (§E).
        row = _row(reference_text="hello world", prediction="hello world", wer=0.99)
        assert row_wer(row) == 0.0

    def test_falls_back_to_stored_wer_without_raw_text(self):
        row = _row(wer=0.42)
        assert row_wer(row) == 0.42


class TestBoardCarriesNormalizerVersion:
    def test_stt_board_has_version(self):
        by_competitor = {"c": [_row(reference_text="a b", prediction="a b")]}
        board = build_benchmark_board("stt", "d", "en-US", by_competitor, "t")
        assert board.wer_normalizer_version == WER_NORMALIZER_VERSION

    def test_non_stt_board_has_no_version(self):
        by_competitor = {"c": [_row(reference_intent="a", prediction="a")]}
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        assert board.wer_normalizer_version is None
