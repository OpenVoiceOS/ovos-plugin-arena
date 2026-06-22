"""Unit tests for arena.metrics — benchmark scoring."""
from __future__ import annotations

import pytest

from arena.metrics import (
    build_benchmark_board,
    row_is_correct,
    row_wer,
    score_intent,
    score_stt,
    score_wake_word,
    ww_row_correct,
)
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestRowIsCorrect:
    def test_exact_match_field_wins(self):
        assert row_is_correct(_row(reference_intent="a", prediction="b",
                                   exact_match=True))

    def test_string_comparison_fallback(self):
        assert row_is_correct(_row(reference_intent="a", prediction="a"))
        assert not row_is_correct(_row(reference_intent="a", prediction="b"))

    def test_ood_correct_rejection(self):
        assert row_is_correct(_row(reference_intent=None, prediction=None))
        assert not row_is_correct(_row(reference_intent=None, prediction="a"))


class TestScoreIntent:
    def test_accuracy_counts_ood_rejections(self):
        rows = [
            _row(reference_intent="a", prediction="a", bucket="template"),
            _row(reference_intent="a", prediction="b", bucket="template"),
            _row(reference_intent=None, prediction=None, bucket="far_ood"),
            _row(reference_intent=None, prediction="a", bucket="far_ood"),
        ]
        metrics = score_intent(rows)
        assert metrics["accuracy"] == 0.5
        assert metrics["ood_fpr"] == 0.5
        assert metrics["acc_template"] == 0.5
        assert metrics["acc_far_ood"] == 0.5

    def test_perfect_run(self):
        rows = [
            _row(reference_intent="a", prediction="a"),
            _row(reference_intent="b", prediction="b"),
            _row(reference_intent=None, prediction=None),
        ]
        metrics = score_intent(rows)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["ood_fpr"] == 0.0

    def test_ood_false_positive_hurts_macro_f1(self):
        clean = score_intent([
            _row(reference_intent="a", prediction="a"),
            _row(reference_intent=None, prediction=None),
        ])
        noisy = score_intent([
            _row(reference_intent="a", prediction="a"),
            _row(reference_intent=None, prediction="a"),
        ])
        assert noisy["macro_f1"] < clean["macro_f1"]

    def test_slot_exact_match(self):
        rows = [
            _row(reference_intent="a", prediction="a",
                 reference_slots={"song": "Africa"},
                 predicted_slots={"song": "africa"}),  # case-insensitive
            _row(reference_intent="a", prediction="a",
                 reference_slots={"song": "x"},
                 predicted_slots={}),
        ]
        assert score_intent(rows)["slot_exact_match"] == 0.5

    def test_latency_median(self):
        rows = [
            _row(reference_intent="a", prediction="a", latency_ms=10.0),
            _row(reference_intent="a", prediction="a", latency_ms=20.0),
            _row(reference_intent="a", prediction="a", latency_ms=30.0),
        ]
        assert score_intent(rows)["latency_ms_median"] == 20.0

    def test_empty(self):
        assert score_intent([])["accuracy"] == 0.0


class TestScoreStt:
    def test_wer_present(self):
        rows = [_row(wer=0.1), _row(wer=0.3)]
        metrics = score_stt(rows)
        assert metrics["wer_mean"] == pytest.approx(0.2)
        assert metrics["wer_median"] == pytest.approx(0.2)

    def test_wer_computed_from_reference(self):
        rows = [_row(reference_text="hello world", prediction="hello world"),
                _row(reference_text="hello world", prediction="hello there")]
        assert score_stt(rows)["wer_mean"] == pytest.approx(0.25)

    def test_row_wer_priority(self):
        assert row_wer(_row(wer=0.42)) == 0.42
        assert row_wer(_row(reference_text="a b", prediction="a c")) == 0.5
        assert row_wer(_row()) is None


class TestScoreWakeWord:
    def _clip(self, label, prediction, **over):
        return _row(label=label, prediction=prediction, **over)

    def test_row_correct(self):
        assert ww_row_correct(self._clip("positive", "detected")) is True
        assert ww_row_correct(self._clip("positive", "not_detected")) is False
        assert ww_row_correct(self._clip("negative", "not_detected")) is True
        assert ww_row_correct(self._clip("negative", "detected")) is False
        assert ww_row_correct(self._clip(None, "detected")) is None

    def test_label_aliases(self):
        # numeric / boolean-ish labels normalise to presence
        assert ww_row_correct(self._clip("1", "detected")) is True
        assert ww_row_correct(self._clip("0", "not_detected")) is True
        assert ww_row_correct(self._clip("adversarial", "not_detected")) is True

    def test_rates(self):
        rows = [
            self._clip("positive", "detected"),       # TP
            self._clip("positive", "not_detected"),   # FR
            self._clip("negative", "not_detected"),   # TN
            self._clip("negative", "detected"),        # FA
        ]
        m = score_wake_word(rows)
        assert m["error_rate"] == 0.5
        assert m["accuracy"] == 0.5
        assert m["false_accept_rate"] == 0.5
        assert m["false_reject_rate"] == 0.5

    def test_perfect(self):
        rows = [self._clip("positive", "detected"),
                self._clip("negative", "not_detected")]
        m = score_wake_word(rows)
        assert m["error_rate"] == 0.0
        assert m["false_accept_rate"] == 0.0
        assert m["false_reject_rate"] == 0.0

    def test_unscorable_rows_ignored(self):
        assert score_wake_word([self._clip(None, None)]) == {}

    def test_latency(self):
        rows = [self._clip("positive", "detected", latency_ms=5.0),
                self._clip("negative", "not_detected", latency_ms=15.0)]
        assert score_wake_word(rows)["latency_ms_median"] == 10.0


class TestBenchmarkBoard:
    def test_intent_ranked_by_accuracy_desc(self):
        by_competitor = {
            "weak": [_row(competitor_id="weak", reference_intent="a",
                          prediction="b")],
            "strong": [_row(competitor_id="strong", reference_intent="a",
                            prediction="a")],
        }
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        assert board.primary_metric == "accuracy"
        assert [e.competitor_id for e in board.entries] == ["strong", "weak"]
        assert [e.rank for e in board.entries] == [1, 2]

    def test_stt_ranked_by_wer_asc(self):
        by_competitor = {
            "bad": [_row(competitor_id="bad", wer=0.9)],
            "good": [_row(competitor_id="good", wer=0.1)],
        }
        board = build_benchmark_board("stt", "d", "pt-PT", by_competitor, "t")
        assert [e.competitor_id for e in board.entries] == ["good", "bad"]

    def test_wake_word_ranked_by_error_rate_asc(self):
        by_competitor = {
            "noisy": [_row(competitor_id="noisy", label="negative",
                           prediction="detected")],
            "clean": [_row(competitor_id="clean", label="negative",
                           prediction="not_detected")],
        }
        board = build_benchmark_board("wake_word", "d", "en", by_competitor, "t")
        assert board.primary_metric == "error_rate"
        assert [e.competitor_id for e in board.entries] == ["clean", "noisy"]

    def test_unscored_modality_yields_empty_board(self):
        board = build_benchmark_board("tts", "d", "en-US", {"x": []}, "t")
        assert board.entries == []
