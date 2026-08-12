"""Unit tests for arena.metrics — benchmark scoring."""
from __future__ import annotations

import pytest

from arena.metrics import (
    build_benchmark_board,
    domain_of,
    intelligibility_scores,
    row_intelligibility_wer,
    row_is_correct,
    row_utmos,
    row_wer,
    score_intent,
    score_stt,
    score_tts,
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


class TestDomainOf:
    """domain_of — the 'text before the first :' rule domain-granularity
    scoring is built on (meteocat and future domain-only corpora)."""

    def test_strips_intent_suffix(self):
        assert domain_of("weather:current_conditions") == "weather"

    def test_bare_domain_passes_through(self):
        assert domain_of("weather") == "weather"

    def test_only_first_colon_splits(self):
        assert domain_of("weather:forecast:tomorrow") == "weather"

    def test_none_stays_none(self):
        assert domain_of(None) is None


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


class TestRowUtmos:
    def test_present(self):
        assert row_utmos(_row(extras={"utmos": 3.5})) == 3.5

    def test_missing(self):
        assert row_utmos(_row()) is None

    def test_non_numeric_ignored(self):
        assert row_utmos(_row(extras={"utmos": "not-a-number"})) is None

    def test_nan_guard(self):
        assert row_utmos(_row(extras={"utmos": float("nan")})) is None


class TestScoreTts:
    def test_mean(self):
        rows = [_row(extras={"utmos": 3.0}), _row(extras={"utmos": 4.0})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(3.5)
        assert metrics["n_scored"] == 2.0

    def test_missing_rows_excluded_not_fatal(self):
        rows = [_row(extras={"utmos": 4.0}), _row(extras={}),
                _row(extras={"utmos": None})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(4.0)
        assert metrics["n_scored"] == 1.0

    def test_all_rows_missing_utmos(self):
        rows = [_row(extras={}), _row(extras={})]
        metrics = score_tts(rows)
        assert "utmos" not in metrics
        assert metrics["n_scored"] == 0.0

    def test_single_row(self):
        metrics = score_tts([_row(extras={"utmos": 4.2})])
        assert metrics["utmos"] == pytest.approx(4.2)

    def test_empty(self):
        assert score_tts([]) == {"n_scored": 0.0}

    def test_latency(self):
        rows = [_row(extras={"utmos": 4.0}, latency_ms=5.0),
                _row(extras={"utmos": 3.0}, latency_ms=15.0)]
        assert score_tts(rows)["latency_ms_median"] == 10.0

    def test_nan_score_excluded(self):
        rows = [_row(extras={"utmos": 4.0}),
                _row(extras={"utmos": float("nan")})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(4.0)
        assert metrics["n_scored"] == 1.0

    def test_intelligibility_wer_mean_and_ci(self):
        rows = [_row(extras={"utmos": 4.0, "intelligibility_wer": 0.0}),
                _row(extras={"utmos": 3.0, "intelligibility_wer": 0.5})]
        metrics = score_tts(rows)
        # UTMOS stays primary, intelligibility_wer rides along as secondary
        assert metrics["utmos"] == pytest.approx(3.5)
        assert metrics["intelligibility_wer"] == pytest.approx(0.25)
        assert metrics["intelligibility_n_scored"] == 2.0
        assert "intelligibility_wer_ci_lower" in metrics
        assert "intelligibility_wer_ci_upper" in metrics
        assert (metrics["intelligibility_wer_ci_lower"]
                <= metrics["intelligibility_wer"]
                <= metrics["intelligibility_wer_ci_upper"])

    def test_intelligibility_wer_missing_rows_excluded_not_fatal(self):
        rows = [_row(extras={"intelligibility_wer": 0.2}), _row(extras={})]
        metrics = score_tts(rows)
        assert metrics["intelligibility_wer"] == pytest.approx(0.2)
        assert metrics["intelligibility_n_scored"] == 1.0

    def test_no_intelligibility_data_omits_secondary_metric(self):
        rows = [_row(extras={"utmos": 4.0})]
        metrics = score_tts(rows)
        assert "intelligibility_wer" not in metrics


class TestRowIntelligibilityWer:
    def test_present(self):
        assert row_intelligibility_wer(_row(extras={"intelligibility_wer": 0.3})) == 0.3

    def test_missing(self):
        assert row_intelligibility_wer(_row()) is None

    def test_non_numeric_ignored(self):
        assert row_intelligibility_wer(
            _row(extras={"intelligibility_wer": "nope"})) is None

    def test_nan_guard(self):
        assert row_intelligibility_wer(
            _row(extras={"intelligibility_wer": float("nan")})) is None


class TestIntelligibilityScores:
    def test_perfect_transcript_zero_wer_cer(self):
        wer, cer = intelligibility_scores("hello there", "hello there")
        assert wer == 0.0
        assert cer == 0.0

    def test_mismatched_transcript_nonzero(self):
        wer, cer = intelligibility_scores("hello there", "goodbye world")
        assert wer > 0.0
        assert cer > 0.0

    def test_reuses_canonical_normalizer_punct_and_case_insensitive(self):
        wer, _cer = intelligibility_scores("Hello, there!", "hello there")
        assert wer == 0.0


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

    def test_no_competitors_yields_empty_board(self):
        board = build_benchmark_board("tts", "d", "en-US", {}, "t")
        assert board.entries == []

    def test_tts_ranked_by_utmos_desc(self):
        by_competitor = {
            "bad": [_row(competitor_id="bad", extras={"utmos": 2.0})],
            "good": [_row(competitor_id="good", extras={"utmos": 4.0})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        assert board.primary_metric == "utmos"
        assert [e.competitor_id for e in board.entries] == ["good", "bad"]

    def test_solo_zero_scored_entry_is_unranked_not_rank_one(self):
        # A fighter whose entire TTS run failed (no row got a usable utmos)
        # has n_scored == 0 and no "utmos" key at all. As the board's only
        # entry it must not land at rank 1 — it has no signal to rank on.
        by_competitor = {
            "phoonnx-dii-es-es": [
                _row(competitor_id="phoonnx-dii-es-es", extras={}),
            ],
        }
        board = build_benchmark_board("tts", "d", "es-ES", by_competitor, "t")
        entry = board.entries[0]
        assert entry.metrics.get("n_scored") == 0.0
        assert "utmos" not in entry.metrics
        assert entry.unranked is True
        assert entry.rank == 0
        assert entry.unranked_reason

    def test_zero_scored_entry_ranked_below_scored_peers(self):
        by_competitor = {
            "failed": [_row(competitor_id="failed", extras={})],
            "good": [_row(competitor_id="good", extras={"utmos": 4.0})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        good = next(e for e in board.entries if e.competitor_id == "good")
        failed = next(e for e in board.entries if e.competitor_id == "failed")
        assert good.rank == 1
        assert good.unranked is False
        assert failed.unranked is True
        assert failed.rank == 0
