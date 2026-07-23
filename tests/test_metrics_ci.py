"""Unit tests for arena.metrics bootstrap confidence intervals (§4 A1.2)."""
from __future__ import annotations

from arena.metrics import (
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    build_benchmark_board,
    primary_metric_ci,
)
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestBootstrapMeanCi:
    def test_empty_returns_none(self):
        assert bootstrap_mean_ci([]) is None

    def test_single_value_collapses(self):
        assert bootstrap_mean_ci([0.7]) == (0.7, 0.7)

    def test_deterministic_for_fixed_seed(self):
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        ci1 = bootstrap_mean_ci(values, seed=5)
        ci2 = bootstrap_mean_ci(values, seed=5)
        assert ci1 == ci2

    def test_ci_contains_point_estimate(self):
        values = [1.0] * 80 + [0.0] * 20
        lo, hi = bootstrap_mean_ci(values, seed=1)
        point = sum(values) / len(values)
        assert lo <= point <= hi

    def test_ci_narrows_with_more_samples(self):
        few = [1.0, 0.0, 1.0, 0.0]
        many = few * 50
        lo_f, hi_f = bootstrap_mean_ci(few, seed=2)
        lo_m, hi_m = bootstrap_mean_ci(many, seed=2)
        assert (hi_m - lo_m) < (hi_f - lo_f)


class TestBootstrapRatioCi:
    def test_empty_returns_none(self):
        assert bootstrap_ratio_ci([]) is None

    def test_zero_denominator_pairs_ignored(self):
        assert bootstrap_ratio_ci([(1.0, 0.0), (2.0, 0.0)]) is None

    def test_single_pair_collapses(self):
        assert bootstrap_ratio_ci([(2.0, 10.0)]) == (0.2, 0.2)

    def test_weighted_by_denominator_not_per_pair_average(self):
        # Many short utterances (1 error / 1 word = WER 1.0 each) and many
        # long, near-perfect utterances (1 error / 100 words = WER 0.01
        # each), in equal *counts*. A naive per-pair average would report
        # ~0.5; weighting by word count (the correct WER aggregation) must
        # land near the long utterances' rate, since they contribute far
        # more of the total word count.
        pairs = [(1.0, 1.0)] * 20 + [(1.0, 100.0)] * 20
        lo, hi = bootstrap_ratio_ci(pairs, seed=0, rounds=2000)
        naive_average = 0.5 * (1.0 + 0.01)
        weighted_point = 40.0 / 2020.0
        assert lo <= weighted_point <= hi
        assert hi < naive_average

    def test_deterministic_for_fixed_seed(self):
        pairs = [(1.0, 10.0), (2.0, 8.0), (0.0, 12.0)]
        ci1 = bootstrap_ratio_ci(pairs, seed=9)
        ci2 = bootstrap_ratio_ci(pairs, seed=9)
        assert ci1 == ci2


class TestPrimaryMetricCi:
    def test_intent_uses_mean_strategy(self):
        rows = [_row(reference_intent="a", prediction="a")] * 8 + [
            _row(reference_intent="a", prediction="b")
        ] * 2
        ci = primary_metric_ci("intent", rows)
        assert ci is not None
        lo, hi = ci
        assert lo <= 0.8 <= hi

    def test_stt_uses_ratio_strategy(self):
        rows = [
            _row(reference_text="hello world", prediction="hello world"),
            _row(reference_text="one two three", prediction="one two four"),
        ]
        ci = primary_metric_ci("stt", rows)
        assert ci is not None
        lo, hi = ci
        assert lo <= 1 / 5 <= hi  # 1 error / 5 reference words total

    def test_wake_word_uses_mean_strategy(self):
        rows = [_row(label="positive", prediction="detected")] * 9 + [
            _row(label="positive", prediction="negative")
        ]
        ci = primary_metric_ci("wake_word", rows)
        assert ci is not None
        lo, hi = ci
        assert lo <= 0.1 <= hi

    def test_unknown_modality_returns_none(self):
        assert primary_metric_ci("unknown_modality", [_row()]) is None

    def test_tts_uses_mean_strategy_over_utmos(self):
        rows = [_row(extras={"utmos": 4.0})] * 9 + [_row(extras={"utmos": 2.0})]
        ci = primary_metric_ci("tts", rows)
        assert ci is not None
        lo, hi = ci
        assert lo <= 3.8 <= hi

    def test_tts_no_scored_rows_returns_none(self):
        assert primary_metric_ci("tts", [_row(extras={})]) is None

    def test_no_scoreable_rows_returns_none(self):
        # wake_word rows with unrecognisable label/prediction tokens score
        # no signal at all.
        rows = [_row(label=None, prediction=None)]
        assert primary_metric_ci("wake_word", rows) is None


class TestBuildBenchmarkBoardCi:
    def test_entries_carry_ci_and_tied_flag(self):
        good_rows = [_row(competitor_id="good", reference_intent="a", prediction="a")] * 20
        ok_rows = [_row(competitor_id="ok", reference_intent="a", prediction="a")] * 19 + [
            _row(competitor_id="ok", reference_intent="a", prediction="b")
        ]
        bad_rows = [_row(competitor_id="bad", reference_intent="a", prediction="b")] * 20

        board = build_benchmark_board(
            "intent", "d", "en-US",
            {"good": good_rows, "ok": ok_rows, "bad": bad_rows},
            "t",
        )
        by_id = {e.competitor_id: e for e in board.entries}

        assert by_id["good"].primary_metric_ci_lower is not None
        assert by_id["good"].primary_metric_ci_upper is not None
        # good (100%) and ok (95%) should read as statistically
        # indistinguishable at this sample size — bad (0%) should not.
        assert by_id["good"].tied_with_leader is True
        assert by_id["ok"].tied_with_leader is True
        assert by_id["bad"].tied_with_leader is False
        assert board.entries[0].competitor_id == "good"
        assert board.entries[0].rank == 1

    def test_tts_board_carries_utmos_ci(self):
        rows = [_row(competitor_id="voice_a", extras={"utmos": 4.0})] * 10
        board = build_benchmark_board("tts", "d", "en-US", {"voice_a": rows}, "t")
        assert board.primary_metric == "utmos"
        assert board.entries[0].primary_metric_ci_lower is not None
        assert board.entries[0].primary_metric_ci_upper is not None
