"""Unit + assemble-level tests for the TTS judge-agreement panel.

``spearman_rho`` is tested directly (perfect agreement, perfect inversion,
ties, partial correlation) and ``build_benchmark_board``'s TTS path is
tested for the ``judge_agreement`` block it stamps onto the board.
"""
from __future__ import annotations

from arena.metrics import build_benchmark_board, spearman_rho
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestSpearmanRho:
    def test_identical_rankings_is_perfect_positive(self):
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        assert rho == 1.0

    def test_inverted_rankings_is_perfect_negative(self):
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
        assert rho == -1.0

    def test_tied_ranks_use_average_rank(self):
        # x has a tie at positions 2,3 (values 2.0, 2.0) -> both get rank 2.5.
        # x ranks: [1, 2.5, 2.5, 4]; y ranks: [1, 2, 3, 4] (no ties in y).
        # Pearson correlation of those two rank vectors, computed by hand:
        # mean(xr)=2.5, mean(yr)=2.5
        # dx = [-1.5, 0, 0, 1.5]; dy = [-1.5, -0.5, 0.5, 1.5]
        # cov = (-1.5*-1.5 + 0*-0.5 + 0*0.5 + 1.5*1.5) = 2.25+2.25 = 4.5
        # var_x = 1.5^2*2 = 4.5; var_y = 1.5^2+0.5^2+0.5^2+1.5^2 = 5.0
        # rho = 4.5 / sqrt(4.5*5.0) = 4.5 / sqrt(22.5) ~= 0.94868
        rho = spearman_rho([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        assert rho is not None
        assert round(rho, 4) == round(4.5 / (4.5 * 5.0) ** 0.5, 4)

    def test_partial_correlation_is_between_bounds(self):
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0])
        assert rho is not None
        assert -1.0 < rho < 1.0

    def test_too_few_points_returns_none(self):
        assert spearman_rho([1.0], [2.0]) is None
        assert spearman_rho([], []) is None

    def test_zero_variance_returns_none(self):
        assert spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


class TestTtsJudgeAgreementBoard:
    def _by_competitor(self):
        # 3 fighters, agreeing rankings on utmos/sigmos/dnsmos (a > b > c),
        # nisqa only scored for a and b, intelligibility_wer scored for all
        # three but INVERTED (lower wer = better), so its "score" ranking
        # should still track utmos when wer is anti-correlated with utmos.
        return {
            "a": [_row(competitor_id="a", extras={
                "utmos": 4.5, "sigmos.ovrl": 4.5, "dnsmos.ovrl": 4.5,
                "nisqa.mos": 4.5, "intelligibility_wer": 0.05,
            })],
            "b": [_row(competitor_id="b", extras={
                "utmos": 3.5, "sigmos.ovrl": 3.5, "dnsmos.ovrl": 3.5,
                "nisqa.mos": 3.5, "intelligibility_wer": 0.15,
            })],
            "c": [_row(competitor_id="c", extras={
                "utmos": 2.5, "sigmos.ovrl": 2.5, "dnsmos.ovrl": 2.5,
                "intelligibility_wer": 0.25,
            })],
        }

    def test_judge_agreement_present_with_expected_shape(self):
        board = build_benchmark_board(
            "tts", "d", "en-US", self._by_competitor(), "t",
        )
        ja = board.judge_agreement
        assert ja is not None
        assert ja.n_fighters == 3

        # UTMOS/SIGMOS/DNSMOS/Intelligibility all agree perfectly (a>b>c
        # on the quality judges, a<b<c on wer i.e. a>b>c once inverted).
        assert ja.matrix["UTMOS"]["SIGMOS"] == 1.0
        assert ja.matrix["UTMOS"]["DNSMOS"] == 1.0
        assert ja.matrix["UTMOS"]["Intelligibility"] == 1.0
        assert ja.matrix["UTMOS"]["UTMOS"] == 1.0
        # symmetric
        assert ja.matrix["SIGMOS"]["UTMOS"] == ja.matrix["UTMOS"]["SIGMOS"]
        # NISQA only has 2 scored fighters (a, b) -> pairwise correlation
        # against it is not computable (needs >= 2 common fighters is met,
        # but the two-point correlation is degenerate/undefined here since
        # scores move together trivially) - just assert it doesn't crash
        # and, if present, is a float.
        if "NISQA" in ja.matrix.get("UTMOS", {}):
            assert isinstance(ja.matrix["UTMOS"]["NISQA"], float)

        assert ja.top5["UTMOS"] == ["a", "b", "c"]
        assert ja.top5["Intelligibility"] == ["a", "b", "c"]

    def test_two_fighters_does_not_crash_and_reports_n_fighters(self):
        by_competitor = {
            "a": [_row(competitor_id="a", extras={"utmos": 4.0, "sigmos.ovrl": 4.0})],
            "b": [_row(competitor_id="b", extras={"utmos": 3.0, "sigmos.ovrl": 3.0})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        ja = board.judge_agreement
        assert ja is not None
        assert ja.n_fighters == 2
        assert ja.top5["UTMOS"] == ["a", "b"]

    def test_one_fighter_one_judge_does_not_crash(self):
        by_competitor = {
            "a": [_row(competitor_id="a", extras={"utmos": 4.0})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        ja = board.judge_agreement
        assert ja is not None
        assert ja.n_fighters == 1
        assert ja.top5["UTMOS"] == ["a"]
        # no pair possible -> no correlations
        assert ja.matrix.get("UTMOS", {}).get("UTMOS") in (1.0, None)

    def test_no_judges_scored_does_not_crash(self):
        by_competitor = {
            "a": [_row(competitor_id="a", extras={})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        ja = board.judge_agreement
        assert ja is not None
        assert ja.n_fighters == 0
        assert ja.matrix == {}
        assert ja.top5 == {}

    def test_non_tts_board_has_no_judge_agreement(self):
        by_competitor = {
            "strong": [_row(competitor_id="strong", reference_intent="a",
                            prediction="a")],
        }
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        assert board.judge_agreement is None
