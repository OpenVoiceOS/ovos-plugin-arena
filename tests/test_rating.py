"""Unit tests for arena.rating — Bradley-Terry fit + bootstrap CIs (§4)."""
from __future__ import annotations

import random

from arena.rating import (
    PairResult,
    bootstrap_confidence_intervals,
    fit_bradley_terry,
    merge_pairwise,
    pairwise_from_results,
    to_rating_scale,
)


def _fit_and_rank(results, competitors):
    wins, games = pairwise_from_results(results)
    strengths = fit_bradley_terry(wins, games, competitors)
    return to_rating_scale(strengths)


class TestFitBradleyTerry:
    def test_recovers_known_dominance(self):
        # A beats B in 90/100 games — A should rank clearly above B.
        results = [PairResult("a", "b", 1.0)] * 90 + [PairResult("a", "b", 0.0)] * 10
        ratings = _fit_and_rank(results, ["a", "b"])
        assert ratings["a"] > ratings["b"]

    def test_transitive_chain_orders_correctly(self):
        # a > b > c, each pair observed decisively.
        results = (
            [PairResult("a", "b", 1.0)] * 20
            + [PairResult("b", "c", 1.0)] * 20
        )
        ratings = _fit_and_rank(results, ["a", "b", "c"])
        assert ratings["a"] > ratings["b"] > ratings["c"]

    def test_undefeated_and_winless_fighters_converge(self):
        # Zero-loss and zero-win fighters must not collapse to 0/inf thanks
        # to the phantom prior — they still get a finite, ordered rating.
        results = [PairResult("champ", "chump", 1.0)] * 5
        wins, games = pairwise_from_results(results)
        strengths = fit_bradley_terry(wins, games, ["champ", "chump", "unplayed"])
        assert all(v > 0 for v in strengths.values())
        ratings = to_rating_scale(strengths)
        assert ratings["champ"] > ratings["unplayed"] > ratings["chump"]

    def test_symmetric_record_ties_ratings(self):
        results = [PairResult("a", "b", 0.5)] * 10
        ratings = _fit_and_rank(results, ["a", "b"])
        assert abs(ratings["a"] - ratings["b"]) < 1e-6

    def test_ab_swap_symmetry(self):
        # Recording "a beats b" vs "b loses to a" (same fact, opposite
        # perspective) must produce identical ratings.
        forward = [PairResult("a", "b", 1.0)] * 7 + [PairResult("a", "b", 0.0)] * 3
        backward = [PairResult("b", "a", 0.0)] * 7 + [PairResult("b", "a", 1.0)] * 3
        r1 = _fit_and_rank(forward, ["a", "b"])
        r2 = _fit_and_rank(backward, ["a", "b"])
        assert abs(r1["a"] - r2["a"]) < 1e-6
        assert abs(r1["b"] - r2["b"]) < 1e-6

    def test_shuffle_invariant(self):
        results = (
            [PairResult("a", "b", 1.0)] * 12
            + [PairResult("b", "c", 1.0)] * 8
            + [PairResult("a", "c", 0.5)] * 5
        )
        shuffled = list(results)
        random.Random(42).shuffle(shuffled)
        r1 = _fit_and_rank(results, ["a", "b", "c"])
        r2 = _fit_and_rank(shuffled, ["a", "b", "c"])
        for c in ("a", "b", "c"):
            assert abs(r1[c] - r2[c]) < 1e-9

    def test_empty_competitors(self):
        assert fit_bradley_terry({}, {}, []) == {}

    def test_single_competitor(self):
        strengths = fit_bradley_terry({}, {}, ["solo"])
        assert strengths == {"solo": 1.0}


class TestBootstrapConfidenceIntervals:
    def test_deterministic_for_fixed_seed(self):
        results = [PairResult("a", "b", 1.0)] * 5 + [PairResult("a", "b", 0.0)] * 5
        ci1 = bootstrap_confidence_intervals(results, {}, {}, ["a", "b"], seed=7)
        ci2 = bootstrap_confidence_intervals(results, {}, {}, ["a", "b"], seed=7)
        assert ci1 == ci2

    def test_ci_narrows_with_more_votes(self):
        few = [PairResult("a", "b", 1.0)] * 2 + [PairResult("a", "b", 0.0)] * 2
        many = few * 25  # same win rate, far more observations
        ci_few = bootstrap_confidence_intervals(few, {}, {}, ["a", "b"], seed=1)
        ci_many = bootstrap_confidence_intervals(many, {}, {}, ["a", "b"], seed=1)
        width_few = ci_few["a"][1] - ci_few["a"][0]
        width_many = ci_many["a"][1] - ci_many["a"][0]
        assert width_many < width_few

    def test_zero_human_votes_collapses_to_seed_point(self):
        # With no human votes, every bootstrap round resamples nothing, so
        # every competitor's CI must collapse to a single point.
        seed_wins, seed_games = pairwise_from_results(
            [PairResult("a", "b", 1.0, weight=0.25)] * 10
        )
        ci = bootstrap_confidence_intervals([], seed_wins, seed_games, ["a", "b"], seed=3)
        for lo, hi in ci.values():
            assert abs(lo - hi) < 1e-9

    def test_no_competitors_returns_empty(self):
        assert bootstrap_confidence_intervals([], {}, {}, []) == {}


class TestPairwiseHelpers:
    def test_accumulate_symmetry(self):
        wins, games = pairwise_from_results([PairResult("a", "b", 1.0)])
        assert wins["a"]["b"] == 1.0
        assert wins["b"]["a"] == 0.0
        assert games["a"]["b"] == games["b"]["a"] == 1.0

    def test_merge_pairwise_does_not_mutate_inputs(self):
        base_wins, base_games = pairwise_from_results([PairResult("a", "b", 1.0)])
        extra_wins, extra_games = pairwise_from_results([PairResult("a", "b", 0.0)])
        merged_wins, merged_games = merge_pairwise(base_wins, base_games, extra_wins, extra_games)
        assert base_wins["a"]["b"] == 1.0  # unmutated
        assert merged_wins["a"]["b"] == 1.0
        assert merged_wins["b"]["a"] == 1.0
        assert merged_games["a"]["b"] == 2.0


class TestToRatingScale:
    def test_empty_input(self):
        assert to_rating_scale({}) == {}

    def test_equal_strengths_anchor_at_mean(self):
        ratings = to_rating_scale({"a": 2.0, "b": 2.0})
        assert abs(ratings["a"] - 1200.0) < 1e-6
        assert abs(ratings["b"] - 1200.0) < 1e-6
