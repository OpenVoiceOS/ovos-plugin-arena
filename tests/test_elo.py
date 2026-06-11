"""Unit tests for the deterministic ELO engine (arena.elo)."""
from __future__ import annotations

import pytest

from arena.elo import (
    AUTO_K_DIVISOR,
    INITIAL_ELO,
    K_FACTOR,
    K_FACTOR_VETERAN,
    VETERAN_THRESHOLD,
    EloLedger,
    expected_score,
    k_factor,
    update_ratings,
)
from arena.models import VoteOutcome


class TestExpectedScore:
    def test_equal_ratings(self):
        assert expected_score(1200, 1200) == 0.5

    def test_higher_rating_favoured(self):
        assert expected_score(1400, 1200) > 0.5
        assert expected_score(1200, 1400) < 0.5

    def test_symmetry(self):
        assert expected_score(1300, 1100) + expected_score(1100, 1300) == pytest.approx(1.0)

    def test_400_point_gap(self):
        # A 400-point gap means ~10:1 odds under the ELO model
        assert expected_score(1600, 1200) == pytest.approx(10 / 11)


class TestKFactor:
    def test_fresh_player(self):
        assert k_factor(0) == K_FACTOR

    def test_veteran(self):
        assert k_factor(VETERAN_THRESHOLD) == K_FACTOR_VETERAN

    def test_auto_reduced(self):
        assert k_factor(0, auto=True) == K_FACTOR / AUTO_K_DIVISOR
        assert k_factor(VETERAN_THRESHOLD, auto=True) == (
            K_FACTOR_VETERAN / AUTO_K_DIVISOR
        )


class TestUpdateRatings:
    def test_win_raises_winner(self):
        new_a, new_b = update_ratings(1200, 1200, VoteOutcome.CANDIDATE_A)
        assert new_a > 1200 > new_b

    def test_loss_lowers_loser(self):
        new_a, new_b = update_ratings(1200, 1200, VoteOutcome.CANDIDATE_B)
        assert new_b > 1200 > new_a

    def test_delta_symmetric_at_equal_k(self):
        new_a, new_b = update_ratings(1200, 1200, VoteOutcome.CANDIDATE_A)
        assert (new_a - 1200) == pytest.approx(1200 - new_b)

    def test_tie_at_equal_ratings_is_noop(self):
        new_a, new_b = update_ratings(1200, 1200, VoteOutcome.TIE)
        assert new_a == pytest.approx(1200)
        assert new_b == pytest.approx(1200)

    def test_both_wrong_scores_as_tie(self):
        tie = update_ratings(1300, 1100, VoteOutcome.TIE)
        both_wrong = update_ratings(1300, 1100, VoteOutcome.BOTH_WRONG)
        assert tie == both_wrong

    def test_tie_pulls_unequal_ratings_together(self):
        new_a, new_b = update_ratings(1400, 1200, VoteOutcome.TIE)
        assert new_a < 1400
        assert new_b > 1200

    def test_upset_win_pays_more(self):
        # Underdog (A at 1100) beating a 1400 gains more than beating a peer
        upset_gain = update_ratings(1100, 1400, VoteOutcome.CANDIDATE_A)[0] - 1100
        peer_gain = update_ratings(1100, 1100, VoteOutcome.CANDIDATE_A)[0] - 1100
        assert upset_gain > peer_gain

    def test_auto_votes_move_less(self):
        human = update_ratings(1200, 1200, VoteOutcome.CANDIDATE_A)[0] - 1200
        auto = update_ratings(1200, 1200, VoteOutcome.CANDIDATE_A, auto=True)[0] - 1200
        assert auto == pytest.approx(human / AUTO_K_DIVISOR)


class TestEloLedger:
    def test_ensure_initialises(self):
        ledger = EloLedger()
        ledger.ensure("x")
        assert ledger.ratings["x"] == INITIAL_ELO
        assert ledger.battles["x"] == 0

    def test_apply_win(self):
        ledger = EloLedger()
        ledger.apply("x", "y", VoteOutcome.CANDIDATE_A)
        assert ledger.ratings["x"] > INITIAL_ELO > ledger.ratings["y"]
        assert ledger.wins["x"] == 1
        assert ledger.losses["y"] == 1
        assert ledger.battles["x"] == ledger.battles["y"] == 1
        assert ledger.human_votes["x"] == 1
        assert ledger.auto_votes["x"] == 0

    def test_apply_auto_counted_separately(self):
        ledger = EloLedger()
        ledger.apply("x", "y", VoteOutcome.CANDIDATE_A, auto=True)
        assert ledger.auto_votes["x"] == 1
        assert ledger.human_votes["x"] == 0

    def test_tie_counted_for_both(self):
        ledger = EloLedger()
        ledger.apply("x", "y", VoteOutcome.TIE)
        assert ledger.ties["x"] == ledger.ties["y"] == 1
        assert ledger.wins["x"] == ledger.wins["y"] == 0

    def test_deterministic_replay(self):
        votes = [
            ("x", "y", VoteOutcome.CANDIDATE_A),
            ("y", "z", VoteOutcome.CANDIDATE_B),
            ("x", "z", VoteOutcome.TIE),
            ("x", "y", VoteOutcome.BOTH_WRONG),
        ]
        ledgers = []
        for _ in range(2):
            ledger = EloLedger()
            for a, b, outcome in votes:
                ledger.apply(a, b, outcome)
            ledgers.append(ledger.ratings)
        assert ledgers[0] == ledgers[1]

    def test_order_matters(self):
        ledger1 = EloLedger()
        ledger1.apply("x", "y", VoteOutcome.CANDIDATE_A)
        ledger1.apply("x", "z", VoteOutcome.CANDIDATE_A)

        ledger2 = EloLedger()
        ledger2.apply("x", "z", VoteOutcome.CANDIDATE_A)
        ledger2.apply("x", "y", VoteOutcome.CANDIDATE_A)

        # Same multiset of votes, different order → different trajectories
        assert ledger1.ratings["y"] != ledger2.ratings["y"]

    def test_many_wins_converge_upward(self):
        ledger = EloLedger()
        for _ in range(100):
            ledger.apply("strong", "weak", VoteOutcome.CANDIDATE_A)
        assert ledger.ratings["strong"] > 1400
        assert ledger.ratings["weak"] < 1000
