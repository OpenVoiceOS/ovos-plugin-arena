"""Unit tests for arena.fraud — vote fraud / dedup resistance (§4 A1.4)."""
from __future__ import annotations

import unittest.mock

from arena.fraud import (
    apply_account_age_gate,
    apply_daily_cap,
    apply_one_sided_downweight,
    resolve_vote_weights,
)


def _vote(author, battle_id="b1", choice="a", created_at="2026-01-01T00:00:00Z"):
    return {"author": author, "battle_id": battle_id, "choice": choice,
            "created_at": created_at, "issue_number": 1}


class TestApplyDailyCap:
    def test_under_cap_all_pass(self):
        votes = [_vote("alice", battle_id=f"b{i}") for i in range(10)]
        decisions = apply_daily_cap(votes, {}, cap=50)
        assert all(d.weight == 1.0 for d in decisions)

    def test_over_cap_excess_discarded(self):
        votes = [_vote("alice", battle_id=f"b{i}") for i in range(5)]
        decisions = apply_daily_cap(votes, {}, cap=3)
        assert [d.weight for d in decisions] == [1.0, 1.0, 1.0, 0.0, 0.0]
        assert decisions[3].discarded_reason == "daily_vote_cap_exceeded"

    def test_cap_is_per_modality(self):
        votes = [_vote("alice", battle_id=f"intent-{i}") for i in range(3)] + \
                [_vote("alice", battle_id=f"stt-{i}") for i in range(3)]
        modality_by_battle = {f"intent-{i}": "intent" for i in range(3)}
        modality_by_battle.update({f"stt-{i}": "stt" for i in range(3)})
        decisions = apply_daily_cap(votes, modality_by_battle, cap=3)
        assert all(d.weight == 1.0 for d in decisions)  # 3 each, under cap

    def test_cap_is_per_day(self):
        votes = [
            _vote("alice", battle_id="b1", created_at="2026-01-01T00:00:00Z"),
            _vote("alice", battle_id="b2", created_at="2026-01-01T12:00:00Z"),
            _vote("alice", battle_id="b3", created_at="2026-01-02T00:00:00Z"),
        ]
        decisions = apply_daily_cap(votes, {}, cap=2)
        assert [d.weight for d in decisions] == [1.0, 1.0, 1.0]  # 2 on day1, 1 on day2

    def test_deterministic_by_input_order(self):
        votes = [_vote("alice", battle_id=f"b{i}") for i in range(5)]
        d1 = [d.weight for d in apply_daily_cap(votes, {}, cap=3)]
        d2 = [d.weight for d in apply_daily_cap(votes, {}, cap=3)]
        assert d1 == d2


class TestApplyAccountAgeGate:
    def test_new_account_gated_to_zero(self):
        votes = [_vote("newbie", created_at="2026-01-05T00:00:00Z")]
        decisions = apply_daily_cap(votes, {})
        gated = apply_account_age_gate(
            decisions, {"newbie": "2026-01-01T00:00:00Z"}, min_days=7,
        )
        assert gated[0].weight == 0.0
        assert gated[0].discarded_reason == "account_too_new"

    def test_established_account_passes(self):
        votes = [_vote("veteran", created_at="2026-06-01T00:00:00Z")]
        decisions = apply_daily_cap(votes, {})
        gated = apply_account_age_gate(
            decisions, {"veteran": "2020-01-01T00:00:00Z"}, min_days=7,
        )
        assert gated[0].weight == 1.0

    def test_unknown_author_not_gated(self):
        votes = [_vote("stranger")]
        decisions = apply_daily_cap(votes, {})
        gated = apply_account_age_gate(decisions, {})
        assert gated[0].weight == 1.0

    def test_already_discarded_vote_unchanged(self):
        votes = [_vote("alice", battle_id=f"b{i}") for i in range(2)]
        decisions = apply_daily_cap(votes, {}, cap=1)
        gated = apply_account_age_gate(decisions, {"alice": "2020-01-01T00:00:00Z"})
        assert gated[1].discarded_reason == "daily_vote_cap_exceeded"
        assert gated[1].weight == 0.0

    def test_pure_no_network(self):
        # arena.fraud must never touch the network — enforce it structurally.
        with unittest.mock.patch("subprocess.run", side_effect=AssertionError("network call!")):
            votes = [_vote("alice")]
            decisions = apply_daily_cap(votes, {})
            apply_account_age_gate(decisions, {"alice": "2020-01-01T00:00:00Z"})
            apply_one_sided_downweight(decisions)


class TestApplyOneSidedDownweight:
    def test_below_min_votes_unaffected(self):
        votes = [_vote("alice", battle_id=f"b{i}", choice="a") for i in range(10)]
        decisions = apply_daily_cap(votes, {})
        result = apply_one_sided_downweight(decisions, min_votes=20)
        assert all(d.weight == 1.0 for d in result)

    def test_always_a_downweighted(self):
        votes = [_vote("bot", battle_id=f"b{i}", choice="a") for i in range(25)]
        decisions = apply_daily_cap(votes, {})
        result = apply_one_sided_downweight(decisions, min_votes=20, threshold=0.95)
        assert all(d.weight == 0.5 for d in result)

    def test_balanced_voter_unaffected(self):
        votes = [
            _vote("fair", battle_id=f"b{i}", choice="a" if i % 2 == 0 else "b")
            for i in range(30)
        ]
        decisions = apply_daily_cap(votes, {})
        result = apply_one_sided_downweight(decisions, min_votes=20, threshold=0.95)
        assert all(d.weight == 1.0 for d in result)

    def test_ties_excluded_from_ratio(self):
        # 15 "a" + 15 ties: ratio among a/b-only votes is 100% a, but there
        # are only 15 a/b votes — below min_votes=20, so unaffected.
        votes = [_vote("half", battle_id=f"a{i}", choice="a") for i in range(15)] + \
                [_vote("half", battle_id=f"t{i}", choice="tie") for i in range(15)]
        decisions = apply_daily_cap(votes, {})
        result = apply_one_sided_downweight(decisions, min_votes=20, threshold=0.95)
        assert all(d.weight == 1.0 for d in result)

    def test_discarded_votes_not_double_counted_but_flag_preserved(self):
        votes = [_vote("alice", battle_id=f"b{i}", choice="a") for i in range(25)]
        decisions = apply_daily_cap(votes, {}, cap=20)
        result = apply_one_sided_downweight(decisions, min_votes=20, threshold=0.95)
        # the 5 capped votes stay discarded (weight 0), not downweighted to 0.5
        assert result[20].discarded_reason == "daily_vote_cap_exceeded"
        assert result[20].weight == 0.0
        # the 20 surviving votes get downweighted (all "a")
        assert all(result[i].weight == 0.5 for i in range(20))


class TestResolveVoteWeights:
    def test_full_pipeline_deterministic(self):
        votes = [_vote("alice", battle_id=f"b{i}", choice="a") for i in range(3)]
        r1 = resolve_vote_weights(votes, {}, {})
        r2 = resolve_vote_weights(votes, {}, {})
        assert [d.weight for d in r1] == [d.weight for d in r2]

    def test_empty_votes(self):
        assert resolve_vote_weights([], {}, {}) == []
