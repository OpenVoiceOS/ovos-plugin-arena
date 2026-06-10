"""
Tests for the ELO ranking math (arena.elo).

Key property verified: replaying the vote log always reproduces identical ratings.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from app.arena.elo import (
    INITIAL_ELO,
    K_FACTOR,
    expected_score,
    process_vote,
    replay_from_votes,
    update_ratings,
)
from app.arena.models import Matchup, PluginFamily, RatingSnapshot, Vote, VoteOutcome


# ---------------------------------------------------------------------------
# Expected score
# ---------------------------------------------------------------------------


def test_expected_score_equal_ratings():
    e = expected_score(1200.0, 1200.0)
    assert e == pytest.approx(0.5)


def test_expected_score_higher_rated_wins_more_often():
    e = expected_score(1400.0, 1200.0)
    assert e > 0.5


def test_expected_score_lower_rated_loses_more_often():
    e = expected_score(1000.0, 1200.0)
    assert e < 0.5


def test_expected_scores_sum_to_one():
    r_a, r_b = 1350.0, 1150.0
    assert expected_score(r_a, r_b) + expected_score(r_b, r_a) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rating update
# ---------------------------------------------------------------------------


def _make_matchup(pid_a=None, pid_b=None):
    return Matchup(
        id=uuid.uuid4(),
        family=PluginFamily.TTS,
        input_ref="test",
        sample_a_id=uuid.uuid4(),
        sample_b_id=uuid.uuid4(),
        plugin_a_id=pid_a or uuid.uuid4(),
        plugin_b_id=pid_b or uuid.uuid4(),
    )


def _make_vote(matchup_id, outcome, dt=None):
    return Vote(
        id=uuid.uuid4(),
        matchup_id=matchup_id,
        outcome=outcome,
        cast_at=dt or datetime.utcnow(),
    )


def test_winner_gains_elo():
    new_a, new_b = update_ratings(1200.0, 1200.0, VoteOutcome.CANDIDATE_A)
    assert new_a > 1200.0
    assert new_b < 1200.0


def test_tie_moves_ratings_slightly():
    new_a, new_b = update_ratings(1200.0, 1200.0, VoteOutcome.TIE)
    assert new_a == pytest.approx(1200.0)  # equal ratings + tie = no change
    assert new_b == pytest.approx(1200.0)


def test_both_wrong_treated_as_tie():
    new_a_t, new_b_t = update_ratings(1200.0, 1200.0, VoteOutcome.TIE)
    new_a_bw, new_b_bw = update_ratings(1200.0, 1200.0, VoteOutcome.BOTH_WRONG)
    assert new_a_t == pytest.approx(new_a_bw)
    assert new_b_t == pytest.approx(new_b_bw)


def test_zero_sum_property():
    """ELO is zero-sum: total rating before == total rating after."""
    r_a, r_b = 1350.0, 1050.0
    new_a, new_b = update_ratings(r_a, r_b, VoteOutcome.CANDIDATE_A)
    assert new_a + new_b == pytest.approx(r_a + r_b)


def test_zero_sum_all_outcomes():
    for outcome in VoteOutcome:
        r_a, r_b = 1200.0, 1300.0
        new_a, new_b = update_ratings(r_a, r_b, outcome)
        assert new_a + new_b == pytest.approx(r_a + r_b)


# ---------------------------------------------------------------------------
# process_vote
# ---------------------------------------------------------------------------


def test_process_vote_mutates_state():
    m = _make_matchup()
    v = _make_vote(m.id, VoteOutcome.CANDIDATE_A)

    ratings = {m.plugin_a_id: 1200.0, m.plugin_b_id: 1200.0}
    battles = {m.plugin_a_id: 0, m.plugin_b_id: 0}

    snap_a, snap_b = process_vote(v, m, ratings, battles)

    assert ratings[m.plugin_a_id] > 1200.0
    assert ratings[m.plugin_b_id] < 1200.0
    assert battles[m.plugin_a_id] == 1
    assert battles[m.plugin_b_id] == 1


def test_process_vote_returns_correct_snapshots():
    m = _make_matchup()
    v = _make_vote(m.id, VoteOutcome.CANDIDATE_B)

    ratings = {m.plugin_a_id: 1300.0, m.plugin_b_id: 1100.0}
    battles = {m.plugin_a_id: 5, m.plugin_b_id: 5}

    snap_a, snap_b = process_vote(v, m, ratings, battles)

    assert snap_a.plugin_id == m.plugin_a_id
    assert snap_b.plugin_id == m.plugin_b_id
    assert snap_a.elo_before == pytest.approx(1300.0)
    assert snap_b.elo_before == pytest.approx(1100.0)
    assert snap_a.delta == pytest.approx(snap_a.elo_after - 1300.0)
    assert snap_b.delta == pytest.approx(snap_b.elo_after - 1100.0)


# ---------------------------------------------------------------------------
# replay_from_votes — KEY PROPERTY TEST
# ---------------------------------------------------------------------------


def _build_vote_sequence(n=10, seed=42):
    """Build a deterministic sequence of votes for replay testing."""
    import random

    rng = random.Random(seed)
    outcomes = list(VoteOutcome)

    pid_a = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000000")
    pid_b = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000000")

    matchups = {}
    votes = []
    base = datetime(2024, 1, 1)

    for i in range(n):
        m = Matchup(
            id=uuid.uuid4(),
            family=PluginFamily.TTS,
            input_ref=f"prompt-{i}",
            sample_a_id=uuid.uuid4(),
            sample_b_id=uuid.uuid4(),
            plugin_a_id=pid_a,
            plugin_b_id=pid_b,
        )
        matchups[m.id] = m

        v = Vote(
            id=uuid.uuid4(),
            matchup_id=m.id,
            outcome=rng.choice(outcomes),
            cast_at=base + timedelta(minutes=i),
        )
        votes.append(v)

    return votes, matchups


def test_replay_is_deterministic():
    """Replaying the same vote log twice yields identical ratings."""
    votes, matchups = _build_vote_sequence(n=20)

    result1 = replay_from_votes(votes, matchups)
    result2 = replay_from_votes(votes, matchups)

    assert result1.keys() == result2.keys()
    for pid in result1:
        assert result1[pid] == pytest.approx(result2[pid])


def test_replay_matches_sequential_processing():
    """
    Ratings from replay_from_votes must match those obtained by applying
    process_vote one step at a time.
    """
    votes, matchups = _build_vote_sequence(n=15)

    # Sequential processing
    pid_a = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000000")
    pid_b = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000000")
    ratings_seq = {pid_a: INITIAL_ELO, pid_b: INITIAL_ELO}
    battles_seq = {pid_a: 0, pid_b: 0}

    for vote in votes:
        m = matchups[vote.matchup_id]
        process_vote(vote, m, ratings_seq, battles_seq)

    # Replay
    ratings_replay = replay_from_votes(votes, matchups)

    for pid in (pid_a, pid_b):
        assert ratings_seq[pid] == pytest.approx(ratings_replay[pid])


def test_replay_empty_vote_log():
    result = replay_from_votes([], {})
    assert result == {}


def test_replay_skips_orphaned_votes():
    votes, matchups = _build_vote_sequence(n=5)
    # Remove one matchup to simulate orphaned vote
    first_vote = votes[0]
    del matchups[first_vote.matchup_id]

    result = replay_from_votes(votes, matchups)
    # Should still work for the remaining 4 votes
    assert len(result) > 0


def test_elo_ordering_reflects_wins():
    """A plugin that wins every matchup should end up with higher ELO."""
    pid_a = uuid.uuid4()
    pid_b = uuid.uuid4()
    matchups = {}
    votes = []
    base = datetime(2024, 1, 1)

    for i in range(10):
        m = Matchup(
            id=uuid.uuid4(),
            family=PluginFamily.TTS,
            input_ref=f"p{i}",
            sample_a_id=uuid.uuid4(),
            sample_b_id=uuid.uuid4(),
            plugin_a_id=pid_a,
            plugin_b_id=pid_b,
        )
        matchups[m.id] = m
        v = Vote(
            id=uuid.uuid4(),
            matchup_id=m.id,
            outcome=VoteOutcome.CANDIDATE_A,  # A always wins
            cast_at=base + timedelta(minutes=i),
        )
        votes.append(v)

    result = replay_from_votes(votes, matchups)
    assert result[pid_a] > result[pid_b]
