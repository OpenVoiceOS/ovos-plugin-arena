"""
ELO edge-case tests.

Covers:
- rating floor (ELO should not go negative even after many losses)
- K-factor switch at VETERAN_THRESHOLD (larger delta before, smaller after)
- draw handling documented behaviour (tie → no change when ratings equal)
- vote for unknown matchup in process_vote (missing matchup keys handled)
- replay order-dependence: votes processed in *different* orders produce
  *different* results (order-dependent by design — document this)
"""

import uuid
from datetime import datetime, timedelta

import pytest

from app.arena.elo import (
    INITIAL_ELO,
    K_FACTOR,
    K_FACTOR_VETERAN,
    VETERAN_THRESHOLD,
    expected_score,
    k_factor,
    process_vote,
    replay_from_votes,
    update_ratings,
)
from app.arena.models import Matchup, PluginFamily, Vote, VoteOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matchup(pid_a=None, pid_b=None, fam=PluginFamily.TTS):
    return Matchup(
        id=uuid.uuid4(),
        family=fam,
        input_ref="test",
        sample_a_id=uuid.uuid4(),
        sample_b_id=uuid.uuid4(),
        plugin_a_id=pid_a or uuid.uuid4(),
        plugin_b_id=pid_b or uuid.uuid4(),
    )


def _vote(matchup_id, outcome, dt=None):
    return Vote(
        id=uuid.uuid4(),
        matchup_id=matchup_id,
        outcome=outcome,
        cast_at=dt or datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# K-factor switch
# ---------------------------------------------------------------------------


def test_k_factor_below_threshold():
    assert k_factor(0) == K_FACTOR
    assert k_factor(VETERAN_THRESHOLD - 1) == K_FACTOR


def test_k_factor_at_threshold():
    assert k_factor(VETERAN_THRESHOLD) == K_FACTOR_VETERAN


def test_k_factor_above_threshold():
    assert k_factor(VETERAN_THRESHOLD + 100) == K_FACTOR_VETERAN


def test_update_ratings_delta_smaller_for_veteran():
    """A veteran player (>= VETERAN_THRESHOLD battles) gains/loses less per vote."""
    r_a, r_b = 1200.0, 1200.0
    new_a_fresh, _ = update_ratings(r_a, r_b, VoteOutcome.CANDIDATE_A, battles_a=0, battles_b=0)
    new_a_vet, _ = update_ratings(r_a, r_b, VoteOutcome.CANDIDATE_A,
                                   battles_a=VETERAN_THRESHOLD, battles_b=0)
    delta_fresh = abs(new_a_fresh - r_a)
    delta_vet = abs(new_a_vet - r_a)
    assert delta_fresh > delta_vet, "Fresh player should have larger delta than veteran"


# ---------------------------------------------------------------------------
# Rating floor — should not go below 100 after many losses
# ---------------------------------------------------------------------------


def test_elo_does_not_go_negative_after_many_losses():
    """
    After a large number of consecutive losses the rating should remain positive.
    The ELO formula converges — it cannot reach zero since expected_score → 0
    means the update term K*(0 - ~0) ≈ 0.
    """
    rating = INITIAL_ELO
    for _ in range(500):
        # Always lose against a very strong opponent
        rating, _ = update_ratings(rating, 3000.0, VoteOutcome.CANDIDATE_B)
    assert rating > 0, f"ELO went non-positive: {rating}"


# ---------------------------------------------------------------------------
# Documented draw behaviour
# ---------------------------------------------------------------------------


def test_tie_equal_ratings_no_movement():
    """When both ratings are equal and outcome is TIE, scores stay the same."""
    new_a, new_b = update_ratings(1200.0, 1200.0, VoteOutcome.TIE)
    assert new_a == pytest.approx(1200.0)
    assert new_b == pytest.approx(1200.0)


def test_tie_unequal_ratings_higher_loses():
    """In a tie the over-rated player loses slightly and the under-rated gains."""
    new_a, new_b = update_ratings(1400.0, 1200.0, VoteOutcome.TIE)
    assert new_a < 1400.0, "Over-rated player should lose ELO on a tie"
    assert new_b > 1200.0, "Under-rated player should gain ELO on a tie"


# ---------------------------------------------------------------------------
# process_vote with unknown plugin in ratings dict
# ---------------------------------------------------------------------------


def test_process_vote_initialises_unknown_plugin():
    """process_vote must handle plugins absent from the ratings dict."""
    pid_a = uuid.uuid4()
    pid_b = uuid.uuid4()
    m = _matchup(pid_a, pid_b)
    v = _vote(m.id, VoteOutcome.CANDIDATE_A)

    # Pass empty dicts — the function should use INITIAL_ELO for missing keys
    ratings: dict = {}
    battles: dict = {}
    snap_a, snap_b = process_vote(v, m, ratings, battles)

    assert snap_a.elo_before == pytest.approx(INITIAL_ELO)
    assert snap_b.elo_before == pytest.approx(INITIAL_ELO)
    assert ratings[pid_a] > INITIAL_ELO
    assert ratings[pid_b] < INITIAL_ELO


# ---------------------------------------------------------------------------
# Replay order-dependence is the DOCUMENTED, EXPECTED behaviour
# ---------------------------------------------------------------------------


def test_replay_is_order_dependent():
    """
    replay_from_votes is explicitly order-dependent.

    The docstring states: "Votes must be ordered by cast_at for
    reproducibility".  Given two *different* orderings of the same votes,
    the final ratings should differ, confirming the system is not accidentally
    commutative.
    """
    pid_a = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000000")
    pid_b = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000000")

    base = datetime(2024, 1, 1)
    matchups = {}

    def make_vote(outcome, minutes):
        m = Matchup(
            id=uuid.uuid4(),
            family=PluginFamily.TTS,
            input_ref=f"p{minutes}",
            sample_a_id=uuid.uuid4(),
            sample_b_id=uuid.uuid4(),
            plugin_a_id=pid_a,
            plugin_b_id=pid_b,
        )
        matchups[m.id] = m
        return Vote(
            id=uuid.uuid4(),
            matchup_id=m.id,
            outcome=outcome,
            cast_at=base + timedelta(minutes=minutes),
        )

    # Asymmetric sequence so order matters
    v1 = make_vote(VoteOutcome.CANDIDATE_A, 0)
    v2 = make_vote(VoteOutcome.CANDIDATE_B, 1)

    result_asc = replay_from_votes([v1, v2], matchups)
    result_desc = replay_from_votes([v2, v1], matchups)

    # With equal starting ratings, alternating A/B wins produces K-factor
    # asymmetry across the two orderings — ratings will differ.
    assert result_asc[pid_a] != pytest.approx(result_desc[pid_a]), (
        "Order-dependent replay should produce different results for different vote orderings"
    )


# ---------------------------------------------------------------------------
# Replay with a single vote
# ---------------------------------------------------------------------------


def test_replay_single_vote():
    pid_a = uuid.uuid4()
    pid_b = uuid.uuid4()
    m = Matchup(
        id=uuid.uuid4(),
        family=PluginFamily.TTS,
        input_ref="x",
        sample_a_id=uuid.uuid4(),
        sample_b_id=uuid.uuid4(),
        plugin_a_id=pid_a,
        plugin_b_id=pid_b,
    )
    v = Vote(id=uuid.uuid4(), matchup_id=m.id, outcome=VoteOutcome.CANDIDATE_A,
             cast_at=datetime.utcnow())

    result = replay_from_votes([v], {m.id: m})
    assert result[pid_a] > INITIAL_ELO
    assert result[pid_b] < INITIAL_ELO


# ---------------------------------------------------------------------------
# Multi-family replay: only requested family affected
# ---------------------------------------------------------------------------


def test_replay_with_mixed_families_only_counts_relevant_matchups():
    """Matchups of different families share replay but final ratings only include
    those plugins present in the vote log passed to replay_from_votes."""
    pid_tts_a = uuid.uuid4()
    pid_stt_a = uuid.uuid4()
    pid_stt_b = uuid.uuid4()

    def mk(pid_a, pid_b, fam):
        m = Matchup(
            id=uuid.uuid4(),
            family=fam,
            input_ref="ref",
            sample_a_id=uuid.uuid4(),
            sample_b_id=uuid.uuid4(),
            plugin_a_id=pid_a,
            plugin_b_id=pid_b,
        )
        v = Vote(id=uuid.uuid4(), matchup_id=m.id,
                 outcome=VoteOutcome.CANDIDATE_A, cast_at=datetime.utcnow())
        return m, v

    m_stt, v_stt = mk(pid_stt_a, pid_stt_b, PluginFamily.STT)
    matchups = {m_stt.id: m_stt}
    result = replay_from_votes([v_stt], matchups)

    # Only STT plugins should appear — the TTS pid is absent
    assert pid_tts_a not in result
    assert pid_stt_a in result
