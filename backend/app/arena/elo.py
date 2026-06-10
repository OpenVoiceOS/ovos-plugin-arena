"""
Deterministic ELO / Bradley-Terry ranking for the OVOS Plugin Arena.

The rating system is fully replayable: given the ordered vote log, calling
``replay_from_votes`` always converges to the same ratings regardless of the
current state of ``elo_current``.

ELO update rules
----------------
K = 32  (standard for new players; reduce to 16 after N battles)
Expected score  E_a = 1 / (1 + 10^((R_b - R_a) / 400))
Actual score    S_a = 1.0 (win), 0.5 (tie / both_wrong), 0.0 (loss)
New rating      R_a' = R_a + K * (S_a - E_a)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.arena.models import (
    Matchup,
    RatingSnapshot,
    Vote,
    VoteOutcome,
)

INITIAL_ELO: float = 1200.0
K_FACTOR: float = 32.0
K_FACTOR_VETERAN: float = 16.0
VETERAN_THRESHOLD: int = 30  # battles before using lower K


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that player A beats player B under the ELO model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(battles: int) -> float:
    return K_FACTOR if battles < VETERAN_THRESHOLD else K_FACTOR_VETERAN


def update_ratings(
    rating_a: float,
    rating_b: float,
    outcome: VoteOutcome,
    battles_a: int = 0,
    battles_b: int = 0,
) -> Tuple[float, float]:
    """
    Compute updated ELO ratings for a single vote outcome.

    Parameters
    ----------
    rating_a, rating_b : current ELO ratings
    outcome : VoteOutcome (from A's perspective)
    battles_a, battles_b : battles fought so far (for K-factor)

    Returns
    -------
    (new_rating_a, new_rating_b)
    """
    e_a = expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a

    if outcome == VoteOutcome.CANDIDATE_A:
        s_a, s_b = 1.0, 0.0
    elif outcome == VoteOutcome.CANDIDATE_B:
        s_a, s_b = 0.0, 1.0
    else:
        # TIE or BOTH_WRONG — treat as draw for ELO purposes
        s_a, s_b = 0.5, 0.5

    ka = k_factor(battles_a)
    kb = k_factor(battles_b)

    new_a = rating_a + ka * (s_a - e_a)
    new_b = rating_b + kb * (s_b - e_b)
    return new_a, new_b


def process_vote(
    vote: Vote,
    matchup: Matchup,
    ratings: Dict[uuid.UUID, float],
    battles: Dict[uuid.UUID, int],
) -> Tuple[RatingSnapshot, RatingSnapshot]:
    """
    Apply *vote* to *ratings* in-place and return two RatingSnapshots.

    Parameters
    ----------
    vote     : the Vote being processed
    matchup  : the Matchup the vote refers to
    ratings  : mutable mapping plugin_id → current ELO (mutated)
    battles  : mutable mapping plugin_id → battles fought (mutated)

    Returns
    -------
    (snapshot_a, snapshot_b) — one per competing plugin
    """
    pid_a = matchup.plugin_a_id
    pid_b = matchup.plugin_b_id

    r_a = ratings.get(pid_a, INITIAL_ELO)
    r_b = ratings.get(pid_b, INITIAL_ELO)
    b_a = battles.get(pid_a, 0)
    b_b = battles.get(pid_b, 0)

    new_a, new_b = update_ratings(r_a, r_b, vote.outcome, b_a, b_b)

    now = datetime.utcnow()

    snap_a = RatingSnapshot(
        vote_id=vote.id,
        plugin_id=pid_a,
        elo_before=r_a,
        elo_after=new_a,
        delta=new_a - r_a,
        snapshot_at=now,
    )
    snap_b = RatingSnapshot(
        vote_id=vote.id,
        plugin_id=pid_b,
        elo_before=r_b,
        elo_after=new_b,
        delta=new_b - r_b,
        snapshot_at=now,
    )

    # Mutate state
    ratings[pid_a] = new_a
    ratings[pid_b] = new_b
    battles[pid_a] = b_a + 1
    battles[pid_b] = b_b + 1

    return snap_a, snap_b


def replay_from_votes(
    votes: List[Vote],
    matchups_by_id: Dict[uuid.UUID, Matchup],
    initial_elo: float = INITIAL_ELO,
) -> Dict[uuid.UUID, float]:
    """
    Deterministically replay the full vote log and return final ratings.

    Votes must be ordered by ``cast_at`` for reproducibility (the DB query
    that feeds this must include ORDER BY cast_at).

    Parameters
    ----------
    votes           : ordered list of all Vote records
    matchups_by_id  : mapping matchup_id → Matchup (pre-fetched)
    initial_elo     : starting ELO for every plugin (default 1200)

    Returns
    -------
    Dict mapping plugin_id → final ELO
    """
    ratings: Dict[uuid.UUID, float] = {}
    battles: Dict[uuid.UUID, int] = {}

    for vote in votes:
        matchup = matchups_by_id.get(vote.matchup_id)
        if matchup is None:
            continue  # orphaned vote — skip

        # Ensure initial state
        for pid in (matchup.plugin_a_id, matchup.plugin_b_id):
            if pid not in ratings:
                ratings[pid] = initial_elo
                battles[pid] = 0

        process_vote(vote, matchup, ratings, battles)

    return ratings
