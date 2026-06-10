"""
Ranking engine for the OVOS Plugin Arena.

Bridges the ELO math (arena.elo) with the SQLite persistence layer
(arena.db).  The two public functions are:

``process_vote_and_update``
    Apply a single vote to the live ELO table and append rating snapshots.
    This is the hot path called right after a vote is cast.

``recompute_all_ratings``
    Replay the full ordered vote log from scratch, reset the elo_current
    table, and re-insert all rating snapshots.  Use this to verify
    determinism or to recover from a corrupted elo_current table.

Automated metric votes
----------------------
When ingesting external benchmark results (WER, RTF, detection F1) the
caller sets ``Vote.automated = True`` and ``Vote.voter_id = "auto:<source>"``.
These votes flow through the same ELO pipeline as human votes so the
leaderboard reflects both signal types.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Optional

from app.arena import db
from app.arena.elo import INITIAL_ELO, process_vote, replay_from_votes
from app.arena.models import Matchup, PluginFamily, Vote, VoteOutcome

logger = logging.getLogger(__name__)


def process_vote_and_update(vote: Vote) -> None:
    """
    Apply *vote* to the live elo_current table and persist rating snapshots.

    Steps
    -----
    1. Load the matchup referenced by the vote.
    2. Fetch current ELO + battle counts for both plugins.
    3. Compute new ratings via the ELO formula.
    4. Persist two RatingSnapshot rows.
    5. Update elo_current rows for both plugins.
    6. Mark the matchup as voted.

    Parameters
    ----------
    vote : an already-persisted Vote (id must exist in the votes table)
    """
    matchup = db.get_matchup(vote.matchup_id)
    if matchup is None:
        logger.error("process_vote_and_update: matchup %s not found", vote.matchup_id)
        return

    pid_a = matchup.plugin_a_id
    pid_b = matchup.plugin_b_id

    stats_a = db.get_elo_stats(pid_a)
    stats_b = db.get_elo_stats(pid_b)

    ratings = {pid_a: stats_a["elo"], pid_b: stats_b["elo"]}
    battles = {pid_a: stats_a["battles"], pid_b: stats_b["battles"]}

    snap_a, snap_b = process_vote(vote, matchup, ratings, battles)

    db.create_rating_snapshot(snap_a)
    db.create_rating_snapshot(snap_b)

    outcome = vote.outcome
    won_a = outcome == VoteOutcome.CANDIDATE_A
    won_b = outcome == VoteOutcome.CANDIDATE_B
    tied = outcome in (VoteOutcome.TIE, VoteOutcome.BOTH_WRONG)

    db.update_elo_stats(pid_a, snap_a.elo_after, won=won_a, tied=tied)
    db.update_elo_stats(pid_b, snap_b.elo_after, won=won_b, tied=tied)

    db.mark_matchup_voted(vote.matchup_id)

    logger.debug(
        "ELO update: %s %.1f→%.1f  %s %.1f→%.1f  outcome=%s",
        pid_a,
        snap_a.elo_before,
        snap_a.elo_after,
        pid_b,
        snap_b.elo_before,
        snap_b.elo_after,
        outcome.value,
    )


def recompute_all_ratings(family: Optional[PluginFamily] = None) -> Dict[uuid.UUID, float]:
    """
    Replay the full vote log deterministically and rebuild elo_current.

    Parameters
    ----------
    family : if given, only recompute for matchups of that family

    Returns
    -------
    Dict mapping plugin_id → final ELO after replay
    """
    # Load all votes ordered by cast_at (deterministic ordering)
    votes = db.list_votes()
    votes.sort(key=lambda v: v.cast_at)

    # Build matchup lookup
    matchups_by_id: Dict[uuid.UUID, Matchup] = {}
    for vote in votes:
        mid = vote.matchup_id
        if mid not in matchups_by_id:
            m = db.get_matchup(mid)
            if m is not None:
                matchups_by_id[mid] = m

    if family is not None:
        # Filter to votes whose matchup belongs to the requested family
        votes = [
            v
            for v in votes
            if matchups_by_id.get(v.matchup_id) is not None
            and matchups_by_id[v.matchup_id].family == family
        ]

    final_ratings = replay_from_votes(votes, matchups_by_id, INITIAL_ELO)

    # Reset elo_current for affected plugins
    for plugin_id, elo in final_ratings.items():
        db.set_elo(plugin_id, elo)

    logger.info(
        "Recomputed ratings for %d plugins (%d votes replayed)",
        len(final_ratings),
        len(votes),
    )
    return final_ratings


def ingest_metric_vote(
    matchup_id: uuid.UUID,
    winner: Optional[str],  # "a", "b", "tie", "both_wrong"
    source: str = "auto",
) -> Optional[Vote]:
    """
    Create and process an automated vote derived from benchmark metrics.

    Parameters
    ----------
    matchup_id : the Matchup being scored
    winner     : "a" → CANDIDATE_A, "b" → CANDIDATE_B,
                 "tie" → TIE, "both_wrong" → BOTH_WRONG, None → skip
    source     : label for the voter_id, e.g. "wer_oracle" or "rtf_auto"

    Returns
    -------
    The created Vote, or None if skipped
    """
    if winner is None:
        return None

    outcome_map = {
        "a": VoteOutcome.CANDIDATE_A,
        "b": VoteOutcome.CANDIDATE_B,
        "tie": VoteOutcome.TIE,
        "both_wrong": VoteOutcome.BOTH_WRONG,
    }
    outcome = outcome_map.get(winner.lower())
    if outcome is None:
        logger.warning("ingest_metric_vote: unknown winner %r, skipping", winner)
        return None

    vote = Vote(
        matchup_id=matchup_id,
        outcome=outcome,
        voter_id=f"auto:{source}",
        automated=True,
    )
    db.create_vote(vote)
    process_vote_and_update(vote)
    return vote
