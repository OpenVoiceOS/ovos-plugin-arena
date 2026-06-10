"""
Tests for the ranking engine (arena.ranking).

Verifies that process_vote_and_update persists ELO changes and that
recompute_all_ratings reproduces the same final ratings as the incremental path.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from app.arena import db as arena_db
from app.arena import ranking
from app.arena.models import (
    EvalRun,
    Matchup,
    Plugin,
    PluginFamily,
    Sample,
    Vote,
    VoteOutcome,
)


def _seed_matchup(plugin_a, plugin_b, input_ref="hello"):
    """Create a complete matchup including eval runs and samples."""
    run_a = EvalRun(plugin_id=plugin_a.id, family=PluginFamily.TTS)
    run_b = EvalRun(plugin_id=plugin_b.id, family=PluginFamily.TTS)
    arena_db.create_eval_run(run_a)
    arena_db.create_eval_run(run_b)

    s_a = Sample(run_id=run_a.id, plugin_id=plugin_a.id, family=PluginFamily.TTS, input_ref=input_ref)
    s_b = Sample(run_id=run_b.id, plugin_id=plugin_b.id, family=PluginFamily.TTS, input_ref=input_ref)
    arena_db.create_sample(s_a)
    arena_db.create_sample(s_b)

    m = Matchup(
        family=PluginFamily.TTS,
        input_ref=input_ref,
        sample_a_id=s_a.id,
        sample_b_id=s_b.id,
        plugin_a_id=plugin_a.id,
        plugin_b_id=plugin_b.id,
    )
    arena_db.create_matchup(m)
    return m


def test_vote_updates_elo_in_db(tmp_db):
    p_a = Plugin(plugin_name="ranka", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="rankb", display_name="B", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)

    m = _seed_matchup(p_a, p_b)

    vote = Vote(matchup_id=m.id, outcome=VoteOutcome.CANDIDATE_A)
    arena_db.create_vote(vote)
    ranking.process_vote_and_update(vote)

    stats_a = arena_db.get_elo_stats(p_a.id)
    stats_b = arena_db.get_elo_stats(p_b.id)

    assert stats_a["elo"] > 1200.0, "winner should gain ELO"
    assert stats_b["elo"] < 1200.0, "loser should lose ELO"
    assert stats_a["wins"] == 1
    assert stats_b["losses"] == 1


def test_recompute_matches_incremental(tmp_db):
    """
    After a sequence of votes processed incrementally, recompute_all_ratings
    must produce the same ELO values.
    """
    p_a = Plugin(plugin_name="repla", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="replb", display_name="B", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)

    outcomes = [
        VoteOutcome.CANDIDATE_A,
        VoteOutcome.CANDIDATE_B,
        VoteOutcome.TIE,
        VoteOutcome.CANDIDATE_A,
        VoteOutcome.CANDIDATE_B,
    ]

    base = datetime(2024, 1, 1)
    for i, outcome in enumerate(outcomes):
        m = _seed_matchup(p_a, p_b, input_ref=f"prompt-{i}")
        vote = Vote(
            matchup_id=m.id,
            outcome=outcome,
            cast_at=base + timedelta(minutes=i),
        )
        arena_db.create_vote(vote)
        ranking.process_vote_and_update(vote)

    # Capture incremental ELO
    elo_a_inc = arena_db.get_elo_stats(p_a.id)["elo"]
    elo_b_inc = arena_db.get_elo_stats(p_b.id)["elo"]

    # Recompute from scratch
    replayed = ranking.recompute_all_ratings()

    assert replayed[p_a.id] == pytest.approx(elo_a_inc, abs=0.01)
    assert replayed[p_b.id] == pytest.approx(elo_b_inc, abs=0.01)


def test_ingest_metric_vote(tmp_db):
    p_a = Plugin(plugin_name="meta", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="metb", display_name="B", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)

    m = _seed_matchup(p_a, p_b)

    vote = ranking.ingest_metric_vote(matchup_id=m.id, winner="a", source="rtf_auto")
    assert vote is not None
    assert vote.automated is True
    assert vote.voter_id == "auto:rtf_auto"

    stats_a = arena_db.get_elo_stats(p_a.id)
    assert stats_a["elo"] > 1200.0


def test_ingest_metric_vote_invalid_winner(tmp_db):
    p_a = Plugin(plugin_name="inv_a", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="inv_b", display_name="B", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)

    m = _seed_matchup(p_a, p_b)

    result = ranking.ingest_metric_vote(matchup_id=m.id, winner="INVALID")
    assert result is None
