"""
Tests for arena Pydantic data models (arena.models).
"""

import uuid

import pytest

from app.arena.models import (
    EvalRun,
    EvalStatus,
    LeaderboardEntry,
    Matchup,
    Plugin,
    PluginFamily,
    RatingSnapshot,
    Sample,
    Vote,
    VoteOutcome,
)


def test_plugin_defaults():
    p = Plugin(plugin_name="ovos-tts-plugin-test", display_name="Test TTS", family=PluginFamily.TTS)
    assert isinstance(p.id, uuid.UUID)
    assert p.config == {}
    assert p.config_hash == ""
    assert p.lang is None


def test_plugin_families():
    for fam in PluginFamily:
        p = Plugin(plugin_name=f"test-{fam.value}", display_name=fam.value, family=fam)
        assert p.family == fam


def test_eval_run_defaults():
    run = EvalRun(plugin_id=uuid.uuid4(), family=PluginFamily.STT)
    assert run.status == EvalStatus.PENDING
    assert run.metrics == {}
    assert run.started_at is None


def test_sample_creation():
    s = Sample(
        run_id=uuid.uuid4(),
        plugin_id=uuid.uuid4(),
        family=PluginFamily.TTS,
        input_ref="hello world",
    )
    assert s.output_ref is None
    assert s.metrics == {}


def test_matchup_fields():
    m = Matchup(
        family=PluginFamily.TTS,
        input_ref="test prompt",
        sample_a_id=uuid.uuid4(),
        sample_b_id=uuid.uuid4(),
        plugin_a_id=uuid.uuid4(),
        plugin_b_id=uuid.uuid4(),
    )
    assert m.status == "pending"


def test_vote_outcomes():
    for outcome in VoteOutcome:
        v = Vote(matchup_id=uuid.uuid4(), outcome=outcome)
        assert v.outcome == outcome
        assert not v.automated


def test_rating_snapshot():
    snap = RatingSnapshot(
        vote_id=uuid.uuid4(),
        plugin_id=uuid.uuid4(),
        elo_before=1200.0,
        elo_after=1216.0,
        delta=16.0,
    )
    assert snap.delta == pytest.approx(16.0)


def test_leaderboard_entry():
    entry = LeaderboardEntry(
        rank=1,
        plugin_id=uuid.uuid4(),
        plugin_name="test-plugin",
        display_name="Test Plugin",
        family=PluginFamily.TTS,
        lang="en-us",
        elo=1320.5,
        battles=10,
        wins=7,
        losses=2,
        ties=1,
        win_rate=70.0,
    )
    assert entry.rank == 1
    assert entry.elo == pytest.approx(1320.5)
