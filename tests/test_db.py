"""
Tests for the SQLite persistence layer (arena.db).
"""

import uuid

import pytest

from app.arena import db as arena_db
from app.arena.models import (
    EvalRun,
    Matchup,
    Plugin,
    PluginFamily,
    Sample,
    Vote,
    VoteOutcome,
)


def test_upsert_and_retrieve_plugin(tmp_db):
    p = Plugin(plugin_name="ovos-tts-plugin-test", display_name="Test TTS", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p)

    retrieved = arena_db.get_plugin_by_name("ovos-tts-plugin-test")
    assert retrieved is not None
    assert retrieved.plugin_name == "ovos-tts-plugin-test"
    assert retrieved.family == PluginFamily.TTS


def test_upsert_plugin_is_idempotent(tmp_db):
    p = Plugin(plugin_name="idempotent-plugin", display_name="v1", family=PluginFamily.STT)
    arena_db.upsert_plugin(p)

    p2 = Plugin(
        id=p.id,
        plugin_name="idempotent-plugin",
        display_name="v2",
        family=PluginFamily.STT,
    )
    arena_db.upsert_plugin(p2)

    retrieved = arena_db.get_plugin_by_name("idempotent-plugin")
    assert retrieved.display_name == "v2"


def test_list_plugins_by_family(tmp_db):
    for i in range(3):
        arena_db.upsert_plugin(
            Plugin(plugin_name=f"tts-{i}", display_name=f"TTS {i}", family=PluginFamily.TTS)
        )
    arena_db.upsert_plugin(
        Plugin(plugin_name="stt-0", display_name="STT 0", family=PluginFamily.STT)
    )

    tts = arena_db.list_plugins(family=PluginFamily.TTS)
    assert len(tts) == 3
    stt = arena_db.list_plugins(family=PluginFamily.STT)
    assert len(stt) == 1


def test_eval_run_lifecycle(tmp_db):
    p = Plugin(plugin_name="run-plugin", display_name="Run Plugin", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p)

    run = EvalRun(plugin_id=p.id, family=PluginFamily.TTS)
    arena_db.create_eval_run(run)

    fetched = arena_db.get_eval_run(run.id)
    assert fetched is not None
    assert fetched.status.value == "pending"

    from app.arena.models import EvalStatus
    from datetime import datetime

    fetched.status = EvalStatus.DONE
    fetched.finished_at = datetime.utcnow()
    fetched.metrics = {"rtf": 0.12}
    arena_db.update_eval_run(fetched)

    updated = arena_db.get_eval_run(run.id)
    assert updated.status == EvalStatus.DONE
    assert updated.metrics["rtf"] == pytest.approx(0.12)


def test_sample_create_and_retrieve(tmp_db):
    p = Plugin(plugin_name="sample-plugin", display_name="P", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p)

    run = EvalRun(plugin_id=p.id, family=PluginFamily.TTS)
    arena_db.create_eval_run(run)

    s = Sample(
        run_id=run.id,
        plugin_id=p.id,
        family=PluginFamily.TTS,
        input_ref="hello",
        output_ref="/tmp/hello.wav",
        metrics={"rtf": 0.05},
    )
    arena_db.create_sample(s)

    samples = arena_db.list_samples_for_run(run.id)
    assert len(samples) == 1
    assert samples[0].output_ref == "/tmp/hello.wav"


def test_matchup_pending_queue(tmp_db):
    p_a = Plugin(plugin_name="pa", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="pb", display_name="B", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)

    run_a = EvalRun(plugin_id=p_a.id, family=PluginFamily.TTS)
    run_b = EvalRun(plugin_id=p_b.id, family=PluginFamily.TTS)
    arena_db.create_eval_run(run_a)
    arena_db.create_eval_run(run_b)

    s_a = Sample(run_id=run_a.id, plugin_id=p_a.id, family=PluginFamily.TTS, input_ref="hi")
    s_b = Sample(run_id=run_b.id, plugin_id=p_b.id, family=PluginFamily.TTS, input_ref="hi")
    arena_db.create_sample(s_a)
    arena_db.create_sample(s_b)

    m = Matchup(
        family=PluginFamily.TTS,
        input_ref="hi",
        sample_a_id=s_a.id,
        sample_b_id=s_b.id,
        plugin_a_id=p_a.id,
        plugin_b_id=p_b.id,
    )
    arena_db.create_matchup(m)

    pending = arena_db.get_pending_matchup(PluginFamily.TTS)
    assert pending is not None
    assert pending.id == m.id

    arena_db.mark_matchup_voted(m.id)
    assert arena_db.get_pending_matchup(PluginFamily.TTS) is None


def test_vote_and_leaderboard(tmp_db):
    p_a = Plugin(plugin_name="la", display_name="LA", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="lb", display_name="LB", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)
        arena_db.update_elo_stats(p.id, 1200.0, won=False, tied=True)  # initialise

    lb = arena_db.get_leaderboard(PluginFamily.TTS)
    assert len(lb) == 2
    # After initialisation both are at 1200 (ties), no winner yet
    assert lb[0].elo == pytest.approx(1200.0)


def test_elo_stats_update(tmp_db):
    p = Plugin(plugin_name="elo-test", display_name="E", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p)

    arena_db.update_elo_stats(p.id, 1232.0, won=True, tied=False)
    stats = arena_db.get_elo_stats(p.id)
    assert stats["elo"] == pytest.approx(1232.0)
    assert stats["wins"] == 1
    assert stats["battles"] == 1
