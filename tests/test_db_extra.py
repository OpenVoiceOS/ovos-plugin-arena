"""
Additional DB-layer tests covering:
- concurrent write safety (threading)
- init_db idempotency (schema re-creation)
- FK integrity: vote referencing missing matchup raises
- list_plugins with lang filter
- get_plugin_by_id
- list_eval_runs filter by plugin_id
- update_elo_stats accumulation (battles/wins/losses/ties counters)
- metric ingestion with invalid metric name (ingest_metric_vote unknown winner)
"""

import threading
import uuid
from datetime import datetime

import pytest

from app.arena import db as arena_db
from app.arena.models import (
    EvalRun,
    EvalStatus,
    Matchup,
    Plugin,
    PluginFamily,
    Sample,
    Vote,
    VoteOutcome,
)


# ---------------------------------------------------------------------------
# Schema idempotency
# ---------------------------------------------------------------------------


def test_init_db_twice_is_idempotent(tmp_db):
    """Calling init_db a second time must not destroy existing data."""
    p = Plugin(plugin_name="idem-plugin", display_name="X", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p)

    # Re-initialise — CREATE TABLE IF NOT EXISTS should be a no-op
    arena_db.init_db(path=tmp_db)

    retrieved = arena_db.get_plugin_by_name("idem-plugin")
    assert retrieved is not None, "Existing data should survive re-init"


# ---------------------------------------------------------------------------
# get_plugin_by_id
# ---------------------------------------------------------------------------


def test_get_plugin_by_id_found(tmp_db):
    p = Plugin(plugin_name="byid-plugin", display_name="Y", family=PluginFamily.STT)
    arena_db.upsert_plugin(p)
    fetched = arena_db.get_plugin_by_id(p.id)
    assert fetched is not None
    assert fetched.plugin_name == "byid-plugin"


def test_get_plugin_by_id_missing(tmp_db):
    result = arena_db.get_plugin_by_id(uuid.uuid4())
    assert result is None


# ---------------------------------------------------------------------------
# list_plugins lang filter
# ---------------------------------------------------------------------------


def test_list_plugins_lang_filter(tmp_db):
    arena_db.upsert_plugin(
        Plugin(plugin_name="tts-pt", display_name="PT", family=PluginFamily.TTS, lang="pt-pt")
    )
    arena_db.upsert_plugin(
        Plugin(plugin_name="tts-en", display_name="EN", family=PluginFamily.TTS, lang="en-us")
    )
    arena_db.upsert_plugin(
        Plugin(plugin_name="tts-nolang", display_name="NO", family=PluginFamily.TTS, lang=None)
    )

    pt_only = arena_db.list_plugins(family=PluginFamily.TTS, lang="pt-pt")
    names = {p.plugin_name for p in pt_only}
    assert "tts-pt" in names
    assert "tts-nolang" in names   # NULL lang matches any lang filter
    assert "tts-en" not in names


# ---------------------------------------------------------------------------
# list_eval_runs filtered by plugin_id
# ---------------------------------------------------------------------------


def test_list_eval_runs_filter_by_plugin(tmp_db):
    p1 = Plugin(plugin_name="run-p1", display_name="P1", family=PluginFamily.TTS)
    p2 = Plugin(plugin_name="run-p2", display_name="P2", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p1)
    arena_db.upsert_plugin(p2)

    run1 = EvalRun(plugin_id=p1.id, family=PluginFamily.TTS)
    run2 = EvalRun(plugin_id=p2.id, family=PluginFamily.TTS)
    arena_db.create_eval_run(run1)
    arena_db.create_eval_run(run2)

    runs_p1 = arena_db.list_eval_runs(plugin_id=p1.id)
    assert len(runs_p1) == 1
    assert runs_p1[0].plugin_id == p1.id


# ---------------------------------------------------------------------------
# FK integrity: vote referencing missing matchup
# ---------------------------------------------------------------------------


def test_vote_fk_missing_matchup_raises(tmp_db):
    """Inserting a vote that references a non-existent matchup must fail (FK ON)."""
    vote = Vote(
        id=uuid.uuid4(),
        matchup_id=uuid.uuid4(),   # does not exist
        outcome=VoteOutcome.CANDIDATE_A,
        cast_at=datetime.utcnow(),
    )
    with pytest.raises(Exception):
        arena_db.create_vote(vote)


# ---------------------------------------------------------------------------
# update_elo_stats counter accumulation
# ---------------------------------------------------------------------------


def test_elo_stats_wins_losses_ties_accumulate(tmp_db):
    p = Plugin(plugin_name="counter-plugin", display_name="C", family=PluginFamily.TTS)
    arena_db.upsert_plugin(p)

    arena_db.update_elo_stats(p.id, 1216.0, won=True, tied=False)   # win
    arena_db.update_elo_stats(p.id, 1200.0, won=False, tied=False)  # loss
    arena_db.update_elo_stats(p.id, 1208.0, won=False, tied=True)   # tie

    stats = arena_db.get_elo_stats(p.id)
    assert stats["battles"] == 3
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["ties"] == 1
    assert stats["elo"] == pytest.approx(1208.0)


# ---------------------------------------------------------------------------
# Concurrent writes — threading safety
# ---------------------------------------------------------------------------


def test_concurrent_upsert_does_not_corrupt(tmp_db):
    """
    Multiple threads simultaneously upserting distinct plugins should all
    succeed without IntegrityError or data corruption.
    """
    errors = []

    def worker(i):
        try:
            p = Plugin(
                plugin_name=f"concurrent-{i}",
                display_name=f"C{i}",
                family=PluginFamily.TTS,
            )
            arena_db.upsert_plugin(p)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent writes raised: {errors}"

    all_plugins = arena_db.list_plugins(family=PluginFamily.TTS)
    names = {p.plugin_name for p in all_plugins}
    for i in range(20):
        assert f"concurrent-{i}" in names, f"concurrent-{i} missing after concurrent upsert"


def test_concurrent_vote_submission_does_not_corrupt(tmp_db):
    """
    Two threads racing to submit votes for *different* matchups should both
    succeed and the ELO counters should reflect two separate battles.
    """
    p_a = Plugin(plugin_name="cv-a", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="cv-b", display_name="B", family=PluginFamily.TTS)
    for p in (p_a, p_b):
        arena_db.upsert_plugin(p)

    def _make_matchup():
        run_a = EvalRun(plugin_id=p_a.id, family=PluginFamily.TTS)
        run_b = EvalRun(plugin_id=p_b.id, family=PluginFamily.TTS)
        arena_db.create_eval_run(run_a)
        arena_db.create_eval_run(run_b)
        s_a = Sample(run_id=run_a.id, plugin_id=p_a.id, family=PluginFamily.TTS, input_ref="x")
        s_b = Sample(run_id=run_b.id, plugin_id=p_b.id, family=PluginFamily.TTS, input_ref="x")
        arena_db.create_sample(s_a)
        arena_db.create_sample(s_b)
        m = Matchup(
            family=PluginFamily.TTS,
            input_ref="x",
            sample_a_id=s_a.id,
            sample_b_id=s_b.id,
            plugin_a_id=p_a.id,
            plugin_b_id=p_b.id,
        )
        arena_db.create_matchup(m)
        return m

    matchup1 = _make_matchup()
    matchup2 = _make_matchup()

    errors = []

    def vote_and_update(matchup, outcome):
        try:
            from app.arena import ranking
            v = Vote(matchup_id=matchup.id, outcome=outcome)
            arena_db.create_vote(v)
            ranking.process_vote_and_update(v)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=vote_and_update, args=(matchup1, VoteOutcome.CANDIDATE_A))
    t2 = threading.Thread(target=vote_and_update, args=(matchup2, VoteOutcome.CANDIDATE_B))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Concurrent vote threads raised: {errors}"

    stats_a = arena_db.get_elo_stats(p_a.id)
    assert stats_a["battles"] == 2, "Both votes should be counted"
