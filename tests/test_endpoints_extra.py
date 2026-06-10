"""
Additional endpoint tests covering:
- blind matchup must NOT leak plugin_a_id / plugin_b_id (identity masking)
- vote submission idempotency / double-vote returns 409
- leaderboard pagination (limit / offset query params)
- leaderboard lang filter
- metric vote ingestion endpoint: valid + unknown matchup_id + invalid winner
- vote on unknown matchup_id returns 404
"""

import uuid
from pathlib import Path
import sys

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture()
def client(tmp_db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.arena.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/arena")
    return TestClient(app)


def _seed_matchup(tmp_db, plugin_a_name="epa", plugin_b_name="epb"):
    from app.arena import db as arena_db
    from app.arena.models import EvalRun, Matchup, Plugin, PluginFamily, Sample

    p_a = Plugin(plugin_name=plugin_a_name, display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name=plugin_b_name, display_name="B", family=PluginFamily.TTS)
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
    return m, p_a, p_b


# ---------------------------------------------------------------------------
# Identity masking
# ---------------------------------------------------------------------------


def test_next_matchup_does_not_expose_plugin_ids(client, tmp_db):
    _seed_matchup(tmp_db)
    resp = client.get("/api/v1/arena/matchups/next/tts")
    assert resp.status_code == 200
    data = resp.json()
    # These internal fields must be absent from the public payload
    assert "plugin_a_id" not in data
    assert "plugin_b_id" not in data


def test_next_matchup_exposes_output_refs(client, tmp_db):
    """Public response must include the output references so the voter can listen."""
    _seed_matchup(tmp_db)
    resp = client.get("/api/v1/arena/matchups/next/tts")
    assert resp.status_code == 200
    data = resp.json()
    # keys exist (may be null if no file path set in this seed)
    assert "output_a_ref" in data
    assert "output_b_ref" in data


# ---------------------------------------------------------------------------
# Double-vote idempotency
# ---------------------------------------------------------------------------


def test_double_vote_returns_409(client, tmp_db):
    m, _, _ = _seed_matchup(tmp_db)
    first = client.post(f"/api/v1/arena/matchups/{m.id}/vote", json={"outcome": "candidate_a"})
    assert first.status_code == 201
    second = client.post(f"/api/v1/arena/matchups/{m.id}/vote", json={"outcome": "candidate_a"})
    assert second.status_code == 409


def test_double_vote_different_outcomes_still_409(client, tmp_db):
    m, _, _ = _seed_matchup(tmp_db, "ep-diff-a", "ep-diff-b")
    client.post(f"/api/v1/arena/matchups/{m.id}/vote", json={"outcome": "candidate_a"})
    resp = client.post(f"/api/v1/arena/matchups/{m.id}/vote", json={"outcome": "candidate_b"})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Vote on unknown matchup
# ---------------------------------------------------------------------------


def test_vote_unknown_matchup_returns_404(client, tmp_db):
    resp = client.post(
        f"/api/v1/arena/matchups/{uuid.uuid4()}/vote",
        json={"outcome": "candidate_a"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Leaderboard pagination
# ---------------------------------------------------------------------------


def _seed_n_plugins(n, family="tts"):
    from app.arena import db as arena_db
    from app.arena.models import Plugin, PluginFamily

    fam = PluginFamily(family)
    ids = []
    for i in range(n):
        p = Plugin(plugin_name=f"page-plugin-{i}", display_name=f"P{i}", family=fam)
        arena_db.upsert_plugin(p)
        arena_db.update_elo_stats(p.id, 1200.0 + i, won=False, tied=True)
        ids.append(p.id)
    return ids


def test_leaderboard_limit(client, tmp_db):
    _seed_n_plugins(10)
    resp = client.get("/api/v1/arena/leaderboard/tts?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entries"]) == 3
    assert data["total"] == 10


def test_leaderboard_offset(client, tmp_db):
    _seed_n_plugins(5)
    resp_full = client.get("/api/v1/arena/leaderboard/tts?limit=5&offset=0")
    resp_offset = client.get("/api/v1/arena/leaderboard/tts?limit=5&offset=2")

    full = resp_full.json()["entries"]
    offset_entries = resp_offset.json()["entries"]

    assert len(offset_entries) == 3  # 5 - 2
    # The third entry of the full list should be first in offset list
    assert full[2]["plugin_name"] == offset_entries[0]["plugin_name"]


def test_leaderboard_lang_filter(client, tmp_db):
    from app.arena import db as arena_db
    from app.arena.models import Plugin, PluginFamily

    p_pt = Plugin(plugin_name="lb-pt", display_name="PT", family=PluginFamily.TTS, lang="pt-pt")
    p_en = Plugin(plugin_name="lb-en", display_name="EN", family=PluginFamily.TTS, lang="en-us")
    for p in (p_pt, p_en):
        arena_db.upsert_plugin(p)
        arena_db.update_elo_stats(p.id, 1200.0, won=False, tied=True)

    resp = client.get("/api/v1/arena/leaderboard/tts?lang=pt-pt")
    assert resp.status_code == 200
    data = resp.json()
    names = {e["plugin_name"] for e in data["entries"]}
    assert "lb-pt" in names
    assert "lb-en" not in names


# ---------------------------------------------------------------------------
# Metric vote endpoint
# ---------------------------------------------------------------------------


def test_metric_vote_valid(client, tmp_db):
    m, _, _ = _seed_matchup(tmp_db, "mv-a", "mv-b")
    resp = client.post(
        "/api/v1/arena/votes/metric",
        json={"matchup_id": str(m.id), "winner": "a", "source": "rtf"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["automated"] is True


def test_metric_vote_invalid_winner_returns_400(client, tmp_db):
    m, _, _ = _seed_matchup(tmp_db, "mvi-a", "mvi-b")
    resp = client.post(
        "/api/v1/arena/votes/metric",
        json={"matchup_id": str(m.id), "winner": "UNKNOWN_WINNER", "source": "test"},
    )
    assert resp.status_code == 400


def test_metric_vote_all_valid_winners(client, tmp_db):
    """Each valid winner token must produce a 201."""
    from app.arena import db as arena_db
    from app.arena.models import EvalRun, Matchup, Plugin, PluginFamily, Sample

    for winner_token in ("a", "b", "tie", "both_wrong"):
        # Fresh plugins + matchup for each winner to avoid double-vote
        suffix = winner_token.replace("_", "")
        p_a = Plugin(plugin_name=f"mva2-{suffix}-a", display_name="A", family=PluginFamily.TTS)
        p_b = Plugin(plugin_name=f"mva2-{suffix}-b", display_name="B", family=PluginFamily.TTS)
        for p in (p_a, p_b):
            arena_db.upsert_plugin(p)

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

        resp = client.post(
            "/api/v1/arena/votes/metric",
            json={"matchup_id": str(m.id), "winner": winner_token, "source": "auto"},
        )
        assert resp.status_code == 201, f"winner={winner_token!r} got {resp.status_code}"
