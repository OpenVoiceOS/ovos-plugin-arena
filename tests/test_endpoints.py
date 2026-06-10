"""
HTTP endpoint tests for the arena core router (arena.router).

Uses FastAPI's TestClient — no live server required.
The arena.db module is pointed at a temporary SQLite file for each test.
"""

import sys
import uuid
from pathlib import Path

import pytest

# Ensure backend/app is importable
BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture()
def client(tmp_db):
    """
    Provide a FastAPI TestClient wired to the arena router, using the
    temporary database created by the tmp_db fixture.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.arena.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/arena")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Plugin listing
# ---------------------------------------------------------------------------


def test_list_plugins_empty(client):
    resp = client.get("/api/v1/arena/plugins")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_plugin_not_found(client):
    resp = client.get(f"/api/v1/arena/plugins/{uuid.uuid4()}")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Eval runs
# ---------------------------------------------------------------------------


def _register_plugin(client, name="test-plugin", family="tts"):
    from app.arena import db as arena_db
    from app.arena.models import Plugin, PluginFamily

    p = Plugin(plugin_name=name, display_name=name, family=PluginFamily(family))
    arena_db.upsert_plugin(p)
    return p


def test_create_run(client, tmp_db):
    p = _register_plugin(client)
    resp = client.post(
        "/api/v1/arena/runs",
        json={"plugin_id": str(p.id), "family": "tts", "lang": "en-us"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "pending"
    assert data["plugin_id"] == str(p.id)


def test_create_run_missing_plugin(client):
    resp = client.post(
        "/api/v1/arena/runs",
        json={"plugin_id": str(uuid.uuid4()), "family": "tts"},
    )
    assert resp.status_code == 404


def test_get_run_not_found(client):
    resp = client.get(f"/api/v1/arena/runs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_list_runs_empty(client):
    resp = client.get("/api/v1/arena/runs")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Matchup + voting flow
# ---------------------------------------------------------------------------


def _seed_full_matchup(client, tmp_db):
    """
    Register two plugins, create runs + samples + matchup via the DB directly,
    and return (matchup_id, plugin_a_id, plugin_b_id).
    """
    from app.arena import db as arena_db
    from app.arena.models import EvalRun, Matchup, Plugin, PluginFamily, Sample

    p_a = Plugin(plugin_name="ep-a", display_name="A", family=PluginFamily.TTS)
    p_b = Plugin(plugin_name="ep-b", display_name="B", family=PluginFamily.TTS)
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


def test_get_next_matchup(client, tmp_db):
    m, _, _ = _seed_full_matchup(client, tmp_db)
    resp = client.get("/api/v1/arena/matchups/next/tts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    # Plugin IDs must NOT be exposed in public response
    assert "plugin_a_id" not in data
    assert "plugin_b_id" not in data


def test_get_next_matchup_none(client):
    resp = client.get("/api/v1/arena/matchups/next/tts")
    assert resp.status_code == 404


def test_submit_vote_happy_path(client, tmp_db):
    m, p_a, p_b = _seed_full_matchup(client, tmp_db)
    resp = client.post(
        f"/api/v1/arena/matchups/{m.id}/vote",
        json={"outcome": "candidate_a"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["outcome"] == "candidate_a"


def test_submit_vote_updates_leaderboard(client, tmp_db):
    m, p_a, p_b = _seed_full_matchup(client, tmp_db)
    client.post(
        f"/api/v1/arena/matchups/{m.id}/vote",
        json={"outcome": "candidate_a"},
    )
    resp = client.get("/api/v1/arena/leaderboard/tts")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    elos = {e["plugin_name"]: e["elo"] for e in entries}
    # Winner should have higher ELO
    assert elos["ep-a"] > elos["ep-b"]


def test_submit_vote_on_voted_matchup_returns_409(client, tmp_db):
    m, _, _ = _seed_full_matchup(client, tmp_db)
    client.post(f"/api/v1/arena/matchups/{m.id}/vote", json={"outcome": "candidate_a"})
    resp = client.post(f"/api/v1/arena/matchups/{m.id}/vote", json={"outcome": "candidate_b"})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def test_leaderboard_empty(client):
    resp = client.get("/api/v1/arena/leaderboard/tts")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_leaderboard_shows_registered_plugins(client, tmp_db):
    _register_plugin(client, "plug1")
    _register_plugin(client, "plug2")
    from app.arena import db as arena_db
    from app.arena.models import PluginFamily

    # initialise ELO entries
    for p in arena_db.list_plugins(family=PluginFamily.TTS):
        arena_db.update_elo_stats(p.id, 1200.0, won=False, tied=True)

    resp = client.get("/api/v1/arena/leaderboard/tts")
    assert resp.json()["total"] == 2


# ---------------------------------------------------------------------------
# Recompute endpoint
# ---------------------------------------------------------------------------


def test_recompute_endpoint(client):
    resp = client.post("/api/v1/arena/ranking/recompute")
    assert resp.status_code == 200
    assert resp.json()["status"] == "recompute scheduled"
