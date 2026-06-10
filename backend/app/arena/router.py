"""
FastAPI router for the OVOS Plugin Arena core (P1 + P2).

All routes are self-contained: they use the SQLite arena.db layer directly
and do NOT require PostgreSQL, Kafka, or authentication (auth is optional
and can be layered on later).

Mount this router in app/main.py::

    from app.arena.router import router as arena_router
    app.include_router(arena_router, prefix="/api/v1/arena")
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.arena import db, ranking
from app.arena.discovery import discover_plugins
from app.arena.models import (
    EvalRun,
    EvalStatus,
    LeaderboardResponse,
    Matchup,
    MatchupPublic,
    Plugin,
    PluginFamily,
    Sample,
    Vote,
    VoteOutcome,
)
from pydantic import BaseModel

router = APIRouter(tags=["arena-core"])


# ---------------------------------------------------------------------------
# Request/response helpers
# ---------------------------------------------------------------------------


class DiscoverRequest(BaseModel):
    families: Optional[List[PluginFamily]] = None


class EvalRunCreate(BaseModel):
    plugin_id: uuid.UUID
    family: PluginFamily
    lang: str = "en-us"
    meta: dict = {}


class VoteRequest(BaseModel):
    outcome: VoteOutcome
    voter_id: Optional[str] = None
    note: Optional[str] = None


class MetricVoteRequest(BaseModel):
    matchup_id: uuid.UUID
    winner: str  # "a" | "b" | "tie" | "both_wrong"
    source: str = "auto"


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


@router.post("/plugins/discover", response_model=List[Plugin], summary="Discover & register plugins via OPM")
def discover_and_register(body: DiscoverRequest) -> List[Plugin]:
    """
    Scan installed OVOS entry points and upsert them into the arena registry.

    Returns the list of discovered Plugin records (including existing ones).
    """
    plugins = discover_plugins(families=body.families)
    for p in plugins:
        db.upsert_plugin(p)
    return plugins


@router.get("/plugins", response_model=List[Plugin], summary="List registered plugins")
def list_plugins(
    family: Optional[PluginFamily] = Query(None),
    lang: Optional[str] = Query(None),
) -> List[Plugin]:
    return db.list_plugins(family=family, lang=lang)


@router.get("/plugins/{plugin_id}", response_model=Plugin, summary="Get plugin by ID")
def get_plugin(plugin_id: uuid.UUID) -> Plugin:
    p = db.get_plugin_by_id(plugin_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    return p


# ---------------------------------------------------------------------------
# Eval runs
# ---------------------------------------------------------------------------


@router.post("/runs", response_model=EvalRun, status_code=201, summary="Create an eval run")
def create_run(body: EvalRunCreate) -> EvalRun:
    """
    Create an EvalRun for a plugin.  The run starts in PENDING state.
    Use the background worker (or the /runs/{id}/execute endpoint) to
    actually run the plugin.
    """
    plugin = db.get_plugin_by_id(body.plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")

    run = EvalRun(
        plugin_id=body.plugin_id,
        family=body.family,
        lang=body.lang,
        meta=body.meta,
    )
    return db.create_eval_run(run)


@router.get("/runs/{run_id}", response_model=EvalRun, summary="Get eval run status")
def get_run(run_id: uuid.UUID) -> EvalRun:
    run = db.get_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs", response_model=List[EvalRun], summary="List eval runs")
def list_runs(plugin_id: Optional[uuid.UUID] = Query(None)) -> List[EvalRun]:
    return db.list_eval_runs(plugin_id=plugin_id)


# ---------------------------------------------------------------------------
# Matchups (blind pairs)
# ---------------------------------------------------------------------------


@router.post("/matchups", response_model=Matchup, status_code=201, summary="Create a blind matchup")
def create_matchup(
    family: PluginFamily,
    input_ref: str,
    sample_a_id: uuid.UUID,
    sample_b_id: uuid.UUID,
) -> Matchup:
    """
    Create a blind A/B matchup from two Sample IDs.

    The voter sees only output_a_ref and output_b_ref — plugin identities
    are hidden until a vote is cast.
    """
    sample_a = db.get_sample(sample_a_id)
    sample_b = db.get_sample(sample_b_id)
    if sample_a is None or sample_b is None:
        raise HTTPException(status_code=404, detail="Sample(s) not found")

    matchup = Matchup(
        family=family,
        input_ref=input_ref,
        sample_a_id=sample_a_id,
        sample_b_id=sample_b_id,
        plugin_a_id=sample_a.plugin_id,
        plugin_b_id=sample_b.plugin_id,
    )
    return db.create_matchup(matchup)


@router.get("/matchups/next/{family}", response_model=MatchupPublic, summary="Fetch next matchup for voting")
def get_next_matchup(family: PluginFamily) -> MatchupPublic:
    """
    Return the oldest pending matchup for *family*.

    Plugin identities are masked — only output refs and metrics are exposed
    so the voter cannot identify which plugin produced which output.
    """
    matchup = db.get_pending_matchup(family)
    if matchup is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pending matchups for family={family.value}",
        )

    sample_a = db.get_sample(matchup.sample_a_id)
    sample_b = db.get_sample(matchup.sample_b_id)

    return MatchupPublic(
        id=matchup.id,
        family=matchup.family,
        input_ref=matchup.input_ref,
        output_a_ref=sample_a.output_ref if sample_a else None,
        output_b_ref=sample_b.output_ref if sample_b else None,
        metrics_a=sample_a.metrics if sample_a else {},
        metrics_b=sample_b.metrics if sample_b else {},
        status=matchup.status,
        created_at=matchup.created_at,
    )


@router.get("/matchups/{matchup_id}", response_model=Matchup, summary="Get matchup details (admin)")
def get_matchup_details(matchup_id: uuid.UUID) -> Matchup:
    m = db.get_matchup(matchup_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Matchup not found")
    return m


# ---------------------------------------------------------------------------
# Votes
# ---------------------------------------------------------------------------


@router.post("/matchups/{matchup_id}/vote", response_model=Vote, status_code=201, summary="Submit a vote")
def submit_vote(matchup_id: uuid.UUID, body: VoteRequest) -> Vote:
    """
    Submit a human vote for a matchup.

    Immediately updates the live ELO table via ``ranking.process_vote_and_update``.
    """
    matchup = db.get_matchup(matchup_id)
    if matchup is None:
        raise HTTPException(status_code=404, detail="Matchup not found")
    if matchup.status == "voted":
        raise HTTPException(status_code=409, detail="Matchup already voted")

    vote = Vote(
        matchup_id=matchup_id,
        outcome=body.outcome,
        voter_id=body.voter_id,
        automated=False,
        note=body.note,
    )
    db.create_vote(vote)
    ranking.process_vote_and_update(vote)
    return vote


@router.post("/votes/metric", response_model=Vote, status_code=201, summary="Ingest automated metric vote")
def ingest_metric_vote(body: MetricVoteRequest) -> Vote:
    """
    Ingest a vote derived from automated benchmark metrics (WER, RTF, F1).

    The vote is marked ``automated=True`` and processed through the same ELO
    pipeline as human votes.
    """
    vote = ranking.ingest_metric_vote(
        matchup_id=body.matchup_id,
        winner=body.winner,
        source=body.source,
    )
    if vote is None:
        raise HTTPException(status_code=400, detail=f"Invalid winner value: {body.winner!r}")
    return vote


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------


@router.get("/leaderboard/{family}", response_model=LeaderboardResponse, summary="Leaderboard for a family")
def leaderboard(
    family: PluginFamily,
    lang: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
) -> LeaderboardResponse:
    """
    Return the current ELO leaderboard for *family*, optionally filtered by *lang*.

    Rankings are computed live from the elo_current table (updated after
    every vote).  For a fully deterministic snapshot call ``/ranking/recompute``
    first.
    """
    entries = db.get_leaderboard(family=family, lang=lang, limit=limit, offset=offset)
    total = db.count_plugins(family=family, lang=lang)
    return LeaderboardResponse(family=family, lang=lang, entries=entries, total=total)


# ---------------------------------------------------------------------------
# Ranking maintenance
# ---------------------------------------------------------------------------


@router.post("/ranking/recompute", summary="Replay vote log and rebuild ELO table")
def recompute_ratings(
    family: Optional[PluginFamily] = Query(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict:
    """
    Replay the full ordered vote log to rebuild ``elo_current``.

    This is deterministic: calling it twice with the same vote log produces
    identical results.  Use it to verify correctness or recover from
    inconsistency.

    Runs in a background task for large logs; returns immediately with a
    confirmation message.
    """
    background_tasks.add_task(ranking.recompute_all_ratings, family)
    return {"status": "recompute scheduled", "family": family}
