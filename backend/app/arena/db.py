"""
SQLite persistence layer for the OVOS Plugin Arena.

Uses the stdlib sqlite3 module directly — no ORM, no external services.
The schema mirrors the Pydantic models in arena.models and is created on
first connect via CREATE TABLE IF NOT EXISTS.

All writes are serialised through a threading.Lock so the module is safe
to use from multiple FastAPI worker threads sharing one SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from app.arena.models import (
    EvalRun,
    EvalStatus,
    IngestedPrediction,
    LeaderboardEntry,
    Matchup,
    Plugin,
    PluginFamily,
    PredictionSource,
    RatingSnapshot,
    Sample,
    Vote,
    VoteOutcome,
    VoteSource,
)

_lock = threading.Lock()

# Default path; caller overrides via init_db(path=...)
_DB_PATH: Path = Path("arena.sqlite3")


def set_db_path(path: Path) -> None:
    global _DB_PATH
    _DB_PATH = path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    with _lock:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plugins (
    id          TEXT PRIMARY KEY,
    plugin_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    family      TEXT NOT NULL,
    lang        TEXT,
    author      TEXT,
    description TEXT,
    homepage_url TEXT,
    config      TEXT NOT NULL DEFAULT '{}',
    config_hash TEXT NOT NULL DEFAULT '',
    discovered_at TEXT NOT NULL,
    extra       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              TEXT PRIMARY KEY,
    plugin_id       TEXT NOT NULL REFERENCES plugins(id),
    family          TEXT NOT NULL,
    lang            TEXT NOT NULL DEFAULT 'en-us',
    status          TEXT NOT NULL DEFAULT 'pending',
    started_at      TEXT,
    finished_at     TEXT,
    failure_reason  TEXT,
    metrics         TEXT NOT NULL DEFAULT '{}',
    meta            TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS samples (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES eval_runs(id),
    plugin_id   TEXT NOT NULL REFERENCES plugins(id),
    family      TEXT NOT NULL,
    input_ref   TEXT NOT NULL,
    output_ref  TEXT,
    metrics     TEXT NOT NULL DEFAULT '{}',
    produced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matchups (
    id          TEXT PRIMARY KEY,
    family      TEXT NOT NULL,
    input_ref   TEXT NOT NULL,
    sample_a_id TEXT NOT NULL,
    sample_b_id TEXT NOT NULL,
    plugin_a_id TEXT NOT NULL REFERENCES plugins(id),
    plugin_b_id TEXT NOT NULL REFERENCES plugins(id),
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS votes (
    id           TEXT PRIMARY KEY,
    matchup_id   TEXT NOT NULL REFERENCES matchups(id),
    outcome      TEXT NOT NULL,
    voter_id     TEXT,
    voter_source TEXT NOT NULL DEFAULT 'human',
    automated    INTEGER NOT NULL DEFAULT 0,
    note         TEXT,
    cast_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_sources (
    id          TEXT PRIMARY KEY,
    hf_dataset  TEXT NOT NULL,
    revision    TEXT NOT NULL DEFAULT 'main',
    modality    TEXT NOT NULL,
    lang        TEXT NOT NULL,
    ingested_at TEXT,
    row_count   INTEGER NOT NULL DEFAULT 0,
    meta        TEXT NOT NULL DEFAULT '{}',
    UNIQUE(hf_dataset, revision)
);

CREATE TABLE IF NOT EXISTS ingested_predictions (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES prediction_sources(id),
    sample_id       TEXT NOT NULL,
    plugin_id       TEXT NOT NULL,
    plugin_version  TEXT NOT NULL,
    prediction      TEXT NOT NULL,
    reference       TEXT,
    wer             REAL,
    metrics         TEXT NOT NULL DEFAULT '{}',
    hf_row_ref      TEXT NOT NULL DEFAULT '',
    ingested_at     TEXT NOT NULL,
    UNIQUE(source_id, sample_id, plugin_version)
);

CREATE TABLE IF NOT EXISTS rating_snapshots (
    id          TEXT PRIMARY KEY,
    vote_id     TEXT NOT NULL REFERENCES votes(id),
    plugin_id   TEXT NOT NULL REFERENCES plugins(id),
    elo_before  REAL NOT NULL,
    elo_after   REAL NOT NULL,
    delta       REAL NOT NULL,
    snapshot_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elo_current (
    plugin_id   TEXT PRIMARY KEY REFERENCES plugins(id),
    elo         REAL NOT NULL DEFAULT 1200.0,
    battles     INTEGER NOT NULL DEFAULT 0,
    wins        INTEGER NOT NULL DEFAULT 0,
    losses      INTEGER NOT NULL DEFAULT 0,
    ties        INTEGER NOT NULL DEFAULT 0
);
"""


def init_db(path: Optional[Path] = None) -> None:
    """Create tables if they don't exist. Call once at startup."""
    if path is not None:
        set_db_path(path)
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Plugin CRUD
# ---------------------------------------------------------------------------


def _plugin_from_row(row: sqlite3.Row) -> Plugin:
    return Plugin(
        id=uuid.UUID(row["id"]),
        plugin_name=row["plugin_name"],
        display_name=row["display_name"],
        family=PluginFamily(row["family"]),
        lang=row["lang"],
        author=row["author"],
        description=row["description"],
        homepage_url=row["homepage_url"],
        config=json.loads(row["config"]),
        config_hash=row["config_hash"],
        discovered_at=datetime.fromisoformat(row["discovered_at"]),
        extra=json.loads(row["extra"]),
    )


def upsert_plugin(plugin: Plugin) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO plugins
                (id, plugin_name, display_name, family, lang, author, description,
                 homepage_url, config, config_hash, discovered_at, extra)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(plugin_name) DO UPDATE SET
                display_name=excluded.display_name,
                lang=excluded.lang,
                author=excluded.author,
                description=excluded.description,
                homepage_url=excluded.homepage_url,
                config=excluded.config,
                config_hash=excluded.config_hash,
                extra=excluded.extra
            """,
            (
                str(plugin.id),
                plugin.plugin_name,
                plugin.display_name,
                plugin.family.value,
                plugin.lang,
                plugin.author,
                plugin.description,
                plugin.homepage_url,
                json.dumps(plugin.config),
                plugin.config_hash,
                plugin.discovered_at.isoformat(),
                json.dumps(plugin.extra),
            ),
        )
        # ensure elo_current row exists
        conn.execute(
            "INSERT OR IGNORE INTO elo_current (plugin_id) VALUES (?)",
            (str(plugin.id),),
        )


def get_plugin_by_name(name: str) -> Optional[Plugin]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM plugins WHERE plugin_name=?", (name,)
        ).fetchone()
    return _plugin_from_row(row) if row else None


def get_plugin_by_id(plugin_id: uuid.UUID) -> Optional[Plugin]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM plugins WHERE id=?", (str(plugin_id),)
        ).fetchone()
    return _plugin_from_row(row) if row else None


def list_plugins(family: Optional[PluginFamily] = None, lang: Optional[str] = None) -> List[Plugin]:
    q = "SELECT * FROM plugins WHERE 1=1"
    params: list = []
    if family:
        q += " AND family=?"
        params.append(family.value)
    if lang:
        q += " AND (lang=? OR lang IS NULL)"
        params.append(lang)
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [_plugin_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# EvalRun CRUD
# ---------------------------------------------------------------------------


def _run_from_row(row: sqlite3.Row) -> EvalRun:
    return EvalRun(
        id=uuid.UUID(row["id"]),
        plugin_id=uuid.UUID(row["plugin_id"]),
        family=PluginFamily(row["family"]),
        lang=row["lang"],
        status=EvalStatus(row["status"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        failure_reason=row["failure_reason"],
        metrics=json.loads(row["metrics"]),
        meta=json.loads(row["meta"]),
    )


def create_eval_run(run: EvalRun) -> EvalRun:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO eval_runs
               (id, plugin_id, family, lang, status, started_at, finished_at,
                failure_reason, metrics, meta)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                str(run.id),
                str(run.plugin_id),
                run.family.value,
                run.lang,
                run.status.value,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.failure_reason,
                json.dumps(run.metrics),
                json.dumps(run.meta),
            ),
        )
    return run


def update_eval_run(run: EvalRun) -> EvalRun:
    with get_conn() as conn:
        conn.execute(
            """UPDATE eval_runs SET status=?, started_at=?, finished_at=?,
               failure_reason=?, metrics=?, meta=? WHERE id=?""",
            (
                run.status.value,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.failure_reason,
                json.dumps(run.metrics),
                json.dumps(run.meta),
                str(run.id),
            ),
        )
    return run


def get_eval_run(run_id: uuid.UUID) -> Optional[EvalRun]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM eval_runs WHERE id=?", (str(run_id),)).fetchone()
    return _run_from_row(row) if row else None


def list_eval_runs(plugin_id: Optional[uuid.UUID] = None) -> List[EvalRun]:
    q = "SELECT * FROM eval_runs WHERE 1=1"
    params: list = []
    if plugin_id:
        q += " AND plugin_id=?"
        params.append(str(plugin_id))
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [_run_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Sample CRUD
# ---------------------------------------------------------------------------


def _sample_from_row(row: sqlite3.Row) -> Sample:
    return Sample(
        id=uuid.UUID(row["id"]),
        run_id=uuid.UUID(row["run_id"]),
        plugin_id=uuid.UUID(row["plugin_id"]),
        family=PluginFamily(row["family"]),
        input_ref=row["input_ref"],
        output_ref=row["output_ref"],
        metrics=json.loads(row["metrics"]),
        produced_at=datetime.fromisoformat(row["produced_at"]),
    )


def create_sample(sample: Sample) -> Sample:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO samples
               (id, run_id, plugin_id, family, input_ref, output_ref, metrics, produced_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                str(sample.id),
                str(sample.run_id),
                str(sample.plugin_id),
                sample.family.value,
                sample.input_ref,
                sample.output_ref,
                json.dumps(sample.metrics),
                sample.produced_at.isoformat(),
            ),
        )
    return sample


def get_sample(sample_id: uuid.UUID) -> Optional[Sample]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM samples WHERE id=?", (str(sample_id),)).fetchone()
    return _sample_from_row(row) if row else None


def list_samples_for_run(run_id: uuid.UUID) -> List[Sample]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM samples WHERE run_id=?", (str(run_id),)
        ).fetchall()
    return [_sample_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Matchup CRUD
# ---------------------------------------------------------------------------


def _matchup_from_row(row: sqlite3.Row) -> Matchup:
    return Matchup(
        id=uuid.UUID(row["id"]),
        family=PluginFamily(row["family"]),
        input_ref=row["input_ref"],
        sample_a_id=uuid.UUID(row["sample_a_id"]),
        sample_b_id=uuid.UUID(row["sample_b_id"]),
        plugin_a_id=uuid.UUID(row["plugin_a_id"]),
        plugin_b_id=uuid.UUID(row["plugin_b_id"]),
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def create_matchup(matchup: Matchup) -> Matchup:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO matchups
               (id, family, input_ref, sample_a_id, sample_b_id,
                plugin_a_id, plugin_b_id, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                str(matchup.id),
                matchup.family.value,
                matchup.input_ref,
                str(matchup.sample_a_id),
                str(matchup.sample_b_id),
                str(matchup.plugin_a_id),
                str(matchup.plugin_b_id),
                matchup.status,
                matchup.created_at.isoformat(),
            ),
        )
    return matchup


def get_matchup(matchup_id: uuid.UUID) -> Optional[Matchup]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM matchups WHERE id=?", (str(matchup_id),)
        ).fetchone()
    return _matchup_from_row(row) if row else None


def get_pending_matchup(family: PluginFamily) -> Optional[Matchup]:
    """Return one un-voted matchup for *family*, oldest first."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM matchups WHERE family=? AND status='pending'
               ORDER BY created_at LIMIT 1""",
            (family.value,),
        ).fetchone()
    return _matchup_from_row(row) if row else None


def mark_matchup_voted(matchup_id: uuid.UUID) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE matchups SET status='voted' WHERE id=?", (str(matchup_id),)
        )


# ---------------------------------------------------------------------------
# Vote CRUD
# ---------------------------------------------------------------------------


def _vote_from_row(row: sqlite3.Row) -> Vote:
    keys = row.keys()
    return Vote(
        id=uuid.UUID(row["id"]),
        matchup_id=uuid.UUID(row["matchup_id"]),
        outcome=VoteOutcome(row["outcome"]),
        voter_id=row["voter_id"],
        voter_source=VoteSource(row["voter_source"]) if "voter_source" in keys and row["voter_source"] else VoteSource.HUMAN,
        automated=bool(row["automated"]),
        note=row["note"],
        cast_at=datetime.fromisoformat(row["cast_at"]),
    )


def create_vote(vote: Vote) -> Vote:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO votes
               (id, matchup_id, outcome, voter_id, voter_source, automated, note, cast_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                str(vote.id),
                str(vote.matchup_id),
                vote.outcome.value,
                vote.voter_id,
                vote.voter_source.value,
                1 if vote.automated else 0,
                vote.note,
                vote.cast_at.isoformat(),
            ),
        )
    return vote


def list_votes(matchup_id: Optional[uuid.UUID] = None) -> List[Vote]:
    q = "SELECT * FROM votes WHERE 1=1"
    params: list = []
    if matchup_id:
        q += " AND matchup_id=?"
        params.append(str(matchup_id))
    q += " ORDER BY cast_at"
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [_vote_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Rating snapshots CRUD
# ---------------------------------------------------------------------------


def create_rating_snapshot(snap: RatingSnapshot) -> RatingSnapshot:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO rating_snapshots
               (id, vote_id, plugin_id, elo_before, elo_after, delta, snapshot_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                str(snap.id),
                str(snap.vote_id),
                str(snap.plugin_id),
                snap.elo_before,
                snap.elo_after,
                snap.delta,
                snap.snapshot_at.isoformat(),
            ),
        )
    return snap


def list_rating_snapshots(plugin_id: Optional[uuid.UUID] = None) -> List[RatingSnapshot]:
    q = "SELECT * FROM rating_snapshots WHERE 1=1"
    params: list = []
    if plugin_id:
        q += " AND plugin_id=?"
        params.append(str(plugin_id))
    q += " ORDER BY snapshot_at"
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [
        RatingSnapshot(
            id=uuid.UUID(r["id"]),
            vote_id=uuid.UUID(r["vote_id"]),
            plugin_id=uuid.UUID(r["plugin_id"]),
            elo_before=r["elo_before"],
            elo_after=r["elo_after"],
            delta=r["delta"],
            snapshot_at=datetime.fromisoformat(r["snapshot_at"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# ELO current table helpers
# ---------------------------------------------------------------------------


def get_elo(plugin_id: uuid.UUID) -> float:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT elo FROM elo_current WHERE plugin_id=?", (str(plugin_id),)
        ).fetchone()
    return row["elo"] if row else 1200.0


def set_elo(plugin_id: uuid.UUID, elo: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO elo_current (plugin_id, elo) VALUES (?,?)",
            (str(plugin_id), elo),
        )


def get_elo_stats(plugin_id: uuid.UUID) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM elo_current WHERE plugin_id=?", (str(plugin_id),)
        ).fetchone()
    if not row:
        return {"elo": 1200.0, "battles": 0, "wins": 0, "losses": 0, "ties": 0}
    return dict(row)


def update_elo_stats(
    plugin_id: uuid.UUID,
    new_elo: float,
    won: bool,
    tied: bool,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO elo_current (plugin_id, elo, battles, wins, losses, ties)
               VALUES (?, ?, 1, ?, ?, ?)
               ON CONFLICT(plugin_id) DO UPDATE SET
                   elo=excluded.elo,
                   battles=battles+1,
                   wins=wins + CASE WHEN ? THEN 1 ELSE 0 END,
                   losses=losses + CASE WHEN (NOT ? AND NOT ?) THEN 1 ELSE 0 END,
                   ties=ties + CASE WHEN ? THEN 1 ELSE 0 END""",
            (
                str(plugin_id),
                new_elo,
                1 if won else 0,
                0 if (won or tied) else 1,
                1 if tied else 0,
                won,
                won,
                tied,
                tied,
            ),
        )


# ---------------------------------------------------------------------------
# Leaderboard query
# ---------------------------------------------------------------------------


def get_leaderboard(
    family: PluginFamily,
    lang: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[LeaderboardEntry]:
    params: list = [family.value]
    lang_clause = ""
    if lang:
        lang_clause = " AND (p.lang=? OR p.lang IS NULL)"
        params.append(lang)

    q = f"""
        SELECT
            p.id, p.plugin_name, p.display_name, p.family, p.lang,
            e.elo, e.battles, e.wins, e.losses, e.ties
        FROM elo_current e
        JOIN plugins p ON p.id = e.plugin_id
        WHERE p.family=?{lang_clause}
        ORDER BY e.elo DESC
        LIMIT ? OFFSET ?
    """
    params += [limit, offset]

    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()

    entries = []
    for rank, row in enumerate(rows, start=offset + 1):
        battles = row["battles"] or 0
        wins = row["wins"] or 0
        entries.append(
            LeaderboardEntry(
                rank=rank,
                plugin_id=uuid.UUID(row["id"]),
                plugin_name=row["plugin_name"],
                display_name=row["display_name"],
                family=PluginFamily(row["family"]),
                lang=row["lang"],
                elo=round(row["elo"], 2),
                battles=battles,
                wins=wins,
                losses=row["losses"] or 0,
                ties=row["ties"] or 0,
                win_rate=round(wins / battles * 100, 2) if battles else 0.0,
            )
        )
    return entries


def count_plugins(family: PluginFamily, lang: Optional[str] = None) -> int:
    params: list = [family.value]
    lang_clause = ""
    if lang:
        lang_clause = " AND (lang=? OR lang IS NULL)"
        params.append(lang)
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM plugins WHERE family=?{lang_clause}", params
        ).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# PredictionSource CRUD (§5)
# ---------------------------------------------------------------------------


def _source_from_row(row: sqlite3.Row) -> PredictionSource:
    return PredictionSource(
        id=uuid.UUID(row["id"]),
        hf_dataset=row["hf_dataset"],
        revision=row["revision"],
        modality=PluginFamily(row["modality"]),
        lang=row["lang"],
        ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
        row_count=row["row_count"],
        meta=json.loads(row["meta"]),
    )


def upsert_prediction_source(source: PredictionSource) -> PredictionSource:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO prediction_sources
               (id, hf_dataset, revision, modality, lang, ingested_at, row_count, meta)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(hf_dataset, revision) DO UPDATE SET
                   modality=excluded.modality,
                   lang=excluded.lang,
                   ingested_at=excluded.ingested_at,
                   row_count=excluded.row_count,
                   meta=excluded.meta""",
            (
                str(source.id),
                source.hf_dataset,
                source.revision,
                source.modality.value,
                source.lang,
                source.ingested_at.isoformat() if source.ingested_at else None,
                source.row_count,
                json.dumps(source.meta),
            ),
        )
    return source


def get_prediction_source(source_id: uuid.UUID) -> Optional[PredictionSource]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM prediction_sources WHERE id=?", (str(source_id),)
        ).fetchone()
    return _source_from_row(row) if row else None


def get_prediction_source_by_dataset(hf_dataset: str, revision: str = "main") -> Optional[PredictionSource]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM prediction_sources WHERE hf_dataset=? AND revision=?",
            (hf_dataset, revision),
        ).fetchone()
    return _source_from_row(row) if row else None


def list_prediction_sources(modality: Optional[PluginFamily] = None) -> List[PredictionSource]:
    q = "SELECT * FROM prediction_sources WHERE 1=1"
    params: list = []
    if modality:
        q += " AND modality=?"
        params.append(modality.value)
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [_source_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# IngestedPrediction CRUD (§P3)
# ---------------------------------------------------------------------------


def _pred_from_row(row: sqlite3.Row) -> IngestedPrediction:
    return IngestedPrediction(
        id=uuid.UUID(row["id"]),
        source_id=uuid.UUID(row["source_id"]),
        sample_id=row["sample_id"],
        plugin_id=row["plugin_id"],
        plugin_version=row["plugin_version"],
        prediction=row["prediction"],
        reference=row["reference"],
        wer=row["wer"],
        metrics=json.loads(row["metrics"]),
        hf_row_ref=row["hf_row_ref"],
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
    )


def upsert_ingested_prediction(pred: IngestedPrediction) -> IngestedPrediction:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ingested_predictions
               (id, source_id, sample_id, plugin_id, plugin_version,
                prediction, reference, wer, metrics, hf_row_ref, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_id, sample_id, plugin_version) DO UPDATE SET
                   prediction=excluded.prediction,
                   reference=excluded.reference,
                   wer=excluded.wer,
                   metrics=excluded.metrics,
                   hf_row_ref=excluded.hf_row_ref,
                   ingested_at=excluded.ingested_at""",
            (
                str(pred.id),
                str(pred.source_id),
                pred.sample_id,
                pred.plugin_id,
                pred.plugin_version,
                pred.prediction,
                pred.reference,
                pred.wer,
                json.dumps(pred.metrics),
                pred.hf_row_ref,
                pred.ingested_at.isoformat(),
            ),
        )
    return pred


def list_predictions_for_source(
    source_id: uuid.UUID,
    plugin_id: Optional[str] = None,
) -> List[IngestedPrediction]:
    q = "SELECT * FROM ingested_predictions WHERE source_id=?"
    params: list = [str(source_id)]
    if plugin_id:
        q += " AND plugin_id=?"
        params.append(plugin_id)
    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
    return [_pred_from_row(r) for r in rows]


def get_predictions_by_sample(
    source_id: uuid.UUID,
    sample_id: str,
) -> List[IngestedPrediction]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ingested_predictions WHERE source_id=? AND sample_id=?",
            (str(source_id), sample_id),
        ).fetchall()
    return [_pred_from_row(r) for r in rows]


def list_sample_ids_with_multiple_plugins(source_id: uuid.UUID) -> List[str]:
    """Return sample_ids that have predictions from at least 2 distinct plugin_versions."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT sample_id FROM ingested_predictions
               WHERE source_id=?
               GROUP BY sample_id
               HAVING COUNT(DISTINCT plugin_version) >= 2""",
            (str(source_id),),
        ).fetchall()
    return [r["sample_id"] for r in rows]
