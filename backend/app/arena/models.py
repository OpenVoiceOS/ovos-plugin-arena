"""
Pydantic data models for the OVOS Plugin Arena core.

These are pure data models (no SQLModel table=True) used by the arena engine.
Persistence is handled by arena.db using SQLite via sqlite3 directly so the
arena engine can run without any external services.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PluginFamily(str, enum.Enum):
    """Supported OVOS plugin families."""

    TTS = "tts"
    STT = "stt"
    WAKE_WORD = "wake_word"
    INTENT = "intent"


class EvalStatus(str, enum.Enum):
    """Lifecycle state of an EvalRun."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class VoteOutcome(str, enum.Enum):
    """Outcome of a single blind matchup vote."""

    CANDIDATE_A = "candidate_a"
    CANDIDATE_B = "candidate_b"
    TIE = "tie"
    BOTH_WRONG = "both_wrong"


class VoteSource(str, enum.Enum):
    """Whether a vote was cast by a human or an auto-battle system."""

    HUMAN = "human"
    AUTO_WER = "system:wer"  # §4 R5 — auto-battle seeding via WER metric


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Plugin(BaseModel):
    """A registered OVOS plugin entry point."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    plugin_name: str  # e.g. "ovos-tts-plugin-phoonnx"
    display_name: str
    family: PluginFamily
    lang: Optional[str] = None  # ISO-639 language tag, None = multi-language
    author: Optional[str] = None
    description: Optional[str] = None
    homepage_url: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""  # SHA-256 of sorted config items
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = Field(default_factory=dict)  # OPM metadata


class EvalRun(BaseModel):
    """A batch evaluation run for one plugin over a fixed prompt set."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    plugin_id: uuid.UUID
    family: PluginFamily
    lang: str = "en-us"
    status: EvalStatus = EvalStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    """A single output produced during an EvalRun."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    run_id: uuid.UUID
    plugin_id: uuid.UUID
    family: PluginFamily
    input_ref: str  # prompt text or audio path
    output_ref: Optional[str] = None  # path to audio/text artifact
    metrics: Dict[str, float] = Field(default_factory=dict)  # WER, RTF, …
    produced_at: datetime = Field(default_factory=datetime.utcnow)


class Matchup(BaseModel):
    """A blind pairwise comparison between two Samples (A vs B)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    family: PluginFamily
    input_ref: str  # shared input for both candidates
    sample_a_id: uuid.UUID
    sample_b_id: uuid.UUID
    plugin_a_id: uuid.UUID
    plugin_b_id: uuid.UUID
    # hidden from voter until after vote
    status: str = "pending"  # pending | voted
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Vote(BaseModel):
    """A human (or automated) vote on a Matchup."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    matchup_id: uuid.UUID
    outcome: VoteOutcome
    voter_id: Optional[str] = None  # anonymous if None; "system:wer" for auto
    voter_source: VoteSource = VoteSource.HUMAN  # §4 R5 — human|auto
    automated: bool = False  # True for metric-derived votes (legacy compat)
    note: Optional[str] = None
    cast_at: datetime = Field(default_factory=datetime.utcnow)


class RatingSnapshot(BaseModel):
    """Immutable ELO snapshot after processing a Vote."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    vote_id: uuid.UUID
    plugin_id: uuid.UUID
    elo_before: float
    elo_after: float
    delta: float
    snapshot_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# M2 — Prediction sources (§5)
# ---------------------------------------------------------------------------


class PredictionSource(BaseModel):
    """A registered HuggingFace prediction dataset.

    Registered by admin before (or during) ingestion.  Pins the exact
    revision so every battle assembled from this source is reproducible.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    hf_dataset: str  # e.g. "OpenVoiceOS/ovos-stt-bench-pt-PT"
    revision: str = "main"  # git ref or commit SHA
    modality: PluginFamily
    lang: str  # BCP-47
    ingested_at: Optional[datetime] = None
    row_count: int = 0  # updated after each ingest
    meta: Dict[str, Any] = Field(default_factory=dict)


class IngestedPrediction(BaseModel):
    """One cached prediction row pulled from an HF dataset.

    Only battle-relevant fields are stored (§P3 — no blobs).
    The ``hf_row_ref`` identifies the exact row in the source dataset for
    full reproducibility without storing audio/large artifacts.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID  # → PredictionSource
    sample_id: str  # dataset_entry_id in the real schema
    plugin_id: str  # plugin_name (OPM entry-point name)
    plugin_version: str  # model_id in the real schema (composite key)
    prediction: str  # prediction_transcript for STT
    reference: Optional[str] = None  # transcript / reference_text
    wer: Optional[float] = None  # computed on ingest if reference present
    metrics: Dict[str, float] = Field(default_factory=dict)  # cer, rtf, etc.
    hf_row_ref: str = ""  # serialized row identifier for re-fetch
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Response/API shapes
# ---------------------------------------------------------------------------


class LeaderboardEntry(BaseModel):
    rank: int
    plugin_id: uuid.UUID
    plugin_name: str
    display_name: str
    family: PluginFamily
    lang: Optional[str]
    elo: float
    battles: int
    wins: int
    losses: int
    ties: int
    win_rate: float


class LeaderboardResponse(BaseModel):
    family: PluginFamily
    lang: Optional[str]
    entries: List[LeaderboardEntry]
    total: int


class MatchupPublic(BaseModel):
    """Matchup returned to voter — plugin identities hidden until after vote."""

    id: uuid.UUID
    family: PluginFamily
    input_ref: str
    output_a_ref: Optional[str]
    output_b_ref: Optional[str]
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    status: str
    created_at: datetime
