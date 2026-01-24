"""
Arena domain models for OVOS Plugin Arena.
Aligned with PostgreSQL schema in docs/models.sql
"""

import enum
import uuid
from datetime import date, datetime

from sqlmodel import Field, Relationship, SQLModel


# ============================================================================
# ENUMS
# ============================================================================


class UserRoleEnum(str, enum.Enum):
    """Authorization roles for authenticated users"""

    ADMIN = "admin"  # Can register plugins and manage system
    VOTER = "voter"  # Can participate in battles and vote


class ModalityEnum(str, enum.Enum):
    """Supported plugin modalities"""

    TTS = "tts"  # Text-to-Speech
    STT = "stt"  # Speech-to-Text
    WAKE_WORD = "wake_word"  # Wake Word Detection
    INTENT = "intent"  # Intent Classification


class BattleStatusEnum(str, enum.Enum):
    """Battle execution lifecycle"""

    PENDING = "PENDING"  # Created, awaiting worker
    RUNNING = "RUNNING"  # Worker executing plugins
    READY = "READY"  # Outputs ready for voting
    FAILED = "FAILED"  # Failed after retries or fatal error


class VoteResultEnum(str, enum.Enum):
    """Vote outcomes"""

    CANDIDATE_1 = "candidate_1"
    CANDIDATE_2 = "candidate_2"
    TIE = "tie"
    BOTH_WRONG = "both_wrong"


# ============================================================================
# PLUGIN MODELS
# ============================================================================


class PluginBase(SQLModel):
    """Shared properties for Plugin"""

    plugin_name: str = Field(unique=True, index=True, max_length=255)
    display_name: str = Field(max_length=255)
    author: str | None = Field(default=None, max_length=255)
    description: str | None = None
    homepage_url: str | None = None
    license: str | None = Field(default=None, max_length=255)
    tags: list[str] | None = Field(default=None, sa_column_kwargs={"type_": "ARRAY(TEXT)"})
    metadata: dict | None = Field(default=None, sa_column_kwargs={"type_": "JSONB"})


class PluginCreate(PluginBase):
    """Properties to receive via API on plugin creation"""

    supported_modalities: list[ModalityEnum] = Field(
        sa_column_kwargs={"type_": "ARRAY(TEXT)"}
    )


class PluginUpdate(SQLModel):
    """Properties to receive via API on plugin update"""

    display_name: str | None = None
    author: str | None = None
    description: str | None = None
    homepage_url: str | None = None
    license: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None


class Plugin(PluginBase, table=True):
    """Database model for plugins table"""

    __tablename__ = "plugins"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    supported_modalities: list[str] = Field(
        sa_column_kwargs={"type_": "ARRAY(TEXT)"}
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    competitors: list["Competitor"] = Relationship(
        back_populates="plugin", cascade_delete=True
    )


class PluginPublic(PluginBase):
    """Properties to return via API"""

    id: uuid.UUID
    supported_modalities: list[ModalityEnum]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# COMPETITOR MODELS
# ============================================================================


class CompetitorBase(SQLModel):
    """Shared properties for Competitor"""

    modality: ModalityEnum
    config_hash: str = Field(max_length=255)
    config_json: dict = Field(sa_column_kwargs={"type_": "JSONB"})


class CompetitorCreate(CompetitorBase):
    """Properties to receive via API on competitor creation"""

    plugin_id: uuid.UUID


class CompetitorUpdate(SQLModel):
    """Properties to receive via API on competitor update"""

    config_json: dict | None = None


class Competitor(CompetitorBase, table=True):
    """Database model for competitors table"""

    __tablename__ = "competitors"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    plugin_id: uuid.UUID = Field(foreign_key="plugins.id", ondelete="CASCADE")
    elo: int = Field(default=1200)
    battles_fought: int = Field(default=0)
    wins: int = Field(default=0)
    losses: int = Field(default=0)
    ties: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    plugin: Plugin = Relationship(back_populates="competitors")
    battles_as_a: list["Battle"] = Relationship(
        back_populates="competitor_a",
        sa_relationship_kwargs={"foreign_keys": "Battle.competitor_a_id"},
    )
    battles_as_b: list["Battle"] = Relationship(
        back_populates="competitor_b",
        sa_relationship_kwargs={"foreign_keys": "Battle.competitor_b_id"},
    )


class CompetitorPublic(CompetitorBase):
    """Properties to return via API"""

    id: uuid.UUID
    plugin_id: uuid.UUID
    elo: int
    battles_fought: int
    wins: int
    losses: int
    ties: int
    created_at: datetime


class CompetitorWithPlugin(CompetitorPublic):
    """Competitor with plugin details"""

    plugin: PluginPublic


# ============================================================================
# BATTLE MODELS
# ============================================================================


class BattleBase(SQLModel):
    """Shared properties for Battle"""

    modality: ModalityEnum
    input_ref: str = Field(max_length=500)


class BattleCreate(BattleBase):
    """Properties to receive via API on battle creation"""

    competitor_a_id: uuid.UUID
    competitor_b_id: uuid.UUID


class BattleUpdate(SQLModel):
    """Properties to receive via API on battle update (worker use)"""

    output_a_ref: str | None = None
    output_b_ref: str | None = None
    result_a_data: dict | None = None
    result_b_data: dict | None = None
    status: BattleStatusEnum | None = None
    failure_reason: str | None = None
    attempt: int | None = None


class Battle(BattleBase, table=True):
    """Database model for battles table"""

    __tablename__ = "battles"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    competitor_a_id: uuid.UUID = Field(foreign_key="competitors.id")
    competitor_b_id: uuid.UUID = Field(foreign_key="competitors.id")
    output_a_ref: str | None = Field(default=None, max_length=500)
    output_b_ref: str | None = Field(default=None, max_length=500)
    result_a_data: dict | None = Field(
        default=None, sa_column_kwargs={"type_": "JSONB"}
    )
    result_b_data: dict | None = Field(
        default=None, sa_column_kwargs={"type_": "JSONB"}
    )
    status: BattleStatusEnum = Field(default=BattleStatusEnum.PENDING)
    failure_reason: str | None = None
    attempt: int = Field(default=0)
    metadata: dict | None = Field(default=None, sa_column_kwargs={"type_": "JSONB"})
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    competitor_a: Competitor = Relationship(
        back_populates="battles_as_a",
        sa_relationship_kwargs={"foreign_keys": "[Battle.competitor_a_id]"},
    )
    competitor_b: Competitor = Relationship(
        back_populates="battles_as_b",
        sa_relationship_kwargs={"foreign_keys": "[Battle.competitor_b_id]"},
    )
    votes: list["Vote"] = Relationship(back_populates="battle", cascade_delete=True)


class BattlePublic(BattleBase):
    """Properties to return via API - MASKED for frontend"""

    id: uuid.UUID
    # competitor_a_id and competitor_b_id are intentionally hidden
    output_a_ref: str | None
    output_b_ref: str | None
    status: BattleStatusEnum
    created_at: datetime


class BattleInternal(BattlePublic):
    """Internal battle representation with full data (for admin/workers)"""

    competitor_a_id: uuid.UUID
    competitor_b_id: uuid.UUID
    result_a_data: dict | None
    result_b_data: dict | None
    failure_reason: str | None
    attempt: int
    metadata: dict | None
    updated_at: datetime


# ============================================================================
# VOTE MODELS
# ============================================================================


class VoteBase(SQLModel):
    """Shared properties for Vote"""

    result: VoteResultEnum


class VoteCreate(VoteBase):
    """Properties to receive via API on vote creation"""

    battle_id: uuid.UUID


class Vote(VoteBase, table=True):
    """Database model for votes table"""

    __tablename__ = "votes"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    battle_id: uuid.UUID = Field(foreign_key="battles.id", ondelete="CASCADE")
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    battle: Battle = Relationship(back_populates="votes")


class VotePublic(VoteBase):
    """Properties to return via API"""

    id: uuid.UUID
    battle_id: uuid.UUID
    created_at: datetime


# ============================================================================
# ELO HISTORY MODELS
# ============================================================================


class EloHistoryBase(SQLModel):
    """Shared properties for EloHistory"""

    old_elo: int
    new_elo: int
    elo_change: int


class EloHistoryCreate(EloHistoryBase):
    """Properties to receive when creating ELO history"""

    competitor_id: uuid.UUID
    battle_id: uuid.UUID


class EloHistory(EloHistoryBase, table=True):
    """Database model for elo_history table"""

    __tablename__ = "elo_history"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    competitor_id: uuid.UUID = Field(foreign_key="competitors.id", ondelete="CASCADE")
    battle_id: uuid.UUID = Field(foreign_key="battles.id", ondelete="CASCADE")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EloHistoryPublic(EloHistoryBase):
    """Properties to return via API"""

    id: uuid.UUID
    competitor_id: uuid.UUID
    battle_id: uuid.UUID
    created_at: datetime


# ============================================================================
# PLUGIN METRICS MODELS
# ============================================================================


class PluginMetricsDailyBase(SQLModel):
    """Shared properties for PluginMetricsDaily"""

    modality: ModalityEnum
    metric_date: date
    avg_elo: int
    max_elo: int
    min_elo: int
    battles_fought: int
    wins: int
    losses: int
    ties: int


class PluginMetricsDailyCreate(PluginMetricsDailyBase):
    """Properties to receive when creating daily metrics"""

    plugin_id: uuid.UUID


class PluginMetricsDaily(PluginMetricsDailyBase, table=True):
    """Database model for plugin_metrics_daily table"""

    __tablename__ = "plugin_metrics_daily"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    plugin_id: uuid.UUID = Field(foreign_key="plugins.id", ondelete="CASCADE")


class PluginMetricsDailyPublic(PluginMetricsDailyBase):
    """Properties to return via API"""

    id: uuid.UUID
    plugin_id: uuid.UUID


# ============================================================================
# LEADERBOARD RESPONSE MODELS
# ============================================================================


class LeaderboardEntry(SQLModel):
    """Single entry in leaderboard response"""

    rank: int
    competitor_id: uuid.UUID
    plugin_name: str
    display_name: str
    modality: ModalityEnum
    elo: int
    battles_fought: int
    wins: int
    losses: int
    ties: int
    win_rate: float


class LeaderboardResponse(SQLModel):
    """Full leaderboard response"""

    modality: ModalityEnum
    entries: list[LeaderboardEntry]
    total: int
