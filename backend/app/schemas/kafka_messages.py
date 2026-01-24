"""
Kafka message schemas for OVOS Plugin Arena.
Type-safe message definitions for event-driven communication.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models_arena import ModalityEnum


# ============================================================================
# BATTLE EXECUTION MESSAGES
# ============================================================================


class BattleExecutionRequested(BaseModel):
    """
    Published when a new battle is created and needs worker execution.
    Workers consume this message to execute both competitors and store results.
    """

    battle_id: uuid.UUID = Field(description="Unique battle identifier")
    modality: ModalityEnum = Field(description="Plugin modality for this battle")
    competitor_a_id: uuid.UUID = Field(description="First competitor UUID")
    competitor_b_id: uuid.UUID = Field(description="Second competitor UUID")
    input_ref: str = Field(description="Reference to input data (text/audio)")
    attempt: int = Field(default=0, description="Retry attempt number")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Message creation timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "battle_id": "550e8400-e29b-41d4-a716-446655440000",
                "modality": "tts",
                "competitor_a_id": "123e4567-e89b-12d3-a456-426614174000",
                "competitor_b_id": "987fcdeb-51a2-43d7-a123-987654321000",
                "input_ref": "dataset/sample_001.txt",
                "attempt": 0,
                "created_at": "2026-01-23T10:30:00Z",
            }
        }


class BattleExecutionCompleted(BaseModel):
    """
    Published by workers when battle execution completes successfully.
    Triggers ELO updates and leaderboard recalculation.
    """

    battle_id: uuid.UUID = Field(description="Completed battle identifier")
    competitor_a_id: uuid.UUID
    competitor_b_id: uuid.UUID
    output_a_ref: str = Field(description="MinIO path to output A")
    output_b_ref: str = Field(description="MinIO path to output B")
    result_a_data: dict | None = Field(
        default=None, description="Structured result data for competitor A"
    )
    result_b_data: dict | None = Field(
        default=None, description="Structured result data for competitor B"
    )
    execution_time_ms: int = Field(description="Total execution time in milliseconds")
    worker_id: str = Field(description="Identifier of worker that executed battle")
    completed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "battle_id": "550e8400-e29b-41d4-a716-446655440000",
                "competitor_a_id": "123e4567-e89b-12d3-a456-426614174000",
                "competitor_b_id": "987fcdeb-51a2-43d7-a123-987654321000",
                "output_a_ref": "minio://battles/550e8400/output_a.wav",
                "output_b_ref": "minio://battles/550e8400/output_b.wav",
                "result_a_data": {"duration_ms": 1234, "sample_rate": 22050},
                "result_b_data": {"duration_ms": 1198, "sample_rate": 22050},
                "execution_time_ms": 3450,
                "worker_id": "worker-tts-001",
                "completed_at": "2026-01-23T10:30:15Z",
            }
        }


class BattleExecutionFailed(BaseModel):
    """
    Published by workers when battle execution fails.
    API layer decides whether to retry or mark as FAILED.
    """

    battle_id: uuid.UUID
    competitor_a_id: uuid.UUID
    competitor_b_id: uuid.UUID
    failure_reason: str = Field(description="Human-readable failure description")
    error_code: str | None = Field(default=None, description="Machine-readable error code")
    attempt: int = Field(description="Failed attempt number")
    is_retryable: bool = Field(
        default=True, description="Whether this failure should trigger a retry"
    )
    worker_id: str
    failed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "battle_id": "550e8400-e29b-41d4-a716-446655440000",
                "competitor_a_id": "123e4567-e89b-12d3-a456-426614174000",
                "competitor_b_id": "987fcdeb-51a2-43d7-a123-987654321000",
                "failure_reason": "Competitor A timed out after 30 seconds",
                "error_code": "EXECUTION_TIMEOUT",
                "attempt": 0,
                "is_retryable": True,
                "worker_id": "worker-tts-001",
                "failed_at": "2026-01-23T10:30:15Z",
            }
        }


# ============================================================================
# VOTE PROCESSING MESSAGES
# ============================================================================


class VoteSubmitted(BaseModel):
    """
    Published when a user submits a vote.
    Triggers ELO calculation and competitor stats updates.
    """

    vote_id: uuid.UUID = Field(description="Unique vote identifier")
    battle_id: uuid.UUID
    user_id: uuid.UUID
    result: str = Field(description="Vote result: candidate_1, candidate_2, tie, both_wrong")
    competitor_a_id: uuid.UUID
    competitor_b_id: uuid.UUID
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "vote_id": "c0a80101-0000-0000-0000-000000000001",
                "battle_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "a0b1c2d3-e4f5-6789-0abc-def012345678",
                "result": "candidate_1",
                "competitor_a_id": "123e4567-e89b-12d3-a456-426614174000",
                "competitor_b_id": "987fcdeb-51a2-43d7-a123-987654321000",
                "submitted_at": "2026-01-23T10:35:00Z",
            }
        }


# ============================================================================
# ELO UPDATE MESSAGES
# ============================================================================


class EloUpdateRequested(BaseModel):
    """
    Published when vote is submitted, requesting ELO recalculation.
    Consumed by dedicated ELO calculation service or worker.
    """

    battle_id: uuid.UUID
    vote_id: uuid.UUID
    competitor_a_id: uuid.UUID
    competitor_b_id: uuid.UUID
    vote_result: str
    requested_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "battle_id": "550e8400-e29b-41d4-a716-446655440000",
                "vote_id": "c0a80101-0000-0000-0000-000000000001",
                "competitor_a_id": "123e4567-e89b-12d3-a456-426614174000",
                "competitor_b_id": "987fcdeb-51a2-43d7-a123-987654321000",
                "vote_result": "candidate_1",
                "requested_at": "2026-01-23T10:35:00Z",
            }
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def serialize_message(message: BaseModel) -> dict:
    """
    Serialize Pydantic message to dict for Kafka publishing.
    Converts UUIDs and datetimes to strings.
    """
    return message.model_dump(mode="json")


def deserialize_message(message_class: type[BaseModel], data: dict) -> BaseModel:
    """
    Deserialize dict from Kafka into typed Pydantic message.
    Validates schema and converts types.
    """
    return message_class.model_validate(data)
