"""
Arena battle endpoints for OVOS Plugin Arena.
Core battle creation and voting functionality.

CRITICAL: API layer never runs plugins. Only creates battle rows and publishes Kafka messages.
"""

import logging
import random
from datetime import datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import IntegrityError

from app import crud_arena
from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import Message
from app.models_arena import (
    BattleCreate,
    BattlePublic,
    ModalityEnum,
    VoteCreate,
    VotePublic,
    VoteResultEnum,
)
from app.schemas.kafka_messages import BattleExecutionRequested
from app.services.kafka_producer import get_kafka_producer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arena", tags=["arena"])


# ============================================================================
# BATTLE CREATION
# ============================================================================


@router.get(
    "/{modality}/battle",
    response_model=BattlePublic,
    summary="Get next battle for voting",
)
async def get_next_battle(
    *,
    session: SessionDep,
    modality: ModalityEnum,
    current_user: CurrentUser,
) -> BattlePublic:
    """
    Get or create the next battle for user to vote on.

    **Flow:**
    1. Check for existing READY battles without user's vote
    2. If none exist, create new battle
    3. Publish to Kafka for worker execution
    4. Return battle with PENDING/READY status

    **IMPORTANT:**
    - API never executes plugins
    - Returns immediately with battle ID
    - Workers handle actual execution asynchronously

    **Idempotent:** Multiple calls return same un-voted battle.
    """
    # First, check for existing READY battles user hasn't voted on
    from sqlmodel import and_, select

    from app.models_arena import Battle, Vote

    statement = (
        select(Battle)
        .where(
            and_(
                Battle.status == "READY",
                Battle.modality == modality,
            )
        )
        .outerjoin(Vote, and_(
            Vote.battle_id == Battle.id,
            Vote.user_id == current_user.id
        ))
        .where(Vote.id.is_(None))  # User hasn't voted
        .order_by(Battle.created_at)
        .limit(1)
    )

    existing_battle = session.exec(statement).first()
    if existing_battle:
        logger.info(
            "Returning existing READY battle: %s for user=%s",
            existing_battle.id,
            current_user.id,
        )
        return BattlePublic.model_validate(existing_battle)

    # No ready battles - create a new one
    # Select two random competitors
    competitors_pair = crud_arena.get_random_competitors_for_battle(
        session=session, modality=modality
    )

    if not competitors_pair:
        raise HTTPException(
            status_code=503,
            detail=f"Not enough competitors registered for modality: {modality.value}. Need at least 2.",
        )

    competitor_a, competitor_b = competitors_pair

    # Select random input from dataset (placeholder for MinIO integration)
    input_datasets = settings.BATTLE_INPUT_DATASETS.get(modality.value, [])
    if not input_datasets:
        raise HTTPException(
            status_code=500,
            detail=f"No input datasets configured for modality: {modality.value}",
        )

    input_ref = random.choice(input_datasets)

    # Create battle row (PostgreSQL is source of truth)
    battle_create = BattleCreate(
        modality=modality,
        competitor_a_id=competitor_a.id,
        competitor_b_id=competitor_b.id,
        input_ref=input_ref,
    )

    battle = crud_arena.create_battle(session=session, battle_create=battle_create)
    logger.info(
        "Created new battle: %s, modality=%s, competitors=(%s, %s)",
        battle.id,
        battle.modality.value,
        competitor_a.id,
        competitor_b.id,
    )

    # Publish to Kafka for worker execution
    try:
        kafka_producer = await get_kafka_producer()
        message = BattleExecutionRequested(
            battle_id=battle.id,
            modality=battle.modality,
            competitor_a_id=competitor_a.id,
            competitor_b_id=competitor_b.id,
            input_ref=input_ref,
            attempt=0,
            created_at=datetime.utcnow(),
        )
        await kafka_producer.publish_battle_execution_requested(message)
        logger.info("Published battle execution request: %s", battle.id)
    except Exception as e:
        logger.error("Failed to publish to Kafka: %s", e)
        # Battle row exists, but Kafka failed
        # Worker can pick it up from DB query for PENDING battles
        logger.warning("Battle created but Kafka publish failed: %s", battle.id)

    # Return immediately - don't wait for workers
    return BattlePublic.model_validate(battle)


# ============================================================================
# VOTING
# ============================================================================


@router.post(
    "/{modality}/vote",
    response_model=VotePublic,
    status_code=201,
    summary="Submit vote for battle",
)
async def submit_vote(
    *,
    session: SessionDep,
    modality: ModalityEnum,
    vote_in: VoteCreate,
    current_user: CurrentUser,
) -> VotePublic:
    """
    Submit user's vote for a battle.

    **Requirements:**
    - Battle must exist and be READY
    - Battle must match modality
    - User cannot vote on same battle twice

    **Effects:**
    - Creates vote record
    - Publishes to Kafka for ELO calculation
    - Workers update competitor stats and ELO

    **Idempotent:** Duplicate votes return 409 Conflict.
    """
    # Verify battle exists and is READY
    battle = crud_arena.get_battle_by_id(session=session, battle_id=vote_in.battle_id)
    if not battle:
        raise HTTPException(status_code=404, detail="Battle not found")

    if battle.modality != modality:
        raise HTTPException(
            status_code=400,
            detail=f"Battle modality mismatch. Expected {modality.value}, got {battle.modality.value}",
        )

    if battle.status != "READY":
        raise HTTPException(
            status_code=400,
            detail=f"Battle not ready for voting. Status: {battle.status}",
        )

    # Check if user already voted
    existing_vote = crud_arena.get_user_vote_for_battle(
        session=session, battle_id=vote_in.battle_id, user_id=current_user.id
    )
    if existing_vote:
        logger.warning(
            "User %s already voted on battle %s",
            current_user.id,
            vote_in.battle_id,
        )
        raise HTTPException(
            status_code=409,
            detail="You have already voted on this battle",
        )

    # Create vote
    try:
        vote = crud_arena.create_vote(
            session=session, vote_create=vote_in, user_id=current_user.id
        )
        logger.info(
            "User %s voted on battle %s: result=%s",
            current_user.id,
            vote.battle_id,
            vote.result.value,
        )
    except IntegrityError:
        # Race condition - user voted between check and insert
        raise HTTPException(
            status_code=409,
            detail="You have already voted on this battle",
        )
    except Exception as e:
        logger.error("Failed to create vote: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

    # Publish vote event to Kafka for ELO calculation
    try:
        kafka_producer = await get_kafka_producer()
        await kafka_producer.publish_vote_submitted(
            vote_id=str(vote.id),
            battle_id=str(battle.id),
            user_id=str(current_user.id),
            result=vote.result.value,
            competitor_a_id=str(battle.competitor_a_id),
            competitor_b_id=str(battle.competitor_b_id),
        )
        logger.info("Published vote event for battle %s", battle.id)
    except Exception as e:
        logger.error("Failed to publish vote to Kafka: %s", e)
        # Vote is persisted, workers can process from DB
        logger.warning("Vote created but Kafka publish failed: %s", vote.id)

    return VotePublic.model_validate(vote)


@router.get(
    "/battles/{battle_id}",
    response_model=BattlePublic,
    summary="Get battle details",
)
def get_battle_details(
    *,
    session: SessionDep,
    battle_id: str,
    current_user: CurrentUser,
) -> BattlePublic:
    """
    Get details of a specific battle.

    **Note:** Competitor IDs are masked in response to preserve blind testing.
    """
    from uuid import UUID

    try:
        battle_uuid = UUID(battle_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid battle ID format")

    battle = crud_arena.get_battle_by_id(session=session, battle_id=battle_uuid)
    if not battle:
        raise HTTPException(status_code=404, detail="Battle not found")

    return BattlePublic.model_validate(battle)
