"""
CRUD operations for OVOS Plugin Arena domain models.
Production-grade database operations with proper error handling.
"""

import hashlib
import random
import uuid
from typing import Any

from sqlmodel import Session, and_, desc, func, select

from app.models_arena import (
    Battle,
    BattleCreate,
    BattleStatusEnum,
    Competitor,
    CompetitorCreate,
    EloHistory,
    EloHistoryCreate,
    LeaderboardEntry,
    ModalityEnum,
    Plugin,
    PluginCreate,
    Vote,
    VoteCreate,
    VoteResultEnum,
)


# ============================================================================
# PLUGIN OPERATIONS
# ============================================================================


def create_plugin(*, session: Session, plugin_create: PluginCreate) -> Plugin:
    """
    Create a new plugin identity.

    Args:
        session: Database session
        plugin_create: Plugin creation data

    Returns:
        Created Plugin instance

    Raises:
        IntegrityError: If plugin_name already exists
    """
    db_plugin = Plugin.model_validate(plugin_create)
    session.add(db_plugin)
    session.commit()
    session.refresh(db_plugin)
    return db_plugin


def get_plugin_by_name(*, session: Session, plugin_name: str) -> Plugin | None:
    """Get plugin by unique plugin_name."""
    statement = select(Plugin).where(Plugin.plugin_name == plugin_name)
    return session.exec(statement).first()


def get_plugin_by_id(*, session: Session, plugin_id: uuid.UUID) -> Plugin | None:
    """Get plugin by UUID."""
    return session.get(Plugin, plugin_id)


# ============================================================================
# COMPETITOR OPERATIONS
# ============================================================================


def compute_config_hash(config_json: dict) -> str:
    """
    Compute deterministic hash of configuration JSON.
    Used to prevent duplicate competitor registrations.
    """
    # Sort keys for deterministic hashing
    config_str = str(sorted(config_json.items()))
    return hashlib.sha256(config_str.encode()).hexdigest()


def create_competitor(
    *, session: Session, competitor_create: CompetitorCreate
) -> Competitor:
    """
    Create a new competitor (plugin + config + modality).

    Args:
        session: Database session
        competitor_create: Competitor creation data

    Returns:
        Created Competitor instance

    Note:
        Automatically computes config_hash from config_json.
        Duplicate (plugin_id, config_hash) raises IntegrityError.
    """
    # Compute config hash
    config_hash = compute_config_hash(competitor_create.config_json)

    db_competitor = Competitor(
        plugin_id=competitor_create.plugin_id,
        modality=competitor_create.modality,
        config_hash=config_hash,
        config_json=competitor_create.config_json,
    )
    session.add(db_competitor)
    session.commit()
    session.refresh(db_competitor)
    return db_competitor


def get_competitor_by_id(
    *, session: Session, competitor_id: uuid.UUID
) -> Competitor | None:
    """Get competitor by UUID."""
    return session.get(Competitor, competitor_id)


def get_competitors_by_modality(
    *, session: Session, modality: ModalityEnum, skip: int = 0, limit: int = 100
) -> list[Competitor]:
    """
    Get all competitors for a specific modality.

    Args:
        session: Database session
        modality: Plugin modality filter
        skip: Number of records to skip
        limit: Maximum records to return

    Returns:
        List of Competitor instances
    """
    statement = (
        select(Competitor)
        .where(Competitor.modality == modality)
        .offset(skip)
        .limit(limit)
    )
    return list(session.exec(statement).all())


def get_random_competitors_for_battle(
    *, session: Session, modality: ModalityEnum
) -> tuple[Competitor, Competitor] | None:
    """
    Select two random competitors for battle.

    Args:
        session: Database session
        modality: Plugin modality

    Returns:
        Tuple of two Competitor instances, or None if < 2 competitors exist

    Note:
        Uses random selection weighted by inverse of battles_fought
        to give newer competitors more exposure.
    """
    statement = select(Competitor).where(Competitor.modality == modality)
    competitors = list(session.exec(statement).all())

    if len(competitors) < 2:
        return None

    # Weight selection by inverse of battles fought (favor newer competitors)
    weights = [1.0 / (c.battles_fought + 1) for c in competitors]
    selected = random.choices(competitors, weights=weights, k=2)

    # Ensure we don't pick the same competitor twice
    while selected[0].id == selected[1].id and len(competitors) > 1:
        selected = random.choices(competitors, weights=weights, k=2)

    return selected[0], selected[1]


# ============================================================================
# BATTLE OPERATIONS
# ============================================================================


def create_battle(*, session: Session, battle_create: BattleCreate) -> Battle:
    """
    Create a new battle (A/B comparison).

    Args:
        session: Database session
        battle_create: Battle creation data

    Returns:
        Created Battle instance with status=PENDING

    Note:
        Battle ID is generated before Kafka publish (system invariant).
    """
    db_battle = Battle.model_validate(battle_create)
    session.add(db_battle)
    session.commit()
    session.refresh(db_battle)
    return db_battle


def get_battle_by_id(*, session: Session, battle_id: uuid.UUID) -> Battle | None:
    """Get battle by UUID."""
    return session.get(Battle, battle_id)


def get_pending_battles(
    *, session: Session, limit: int = 10
) -> list[Battle]:
    """
    Get battles awaiting worker execution.

    Args:
        session: Database session
        limit: Maximum battles to return

    Returns:
        List of Battle instances with status=PENDING
    """
    statement = (
        select(Battle)
        .where(Battle.status == BattleStatusEnum.PENDING)
        .order_by(Battle.created_at)
        .limit(limit)
    )
    return list(session.exec(statement).all())


def get_ready_battles_for_voting(
    *, session: Session, modality: ModalityEnum, limit: int = 10
) -> list[Battle]:
    """
    Get battles ready for user voting.

    Args:
        session: Database session
        modality: Filter by modality
        limit: Maximum battles to return

    Returns:
        List of Battle instances with status=READY and no votes yet
    """
    statement = (
        select(Battle)
        .where(
            and_(
                Battle.status == BattleStatusEnum.READY,
                Battle.modality == modality,
            )
        )
        .outerjoin(Vote)
        .where(Vote.id.is_(None))  # No votes yet
        .order_by(Battle.created_at)
        .limit(limit)
    )
    return list(session.exec(statement).all())


def update_battle_status(
    *,
    session: Session,
    battle_id: uuid.UUID,
    status: BattleStatusEnum,
    failure_reason: str | None = None,
    output_a_ref: str | None = None,
    output_b_ref: str | None = None,
    result_a_data: dict | None = None,
    result_b_data: dict | None = None,
) -> Battle | None:
    """
    Update battle status and related fields (called by workers).

    Args:
        session: Database session
        battle_id: Battle UUID
        status: New status
        failure_reason: Optional failure description
        output_a_ref: Optional MinIO path for output A
        output_b_ref: Optional MinIO path for output B
        result_a_data: Optional structured result data for A
        result_b_data: Optional structured result data for B

    Returns:
        Updated Battle instance or None if not found
    """
    battle = get_battle_by_id(session=session, battle_id=battle_id)
    if not battle:
        return None

    battle.status = status
    if failure_reason:
        battle.failure_reason = failure_reason
    if output_a_ref:
        battle.output_a_ref = output_a_ref
    if output_b_ref:
        battle.output_b_ref = output_b_ref
    if result_a_data:
        battle.result_a_data = result_a_data
    if result_b_data:
        battle.result_b_data = result_b_data

    session.add(battle)
    session.commit()
    session.refresh(battle)
    return battle


# ============================================================================
# VOTE OPERATIONS
# ============================================================================


def create_vote(
    *, session: Session, vote_create: VoteCreate, user_id: uuid.UUID
) -> Vote:
    """
    Create a vote for a battle.

    Args:
        session: Database session
        vote_create: Vote creation data
        user_id: User who is voting

    Returns:
        Created Vote instance

    Raises:
        IntegrityError: If user already voted on this battle
    """
    db_vote = Vote.model_validate(vote_create, update={"user_id": user_id})
    session.add(db_vote)
    session.commit()
    session.refresh(db_vote)
    return db_vote


def get_user_vote_for_battle(
    *, session: Session, battle_id: uuid.UUID, user_id: uuid.UUID
) -> Vote | None:
    """Check if user already voted on a battle."""
    statement = select(Vote).where(
        and_(Vote.battle_id == battle_id, Vote.user_id == user_id)
    )
    return session.exec(statement).first()


# ============================================================================
# ELO OPERATIONS
# ============================================================================


def create_elo_history(
    *, session: Session, elo_history_create: EloHistoryCreate
) -> EloHistory:
    """
    Record ELO change in history table.

    Args:
        session: Database session
        elo_history_create: ELO history data

    Returns:
        Created EloHistory instance
    """
    db_elo_history = EloHistory.model_validate(elo_history_create)
    session.add(db_elo_history)
    session.commit()
    session.refresh(db_elo_history)
    return db_elo_history


def update_competitor_elo(
    *,
    session: Session,
    competitor_id: uuid.UUID,
    new_elo: int,
    battle_id: uuid.UUID,
) -> Competitor | None:
    """
    Update competitor ELO and record change in history.

    Args:
        session: Database session
        competitor_id: Competitor UUID
        new_elo: New ELO rating
        battle_id: Battle that triggered the change

    Returns:
        Updated Competitor instance or None if not found

    Note:
        Automatically creates EloHistory record for audit trail.
    """
    competitor = get_competitor_by_id(session=session, competitor_id=competitor_id)
    if not competitor:
        return None

    old_elo = competitor.elo
    elo_change = new_elo - old_elo

    # Update competitor ELO
    competitor.elo = new_elo
    session.add(competitor)

    # Record history
    elo_history = EloHistoryCreate(
        competitor_id=competitor_id,
        battle_id=battle_id,
        old_elo=old_elo,
        new_elo=new_elo,
        elo_change=elo_change,
    )
    create_elo_history(session=session, elo_history_create=elo_history)

    session.commit()
    session.refresh(competitor)
    return competitor


def update_competitor_stats(
    *,
    session: Session,
    competitor_id: uuid.UUID,
    result: VoteResultEnum,
    is_competitor_a: bool,
) -> Competitor | None:
    """
    Update competitor win/loss/tie statistics.

    Args:
        session: Database session
        competitor_id: Competitor UUID
        result: Vote result
        is_competitor_a: True if this competitor was "A" in battle

    Returns:
        Updated Competitor instance or None if not found
    """
    competitor = get_competitor_by_id(session=session, competitor_id=competitor_id)
    if not competitor:
        return None

    competitor.battles_fought += 1

    # Determine outcome from competitor's perspective
    if result == VoteResultEnum.TIE or result == VoteResultEnum.BOTH_WRONG:
        competitor.ties += 1
    elif (
        result == VoteResultEnum.CANDIDATE_1 and is_competitor_a
    ) or (result == VoteResultEnum.CANDIDATE_2 and not is_competitor_a):
        competitor.wins += 1
    else:
        competitor.losses += 1

    session.add(competitor)
    session.commit()
    session.refresh(competitor)
    return competitor


# ============================================================================
# LEADERBOARD OPERATIONS
# ============================================================================


def get_leaderboard(
    *,
    session: Session,
    modality: ModalityEnum,
    skip: int = 0,
    limit: int = 50,
) -> list[LeaderboardEntry]:
    """
    Get leaderboard sorted by ELO for a modality.

    Args:
        session: Database session
        modality: Filter by modality
        skip: Number of records to skip
        limit: Maximum records to return

    Returns:
        List of LeaderboardEntry instances sorted by ELO descending

    Note:
        PostgreSQL is the source of truth - recomputed on every request.
    """
    statement = (
        select(
            Competitor.id.label("competitor_id"),
            Plugin.plugin_name,
            Plugin.display_name,
            Competitor.modality,
            Competitor.elo,
            Competitor.battles_fought,
            Competitor.wins,
            Competitor.losses,
            Competitor.ties,
        )
        .join(Plugin, Competitor.plugin_id == Plugin.id)
        .where(Competitor.modality == modality)
        .order_by(desc(Competitor.elo))
        .offset(skip)
        .limit(limit)
    )

    results = session.exec(statement).all()

    entries = []
    for rank, row in enumerate(results, start=skip + 1):
        win_rate = (
            (row.wins / row.battles_fought * 100) if row.battles_fought > 0 else 0.0
        )
        entries.append(
            LeaderboardEntry(
                rank=rank,
                competitor_id=row.competitor_id,
                plugin_name=row.plugin_name,
                display_name=row.display_name,
                modality=row.modality,
                elo=row.elo,
                battles_fought=row.battles_fought,
                wins=row.wins,
                losses=row.losses,
                ties=row.ties,
                win_rate=round(win_rate, 2),
            )
        )

    return entries


def get_total_competitors_by_modality(
    *, session: Session, modality: ModalityEnum
) -> int:
    """Get total count of competitors for a modality."""
    statement = select(func.count(Competitor.id)).where(
        Competitor.modality == modality
    )
    return session.exec(statement).one()
