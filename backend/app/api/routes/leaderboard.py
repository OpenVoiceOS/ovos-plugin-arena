"""
Leaderboard endpoints for OVOS Plugin Arena.
Public-facing ranking and statistics.
"""

import logging

from fastapi import APIRouter, HTTPException

from app import crud_arena
from app.api.deps import SessionDep
from app.models_arena import LeaderboardEntry, LeaderboardResponse, ModalityEnum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get(
    "/{modality}",
    response_model=LeaderboardResponse,
    summary="Get leaderboard for modality",
)
def get_leaderboard(
    *,
    session: SessionDep,
    modality: ModalityEnum,
    skip: int = 0,
    limit: int = 50,
) -> LeaderboardResponse:
    """
    Get ranked leaderboard for a specific modality.

    **Query params:**
    - `skip`: Pagination offset (default 0)
    - `limit`: Max results (default 50, max 100)

    **Sorting:**
    - Primary: ELO rating (descending)
    - Secondary: Battles fought (more battles = higher rank in ties)

    **Response:**
    - Rank (1-indexed)
    - Plugin name and display name
    - Current ELO
    - Battle statistics (total, wins, losses, ties)
    - Win rate percentage

    **PostgreSQL is source of truth:**
    Leaderboard recomputed on every request from current database state.
    """
    if limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit cannot exceed 100",
        )

    # Get leaderboard entries
    entries = crud_arena.get_leaderboard(
        session=session,
        modality=modality,
        skip=skip,
        limit=limit,
    )

    # Get total count for pagination
    total = crud_arena.get_total_competitors_by_modality(
        session=session,
        modality=modality,
    )

    logger.info(
        "Leaderboard query: modality=%s, skip=%d, limit=%d, total=%d",
        modality.value,
        skip,
        limit,
        total,
    )

    return LeaderboardResponse(
        modality=modality,
        entries=entries,
        total=total,
    )
