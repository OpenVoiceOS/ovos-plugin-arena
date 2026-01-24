"""
Competitor management endpoints for OVOS Plugin Arena.
Admin-only routes for registering and managing competitors.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import IntegrityError

from app import crud_arena
from app.api.deps import AdminUser, SessionDep
from app.models import Message
from app.models_arena import (
    Competitor,
    CompetitorCreate,
    CompetitorPublic,
    CompetitorWithPlugin,
    ModalityEnum,
    Plugin,
    PluginCreate,
    PluginPublic,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/competitors", tags=["admin", "competitors"])


# ============================================================================
# PLUGIN REGISTRATION
# ============================================================================


@router.post(
    "/plugins",
    response_model=PluginPublic,
    status_code=201,
    summary="Register a new plugin",
)
def register_plugin(
    *,
    session: SessionDep,
    plugin_in: PluginCreate,
    _admin: AdminUser,
) -> Plugin:
    """
    Register a new plugin identity.

    **Admin only.** Creates a plugin with metadata before registering competitors.

    **Idempotent:** Returns existing plugin if plugin_name already exists.
    """
    # Check if plugin already exists
    existing = crud_arena.get_plugin_by_name(
        session=session, plugin_name=plugin_in.plugin_name
    )
    if existing:
        logger.info("Plugin already exists: %s", plugin_in.plugin_name)
        return existing

    try:
        plugin = crud_arena.create_plugin(session=session, plugin_create=plugin_in)
        logger.info("Registered new plugin: %s", plugin.plugin_name)
        return plugin
    except IntegrityError as e:
        logger.error("Failed to register plugin: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Plugin with name '{plugin_in.plugin_name}' already exists",
        )
    except Exception as e:
        logger.error("Unexpected error registering plugin: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# COMPETITOR REGISTRATION
# ============================================================================


@router.post(
    "/",
    response_model=CompetitorPublic,
    status_code=201,
    summary="Register a new competitor",
)
def register_competitor(
    *,
    session: SessionDep,
    competitor_in: CompetitorCreate,
    _admin: AdminUser,
) -> Competitor:
    """
    Register a new competitor (plugin + config + modality).

    **Admin only.** Creates a concrete evaluation unit for battles.

    **Requirements:**
    - `plugin_id` must exist
    - `config_json` will be hashed to prevent duplicates
    - Duplicate (plugin_id, config_hash) returns 400

    **Idempotent:** Same plugin+config returns existing competitor on retry.

    **Response:**
    - Initial ELO: 1200
    - Initial stats: 0 battles, wins, losses, ties
    """
    # Verify plugin exists
    plugin = crud_arena.get_plugin_by_id(
        session=session, plugin_id=competitor_in.plugin_id
    )
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    # Verify modality is supported by plugin
    if competitor_in.modality.value not in plugin.supported_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Plugin does not support modality: {competitor_in.modality.value}",
        )

    try:
        competitor = crud_arena.create_competitor(
            session=session, competitor_create=competitor_in
        )
        logger.info(
            "Registered competitor: plugin=%s, modality=%s, elo=%d",
            plugin.plugin_name,
            competitor.modality.value,
            competitor.elo,
        )
        return competitor
    except IntegrityError:
        # Duplicate config hash - return existing
        logger.warning(
            "Competitor with same config already exists for plugin_id=%s",
            competitor_in.plugin_id,
        )
        raise HTTPException(
            status_code=400,
            detail="Competitor with this configuration already exists",
        )
    except Exception as e:
        logger.error("Unexpected error registering competitor: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# COMPETITOR QUERIES
# ============================================================================


@router.get(
    "/",
    response_model=list[CompetitorWithPlugin],
    summary="List all competitors",
)
def list_competitors(
    *,
    session: SessionDep,
    modality: ModalityEnum | None = None,
    skip: int = 0,
    limit: int = 100,
    _admin: AdminUser,
) -> list[CompetitorWithPlugin]:
    """
    List all registered competitors.

    **Admin only.** For monitoring and debugging.

    **Query params:**
    - `modality`: Filter by modality (optional)
    - `skip`: Pagination offset
    - `limit`: Max results (default 100)
    """
    if modality:
        competitors = crud_arena.get_competitors_by_modality(
            session=session, modality=modality, skip=skip, limit=limit
        )
    else:
        # Get all competitors across modalities
        all_competitors: list[Competitor] = []
        for mod in ModalityEnum:
            comps = crud_arena.get_competitors_by_modality(
                session=session, modality=mod, skip=skip, limit=limit
            )
            all_competitors.extend(comps)

        competitors = all_competitors[:limit]

    # Fetch plugin details for each competitor
    results: list[CompetitorWithPlugin] = []
    for comp in competitors:
        plugin = crud_arena.get_plugin_by_id(session=session, plugin_id=comp.plugin_id)
        if plugin:
            comp_dict = comp.model_dump()
            comp_dict["plugin"] = PluginPublic.model_validate(plugin)
            results.append(CompetitorWithPlugin.model_validate(comp_dict))

    return results


@router.get(
    "/{competitor_id}",
    response_model=CompetitorWithPlugin,
    summary="Get competitor details",
)
def get_competitor(
    *,
    session: SessionDep,
    competitor_id: str,
    _admin: AdminUser,
) -> CompetitorWithPlugin:
    """
    Get detailed information about a specific competitor.

    **Admin only.**
    """
    from uuid import UUID

    try:
        competitor_uuid = UUID(competitor_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid competitor ID format")

    competitor = crud_arena.get_competitor_by_id(
        session=session, competitor_id=competitor_uuid
    )
    if not competitor:
        raise HTTPException(status_code=404, detail="Competitor not found")

    plugin = crud_arena.get_plugin_by_id(session=session, plugin_id=competitor.plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail="Associated plugin not found")

    comp_dict = competitor.model_dump()
    comp_dict["plugin"] = PluginPublic.model_validate(plugin)
    return CompetitorWithPlugin.model_validate(comp_dict)
