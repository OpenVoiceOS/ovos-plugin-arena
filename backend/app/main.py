import sentry_sdk
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.arena.router import router as arena_core_router
from app.core.config import settings
from app.services.kafka_producer import kafka_producer


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

# Arena core — self-contained SQLite evaluation engine (P1 + P2)
# Accessible at /api/v1/arena/…  (no auth, no Kafka, no PostgreSQL required)
app.include_router(arena_core_router, prefix=f"{settings.API_V1_STR}/arena")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize arena SQLite DB and Kafka producer on startup."""
    import logging
    from pathlib import Path

    from app.arena import db as arena_db

    logger = logging.getLogger(__name__)

    # Arena SQLite — zero-infra, always available
    db_path = Path(settings.ARENA_DB_PATH) if hasattr(settings, "ARENA_DB_PATH") else Path("arena.sqlite3")
    arena_db.init_db(path=db_path)
    logger.info("Arena SQLite DB initialised at %s", db_path)

    try:
        await kafka_producer.start()
    except Exception as e:
        # Log error but don't crash - app can still serve API
        logger.error("Failed to start Kafka producer: %s", e)
        logger.warning("API will continue but Kafka publishing will fail")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup Kafka producer on application shutdown."""
    try:
        await kafka_producer.stop()
    except Exception:
        pass  # Ignore errors during shutdown
