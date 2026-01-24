import sentry_sdk
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
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


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize Kafka producer on application startup."""
    try:
        await kafka_producer.start()
    except Exception as e:
        # Log error but don't crash - app can still serve API
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Failed to start Kafka producer: %s", e)
        logger.warning("API will continue but Kafka publishing will fail")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup Kafka producer on application shutdown."""
    try:
        await kafka_producer.stop()
    except Exception:
        pass  # Ignore errors during shutdown
