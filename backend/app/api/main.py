from fastapi import APIRouter

from app.api.routes import arena, competitors, leaderboard, login, private, users, utils
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)

# Arena v1 routes
api_router.include_router(arena.router)
api_router.include_router(leaderboard.router)
api_router.include_router(competitors.router)


if settings.ENVIRONMENT == "local":
    api_router.include_router(private.router)
