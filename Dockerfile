# Multi-stage build: Astro static site + FastAPI backend
# Works with legacy Docker builder (no BuildKit required).

# Stage 1: build the Astro frontend
FROM node:22-slim AS frontend-build
WORKDIR /frontend
COPY frontend-static/package*.json ./
RUN npm ci
COPY frontend-static/ ./
ARG VITE_BACKEND_URL=/api/v1
ARG ASTRO_SITE=http://localhost:8000
ARG ASTRO_BASE=/
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}
ENV ASTRO_SITE=${ASTRO_SITE}
ENV ASTRO_BASE=${ASTRO_BASE}
ENV VITE_GITHUB_REPO=OpenVoiceOS/ovos-plugin-arena
RUN npm run build

# Stage 2: Python backend serving the static frontend
FROM python:3.11-slim AS backend

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONPATH=/app

# Copy project files for dependency install
COPY backend/pyproject.toml backend/uv.lock backend/alembic.ini ./

# Install deps (no BuildKit cache mounts needed)
RUN uv sync --frozen --no-install-project

# Copy application source
COPY backend/app ./app
COPY backend/tests ./tests
COPY backend/scripts ./scripts

# Final sync (installs the project itself)
RUN uv sync

# Copy built static frontend
COPY --from=frontend-build /frontend/dist ./static

# Mount point for SQLite vote database
VOLUME ["/data"]

EXPOSE 8000

CMD ["fastapi", "run", "--workers", "4", "app/main.py"]
