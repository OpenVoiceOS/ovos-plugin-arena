from sqlalchemy.orm import DeclarativeBase

from ovos_plugin_arena.db.meta import meta


class Base(DeclarativeBase):
    """Base for all models."""

    metadata = meta
