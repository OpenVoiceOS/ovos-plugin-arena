"""
Shared pytest fixtures for the OVOS Plugin Arena tests.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

# Ensure backend/app is importable without installing the package
BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture()
def tmp_db(tmp_path):
    """
    Provide an isolated SQLite arena database for each test.

    Yields the Path to the DB file.  The arena.db module is reconfigured
    to point at this path before the test and restored after.
    """
    from app.arena import db as arena_db

    db_path = tmp_path / "test_arena.sqlite3"
    original_path = arena_db._DB_PATH
    arena_db.init_db(path=db_path)
    yield db_path
    # Restore (important when tests run in-process sequentially)
    arena_db._DB_PATH = original_path
