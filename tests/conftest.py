"""
Shared pytest fixtures for the OVOS Plugin Arena tests.
"""

import sys
from pathlib import Path

# Ensure the repo root is importable without installing the package
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
