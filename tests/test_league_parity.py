"""Drift guard: every registry league must be wired into the frontend.

A new ``registry/competitors/<modality>/`` directory is only half a league —
the static frontend keeps its own modality maps.  This test parses those maps
out of the Astro pages (regex extraction over the inline scripts) and asserts
each registry modality appears in every one of them, so adding a league
without updating the UI fails CI instead of silently hiding fighters.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
COMPETITORS_DIR = REPO_ROOT / "registry" / "competitors"
PAGES_DIR = REPO_ROOT / "frontend-static" / "src" / "pages"


def registry_modalities() -> set:
    """Every modality that has at least one competitor definition."""
    return {
        d.name for d in COMPETITORS_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.json"))
    }


def _extract_js_map_keys(source: str, name: str) -> set:
    """Keys of an inline ``const <name> = { key: …, 'key': … }`` object."""
    m = re.search(rf"const\s+{name}\s*=\s*\{{(.*?)\}}", source, re.DOTALL)
    assert m, f"could not find 'const {name} = {{…}}'"
    return set(re.findall(r"['\"]?([A-Za-z0-9_]+)['\"]?\s*:", m.group(1)))


def _extract_js_array(source: str, name: str) -> set:
    """Elements of an inline ``const <name> = [ 'a', 'b', … ]`` array."""
    m = re.search(rf"const\s+{name}\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert m, f"could not find 'const {name} = […]'"
    return set(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", m.group(1)))


class TestLeagueParity:
    def test_registry_has_leagues(self):
        assert registry_modalities()

    def test_leaderboard_league_order(self):
        source = (PAGES_DIR / "leaderboard" / "index.astro").read_text()
        leagues = _extract_js_array(source, "LEAGUE_ORDER")
        missing = registry_modalities() - leagues
        assert not missing, f"LEAGUE_ORDER (leaderboard) missing: {missing}"

    def test_leaderboard_battle_group(self):
        source = (PAGES_DIR / "leaderboard" / "index.astro").read_text()
        groups = _extract_js_map_keys(source, "BATTLE_GROUP")
        missing = registry_modalities() - groups
        assert not missing, f"BATTLE_GROUP (leaderboard) missing: {missing}"

    def test_battle_modality_labels(self):
        source = (PAGES_DIR / "battle" / "index.astro").read_text()
        labels = _extract_js_map_keys(source, "MODALITY_LABELS")
        missing = registry_modalities() - labels
        assert not missing, f"MODALITY_LABELS (battle) missing: {missing}"

    def test_fighters_modality_labels(self):
        source = (PAGES_DIR / "fighters" / "index.astro").read_text()
        labels = _extract_js_map_keys(source, "MODALITY_LABELS")
        missing = registry_modalities() - labels
        assert not missing, f"MODALITY_LABELS (fighters) missing: {missing}"

    @pytest.mark.parametrize("page", ["battle", "fighters"])
    def test_pages_exist(self, page):
        assert (PAGES_DIR / page / "index.astro").exists()
