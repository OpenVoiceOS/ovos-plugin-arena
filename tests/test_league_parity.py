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


def _default_leagues_block(source: str) -> str:
    """The inline ``const DEFAULT_LEAGUES = [ … ];`` array body."""
    m = re.search(r"const\s+DEFAULT_LEAGUES\s*=\s*\[(.*?)\];", source, re.DOTALL)
    assert m, "could not find 'const DEFAULT_LEAGUES = […]'"
    return m.group(1)


class TestLeagueParity:
    def test_registry_has_leagues(self):
        assert registry_modalities()

    def test_no_hardcoded_league_fallback(self):
        """The league list is data. A page that ships its own copy goes stale
        the moment a league is added, renamed or left unswept — which is how
        a tab for an empty league reached the site."""
        for page in ("leaderboard", "matchups"):
            source = (PAGES_DIR / page / "index.astro").read_text()
            assert _default_leagues_block(source).strip() == "", page

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
