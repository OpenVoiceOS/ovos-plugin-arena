"""
Tests for OPM-based plugin discovery (arena.discovery).

These tests do NOT require any specific plugin to be installed.  They
verify that the discovery functions run without crashing, return Plugin
objects with the expected structure, and that the family filter works.
"""

import pytest

from app.arena.discovery import discover_plugins
from app.arena.models import Plugin, PluginFamily


def test_discover_returns_list():
    plugins = discover_plugins()
    assert isinstance(plugins, list)


def test_discover_all_items_are_plugin_instances():
    plugins = discover_plugins()
    for p in plugins:
        assert isinstance(p, Plugin)


def test_discover_families_filter():
    tts_only = discover_plugins(families=[PluginFamily.TTS])
    for p in tts_only:
        assert p.family == PluginFamily.TTS


def test_discover_tts_has_plugin_name():
    tts = discover_plugins(families=[PluginFamily.TTS])
    for p in tts:
        assert p.plugin_name
        assert p.display_name


def test_discover_intent_includes_padatious_if_installed():
    """
    If padatious is installed (it is in the shared venv), the intent
    discovery should find it via the fallback scanner.
    """
    try:
        import padatious  # noqa: F401
        padatious_available = True
    except ImportError:
        padatious_available = False

    if not padatious_available:
        pytest.skip("padatious not installed")

    intent_plugins = discover_plugins(families=[PluginFamily.INTENT])
    names = [p.plugin_name for p in intent_plugins]
    assert "padatious" in names


def test_discover_empty_families_list():
    """Empty families list should return nothing."""
    plugins = discover_plugins(families=[])
    assert plugins == []
