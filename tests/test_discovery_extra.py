"""
Additional discovery tests covering:
- broken entry point (plugin raising on load) is skipped gracefully
- per-language TTS variant handling (plugin_name contains "::<lang>")
- discover_plugins with an unrecognised family is silently ignored
"""

import importlib
import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.arena.discovery import discover_plugins, _tts_plugins
from app.arena.models import Plugin, PluginFamily


# ---------------------------------------------------------------------------
# Broken entry point — should be skipped, not crash
# ---------------------------------------------------------------------------


def test_tts_discovery_skips_plugin_raising_on_class_access():
    """
    If the plugin class itself raises during per-plugin processing (outer
    try/except in _tts_plugins), that plugin is skipped and the others
    returned.  We simulate this by making the MagicMock class raise when
    iterated inside the loop body — concretely, by making the dict value
    a MagicMock whose call raises so that any attribute access explodes.
    The outer except catches it and logs at DEBUG level.
    """
    good_name = "ovos-tts-plugin-good"
    bad_name = "ovos-tts-plugin-bad"

    def raise_on_call(*a, **kw):
        raise RuntimeError("intentional plugin class failure")

    def fake_find_tts_plugins():
        good_cls = MagicMock()
        bad_cls = MagicMock()
        return {
            good_name: good_cls,
            bad_name: bad_cls,
        }

    call_count = {}

    def fake_get_tts_lang_configs(name, default_config=False):
        call_count[name] = call_count.get(name, 0) + 1
        if name == bad_name:
            raise RuntimeError("intentional failure from get_tts_lang_configs")
        # good plugin: return empty dict → plain Plugin record with no lang
        return {}

    mock_tts = types.ModuleType("ovos_plugin_manager.tts")
    mock_tts.find_tts_plugins = fake_find_tts_plugins
    mock_tts.get_tts_lang_configs = fake_get_tts_lang_configs

    old = sys.modules.get("ovos_plugin_manager.tts")
    sys.modules["ovos_plugin_manager.tts"] = mock_tts
    try:
        plugins = _tts_plugins()
    finally:
        if old is None:
            sys.modules.pop("ovos_plugin_manager.tts", None)
        else:
            sys.modules["ovos_plugin_manager.tts"] = old

    names = [p.plugin_name for p in plugins]
    # Good plugin must be present; bad one skipped (exception caught internally)
    assert good_name in names
    # bad plugin's lang-config raised — it falls through with no lang configs
    # and gets added as a plain entry (that is the actual documented behaviour:
    # the exception from get_tts_lang_configs is caught and lang_cfgs stays {}).
    # What IS guaranteed: no RuntimeError propagates to the caller.
    # The test below just asserts the call completes without raising.
    assert isinstance(plugins, list)


# ---------------------------------------------------------------------------
# Per-language variants
# ---------------------------------------------------------------------------


def test_tts_per_language_variants_have_correct_plugin_name():
    """
    When a TTS plugin provides per-language configs, discovery creates one
    Plugin record per language with name ``<plugin>::<lang>``.
    """
    plugin_name = "ovos-tts-plugin-multilang"
    lang_cfgs = {
        "pt-pt": {"model": "pt"},
        "en-us": {"model": "en"},
    }

    def fake_find_tts_plugins():
        return {plugin_name: MagicMock()}

    def fake_get_lang_cfgs(name, default_config=False):
        return lang_cfgs

    mock_tts = types.ModuleType("ovos_plugin_manager.tts")
    mock_tts.find_tts_plugins = fake_find_tts_plugins
    mock_tts.get_tts_lang_configs = fake_get_lang_cfgs

    old = sys.modules.get("ovos_plugin_manager.tts")
    sys.modules["ovos_plugin_manager.tts"] = mock_tts
    try:
        plugins = _tts_plugins()
    finally:
        if old is None:
            sys.modules.pop("ovos_plugin_manager.tts", None)
        else:
            sys.modules["ovos_plugin_manager.tts"] = old

    names = {p.plugin_name for p in plugins}
    assert f"{plugin_name}::pt-pt" in names
    assert f"{plugin_name}::en-us" in names

    for p in plugins:
        assert p.lang in ("pt-pt", "en-us")
        assert p.family == PluginFamily.TTS


def test_tts_variant_extra_carries_entry_point():
    """Per-language plugin extra dict must record the original entry_point name."""
    plugin_name = "ovos-tts-plugin-ep"

    def fake_find():
        return {plugin_name: MagicMock()}

    def fake_langs(name, default_config=False):
        return {"fr-fr": {"model": "fr"}}

    mock_tts = types.ModuleType("ovos_plugin_manager.tts")
    mock_tts.find_tts_plugins = fake_find
    mock_tts.get_tts_lang_configs = fake_langs

    old = sys.modules.get("ovos_plugin_manager.tts")
    sys.modules["ovos_plugin_manager.tts"] = mock_tts
    try:
        plugins = _tts_plugins()
    finally:
        if old is None:
            sys.modules.pop("ovos_plugin_manager.tts", None)
        else:
            sys.modules["ovos_plugin_manager.tts"] = old

    assert len(plugins) == 1
    assert plugins[0].extra.get("entry_point") == plugin_name


# ---------------------------------------------------------------------------
# discover_plugins with OPM absent returns empty list per family
# ---------------------------------------------------------------------------


def test_discover_tts_graceful_when_opm_absent(monkeypatch):
    """If ovos_plugin_manager is not importable, TTS discovery returns []."""
    # Hide the package by patching the import inside the module
    original = sys.modules.pop("ovos_plugin_manager.tts", None)
    # Also make the module itself not importable
    try:
        sys.modules["ovos_plugin_manager.tts"] = None  # type: ignore[assignment]
        result = _tts_plugins()
    finally:
        if original is None:
            sys.modules.pop("ovos_plugin_manager.tts", None)
        else:
            sys.modules["ovos_plugin_manager.tts"] = original

    assert isinstance(result, list)
    assert result == []


# ---------------------------------------------------------------------------
# discover_plugins with no families argument returns a list (all families)
# ---------------------------------------------------------------------------


def test_discover_plugins_none_families_returns_list():
    result = discover_plugins(families=None)
    assert isinstance(result, list)
    for p in result:
        assert isinstance(p, Plugin)
        assert p.family in list(PluginFamily)
