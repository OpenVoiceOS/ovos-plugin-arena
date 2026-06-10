"""
OPM-based plugin discovery for the OVOS Plugin Arena.

Scans installed entry points via ovos-plugin-manager and returns Plugin
records ready to be upserted into the arena database.

Supported families
------------------
* TTS   — opm.tts / mycroft.plugin.tts
* STT   — opm.stt / mycroft.plugin.stt
* WW    — opm.wake_word / mycroft.plugin.wake_word
* Intent — opm.intent / padatious (fallback scan)

The function ``discover_plugins`` is the main public API.  It returns a
list of ``Plugin`` objects and does NOT write to the database itself;
callers decide whether to upsert.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from app.arena.models import Plugin, PluginFamily

logger = logging.getLogger(__name__)


def _config_hash(config: Dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _tts_plugins() -> List[Plugin]:
    try:
        from ovos_plugin_manager.tts import find_tts_plugins, get_tts_lang_configs

        plugins = []
        for name, cls in find_tts_plugins().items():
            try:
                # Try to get language-specific configs
                lang_cfgs: Dict[str, Any] = {}
                try:
                    lang_cfgs = get_tts_lang_configs(name, default_config=True)
                except Exception:
                    pass

                if lang_cfgs:
                    # One Plugin record per language variant
                    for lang, cfg in lang_cfgs.items():
                        base_cfg = cfg if isinstance(cfg, dict) else {}
                        plugins.append(
                            Plugin(
                                plugin_name=f"{name}::{lang}",
                                display_name=f"{name} ({lang})",
                                family=PluginFamily.TTS,
                                lang=lang,
                                config=base_cfg,
                                config_hash=_config_hash(base_cfg),
                                extra={"entry_point": name},
                            )
                        )
                else:
                    plugins.append(
                        Plugin(
                            plugin_name=name,
                            display_name=name,
                            family=PluginFamily.TTS,
                            extra={"entry_point": name},
                        )
                    )
            except Exception as e:
                logger.debug("Skipping TTS plugin %s: %s", name, e)

        return plugins
    except ImportError:
        logger.warning("ovos-plugin-manager not available — TTS discovery skipped")
        return []


def _stt_plugins() -> List[Plugin]:
    try:
        from ovos_plugin_manager.stt import find_stt_plugins, get_stt_lang_configs

        plugins = []
        for name, cls in find_stt_plugins().items():
            try:
                lang_cfgs: Dict[str, Any] = {}
                try:
                    lang_cfgs = get_stt_lang_configs(name, default_config=True)
                except Exception:
                    pass

                if lang_cfgs:
                    for lang, cfg in lang_cfgs.items():
                        base_cfg = cfg if isinstance(cfg, dict) else {}
                        plugins.append(
                            Plugin(
                                plugin_name=f"{name}::{lang}",
                                display_name=f"{name} ({lang})",
                                family=PluginFamily.STT,
                                lang=lang,
                                config=base_cfg,
                                config_hash=_config_hash(base_cfg),
                                extra={"entry_point": name},
                            )
                        )
                else:
                    plugins.append(
                        Plugin(
                            plugin_name=name,
                            display_name=name,
                            family=PluginFamily.STT,
                            extra={"entry_point": name},
                        )
                    )
            except Exception as e:
                logger.debug("Skipping STT plugin %s: %s", name, e)

        return plugins
    except ImportError:
        logger.warning("ovos-plugin-manager not available — STT discovery skipped")
        return []


def _ww_plugins() -> List[Plugin]:
    try:
        from ovos_plugin_manager.wakewords import find_wake_word_plugins

        plugins = []
        for name, cls in find_wake_word_plugins().items():
            plugins.append(
                Plugin(
                    plugin_name=name,
                    display_name=name,
                    family=PluginFamily.WAKE_WORD,
                    extra={"entry_point": name},
                )
            )
        return plugins
    except ImportError:
        logger.warning("ovos-plugin-manager not available — WW discovery skipped")
        return []


def _intent_plugins() -> List[Plugin]:
    """
    Intent plugins are discovered from the ``opm.intent`` entry point group.
    Falls back to scanning for known adapters (padatious, adapt).
    """
    plugins: List[Plugin] = []

    # Primary: OPM entry points
    try:
        import importlib.metadata as importlib_metadata

        eps = importlib_metadata.entry_points(group="opm.intent")
        for ep in eps:
            plugins.append(
                Plugin(
                    plugin_name=ep.name,
                    display_name=ep.name,
                    family=PluginFamily.INTENT,
                    extra={"entry_point": ep.value},
                )
            )
    except Exception as e:
        logger.debug("OPM intent EP scan failed: %s", e)

    # Fallback: check for well-known intent engines
    fallbacks = [
        ("padatious", "padatious.intent_container", "IntentContainer"),
        ("ovos-adapt-plugin", "ovos_adapt_plugin", "AdaptExtractor"),
    ]
    for name, module, cls_name in fallbacks:
        if any(p.plugin_name == name for p in plugins):
            continue
        try:
            mod = __import__(module, fromlist=[cls_name])
            getattr(mod, cls_name)
            plugins.append(
                Plugin(
                    plugin_name=name,
                    display_name=name,
                    family=PluginFamily.INTENT,
                    extra={"fallback_detection": True},
                )
            )
        except (ImportError, AttributeError):
            pass

    return plugins


def discover_plugins(
    families: Optional[List[PluginFamily]] = None,
) -> List[Plugin]:
    """
    Discover all installed OVOS plugins for the requested families.

    Parameters
    ----------
    families : list of PluginFamily to scan; None = all four families

    Returns
    -------
    List of Plugin objects (not yet persisted)
    """
    if families is None:
        families = list(PluginFamily)

    results: List[Plugin] = []

    scanners = {
        PluginFamily.TTS: _tts_plugins,
        PluginFamily.STT: _stt_plugins,
        PluginFamily.WAKE_WORD: _ww_plugins,
        PluginFamily.INTENT: _intent_plugins,
    }

    for family in families:
        scanner = scanners.get(family)
        if scanner:
            found = scanner()
            logger.info("Discovered %d %s plugins", len(found), family.value)
            results.extend(found)

    return results
