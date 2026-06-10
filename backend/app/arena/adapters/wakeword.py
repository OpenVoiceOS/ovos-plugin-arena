"""
Wake-word adapter stub for the OVOS Plugin Arena.

Runs an OVOS wake-word plugin over a labelled audio clip and computes
detection F1 (true positives vs false positives/negatives).

Status: STUB — interface complete, compute path stubbed pending labelled
audio clips and a concrete WW engine install in the arena environment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.arena.adapters.base import BaseAdapter
from app.arena.models import PluginFamily

logger = logging.getLogger(__name__)


class WakeWordAdapter(BaseAdapter):
    """
    Stub adapter for OVOS wake-word plugins.

    Parameters
    ----------
    plugin_name : e.g. "ovos-ww-plugin-precise-onnx"
    config      : plugin configuration dict (must include "wake_word" key)
    """

    family = PluginFamily.WAKE_WORD

    def __init__(
        self,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(plugin_name, config)
        self._plugin: Any = None

    def _load_plugin(self) -> None:
        from ovos_plugin_manager.wakewords import OVOSWakeWordFactory

        self._plugin = OVOSWakeWordFactory.create(
            {"module": self.plugin_name, **self.config}
        )
        logger.info("Loaded WW plugin: %s", self.plugin_name)

    def _unload_plugin(self) -> None:
        if self._plugin is not None:
            try:
                self._plugin.stop()
            except Exception:
                pass
            self._plugin = None

    def _run_one(
        self,
        input_ref: str,
        output_dir: Optional[str],
    ) -> tuple[Optional[str], Dict[str, float]]:
        """
        Check *input_ref* (path to labelled audio clip JSON) for wake word.

        Expected *input_ref* format::

            {"audio_path": "…/clip.wav", "label": 1}  # label 1=positive, 0=negative

        Returns
        -------
        (None, {"detected": 0|1, "label": 0|1, "tp": …, "fp": …, "fn": …})
        """
        logger.warning(
            "WakeWordAdapter._run_one is stubbed; returning placeholder metrics"
        )
        return None, {"detected": -1.0, "label": -1.0, "stub": 1.0}
