"""
STT adapter stub for the OVOS Plugin Arena.

Runs an OVOS STT plugin over a labelled audio clip and computes
Word Error Rate (WER) against a reference transcript.

Status: STUB — interface complete, compute path stubbed pending a
concrete STT engine install in the arena environment.

WER formula
-----------
WER = (S + D + I) / N
where S=substitutions, D=deletions, I=insertions, N=reference word count.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.arena.adapters.base import BaseAdapter
from app.arena.models import PluginFamily

logger = logging.getLogger(__name__)


class SttAdapter(BaseAdapter):
    """
    Stub adapter for OVOS STT plugins.

    Parameters
    ----------
    plugin_name : e.g. "ovos-stt-plugin-whisper"
    config      : plugin configuration dict
    lang        : language tag
    """

    family = PluginFamily.STT

    def __init__(
        self,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None,
        lang: str = "en-us",
    ) -> None:
        super().__init__(plugin_name, config)
        self.lang = lang
        self._plugin: Any = None

    def _load_plugin(self) -> None:
        from ovos_plugin_manager.stt import OVOSSTTFactory

        self._plugin = OVOSSTTFactory.create(
            {"module": self.plugin_name, **self.config}
        )
        logger.info("Loaded STT plugin: %s", self.plugin_name)

    def _unload_plugin(self) -> None:
        if self._plugin is not None:
            try:
                self._plugin.shutdown()
            except Exception:
                pass
            self._plugin = None

    def _run_one(
        self,
        input_ref: str,
        output_dir: Optional[str],
    ) -> tuple[Optional[str], Dict[str, float]]:
        """
        Transcribe *input_ref* (path to a WAV file or JSON with reference).

        Expected *input_ref* format::

            {"audio_path": "…/clip.wav", "reference": "the reference transcript"}

        Returns
        -------
        (None, {"wer": float, "cer": float, "confidence": float})
        """
        import json as _json

        try:
            data = _json.loads(input_ref)
            audio_path = data["audio_path"]
            reference = data.get("reference", "")
        except Exception:
            audio_path = input_ref
            reference = ""

        # TODO: call self._plugin.execute(audio_data, language=self.lang)
        # Stubbed — return placeholder metrics
        logger.warning(
            "SttAdapter._run_one is stubbed; returning placeholder metrics for %s",
            audio_path,
        )
        return None, {"wer": -1.0, "cer": -1.0, "confidence": 0.0, "stub": 1.0}
