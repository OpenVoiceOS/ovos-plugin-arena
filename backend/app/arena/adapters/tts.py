"""
TTS adapter for the OVOS Plugin Arena.

Loads any installed OVOS TTS plugin, synthesises a fixed prompt set to
audio files, and optionally measures Real-Time Factor (RTF).

RTF = synthesis_wall_time / audio_duration

The adapter accepts any plugin that implements the ``OVOSTTSPlugin``
interface from ``ovos_plugin_manager.templates.tts``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.arena.adapters.base import BaseAdapter
from app.arena.models import PluginFamily

logger = logging.getLogger(__name__)


class TtsAdapter(BaseAdapter):
    """
    Concrete adapter for OVOS TTS plugins.

    Parameters
    ----------
    plugin_name : OPM entry-point name, e.g. "ovos-tts-plugin-phoonnx"
    config      : plugin configuration dict passed straight to the plugin
    lang        : BCP-47 language tag, e.g. "en-us"
    """

    family = PluginFamily.TTS

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
        from ovos_plugin_manager.tts import OVOSTTSFactory

        try:
            self._plugin = OVOSTTSFactory.create(
                {"module": self.plugin_name, **self.config}
            )
            logger.info("Loaded TTS plugin: %s", self.plugin_name)
        except Exception as e:
            logger.error("Failed to load TTS plugin %s: %s", self.plugin_name, e)
            raise

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
        Synthesise *input_ref* (a text prompt) to a WAV file.

        Returns
        -------
        (wav_path, {"rtf": float, "duration_s": float, "chars": int})
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # sanitise prompt to filename
            slug = "".join(c if c.isalnum() else "_" for c in input_ref[:40])
            wav_path = str(Path(output_dir) / f"{slug}.wav")
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = tmp.name
            tmp.close()

        t0 = time.perf_counter()
        try:
            self._plugin.get_tts(input_ref, wav_path)
        except Exception as e:
            logger.warning("TTS synthesis failed for '%s': %s", input_ref[:60], e)
            raise

        elapsed = time.perf_counter() - t0

        # Measure audio duration if scipy/wave is available
        duration = _wav_duration(wav_path)
        rtf = elapsed / duration if duration > 0 else 0.0

        return wav_path, {
            "rtf": round(rtf, 4),
            "duration_s": round(duration, 3),
            "synthesis_time_s": round(elapsed, 3),
            "chars": len(input_ref),
        }


def _wav_duration(path: str) -> float:
    """Return duration in seconds of a WAV file, or 0 on failure."""
    try:
        import wave

        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0
