"""
VAD benchmark adapter (§3.2) for :mod:`runner.media_bench`.

Streams labelled clips (speech vs non-speech), feeds each through the registry
fighter's real OVOS ``VADEngine`` frame by frame, and records a per-clip
speech/silence decision against the ground-truth label. The arena scores both
error directions — false-accept (firing *speech* on non-speech: music, noise)
and false-reject (missing real speech) — exactly as the wake-word league does,
since VAD is the same binary-detection task.

The fighter owns its own threshold; the arena records only the decision and
latency. A clip counts as *speech* when any frame is voiced.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator, Tuple

from runner.audio_io import stream_vad
from runner.media_bench import (
    MediaBenchAdapter,
    PredictContext,
    load_plugin_class,
)

log = logging.getLogger("vad-bench")

FRAME_SAMPLES = 480   # 30 ms @ 16 kHz — accepted by webrtcvad and handled by silero
SAMPLE_RATE = 16000


class VADBench(MediaBenchAdapter):
    modality = "vad"
    card_tags = ("voice-activity-detection", "vad")
    card_task = "Per-clip speech / non-speech decisions"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[Tuple[str, dict]]:
        yield from stream_vad(dataset_def, revision, max_per_class=max_samples)

    def load_engine(self, competitor, lang: str):
        from ovos_plugin_manager.vad import load_vad_plugin

        cfg = dict(competitor.config.get("VAD", {})
                   or competitor.config.get("listener", {}).get("VAD", {}))
        module = cfg.get("module") or competitor.plugin
        clazz = load_plugin_class(load_vad_plugin, module)
        return clazz(cfg)

    def predict(self, engine, sample: dict, ctx: PredictContext) -> dict:
        start = time.perf_counter()
        speech = _has_speech(engine, sample["array"])
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "label": _norm_label(sample.get("label")),
            "prediction": "speech" if speech else "silence",
            "audio_url": sample.get("audio_url"),  # playable source clip in battles
            "latency_ms": round(latency_ms, 3),
        }


def _has_speech(engine, array) -> bool:
    """Stream a clip frame by frame; True if the VAD calls any frame voiced.

    The OVOS contract is ``is_silence(chunk) -> bool`` per frame, exactly how
    the listening loop end-points speech. A clip is speech if at least one frame
    is non-silent.
    """
    if hasattr(engine, "reset"):
        try:
            engine.reset()
        except Exception:
            pass
    pcm = _to_pcm16(array)
    step = FRAME_SAMPLES * 2  # 2 bytes / sample (int16)
    voiced = False
    for off in range(0, len(pcm) - step + 1, step):
        chunk = pcm[off:off + step]
        try:
            if not engine.is_silence(chunk):
                voiced = True
                break
        except Exception as exc:
            log.debug("vad frame failed: %s", exc)
    return voiced


def _to_pcm16(array) -> bytes:
    """Float32 [-1, 1] mono array → 16-bit little-endian PCM bytes."""
    import numpy as np

    arr = np.clip(np.asarray(array, dtype="float32"), -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _norm_label(raw) -> str:
    """Map a dataset label to ``speech`` / ``non_speech``."""
    from arena.metrics import _ww_is_positive

    return "speech" if _ww_is_positive(raw) else "non_speech"
