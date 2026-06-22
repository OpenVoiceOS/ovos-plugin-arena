"""
Wake-word benchmark adapter (§3.2) for :mod:`runner.media_bench`.

Streams labelled clips (wake word present vs absent), feeds each through the
registry fighter's real OVOS hotword engine frame by frame exactly as the
listening loop does, and records the binary detection decision against the
ground-truth label.  The arena scores detection error / false-accept /
false-reject from these rows; the plugin owns its own threshold.

The eval dataset's ``label`` column MUST be binary — ``1``/``positive``/the
wake phrase for clips that contain the wake word, ``0``/``negative`` for the
rest — so ground truth is competitor-independent.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator, Tuple

from runner.audio_io import stream_audio_dataset, stream_manifest_audio
from runner.media_bench import (
    MediaBenchAdapter,
    PredictContext,
    load_plugin_class,
)

log = logging.getLogger("ww-bench")

FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz, the OVOS listener chunk size


class WakeWordBench(MediaBenchAdapter):
    modality = "wake_word"
    card_tags = ("keyword-spotting", "wake-word")
    card_task = "Per-clip detection decisions"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[Tuple[str, dict]]:
        fields = dataset_def.reference_fields or {}
        audio_key = fields.get("audio", "audio")
        label_col = fields.get("label", "label")
        source = dataset_def.source
        # ww-bench ships a per-sample manifest.jsonl beside the audio files;
        # plain HF audio corpora carry the clip + label in parquet columns.
        if getattr(source, "file_pattern", None):
            stream = stream_manifest_audio(
                source, audio_key=audio_key,
                extra_keys={"label": label_col}, revision=revision,
                max_samples=max_samples)
        else:
            stream = stream_audio_dataset(
                source, audio_key=audio_key,
                extra_keys={"label": label_col}, revision=revision,
                max_samples=max_samples)
        yield from stream

    def load_engine(self, competitor, lang: str):
        from ovos_plugin_manager.wakewords import load_wake_word_plugin

        hotwords = competitor.config.get("hotwords", {})
        # one hotword block per wake-word fighter
        key_phrase, hw_cfg = next(iter(hotwords.items()), ("hey_mycroft", {}))
        module = hw_cfg.get("module") or competitor.plugin
        clazz = load_plugin_class(load_wake_word_plugin, module)
        return clazz(key_phrase.replace("_", " "), dict(hw_cfg))

    def predict(self, engine, sample: dict, ctx: PredictContext) -> dict:
        pcm = _to_pcm16(sample["array"])
        engine.reset()
        start = time.perf_counter()
        detected = False
        for off in range(0, len(pcm), FRAME_SAMPLES * 2):
            engine.update(pcm[off:off + FRAME_SAMPLES * 2])
            if engine.found_wake_word():
                detected = True
                break
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "label": _norm_label(sample.get("label")),
            "prediction": "detected" if detected else "not_detected",
            "audio_url": sample.get("audio_url"),  # playable source clip in battles
            "latency_ms": round(latency_ms, 3),
        }


def _to_pcm16(array) -> bytes:
    """Float32 [-1, 1] mono array → 16-bit little-endian PCM bytes."""
    import numpy as np

    arr = np.asarray(array, dtype="float32")
    if arr.size and arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _norm_label(raw) -> str:
    """Map a dataset label to ``positive`` / ``negative``."""
    from arena.metrics import _ww_is_positive

    val = _ww_is_positive(raw)
    if val is None:
        return "negative"
    return "positive" if val else "negative"
