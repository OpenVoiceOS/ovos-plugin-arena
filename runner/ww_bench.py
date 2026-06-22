"""
Wake-word benchmark adapter (§3.2) for :mod:`runner.media_bench`.

Streams labelled clips (wake word present vs absent), feeds each through the
registry fighter's real OVOS hotword engine frame by frame exactly as the
listening loop does, and records the binary detection decision against the
ground-truth label.  The arena scores detection error / false-accept /
false-reject from these rows; the plugin owns its own threshold.

Three corpus layouts are supported:

- **audiofolder** (one folder per phrase, e.g. ``OpenVoiceOS/synthetic-wakewords``):
  the dataset names its positive folder via ``wakeword``; every other folder is
  a negative (other wake phrases are strong adversarial hard negatives).
- **manifest** (``manifest.jsonl`` beside the audio, the ww-bench layout): the
  ``role`` field labels each clip.
- **parquet** (audio + a binary ``label`` column).
"""
from __future__ import annotations

import inspect
import logging
import time
from typing import Iterator, Tuple

from runner.audio_io import (
    stream_audio_dataset,
    stream_audiofolder_ww,
    stream_manifest_audio,
    stream_metadata_csv_ww,
)
from runner.media_bench import (
    MediaBenchAdapter,
    PredictContext,
    load_plugin_class,
)

log = logging.getLogger("ww-bench")

FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz, the OVOS listener chunk size
PRIME_SECONDS = 0.7   # leading silence that warms streaming feature buffers
TAIL_SECONDS = 0.3    # trailing silence so the activation can settle
SAMPLE_RATE = 16000


def _apply_hotword_compat() -> None:
    """Let hotword plugins written for a newer plugin-manager load here.

    Recent wake-word plugins call ``super().__init__(key_phrase, config, lang)``;
    older ``HotWordEngine`` bases accept only ``(key_phrase, config)``.  Widen
    the base signature to ignore the extra argument.  A no-op when the installed
    base already accepts ``lang``.
    """
    from ovos_plugin_manager.templates import hotwords as hw

    base = hw.HotWordEngine
    if "lang" in inspect.signature(base.__init__).parameters:
        return
    _orig = base.__init__

    def _compat(self, key_phrase="hey_mycroft", config=None, lang=None,
                *args, **kwargs):
        _orig(self, key_phrase, config)

    base.__init__ = _compat


class WakeWordBench(MediaBenchAdapter):
    modality = "wake_word"
    card_tags = ("keyword-spotting", "wake-word")
    card_task = "Per-clip detection decisions"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[Tuple[str, dict]]:
        source = dataset_def.source
        fields = dataset_def.reference_fields or {}
        if getattr(dataset_def, "wakeword", None):
            # audiofolder: positive folder vs the rest. max_samples caps each
            # class, so a battle pool is balanced. A metadata.csv corpus lists
            # clips in a CSV (folder tree may be too large to enumerate).
            negs = getattr(dataset_def, "negative_dirs", None)
            if (source.file_pattern or "").endswith(".csv"):
                yield from stream_metadata_csv_ww(
                    source, wakeword=dataset_def.wakeword, negative_labels=negs,
                    revision=revision, max_per_class=max_samples,
                    audio_col=fields.get("audio", "file_name"),
                    label_col=fields.get("label", "label"))
            else:
                yield from stream_audiofolder_ww(
                    source, wakeword=dataset_def.wakeword, negative_dirs=negs,
                    revision=revision, max_per_class=max_samples)
            return
        audio_key = fields.get("audio", "audio")
        label_col = fields.get("label", "label")
        if getattr(source, "file_pattern", None):
            yield from stream_manifest_audio(
                source, audio_key=audio_key,
                extra_keys={"label": label_col}, revision=revision,
                max_samples=max_samples)
        else:
            yield from stream_audio_dataset(
                source, audio_key=audio_key,
                extra_keys={"label": label_col}, revision=revision,
                max_samples=max_samples)

    def load_engine(self, competitor, lang: str):
        from ovos_plugin_manager.wakewords import load_wake_word_plugin

        _apply_hotword_compat()
        hotwords = competitor.config.get("hotwords", {})
        # one hotword block per wake-word fighter; the key is the phrase id
        # (underscored), which engines like openWakeWord match against their
        # pretrained model filenames — pass it through unchanged.
        key_phrase, hw_cfg = next(iter(hotwords.items()), ("hey_mycroft", {}))
        module = hw_cfg.get("module") or competitor.plugin
        clazz = load_plugin_class(load_wake_word_plugin, module)
        return clazz(key_phrase, dict(hw_cfg))

    def predict(self, engine, sample: dict, ctx: PredictContext) -> dict:
        start = time.perf_counter()
        detected = _detect(engine, sample["array"])
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "label": _norm_label(sample.get("label")),
            "prediction": "detected" if detected else "not_detected",
            "audio_url": sample.get("audio_url"),  # playable source clip in battles
            "latency_ms": round(latency_ms, 3),
        }


def _prime_pad(array):
    """Wrap a clip in leading + trailing silence.

    Streaming wake-word engines build mel/feature buffers over time; a cold
    engine fed an isolated clip misses activations.  Leading silence primes the
    buffers exactly as continuous mic audio would, the way a real listener sees
    it (mirrors ovoscope's file-driven listener); trailing silence lets a late
    activation settle.
    """
    import numpy as np

    arr = np.asarray(array, dtype="float32")
    lead = np.zeros(int(SAMPLE_RATE * PRIME_SECONDS), dtype="float32")
    tail = np.zeros(int(SAMPLE_RATE * TAIL_SECONDS), dtype="float32")
    return np.concatenate([lead, arr, tail])


def _detect(engine, array) -> bool:
    """Run one clip through a hotword engine, tolerant of both plugin APIs.

    Some engines expose ``update(chunk_bytes)`` then ``found_wake_word()``;
    others do everything in ``found_wake_word(frame)`` taking an int16 frame.
    The clip is primed with silence first so streaming engines activate.
    """
    array = _prime_pad(array)
    if hasattr(engine, "reset"):
        try:
            engine.reset()
        except Exception:
            pass
    takes_frame = len(inspect.signature(engine.found_wake_word).parameters) >= 1
    if takes_frame:
        frames = _int16_frames(array)
        for frame in frames:
            if engine.found_wake_word(frame):
                return True
        return False
    pcm = _to_pcm16(array)
    for off in range(0, len(pcm), FRAME_SAMPLES * 2):
        engine.update(pcm[off:off + FRAME_SAMPLES * 2])
        if engine.found_wake_word():
            return True
    return False


def _to_pcm16(array) -> bytes:
    """Float32 [-1, 1] mono array → 16-bit little-endian PCM bytes."""
    import numpy as np

    arr = np.clip(np.asarray(array, dtype="float32"), -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _int16_frames(array):
    """Float32 mono array → list of int16 numpy frames of FRAME_SAMPLES each."""
    import numpy as np

    arr = (np.clip(np.asarray(array, dtype="float32"), -1.0, 1.0)
           * 32767.0).astype("<i2")
    return [arr[i:i + FRAME_SAMPLES]
            for i in range(0, len(arr), FRAME_SAMPLES)]


def _norm_label(raw) -> str:
    """Map a dataset label to ``positive`` / ``negative``."""
    from arena.metrics import _ww_is_positive

    val = _ww_is_positive(raw)
    return "positive" if val else "negative"
