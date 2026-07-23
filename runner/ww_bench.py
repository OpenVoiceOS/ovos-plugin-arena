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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from runner.audio_io import (
    stream_audio_dataset,
    stream_manifest_audio,
    stream_ww,
)
from runner.media_bench import (
    MediaBenchAdapter,
    PredictContext,
    load_plugin_class,
)

log = logging.getLogger("ww-bench")

FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz, the OVOS listener chunk size
# Leading silence that warms streaming feature buffers. Streaming detectors
# (openWakeWord) only emit predictions once their rolling mel/embedding window
# is full (~2.5 s of frames); a clip fed with too little lead never fills it,
# so the activation is missed (false reject) — and on the shortest clips the
# half-full buffer raises a shape mismatch that drops the sample entirely,
# biasing the false-reject rate. A few seconds of leading silence primes the
# window before the keyword arrives, exactly as a live mic keeps the loop warm.
# Uniform across fighters, so it stays fair-per-task.
PRIME_SECONDS = 3.0
TAIL_SECONDS = 0.5    # trailing silence so a late activation can settle
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


@dataclass
class WWStack:
    """A wake-word fighter as the listener actually stacks it.

    The bare hotword engine, optionally fronted by a **pre-wake VAD** gate (only
    voiced clips ever reach the detector — suppresses false-accepts on
    non-speech) and/or followed by a **verifier** that must confirm an
    activation (e.g. a speaker check). Each distinct (engine, VAD, verifier)
    combination is its own competitor, with its own false-accept / false-reject
    trade-off.
    """

    ww: Any
    vad: Any | None = None
    verifier: Any | None = None


class WakeWordBench(MediaBenchAdapter):
    modality = "wake_word"
    card_tags = ("keyword-spotting", "wake-word")
    card_task = "Per-clip detection decisions"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[tuple[str, dict]]:
        source = dataset_def.source
        fields = dataset_def.reference_fields or {}
        if getattr(dataset_def, "wakeword", None):
            # positives = the wakeword clips; negatives = a not-wake-word corpus
            # (false-accept test) or the same corpus's other phrases. max_samples
            # caps each class so the battle pool stays balanced.
            yield from stream_ww(dataset_def, revision, max_per_class=max_samples)
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

    def load_engine(self, competitor, lang: str) -> WWStack:
        from ovos_plugin_manager.wakewords import load_wake_word_plugin

        _apply_hotword_compat()
        hotwords = competitor.config.get("hotwords", {})
        # one hotword block per wake-word fighter; the key is the phrase id
        # (underscored), which engines like openWakeWord match against their
        # pretrained model filenames — pass it through unchanged.
        key_phrase, hw_cfg = next(iter(hotwords.items()), ("hey_mycroft", {}))
        module = hw_cfg.get("module") or competitor.plugin
        clazz = load_plugin_class(load_wake_word_plugin, module)
        ww = clazz(key_phrase, dict(hw_cfg))
        return WWStack(ww,
                       vad=_load_vad(competitor.config),
                       verifier=_load_verifier(competitor.config))

    def predict(self, stack, sample: dict, ctx: PredictContext) -> dict:
        start = time.perf_counter()
        detected = _detect_stack(stack, sample["array"])
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "label": _norm_label(sample.get("label")),
            "prediction": "detected" if detected else "not_detected",
            "audio_url": sample.get("audio_url"),  # playable source clip in battles
            "latency_ms": round(latency_ms, 3),
        }


def _load_vad(config: dict):
    """Optional pre-wake VAD gate from a fighter config, or None."""
    vad_cfg = config.get("VAD") or (config.get("listener") or {}).get("VAD")
    if not vad_cfg:
        return None
    from ovos_plugin_manager.vad import load_vad_plugin

    module = vad_cfg.get("module")
    clazz = load_plugin_class(load_vad_plugin, module)
    return clazz(dict(vad_cfg))


def _load_verifier(config: dict):
    """Optional hotword verifier from a fighter config, or None."""
    ver_cfg = config.get("hotword_verifier") or config.get("verifier")
    if not ver_cfg:
        return None
    from ovos_plugin_manager.wakewords import load_wake_word_verifier_plugin

    module = ver_cfg.get("module")
    clazz = load_plugin_class(load_wake_word_verifier_plugin, module)
    return clazz(dict(ver_cfg))


def _detect_stack(stack: WWStack, array) -> bool:
    """Run a clip through the full pre-VAD → wake-word → verifier stack.

    Pre-wake VAD gates the clip: if the VAD hears no speech, the detector never
    runs (this is how the stack suppresses false-accepts on non-speech). A
    verifier, if present, must confirm an activation for it to count.
    """
    if stack.vad is not None:
        from runner.vad_bench import _has_speech

        if not _has_speech(stack.vad, array):
            return False
    detected = _detect(stack.ww, array)
    if detected and stack.verifier is not None:
        detected = _verify(stack.verifier, array)
    return detected


def _verify(verifier, array) -> bool:
    """Run a hotword verifier over the primed clip; default to confirm on error."""
    try:
        return bool(verifier.verify(_to_pcm16(_prime_pad(array))))
    except Exception as exc:
        log.debug("verifier failed, accepting activation: %s", exc)
        return True


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
    out = np.concatenate([lead, arr, tail])
    # pad up to a whole number of frames — a short final frame (<25 ms) makes
    # openWakeWord raise and abort the whole clip
    rem = len(out) % FRAME_SAMPLES
    if rem:
        out = np.concatenate([out, np.zeros(FRAME_SAMPLES - rem, dtype="float32")])
    return out


def _detect(engine, array) -> bool:
    """Run one clip through a hotword engine, streaming it as a live mic would.

    The OVOS contract is ``update(chunk_bytes)`` to feed audio, then
    ``found_wake_word()`` to read the latch.  Some plugins (openWakeWord) keep a
    vestigial ``found_wake_word(frame_data)`` argument that they ignore — we
    pass the chunk through so the signature matches, but the detection still
    happens in ``update``.  The clip is primed with silence first so streaming
    engines warm their buffers and activate.
    """
    array = _prime_pad(array)
    if hasattr(engine, "reset"):
        try:
            engine.reset()
        except Exception:
            pass
    fww = engine.found_wake_word
    fww_takes_arg = len(inspect.signature(fww).parameters) >= 1
    has_update = hasattr(engine, "update")
    pcm = _to_pcm16(array)
    for off in range(0, len(pcm), FRAME_SAMPLES * 2):
        chunk = pcm[off:off + FRAME_SAMPLES * 2]
        if has_update:
            engine.update(chunk)
        if (fww(chunk) if fww_takes_arg else fww()):
            return True
    return False


def _to_pcm16(array) -> bytes:
    """Float32 [-1, 1] mono array → 16-bit little-endian PCM bytes."""
    import numpy as np

    arr = np.clip(np.asarray(array, dtype="float32"), -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


class WakeWordStreamBench(WakeWordBench):
    """Continuous-stream wake-word benchmark adapter (§A3.2 / R14).

    Same real OVOS `HotWordEngine` stack as :class:`WakeWordBench`, but each
    sample is one long continuous clip (many onsets, or none — negative-hours
    audio) instead of one isolated labelled clip. The engine runs
    continuously over the whole clip and every activation is recorded as a
    ``(timestamp_s, score)`` event (:func:`_detect_stream`); the arena scores
    those events against the clip's ground-truth onset timestamps
    (:func:`arena.metrics.score_ww_stream`) rather than a single per-clip
    binary decision.

    Only competitors that declare ``"stream"`` in their registry
    ``capabilities`` are eligible here — the rest are clip-only and are
    excluded outright, never zero-scored (see :meth:`filter_competitors`).
    """

    modality = "ww_stream"
    # Fighters live in the wake_word registry pool — same plugin configs,
    # just a different benchmark shape (see WakeWordBench docstring).
    competitor_modality = "wake_word"
    card_tags = ("keyword-spotting", "wake-word", "streaming")
    card_task = "Continuous-stream detection events"

    def filter_competitors(self, competitors: list) -> list:
        return [c for c in competitors if "stream" in (c.capabilities or [])]

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[tuple[str, dict]]:
        fields = dataset_def.reference_fields or {}
        yield from stream_manifest_audio(
            dataset_def.source,
            audio_key=fields.get("audio", "audio"),
            extra_keys={"truth_onsets": fields.get("onsets", "onsets")},
            revision=revision,
            max_samples=max_samples,
        )

    def predict(self, stack: WWStack, sample: dict, ctx: PredictContext) -> dict:
        array = sample["array"]
        events = _detect_stream(stack, array)
        truth_onsets = [float(t) for t in (sample.get("truth_onsets") or [])]
        duration_s = round(len(array) / SAMPLE_RATE, 3)
        return {
            "label": "positive" if truth_onsets else "negative",
            "prediction": "WW_STREAM",
            "audio_url": sample.get("audio_url"),
            "extras": {
                "events": [list(e) for e in events],
                "truth_onsets": truth_onsets,
                "duration_s": duration_s,
            },
        }


def _detect_stream(stack: WWStack, array) -> list[tuple[float, float]]:
    """Run one long continuous clip through the wake-word stack, frame by
    frame, recording every activation as ``(timestamp_s, score)``.

    Mirrors :func:`_detect`: ``reset()`` is called once up front (same as an
    isolated clip), then every frame is fed through ``update()`` /
    ``found_wake_word()`` for the whole clip — no per-firing reset. A single
    continuous clip can still hold many onsets because each plugin's own
    latch/refractory logic (patience, debounce, cooldown — the engine owns
    its own threshold and re-arm behaviour, same as P2) decides when it is
    ready to fire again, exactly as it would against a live mic; the arena
    does not force a re-arm the plugin didn't ask for. An activation still
    has to clear a configured verifier, if any. Score defaults to 1.0: most
    OVOS hotword engines only expose a binary latch (``found_wake_word()``),
    not a calibrated confidence — the arena does not invent one. Engines
    that do expose a numeric ``.confidence``/``.score`` after firing get it
    recorded, enabling a real DET curve for those fighters instead of a
    degenerate single-point one.
    """
    events: list[tuple[float, float]] = []
    engine = stack.ww
    if hasattr(engine, "reset"):
        try:
            engine.reset()
        except Exception:
            pass
    fww = engine.found_wake_word
    fww_takes_arg = len(inspect.signature(fww).parameters) >= 1
    has_update = hasattr(engine, "update")
    pcm = _to_pcm16(array)
    frame_dur = FRAME_SAMPLES / SAMPLE_RATE
    t = 0.0
    for off in range(0, len(pcm), FRAME_SAMPLES * 2):
        chunk = pcm[off:off + FRAME_SAMPLES * 2]
        if has_update:
            engine.update(chunk)
        fired = fww(chunk) if fww_takes_arg else fww()
        if fired and (stack.verifier is None or _verify(stack.verifier, array)):
            score = getattr(engine, "confidence", None) or getattr(engine, "score", None)
            events.append((round(t, 3), float(score) if score is not None else 1.0))
        t += frame_dur
    return events


def _norm_label(raw) -> str:
    """Map a dataset label to ``positive`` / ``negative``."""
    from arena.metrics import _ww_is_positive

    val = _ww_is_positive(raw)
    return "positive" if val else "negative"
