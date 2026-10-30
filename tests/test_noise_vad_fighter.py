"""The noise-vad fighter scores each clip on its own energy, not on the bench's reset.

The VAD bench calls ``reset()`` before every clip (``runner.vad_bench._has_speech``).
The plugin's ``all`` method sets ``max_energy`` from the first frame after a
reset, so that frame always has a max/current ratio of 1.0 and scores as speech.
These tests drive the registry config through the real plugin and the real
bench loop. The expected decision comes from the plugin's energy formula
redone in numpy, ``rms(x - rms(x))`` per frame (``SilenceDetector.get_debiased_energy``),
against the literal threshold 1000. For zero-mean noise that energy is about
1.41 times the RMS, so the boundary sits near an RMS of 707.
"""

from __future__ import annotations

import numpy as np
import pytest
from ovos_vad_plugin_noise import NoiseVAD

from registry.loaders import load_competitor
from runner.vad_bench import FRAME_SAMPLES, SAMPLE_RATE, _has_speech, _to_pcm16

VAD_CONFIG = load_competitor("vad", "noise-vad").config["listener"]["VAD"]
ENERGY_THRESHOLD = 1000


def _noise(rms: float, seconds: float = 1.0) -> np.ndarray:
    """Gaussian noise whose int16 RMS is ``rms``."""
    rng = np.random.default_rng(int(rms))
    return rng.normal(0.0, rms / 32767.0, int(SAMPLE_RATE * seconds)).astype("float32")


def _loudest_frame_energy(clip: np.ndarray) -> float:
    """Highest plugin energy over the 30 ms int16 frames the bench feeds.

    audioop.rms floors to an integer, and audioop.add saturates at int16.
    """
    pcm = np.frombuffer(_to_pcm16(clip), dtype="<i2").astype(np.int64)
    frames = pcm[: len(pcm) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    rms = np.floor(np.sqrt((frames**2).mean(axis=1))).astype(np.int64)
    shifted = np.clip(frames - rms[:, None], -32768, 32767)
    return float(np.floor(np.sqrt((shifted**2).mean(axis=1))).max())


CLIPS = {
    "digital silence": np.zeros(SAMPLE_RATE, dtype="float32"),
    "room noise at RMS 150": _noise(150.0),
    "noise at RMS 500": _noise(500.0),
    "noise at RMS 600": _noise(600.0),
    "noise at RMS 700": _noise(700.0),
    "noise at RMS 850": _noise(850.0),
    "noise at RMS 2500": _noise(2500.0),
    "half second of quiet then a loud tone": np.concatenate(
        [
            _noise(150.0, 0.5),
            (0.3 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE)).astype(
                "float32"
            ),
        ]
    ),
}


@pytest.mark.parametrize("name", sorted(CLIPS))
def test_decision_follows_frame_energy(name):
    clip = CLIPS[name]
    expected = _loudest_frame_energy(clip) >= ENERGY_THRESHOLD
    assert _has_speech(NoiseVAD(VAD_CONFIG), clip) is expected


def test_quiet_clips_are_not_speech():
    engine = NoiseVAD(VAD_CONFIG)
    for name in (
        "digital silence",
        "room noise at RMS 150",
        "noise at RMS 500",
        "noise at RMS 600",
    ):
        assert _has_speech(engine, CLIPS[name]) is False, name


def test_noise_above_the_boundary_is_speech():
    assert _has_speech(NoiseVAD(VAD_CONFIG), CLIPS["noise at RMS 850"]) is True


def test_decision_does_not_depend_on_the_previous_clip():
    engine = NoiseVAD(VAD_CONFIG)
    quiet = CLIPS["room noise at RMS 150"]
    before = _has_speech(engine, quiet)
    _has_speech(engine, CLIPS["noise at RMS 2500"])
    assert _has_speech(engine, quiet) is before is False
