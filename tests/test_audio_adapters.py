"""Unit tests for the STT / wake-word / TTS adapter pure helpers.

These exercise the normalisation logic that turns plugin output into §3.2 row
fields, without loading any plugin or decoding any audio.
"""
from __future__ import annotations

from runner.stt_bench import STTBench, _first_hypothesis
from runner.tts_bench import TTSBench, _safe
from runner.ww_bench import WakeWordBench, _norm_label, _to_pcm16


class TestSTTHypothesis:
    def test_list_of_tuples(self):
        assert _first_hypothesis([("hello", 0.9)]) == ("hello", 0.9)

    def test_list_of_str(self):
        assert _first_hypothesis(["hello"]) == ("hello", 1.0)

    def test_tuple(self):
        assert _first_hypothesis(("hi", 0.5)) == ("hi", 0.5)

    def test_plain_string(self):
        assert _first_hypothesis("hi") == ("hi", 1.0)

    def test_empty(self):
        assert _first_hypothesis([]) == ("", 1.0)
        assert _first_hypothesis(None) == ("", 1.0)


class TestWakeWordHelpers:
    def test_norm_label(self):
        assert _norm_label("positive") == "positive"
        assert _norm_label(1) == "positive"
        assert _norm_label("negative") == "negative"
        assert _norm_label("adversarial") == "negative"
        assert _norm_label(None) == "negative"

    def test_to_pcm16_length_and_type(self):
        import numpy as np

        pcm = _to_pcm16(np.array([0.0, 1.0, -1.0], dtype="float32"))
        assert isinstance(pcm, bytes)
        assert len(pcm) == 3 * 2  # 16-bit samples
        # full-scale maps near int16 max/min
        vals = np.frombuffer(pcm, dtype="<i2")
        assert vals[1] > 32000 and vals[2] < -32000


class FakeWWEngine:
    """Minimal hotword engine: fires once `update` has been called N times."""

    def __init__(self, fire_after=None):
        self.fire_after = fire_after
        self.updates = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.updates = 0

    def update(self, chunk):
        self.updates += 1

    def found_wake_word(self):
        return self.fire_after is not None and self.updates >= self.fire_after


class TestWakeWordPredict:
    def _ctx(self):
        from pathlib import Path

        from runner.media_bench import PredictContext
        return PredictContext(None, "en", "d", "wake_word", Path("/tmp"), "o/r")

    def _sample(self, **over):
        import numpy as np

        s = {"array": np.zeros(4096, dtype="float32"), "sr": 16000,
             "label": "positive", "audio_url": "https://hf/clip.wav"}
        s.update(over)
        return s

    def test_detection_and_passthrough(self):
        adapter = WakeWordBench()
        out = adapter.predict(FakeWWEngine(fire_after=1), self._sample(), self._ctx())
        assert out["prediction"] == "detected"
        assert out["label"] == "positive"
        assert out["audio_url"] == "https://hf/clip.wav"
        assert "latency_ms" in out

    def test_no_detection(self):
        adapter = WakeWordBench()
        out = adapter.predict(FakeWWEngine(fire_after=None),
                              self._sample(label="negative"), self._ctx())
        assert out["prediction"] == "not_detected"
        assert out["label"] == "negative"

    def test_engine_reset_per_clip(self):
        engine = FakeWWEngine(fire_after=None)
        WakeWordBench().predict(engine, self._sample(), self._ctx())
        assert engine.resets == 1


class FrameStyleWW:
    """Hotword engine that does everything in found_wake_word(frame)."""

    def __init__(self, fire_on_index=None):
        self.fire_on_index = fire_on_index
        self.i = -1
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.i = -1

    def found_wake_word(self, frame):  # frame-style API
        self.i += 1
        return self.i == self.fire_on_index


class UpdateStyleWW:
    """Hotword engine with update(chunk) + found_wake_word()."""

    def __init__(self, fire_after=None):
        self.fire_after = fire_after
        self.n = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.n = 0

    def update(self, chunk):
        self.n += 1

    def found_wake_word(self):  # update-style API
        return self.fire_after is not None and self.n >= self.fire_after


class TestWWDetect:
    def _audio(self, n=6400):
        import numpy as np
        return np.zeros(n, dtype="float32")

    def test_frame_style_detection(self):
        from runner.ww_bench import _detect
        assert _detect(FrameStyleWW(fire_on_index=1), self._audio()) is True
        assert _detect(FrameStyleWW(fire_on_index=None), self._audio()) is False

    def test_update_style_detection(self):
        from runner.ww_bench import _detect
        assert _detect(UpdateStyleWW(fire_after=1), self._audio()) is True
        assert _detect(UpdateStyleWW(fire_after=None), self._audio()) is False

    def test_reset_called_each_clip(self):
        from runner.ww_bench import _detect
        eng = UpdateStyleWW(fire_after=None)
        _detect(eng, self._audio())
        assert eng.resets == 1


class TestEvenSampler:
    def test_spans_full_range(self):
        from runner.audio_io import _even
        items = [f"x/{i:03d}.wav" for i in range(100)]
        picked = _even(items, 5)
        assert len(picked) == 5
        assert picked[0] == "x/000.wav"      # starts at the front
        assert picked[-1] != "x/004.wav"     # not a same-run head slice
        assert picked == sorted(picked)       # deterministic, in order

    def test_returns_all_when_under_cap(self):
        from runner.audio_io import _even
        items = ["a", "b", "c"]
        assert _even(items, 10) == items
        assert _even(items, 0) == items


class TestTTSHelpers:
    def test_safe_is_stable_and_short(self):
        a = _safe("hello world")
        b = _safe("hello world")
        c = _safe("different")
        assert a == b and a != c
        assert len(a) == 16 and a.isalnum()


class TestAdapterMetadata:
    def test_modalities(self):
        assert STTBench().modality == "stt"
        assert WakeWordBench().modality == "wake_word"
        assert TTSBench().modality == "tts"

    def test_competitor_langs_intersection(self):
        from types import SimpleNamespace

        adapter = STTBench()
        any_lang = SimpleNamespace(langs=[])
        specific = SimpleNamespace(langs=["pt-PT"])
        assert adapter.competitor_langs(any_lang, ["pt-PT", "en-US"]) == \
            ["pt-PT", "en-US"]
        assert adapter.competitor_langs(specific, ["pt-PT", "en-US"]) == ["pt-PT"]
