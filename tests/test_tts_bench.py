"""Unit tests for runner.tts_bench — TTS synthesis + objective UTMOS scoring.

The real ``speechonnxmetrics`` judge is never imported or downloaded here: a
fake judge is monkeypatched in via ``runner.tts_bench._get_utmos_judge`` so
these tests run with no network and no model install, per §4 test policy
(never skip/importorskip on optional deps — monkeypatch instead).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from runner import tts_bench


class FakeEngine:
    """Stand-in for a real OVOS TTS plugin instance."""

    def __init__(self):
        self.calls = []

    def get_tts(self, text, wav_path, lang=None):
        self.calls.append((text, wav_path, lang))
        # write something so downstream code that stats the file doesn't choke
        with open(wav_path, "wb") as fh:
            fh.write(b"RIFF....WAVEfmt ")


class FakeJudge:
    """Stand-in for speechonnxmetrics.mos.utmos.UTMOS — no ONNX, no network."""

    sample_rate = 16000

    def __init__(self, score=4.2):
        self.score = score
        self.calls = []

    def __call__(self, wav_path, sr):
        self.calls.append((wav_path, sr))
        return self.score


@pytest.fixture(autouse=True)
def _reset_judge_cache(monkeypatch):
    # the judge is a module-level cached singleton — make sure tests don't
    # leak a fake instance into each other.
    monkeypatch.setattr(tts_bench, "_utmos_judge", None)
    yield
    monkeypatch.setattr(tts_bench, "_utmos_judge", None)


def _ctx(tmp_path, competitor_id="voice_a", lang="en-US"):
    competitor = SimpleNamespace(competitor_id=competitor_id)
    return tts_bench.PredictContext(
        competitor, lang, "d", "tts", tmp_path,
        "OpenVoiceOS/ovos-tts-bench-d",
    )


class TestGetUtmosJudge:
    def test_lazy_import_error_is_clear(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "speechonnxmetrics.mos.utmos":
                raise ImportError("no module")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="speechonnxmetrics"):
            tts_bench._get_utmos_judge()

    def test_cached_across_calls(self, monkeypatch):
        judge = FakeJudge()
        monkeypatch.setattr(tts_bench, "_utmos_judge", None)
        calls = {"n": 0}

        def factory(*a, **kw):
            calls["n"] += 1
            return judge

        # patch the class the function would import, by patching the whole
        # helper's internal state directly via a stub module
        import sys
        import types
        fake_mod = types.ModuleType("speechonnxmetrics.mos.utmos")
        fake_mod.UTMOS = factory
        sys.modules["speechonnxmetrics.mos.utmos"] = fake_mod
        sys.modules.setdefault(
            "speechonnxmetrics", types.ModuleType("speechonnxmetrics"))
        sys.modules.setdefault(
            "speechonnxmetrics.mos", types.ModuleType("speechonnxmetrics.mos"))

        first = tts_bench._get_utmos_judge()
        second = tts_bench._get_utmos_judge()
        assert first is second is judge
        assert calls["n"] == 1


class TestPredict:
    def test_extras_carry_utmos_and_judge_provenance(self, tmp_path, monkeypatch):
        fake_judge = FakeJudge(score=4.37)
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: fake_judge)

        engine = FakeEngine()
        ctx = _ctx(tmp_path)
        fields = tts_bench.TTSBench().predict(engine, {"input_text": "hello"}, ctx)

        assert fields["input_text"] == "hello"
        assert fields["prediction"].endswith(".wav")
        assert fields["utmos"] == pytest.approx(4.37)
        assert fields["utmos_judge"] == "TigreGotico/utmos-onnx"
        assert fields["utmos_judge_revision"] == (
            "ff41b8f440cb12ecda18261f9ff7326d058275ce"
        )
        assert "latency_ms" in fields
        assert len(engine.calls) == 1
        assert len(fake_judge.calls) == 1

    def test_score_is_rounded_float(self, tmp_path, monkeypatch):
        fake_judge = FakeJudge(score=3.999999)
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: fake_judge)
        fields = tts_bench.TTSBench().predict(
            FakeEngine(), {"input_text": "x"}, _ctx(tmp_path))
        assert isinstance(fields["utmos"], float)
        assert fields["utmos"] == pytest.approx(4.0)

    def test_scoring_is_not_skipped_when_judge_missing(self, tmp_path, monkeypatch):
        # scoring is NOT optional for TTS runs — a missing dependency must
        # raise, not silently omit utmos from the row.
        def boom():
            raise RuntimeError("TTS benchmarking requires 'speechonnxmetrics'")

        monkeypatch.setattr(tts_bench, "_get_utmos_judge", boom)
        with pytest.raises(RuntimeError, match="speechonnxmetrics"):
            tts_bench.TTSBench().predict(
                FakeEngine(), {"input_text": "x"}, _ctx(tmp_path))
