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
        # §3.2: PredictionRow has no modeled utmos field — these MUST live
        # under "extras", not as flat top-level keys, or pydantic silently
        # drops them (see TestPredictionRowRoundTrip below).
        assert "utmos" not in fields
        assert fields["extras"]["utmos"] == pytest.approx(4.37)
        assert fields["extras"]["utmos_judge"] == "TigreGotico/utmos-onnx"
        assert fields["extras"]["utmos_judge_revision"] == (
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
        assert isinstance(fields["extras"]["utmos"], float)
        assert fields["extras"]["utmos"] == pytest.approx(4.0)

    def test_scoring_is_not_skipped_when_judge_missing(self, tmp_path, monkeypatch):
        # scoring is NOT optional for TTS runs — a missing dependency must
        # raise, not silently omit utmos from the row.
        def boom():
            raise RuntimeError("TTS benchmarking requires 'speechonnxmetrics'")

        monkeypatch.setattr(tts_bench, "_get_utmos_judge", boom)
        with pytest.raises(RuntimeError, match="speechonnxmetrics"):
            tts_bench.TTSBench().predict(
                FakeEngine(), {"input_text": "x"}, _ctx(tmp_path))


class TestPredictionRowRoundTrip:
    """Regression: predict()'s output must survive the real make_row /
    JSONL / parse_row round-trip with utmos still readable via row_utmos().

    A hand-built ``PredictionRow(extras={...})`` (as in test_metrics.py)
    would NOT have caught the original bug — the bug was that ``predict()``
    returned flat top-level keys, which ``PredictionRow`` silently drops.
    This test builds the row exactly the way the real pipeline does:
    ``make_row(competitor, ..., fields=TTSBench.predict(...))`` -> JSON ->
    ``PredictionRow(**parsed)``.
    """

    def test_utmos_survives_make_row_and_parse(self, tmp_path, monkeypatch):
        import json

        from arena.metrics import row_utmos
        from arena.models import PredictionRow
        from registry.loaders import load_competitor
        from runner.media_bench import make_row

        fake_judge = FakeJudge(score=4.3755)
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: fake_judge)

        competitor = load_competitor("tts", "piper-amy-en-us")
        ctx = _ctx(tmp_path, competitor_id=competitor.competitor_id)
        ctx.competitor = competitor
        fields = tts_bench.TTSBench().predict(
            FakeEngine(), {"input_text": "hello there"}, ctx)

        row = make_row(competitor, "d", "en-US", "en-US/00000", "rev", fields)
        # round-trip through JSON, exactly like the JSONL file on disk
        parsed = PredictionRow(**json.loads(json.dumps(row)))

        assert row_utmos(parsed) == pytest.approx(4.3755)
        assert parsed.extras["utmos_judge"] == "TigreGotico/utmos-onnx"
        assert parsed.extras["utmos_judge_revision"] == (
            "ff41b8f440cb12ecda18261f9ff7326d058275ce"
        )


# ---------------------------------------------------------------------------
# Intelligibility (STT round-trip WER/CER, §4 R16)
# ---------------------------------------------------------------------------
#
# The real faster-whisper judge is never loaded here (no network, no model
# download): a fake judge is monkeypatched in via
# ``runner.tts_bench._get_intelligibility_judge``, per §4 test policy.
# ``runner.audio_io.decode_audio_bytes`` IS exercised for real (it is pure
# numpy/soundfile/av, no network) so the resample/transcode pitfalls are
# actually caught rather than assumed away by a mock.


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeIntelligibilityJudge:
    """Stand-in for faster_whisper.WhisperModel — records the array it saw."""

    def __init__(self, text="hello there"):
        self.text = text
        self.calls = []

    def transcribe(self, array):
        self.calls.append(array)
        return [FakeSegment(self.text)], SimpleNamespace(language="en")


def _write_wav(path, seconds=0.5, sr=16000, channels=1):
    import numpy as np
    import soundfile as sf

    n = int(seconds * sr)
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype("float32")
    if channels > 1:
        tone = np.stack([tone] * channels, axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone, sr)


class RealWavEngine:
    """Fake engine that renders a real (decodable) audio file, at a given
    sample rate/channel count/format — lets tests exercise the real
    ``decode_audio_bytes`` resample/transcode path instead of mocking it away."""

    def __init__(self, seconds=0.5, sr=16000, channels=1, fmt=None):
        self.seconds = seconds
        self.sr = sr
        self.channels = channels
        self.fmt = fmt
        self.calls = []

    def get_tts(self, text, wav_path, lang=None):
        self.calls.append((text, wav_path, lang))
        import numpy as np
        import soundfile as sf

        n = int(self.seconds * self.sr)
        tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / self.sr)).astype("float32")
        if self.channels > 1:
            tone = np.stack([tone] * self.channels, axis=-1)
        sf.write(wav_path, tone, self.sr, format=self.fmt)


class CrashingEngine:
    """Fake engine whose synthesis always raises — the crash-must-still-emit-
    a-row pitfall."""

    def get_tts(self, text, wav_path, lang=None):
        raise RuntimeError("synthesis backend exploded")


@pytest.fixture(autouse=True)
def _reset_intelligibility_judge_cache(monkeypatch):
    monkeypatch.setattr(tts_bench, "_intelligibility_judge", None)
    yield
    monkeypatch.setattr(tts_bench, "_intelligibility_judge", None)


class TestIntelligibility:
    def test_extras_carry_wer_cer_and_judge_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        fake_judge = FakeIntelligibilityJudge(text="hello there")
        monkeypatch.setattr(
            tts_bench, "_get_intelligibility_judge", lambda: fake_judge)

        engine = RealWavEngine()
        fields = tts_bench.TTSBench().predict(
            engine, {"input_text": "hello there"}, _ctx(tmp_path))

        assert fields["extras"]["intelligibility_wer"] == pytest.approx(0.0)
        assert fields["extras"]["intelligibility_cer"] == pytest.approx(0.0)
        assert fields["extras"]["intelligibility_judge"] == (
            tts_bench.INTELLIGIBILITY_JUDGE
        )
        assert fields["extras"]["intelligibility_judge_revision"] == (
            tts_bench.INTELLIGIBILITY_JUDGE_REVISION
        )
        assert len(fake_judge.calls) == 1

    def test_wer_reflects_mismatched_transcript(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        # judge mishears everything — WER/CER should be nonzero, not silently
        # clamped or dropped.
        fake_judge = FakeIntelligibilityJudge(text="completely different words")
        monkeypatch.setattr(
            tts_bench, "_get_intelligibility_judge", lambda: fake_judge)

        fields = tts_bench.TTSBench().predict(
            RealWavEngine(), {"input_text": "hello there"}, _ctx(tmp_path))

        assert fields["extras"]["intelligibility_wer"] > 0.0
        assert fields["extras"]["intelligibility_cer"] > 0.0

    def test_synthesis_crash_still_emits_a_row_with_worst_case_wer(
        self, tmp_path, monkeypatch
    ):
        # Pitfall: a synthesis exception must never propagate up and result
        # in a silently-dropped sample (media_bench.run_competitor_lang's
        # blanket except just `continue`s on an uncaught exception) — it
        # must come back as a real row scored WER/CER = 1.0.
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        fields = tts_bench.TTSBench().predict(
            CrashingEngine(), {"input_text": "hello there"}, _ctx(tmp_path))

        assert fields["extras"]["intelligibility_wer"] == 1.0
        assert fields["extras"]["intelligibility_cer"] == 1.0
        assert "synthesis_error" in fields["extras"]
        assert fields["prediction"] is None
        # utmos was never attempted — there is no valid clip to score
        assert "utmos" not in fields["extras"]

    def test_intelligibility_judge_crash_forces_worst_case_not_missing(
        self, tmp_path, monkeypatch
    ):
        # A judge-side crash (e.g. on silence/noise for a low-resource
        # language) is warn-only for the *board* — but the row must not
        # silently omit the metric; it forces the worst case instead.
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())

        def boom():
            raise RuntimeError("judge blew up")

        monkeypatch.setattr(tts_bench, "_get_intelligibility_judge", boom)
        fields = tts_bench.TTSBench().predict(
            RealWavEngine(), {"input_text": "hello there"}, _ctx(tmp_path))

        assert fields["extras"]["intelligibility_wer"] == 1.0
        assert fields["extras"]["intelligibility_cer"] == 1.0
        assert "intelligibility_error" in fields["extras"]
        # utmos scoring — a separate concern — still succeeded
        assert "utmos" in fields["extras"]

    def test_resamples_44khz_stereo_to_16k_mono_before_transcription(
        self, tmp_path, monkeypatch
    ):
        # Pitfall: reading a 44.1 kHz file as if it were already 16 kHz
        # gives a false ~1.7x-longer array and garbage WER. The judge must
        # only ever see a 16 kHz mono array regardless of the source format.
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        fake_judge = FakeIntelligibilityJudge(text="hello there")
        monkeypatch.setattr(
            tts_bench, "_get_intelligibility_judge", lambda: fake_judge)

        engine = RealWavEngine(seconds=0.5, sr=44100, channels=2)
        tts_bench.TTSBench().predict(
            engine, {"input_text": "hello there"}, _ctx(tmp_path))

        assert len(fake_judge.calls) == 1
        seen_array = fake_judge.calls[0]
        assert seen_array.ndim == 1  # mono
        # 0.5s @ 16kHz == 8000 samples; at the wrong (44.1kHz-assumed) rate
        # this would be ~22050 samples instead.
        assert abs(len(seen_array) - 8000) < 200

    def test_mp3_output_is_transcoded_not_read_as_raw_pcm(
        self, tmp_path, monkeypatch
    ):
        # Pitfall: a plugin that renders mp3 (or any non-wav container) must
        # be transcoded via the real decoder, never read as if the bytes on
        # disk were already headerless PCM samples.
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        fake_judge = FakeIntelligibilityJudge(text="hello there")
        monkeypatch.setattr(
            tts_bench, "_get_intelligibility_judge", lambda: fake_judge)

        engine = RealWavEngine(seconds=0.5, sr=16000, channels=1, fmt="MP3")
        fields = tts_bench.TTSBench().predict(
            engine, {"input_text": "hello there"}, _ctx(tmp_path))

        # decoded cleanly through the real audio pipeline, scored normally —
        # not skipped, not crashed into the worst-case fallback.
        assert "intelligibility_error" not in fields["extras"]
        assert fields["extras"]["intelligibility_wer"] == pytest.approx(0.0)
        seen_array = fake_judge.calls[0]
        assert seen_array.ndim == 1
        assert abs(len(seen_array) - 8000) < 400  # ~0.5s @ 16kHz, mp3-lossy

    def test_no_playback_path_only_direct_get_tts_called(self, tmp_path, monkeypatch):
        # §4 R16: the benchmark calls the TTS plugin's direct get_tts
        # synthesis path — no audio-player / playback-mode round trip.
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        monkeypatch.setattr(
            tts_bench, "_get_intelligibility_judge",
            lambda: FakeIntelligibilityJudge())

        engine = RealWavEngine()
        engine.play = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("playback must never be invoked by the benchmark"))
        tts_bench.TTSBench().predict(
            engine, {"input_text": "hello there"}, _ctx(tmp_path))
        assert len(engine.calls) == 1
