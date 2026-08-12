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
# The real onnx-asr judge is never loaded here (no network, no model
# download): a fake judge is monkeypatched in via
# ``runner.tts_bench._get_intelligibility_judge``, per §4 test policy.
# ``runner.audio_io.decode_audio_bytes`` IS exercised for real (it is pure
# numpy/soundfile/av, no network) so the resample/transcode pitfalls are
# actually caught rather than assumed away by a mock.


class FakeIntelligibilityJudge:
    """Stand-in for an onnx_asr TextResultsAsrAdapter — records the array it saw."""

    def __init__(self, text="hello there"):
        self.text = text
        self.calls = []

    def recognize(self, array, sample_rate=16000):
        self.calls.append(array)
        return self.text


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
    monkeypatch.setattr(tts_bench, "_intelligibility_judges", {})
    yield
    monkeypatch.setattr(tts_bench, "_intelligibility_judges", {})


def _patch_judge(monkeypatch, fake_judge, model_id="fake-model", revision="fake-rev"):
    """Patch _get_intelligibility_judge to return a fake judge for any lang."""
    monkeypatch.setattr(
        tts_bench, "_get_intelligibility_judge",
        lambda lang: (fake_judge, revision, model_id))


class TestIntelligibility:
    def test_extras_carry_wer_cer_and_judge_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        fake_judge = FakeIntelligibilityJudge(text="hello there")
        _patch_judge(monkeypatch, fake_judge, "nemo-parakeet-tdt-0.6b-v3",
                     "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce")

        engine = RealWavEngine()
        fields = tts_bench.TTSBench().predict(
            engine, {"input_text": "hello there"}, _ctx(tmp_path))

        assert fields["extras"]["intelligibility_wer"] == pytest.approx(0.0)
        assert fields["extras"]["intelligibility_cer"] == pytest.approx(0.0)
        assert fields["extras"]["intelligibility_judge"] == "nemo-parakeet-tdt-0.6b-v3"
        assert fields["extras"]["intelligibility_judge_revision"] == (
            "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
        )
        assert len(fake_judge.calls) == 1

    def test_wer_reflects_mismatched_transcript(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())
        # judge mishears everything — WER/CER should be nonzero, not silently
        # clamped or dropped.
        fake_judge = FakeIntelligibilityJudge(text="completely different words")
        _patch_judge(monkeypatch, fake_judge)

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
        # judge provenance is still recorded even though scoring never ran —
        # resolve_judge_model() is pure lookup, no model load required.
        assert fields["extras"]["intelligibility_judge"]
        assert fields["extras"]["intelligibility_judge_revision"]
        # utmos was never attempted — there is no valid clip to score
        assert "utmos" not in fields["extras"]

    def test_intelligibility_judge_crash_forces_worst_case_not_missing(
        self, tmp_path, monkeypatch
    ):
        # A judge-side crash (e.g. on silence/noise for a low-resource
        # language) is warn-only for the *board* — but the row must not
        # silently omit the metric; it forces the worst case instead.
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: FakeJudge())

        def boom(lang):
            raise RuntimeError("judge blew up")

        monkeypatch.setattr(tts_bench, "_get_intelligibility_judge", boom)
        fields = tts_bench.TTSBench().predict(
            RealWavEngine(), {"input_text": "hello there"}, _ctx(tmp_path))

        assert fields["extras"]["intelligibility_wer"] == 1.0
        assert fields["extras"]["intelligibility_cer"] == 1.0
        assert "intelligibility_error" in fields["extras"]
        assert fields["extras"]["intelligibility_judge"]
        assert fields["extras"]["intelligibility_judge_revision"]
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
        _patch_judge(monkeypatch, fake_judge)

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
        _patch_judge(monkeypatch, fake_judge)

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
        _patch_judge(monkeypatch, FakeIntelligibilityJudge())

        engine = RealWavEngine()
        engine.play = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("playback must never be invoked by the benchmark"))
        tts_bench.TTSBench().predict(
            engine, {"input_text": "hello there"}, _ctx(tmp_path))
        assert len(engine.calls) == 1

    def test_judge_constructed_with_full_model_id_and_pinned_revision(self, monkeypatch):
        # Regression (was: faster-whisper basename bug): the onnx-asr judge
        # must be constructed with the FULL resolved model id, and the row
        # must carry the pinned revision from asr_judges._REVISIONS even
        # though onnx_asr.load_model itself has no revision parameter.
        constructed = {}

        def fake_load_model(model, **kwargs):
            constructed["model"] = model
            return object()

        import sys, types
        fake_mod = types.ModuleType("onnx_asr")
        fake_mod.load_model = fake_load_model
        monkeypatch.setitem(sys.modules, "onnx_asr", fake_mod)
        monkeypatch.setattr(tts_bench, "_intelligibility_judges", {})

        judge, revision, model_id = tts_bench._get_intelligibility_judge("en-US")
        assert model_id == "nemo-parakeet-tdt-0.6b-v3"
        assert constructed["model"] == "nemo-parakeet-tdt-0.6b-v3"
        assert revision == "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"


class TestJudgeResolution:
    """runner.asr_judges.resolve_judge_model — per-lang judge selection."""

    def test_recommended_model_used_for_lang_in_ovos_config_recommends(self):
        # pt-PT is one of ovos-config's offline_stt recommends — a dedicated
        # whisper-medium-pt export, not the generic multilingual fallback.
        from runner.asr_judges import resolve_judge_model

        model_id, revision = resolve_judge_model("pt-PT")
        assert model_id == "OpenVoiceOS/whisper-medium-pt-onnx"
        assert revision == "7db38a22790ba3f831702db12cb19dd684642bf5"

    def test_fallback_table_used_when_lang_missing_from_recommends(self):
        # Vietnamese has no offline_stt/*.conf in ovos-config; the
        # onnx-stt-plugin-onnx-asr LANG_DEFAULTS fallback picks the
        # dedicated NVIDIA Vietnamese fine-tune instead of whisper-base.
        from runner.asr_judges import resolve_judge_model

        model_id, revision = resolve_judge_model("vi-VN")
        assert model_id == "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx"
        assert revision == "ed9f55ba980eb1c9eeba02a5733eba7cba02f6e7"

    def test_universal_whisper_base_fallback_for_long_tail_lang(self):
        # No dedicated onnx-asr export exists for Malagasy — it falls all
        # the way through to onnx-asr's own bundled whisper-base wrapper
        # (still the onnx-asr package, never faster-whisper).
        from runner.asr_judges import resolve_judge_model

        model_id, revision = resolve_judge_model("mg-MG")
        assert model_id == "whisper-base"
        assert revision == "998334d3bfe2deba3c8e6821f05388dbf2b706d2"

    def test_full_tag_beats_primary_subtag_prefix_match(self):
        # en-US is an exact recommends key; a made-up en-XX must still
        # resolve via the primary-subtag prefix match to the same model.
        from runner.asr_judges import resolve_judge_model

        exact_id, exact_rev = resolve_judge_model("en-US")
        prefix_id, prefix_rev = resolve_judge_model("en-XX")
        assert exact_id == prefix_id == "nemo-parakeet-tdt-0.6b-v3"
        assert exact_rev == prefix_rev

    def test_two_langs_sharing_a_model_reuse_the_cached_judge(self, monkeypatch):
        # en-US and de-DE both resolve to nemo-parakeet-tdt-0.6b-v3 — the
        # model must load once per process, not once per language.
        loads = {"n": 0}

        def fake_load_model(model, **kwargs):
            loads["n"] += 1
            return object()

        import sys, types
        fake_mod = types.ModuleType("onnx_asr")
        fake_mod.load_model = fake_load_model
        monkeypatch.setitem(sys.modules, "onnx_asr", fake_mod)
        monkeypatch.setattr(tts_bench, "_intelligibility_judges", {})

        judge_en, _rev_en, id_en = tts_bench._get_intelligibility_judge("en-US")
        judge_de, _rev_de, id_de = tts_bench._get_intelligibility_judge("de-DE")

        assert id_en == id_de == "nemo-parakeet-tdt-0.6b-v3"
        assert judge_en is judge_de
        assert loads["n"] == 1


# ---------------------------------------------------------------------------
# elapsed_ms scope (performance-metrics campaign M1) — must exclude judging
# ---------------------------------------------------------------------------


class _SlowJudge(FakeJudge):
    """A UTMOS judge stand-in that sleeps to simulate real scoring latency."""

    def __init__(self, seconds, score=4.2):
        super().__init__(score=score)
        self.seconds = seconds

    def __call__(self, wav_path, sr):
        import time as _time

        _time.sleep(self.seconds)
        return super().__call__(wav_path, sr)


class TestElapsedMsExcludesJudging:
    """elapsed_ms/peak_rss_mb must be scoped to ONLY the synthesis call
    (``engine.get_tts``), not the whole ``predict()`` — media_bench wraps
    the whole thing in its own measure_call for the `latency_ms` field, but
    TTSBench.predict() also runs UTMOS scoring and STT round-trip judging
    inside the same call, and folding that judging time into elapsed_ms
    would distort RTF (elapsed_ms / audio_secs), the exact metric this
    capture exists to compute.

    A fake judge that sleeps stands in for slow real scoring; the fake
    synthesis engine is effectively instantaneous. If elapsed_ms is
    contaminated by judging time, it will be at least ``sleep_seconds``
    long — this test fails loudly in that case.
    """

    SLEEP_SECONDS = 0.2

    def test_elapsed_ms_excludes_utmos_and_intelligibility_judging(
        self, tmp_path, monkeypatch
    ):
        slow_judge = _SlowJudge(seconds=self.SLEEP_SECONDS)
        monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: slow_judge)
        # Intelligibility scoring is a second judging stage — keep it fast
        # so the test isolates the UTMOS-sleep contribution cleanly, but it
        # would contaminate elapsed_ms exactly the same way if included.
        monkeypatch.setattr(
            tts_bench, "_score_intelligibility",
            lambda wav_path, text, lang: (0.0, 0.0, "fake-model", "fake-rev"),
        )

        engine = FakeEngine()  # near-instant synthesis
        fields = tts_bench.TTSBench().predict(
            engine, {"input_text": "hello"}, _ctx(tmp_path)
        )

        assert fields["elapsed_ms"] is not None
        # Synthesis-only span: comfortably under the judge's artificial
        # sleep. A regression that measures the whole predict() call would
        # report elapsed_ms >= SLEEP_SECONDS * 1000 here.
        assert fields["elapsed_ms"] < (self.SLEEP_SECONDS * 1000) / 2
        # latency_ms was already synthesis-scoped before this fix and must
        # stay that way too.
        assert fields["latency_ms"] < (self.SLEEP_SECONDS * 1000) / 2
