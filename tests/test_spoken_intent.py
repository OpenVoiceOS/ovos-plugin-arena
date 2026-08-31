"""Unit tests for the spoken-intent (audio-input) datasets and runner path.

No network: HF/STT calls are stubbed. Proves (1) every registered
Speech-MASSIVE dataset def validates and carries a full STT pin, (2) the
schema rejects malformed audio-input defs, and (3) the runner transcribes
each clip ONCE per (dataset, lang) and reuses that SAME cached transcript
across fighters, stamping STT provenance on every row.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

import runner.intent_bench as intent_bench
from registry.loaders import REGISTRY_ROOT, load_all_datasets, load_dataset
from registry.schemas import DatasetDef

SPEECH_MASSIVE_LANGS = (
    "ar-SA", "de-DE", "es-ES", "fr-FR", "hu-HU", "ko-KR",
    "nl-NL", "pl-PL", "pt-PT", "ru-RU", "vi-VN",
)


class TestDatasetDefs:
    @pytest.mark.parametrize("lang", SPEECH_MASSIVE_LANGS)
    def test_registered_dataset_validates(self, lang):
        d = load_dataset("intent", f"speech-massive-{lang}")
        assert d.input == "audio"
        assert d.stt_plugin == "ovos-stt-plugin-onnx-asr"
        assert d.stt_config and "model" in d.stt_config
        assert d.reference_fields["audio"] == "audio"
        assert d.reference_fields["intent"] == "intent_str"
        assert d.source.hf_id == "FBK-MT/Speech-MASSIVE-test"
        assert d.source.subset == lang
        assert d.train_datasets == {"template": "massive-templates-train"}
        assert d.predictions_hf == (
            f"OpenVoiceOS/ovos-intent-bench-speech-massive-{lang}"
        )

    def test_tr_tr_not_registered_v1(self):
        path = REGISTRY_ROOT / "datasets" / "intent" / "speech-massive-tr-TR.json"
        assert not path.exists(), (
            "tr-TR has no pinned onnx-asr default in the STT league yet — "
            "must stay unregistered until one exists"
        )

    def test_all_speech_massive_defs_present_in_full_registry_load(self):
        ids = {d.dataset_id for d in load_all_datasets()}
        for lang in SPEECH_MASSIVE_LANGS:
            assert f"speech-massive-{lang}" in ids


class TestAudioInputValidator:
    def _base(self, **overrides):
        base = dict(
            dataset_id="x",
            modality="intent",
            source={"type": "huggingface", "hf_id": "org/ds", "split": "test",
                    "subset": "en-US"},
            reference_fields={"utterance": "utt", "intent": "intent_str",
                              "audio": "audio"},
            lang="en-US",
            input="audio",
            stt_plugin="ovos-stt-plugin-onnx-asr",
            stt_config={"model": "m"},
        )
        base.update(overrides)
        return base

    def test_valid_audio_dataset(self):
        DatasetDef.model_validate(self._base())

    def test_missing_stt_plugin_rejected(self):
        with pytest.raises(ValidationError, match="stt_plugin"):
            DatasetDef.model_validate(self._base(stt_plugin=None))

    def test_missing_stt_config_rejected(self):
        with pytest.raises(ValidationError, match="stt_plugin"):
            DatasetDef.model_validate(self._base(stt_config=None))

    def test_missing_audio_field_rejected(self):
        with pytest.raises(ValidationError, match="reference_fields\\['audio'\\]"):
            DatasetDef.model_validate(self._base(
                reference_fields={"utterance": "utt", "intent": "intent_str"}))

    def test_audio_input_requires_intent_league(self):
        with pytest.raises(ValidationError, match="intent-league"):
            DatasetDef.model_validate(self._base(modality="stt"))

    def test_text_input_default_unaffected(self):
        d = DatasetDef.model_validate(dict(
            dataset_id="y", modality="intent",
            source={"type": "huggingface", "hf_id": "org/ds", "split": "test"},
            reference_fields={"utterance": "utterance", "intent": "expected_intent"},
            lang="en-US",
        ))
        assert d.input == "text"
        assert d.stt_plugin is None


class _StubEngine:
    def __init__(self, transcripts):
        self._transcripts = transcripts
        self.calls = 0

    def transcribe(self, audio, lang=None):
        self.calls += 1
        return [(self._transcripts[self.calls - 1], 0.9)]


class TestTranscribeDataset:
    """No network: audio_io streaming and the STT plugin loader are stubbed."""

    def _make_dataset_def(self):
        return DatasetDef.model_validate(dict(
            dataset_id="speech-massive-en-US", modality="intent",
            source={"type": "huggingface", "hf_id": "FBK-MT/Speech-MASSIVE-test",
                    "split": "test", "subset": "en-US"},
            reference_fields={"utterance": "utt", "intent": "intent_str",
                              "audio": "audio"},
            lang="en-US", input="audio",
            stt_plugin="ovos-stt-plugin-onnx-asr",
            stt_config={"model": "m"},
        ))

    def _samples(self):
        arr = np.zeros(1600, dtype=np.float32)
        yield "s1", {"array": arr, "sr": 16000, "expected_intent": "weather_query"}
        yield "s2", {"array": arr, "sr": 16000, "expected_intent": "alarm_set"}

    def test_transcribes_once_and_caches(self, tmp_path, monkeypatch):
        d = self._make_dataset_def()
        engine = _StubEngine(["what's the weather", "set an alarm"])
        monkeypatch.setattr(intent_bench, "_iter_audio_samples",
                            lambda *a, **k: self._samples())
        monkeypatch.setattr(intent_bench, "load_stt_engine",
                            lambda dataset_def, lang: engine)

        rows, stt_revision = intent_bench.transcribe_dataset(
            d, "en-US", "rev123", tmp_path)
        assert engine.calls == 2
        assert [r["utterance"] for r in rows] == [
            "what's the weather", "set an alarm"]
        assert [r["expected_intent"] for r in rows] == [
            "weather_query", "alarm_set"]

        cache_file = intent_bench.transcript_cache_path(
            tmp_path, d.dataset_id, "en-US")
        assert cache_file.exists()
        cached = [json.loads(line) for line in cache_file.read_text().splitlines()]
        assert len(cached) == 2

        # Second call (e.g. a second fighter) MUST reuse the cache — no new
        # STT calls — proving one transcription pass is shared across fighters.
        rows2, _ = intent_bench.transcribe_dataset(d, "en-US", "rev123", tmp_path)
        assert engine.calls == 2  # unchanged
        assert rows2 == rows

    def test_row_carries_stt_provenance(self):
        row = intent_bench.make_row(
            _FakeCompetitor(), "speech-massive-en-US", "en-US", 0,
            {"utterance": "what's the weather", "expected_intent": "weather_query",
             "split": "test"},
            "weather_query", {}, None, 5.0,
            "ovos-padacioso-pipeline-plugin-medium", "rev123",
            stt_provenance={
                "stt_plugin": "ovos-stt-plugin-onnx-asr",
                "stt_config": {"model": "m"},
                "stt_revision": "ovos-stt-plugin-onnx-asr==0.2.1a1",
            },
        )
        assert row["stt_plugin"] == "ovos-stt-plugin-onnx-asr"
        assert row["stt_config"] == {"model": "m"}
        assert row["stt_revision"] == "ovos-stt-plugin-onnx-asr==0.2.1a1"
        assert row["exact_match"] is True


class _FakeCompetitor:
    competitor_id = "padacioso-medium"
    pipeline_plugins = ["ovos-padacioso-pipeline-plugin"]

    class modality:
        value = "intent_template"

    plugin = "ovos-padacioso-pipeline-plugin"
    pipeline = ["ovos-padacioso-pipeline-plugin-medium"]
