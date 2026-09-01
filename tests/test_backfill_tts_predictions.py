"""Unit tests for runner.backfill_tts_predictions — reconstructing TTS
prediction rows for (lang, competitor) pairs whose stored audio has no row
(or is missing some rows) in predictions/<lang>/<competitor>.jsonl.

The real judges (UTMOS/SIGMOS/DNSMOS/NISQA/onnx-asr) are monkeypatched, per
§4 test policy (never skip/importorskip on optional deps). A schema-parity
test proves a backfilled row carries the exact same keys as a row written
by the live ``TTSBench.predict`` — it MUST fail if the two writers' row
shapes drift apart. A partial-shard test proves a pair with SOME existing
rows only gets new rows for the wavs missing one, and its existing rows
survive byte-for-byte in the merged output.
"""
from __future__ import annotations

import json

import pytest

from registry.schemas import CompetitorDef, DatasetDef
from runner import backfill_tts_predictions as backfill
from runner import tts_bench


class FakeJudge:
    sample_rate = 16000

    def __init__(self, score=4.2):
        self.score = score

    def __call__(self, wav_path, sr):
        return self.score


@pytest.fixture(autouse=True)
def _fake_judges(monkeypatch):
    fake_utmos = FakeJudge(4.37)
    monkeypatch.setattr(tts_bench, "_utmos_judge", None)
    monkeypatch.setattr(tts_bench, "_get_utmos_judge", lambda: fake_utmos)
    monkeypatch.setattr(backfill, "_get_utmos_judge", lambda: fake_utmos)

    quality_extras = {"sigmos.ovrl": 4.1, "dnsmos.ovrl": 3.3, "nisqa.mos": 4.4}
    monkeypatch.setattr(tts_bench, "_score_quality_dimensions",
                         lambda p: dict(quality_extras))
    monkeypatch.setattr(backfill, "_score_quality_dimensions",
                         lambda p: dict(quality_extras))

    def fake_intelligibility(wav_path, prompt_text, lang):
        return {
            "wer": 0.1,
            "cer": 0.05,
            "judge_model_id": "fake-asr-model",
            "judge_revision": "abc123",
            "judges": [
                {"model": "fake-asr-model", "revision": "abc123",
                 "transcript": prompt_text},
            ],
            "consensus": prompt_text,
            "agreement": 1.0,
        }

    monkeypatch.setattr(tts_bench, "_score_intelligibility", fake_intelligibility)
    monkeypatch.setattr(backfill, "_score_intelligibility", fake_intelligibility)
    yield


@pytest.fixture
def dataset_def():
    return DatasetDef.model_validate({
        "dataset_id": "fake-prompts",
        "modality": "tts",
        "source": {
            "type": "huggingface",
            "hf_id": "OpenVoiceOS/fake-prompts",
            "revision": "main",
            "file_pattern": "{lang}/test.jsonl",
        },
        "reference_fields": {"text": "utterance"},
        "lang": "multi",
        "langs": ["en-US"],
        "license": "apache-2.0",
        "role": "eval",
        "predictions_hf": "OpenVoiceOS/ovos-tts-bench-fake-prompts",
    })


@pytest.fixture
def competitor():
    return CompetitorDef.model_validate({
        "competitor_id": "fake-voice",
        "modality": "tts",
        "plugin": "ovos-tts-plugin-fake",
        "config": {"lang": "en-US", "tts": {"module": "ovos-tts-plugin-fake"}},
        "langs": ["en-US"],
        "display_name": "Fake voice",
    })


def _write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF....WAVEfmt ")


class TestFindPairs:
    def test_maps_pairs_to_their_wav_hashes(self, monkeypatch):
        files = [
            "audio/en-US/voice_a/hash1.wav",
            "audio/en-US/voice_a/hash2.wav",
            "audio/en-US/voice_b/hash3.wav",
            "predictions/en-US/voice_a.jsonl",
        ]

        class FakeApi:
            def list_repo_files(self, repo_id, repo_type):
                return files

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

        pairs = backfill.find_pairs("OpenVoiceOS/ovos-tts-bench-d")
        assert pairs == {
            ("en-US", "voice_a"): {"hash1", "hash2"},
            ("en-US", "voice_b"): {"hash3"},
        }

    def test_no_audio_returns_empty(self, monkeypatch):
        class FakeApi:
            def list_repo_files(self, repo_id, repo_type):
                return ["predictions/en-US/voice_a.jsonl"]

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

        assert backfill.find_pairs("OpenVoiceOS/ovos-tts-bench-d") == {}


class TestPairStatus:
    def test_wav_with_no_row_is_missing(self, monkeypatch, dataset_def):
        prompts = ["hello world", "how are you"]
        monkeypatch.setattr(backfill, "_load_prompts",
                             lambda dd, lang, rev, col: prompts)
        wav_hashes = {tts_bench._safe(p) for p in prompts}

        status = backfill.pair_status(dataset_def, "deadbeef", "en-US",
                                       "fake-voice", wav_hashes,
                                       existing_sample_ids=set())

        assert status["wavs"] == 2
        assert status["existing"] == 0
        assert status["missing"] == [(0, prompts[0]), (1, prompts[1])]

    def test_wav_with_existing_row_is_not_missing(self, monkeypatch, dataset_def):
        prompts = ["hello world", "how are you"]
        monkeypatch.setattr(backfill, "_load_prompts",
                             lambda dd, lang, rev, col: prompts)
        wav_hashes = {tts_bench._safe(p) for p in prompts}

        status = backfill.pair_status(
            dataset_def, "deadbeef", "en-US", "fake-voice", wav_hashes,
            existing_sample_ids={"en-US/00000"})

        assert status["existing"] == 1
        assert status["missing"] == [(1, prompts[1])]

    def test_prompt_with_no_wav_is_not_missing(self, monkeypatch, dataset_def):
        prompts = ["hello world", "never rendered"]
        monkeypatch.setattr(backfill, "_load_prompts",
                             lambda dd, lang, rev, col: prompts)
        wav_hashes = {tts_bench._safe(prompts[0])}  # only the first was rendered

        status = backfill.pair_status(dataset_def, "deadbeef", "en-US",
                                       "fake-voice", wav_hashes,
                                       existing_sample_ids=set())

        assert status["missing"] == [(0, prompts[0])]


class TestBackfillPair:
    def test_produces_one_row_per_missing_entry(
            self, tmp_path, monkeypatch, dataset_def, competitor):
        text = "hello world"
        monkeypatch.setattr(backfill, "load_competitor",
                             lambda modality, cid: competitor)

        audio_dir = tmp_path / "audio"
        _write_wav(audio_dir / f"{tts_bench._safe(text)}.wav")

        rows = backfill.backfill_pair(dataset_def, "en-US", "fake-voice",
                                       audio_dir, "deadbeef",
                                       missing=[(0, text)])

        assert len(rows) == 1
        assert rows[0]["sample_id"] == "en-US/00000"
        assert rows[0]["input_text"] == text

    def test_row_schema_matches_live_bench_writer(
            self, tmp_path, monkeypatch, dataset_def, competitor):
        """The backfilled row must carry exactly the keys a live
        TTSBench.predict()+make_row() row carries (module/extras keys
        included) — this is the regression guard against schema drift
        between the live writer and the backfill tool."""
        text = "hello world"
        monkeypatch.setattr(backfill, "load_competitor",
                             lambda modality, cid: competitor)

        audio_dir = tmp_path / "audio"
        wav_path = audio_dir / f"{tts_bench._safe(text)}.wav"
        _write_wav(wav_path)

        backfilled_rows = backfill.backfill_pair(
            dataset_def, "en-US", "fake-voice", audio_dir, "deadbeef",
            missing=[(0, text)])
        assert len(backfilled_rows) == 1
        backfilled = backfilled_rows[0]

        # Live writer: same competitor, same wav, run through the real
        # adapter (its own get_tts is faked to reuse the identical bytes).
        class ReplayEngine:
            def get_tts(self, text, wav_path_out, lang=None):
                pass  # wav_path_out already exists (ctx.audio_dir == audio_dir)

        ctx = tts_bench.PredictContext(
            competitor, "en-US", "fake-prompts", "tts", tmp_path,
            "OpenVoiceOS/ovos-tts-bench-fake-prompts",
        )
        # TTSBench.predict writes to ctx.audio_dir / lang / competitor_id / hash.wav
        live_wav = tmp_path / "en-US" / "fake-voice" / f"{tts_bench._safe(text)}.wav"
        live_wav.parent.mkdir(parents=True)
        live_wav.write_bytes(wav_path.read_bytes())
        live_fields = tts_bench.TTSBench().predict(
            ReplayEngine(), {"input_text": text}, ctx)
        live_row = backfill.make_row(
            competitor, "fake-prompts", "en-US", "en-US/00000", "deadbeef",
            live_fields, modality="tts")

        assert set(backfilled.keys()) == set(live_row.keys())
        assert set(backfilled["extras"].keys()) >= {
            "utmos", "utmos_judge", "utmos_judge_revision",
            "sigmos.ovrl", "dnsmos.ovrl", "nisqa.mos",
            "intelligibility_wer", "intelligibility_cer",
            "intelligibility_judge", "intelligibility_judge_revision",
        }
        assert set(backfilled["extras"].keys()) == set(live_row["extras"].keys())
        assert backfilled["competitor_id"] == live_row["competitor_id"]
        assert backfilled["plugin_id"] == live_row["plugin_id"]
        assert backfilled["modality"] == live_row["modality"]


class TestWriteMerged:
    def test_partial_shard_keeps_existing_rows_and_adds_only_missing_ones(
            self, tmp_path, monkeypatch, dataset_def, competitor):
        """5 wavs on disk, 3 already have rows: the merged output must have
        exactly 2 new rows and the original 3 rows byte-for-byte unchanged.

        This is the regression this fix is for — the old tool only checked
        whether a predictions file existed at all, so a partially-scored
        pair (JSONL present, some rows missing because a judge raised
        mid-run) was invisible to it. This test FAILS against that code.
        """
        monkeypatch.setattr(backfill, "load_competitor",
                             lambda modality, cid: competitor)

        prompts = [f"prompt {i}" for i in range(5)]
        monkeypatch.setattr(backfill, "_load_prompts",
                             lambda dd, lang, rev, col: prompts)
        audio_dir = tmp_path / "audio"
        for text in prompts:
            _write_wav(audio_dir / f"{tts_bench._safe(text)}.wav")

        # rows 0,1,2 already exist verbatim (arbitrary key order/spacing,
        # to prove they are copied through untouched, not re-serialised)
        existing_lines = [
            json.dumps({"sample_id": "en-US/00000", "input_text": prompts[0],
                        "extras": {"utmos": 3.9}}),
            json.dumps({"sample_id": "en-US/00001", "input_text": prompts[1],
                        "extras": {"utmos": 4.0}}),
            json.dumps({"sample_id": "en-US/00002", "input_text": prompts[2],
                        "extras": {"utmos": 4.1}}),
        ]
        existing_sample_ids = {"en-US/00000", "en-US/00001", "en-US/00002"}

        status = backfill.pair_status(dataset_def, "deadbeef", "en-US",
                                       "fake-voice",
                                       {tts_bench._safe(p) for p in prompts},
                                       existing_sample_ids)
        assert status["wavs"] == 5
        assert status["existing"] == 3
        assert [i for i, _ in status["missing"]] == [3, 4]

        new_rows = backfill.backfill_pair(dataset_def, "en-US", "fake-voice",
                                          audio_dir, "deadbeef",
                                          status["missing"])
        assert len(new_rows) == 2
        assert {r["sample_id"] for r in new_rows} == {"en-US/00003", "en-US/00004"}

        out_path = tmp_path / "out" / "en-US" / "fake-voice.jsonl"
        backfill.write_merged(existing_lines, new_rows, out_path)

        written_lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(written_lines) == 5
        # the first 3 lines are exactly what was already there, unchanged
        assert written_lines[:3] == existing_lines
        new_written = [json.loads(line) for line in written_lines[3:]]
        assert {r["sample_id"] for r in new_written} == {"en-US/00003", "en-US/00004"}


class TestUploadPair:
    def test_uploads_jsonl_to_predictions_path(self, tmp_path, monkeypatch):
        jsonl_path = tmp_path / "en-US" / "voice_a.jsonl"
        jsonl_path.parent.mkdir(parents=True)
        jsonl_path.write_text('{"sample_id": "en-US/00000"}\n')

        calls = []

        class FakeApi:
            def upload_file(self, **kwargs):
                calls.append(kwargs)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

        backfill.upload_pair("OpenVoiceOS/ovos-tts-bench-d", jsonl_path,
                              "en-US", "voice_a")
        assert len(calls) == 1
        assert calls[0]["path_in_repo"] == "predictions/en-US/voice_a.jsonl"
        assert calls[0]["repo_id"] == "OpenVoiceOS/ovos-tts-bench-d"
