"""Unit tests for the prediction runner — pure logic only (no STT inference)."""
from __future__ import annotations

import json
import textwrap



# ---------------------------------------------------------------------------
# schema.STTRow
# ---------------------------------------------------------------------------

class TestSTTRow:
    def _make(self, **kwargs):
        from runner.schema import STTRow
        defaults = dict(
            dataset_entry_id="audio_001.wav",
            plugin_name="ovos-stt-plugin-fasterwhisper",
            model_id="ovos-stt-plugin-fasterwhisper/small",
            prediction_transcript="olá mundo",
            transcript="olá mundo",
            prediction_confidence=0.95,
        )
        defaults.update(kwargs)
        return STTRow(**defaults)

    def test_to_dict_has_required_columns(self):
        row = self._make()
        d = row.to_dict()
        for col in [
            "dataset_entry_id", "plugin_name", "model_id",
            "prediction_transcript", "transcript",
            "prediction_confidence", "prediction_type",
        ]:
            assert col in d, f"missing column: {col}"

    def test_prediction_type_default(self):
        row = self._make()
        assert row.to_dict()["prediction_type"] == "STT"

    def test_to_jsonl_round_trip(self):
        row = self._make(lang="pt-PT", dataset_id="PolyAI/minds14/pt-PT/train/pt-PT")
        obj = json.loads(row.to_jsonl())
        assert obj["dataset_entry_id"] == "audio_001.wav"
        assert obj["lang"] == "pt-PT"

    def test_fields_list(self):
        from runner.schema import STTRow
        fields = STTRow.fields()
        assert "dataset_entry_id" in fields
        assert "prediction_transcript" in fields
        assert "transcript" in fields


# ---------------------------------------------------------------------------
# schema.JobManifest
# ---------------------------------------------------------------------------

class TestJobManifest:
    def test_new_manifest_empty(self, tmp_path):
        from runner.schema import JobManifest
        m = JobManifest.load(tmp_path, "plugin|model|dataset")
        assert len(m.done_ids) == 0

    def test_mark_done_persists(self, tmp_path):
        from runner.schema import JobManifest
        m = JobManifest.load(tmp_path, "p|m|d")
        m.mark_done("sample_001.wav", tmp_path)
        # Reload from disk
        m2 = JobManifest.load(tmp_path, "p|m|d")
        assert m2.is_done("sample_001.wav")

    def test_is_done_false_for_unknown(self, tmp_path):
        from runner.schema import JobManifest
        m = JobManifest.load(tmp_path, "p|m|d")
        assert not m.is_done("never_seen.wav")

    def test_save_and_reload_idempotent(self, tmp_path):
        from runner.schema import JobManifest
        m = JobManifest.load(tmp_path, "p|m|d")
        for i in range(5):
            m.mark_done(f"s_{i}.wav", tmp_path)
        m2 = JobManifest.load(tmp_path, "p|m|d")
        assert m2.done_ids == {f"s_{i}.wav" for i in range(5)}

    def test_job_key_special_chars(self, tmp_path):
        from runner.schema import JobManifest
        key = "ovos-stt-plugin-fasterwhisper|my-north-ai/whisper|PolyAI/minds14/pt-PT/train/pt-PT"
        m = JobManifest.load(tmp_path, key)
        m.mark_done("x.wav", tmp_path)
        m2 = JobManifest.load(tmp_path, key)
        assert m2.is_done("x.wav")


# ---------------------------------------------------------------------------
# queue_config.load_queue
# ---------------------------------------------------------------------------

class TestLoadQueue:
    def test_load_minimal_queue(self, tmp_path):
        from runner.queue_config import load_queue
        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - plugin:
                  plugin_name: ovos-stt-plugin-fasterwhisper
                  model_name: small
                  lang: pt-PT
                dataset:
                  hf_repo: PolyAI/minds14
                  subset: pt-PT
                  split: train
                  ground_truth_key: transcription
                hf_output_dataset: OpenVoiceOS/ovos-stt-bench-pt-PT
        """))
        jobs = load_queue(q)
        assert len(jobs) == 1
        assert jobs[0].plugin.plugin_name == "ovos-stt-plugin-fasterwhisper"
        assert jobs[0].plugin.model_name == "small"
        assert jobs[0].plugin.lang == "pt-PT"
        assert jobs[0].hf_output_dataset == "OpenVoiceOS/ovos-stt-bench-pt-PT"

    def test_dataset_id_construction(self, tmp_path):
        from runner.queue_config import load_queue
        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - plugin:
                  plugin_name: ovos-stt-plugin-fasterwhisper
                  model_name: small
                  lang: pt-PT
                dataset:
                  hf_repo: PolyAI/minds14
                  subset: pt-PT
                  split: train
                  ground_truth_key: transcription
        """))
        jobs = load_queue(q)
        assert jobs[0].dataset.dataset_id == "PolyAI/minds14/pt-PT/train"

    def test_default_hf_output_dataset(self, tmp_path):
        from runner.queue_config import load_queue
        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - plugin:
                  plugin_name: ovos-stt-plugin-fasterwhisper
                  model_name: small
                  lang: pt-PT
                dataset:
                  hf_repo: PolyAI/minds14
        """))
        jobs = load_queue(q)
        assert jobs[0].hf_output_dataset == "OpenVoiceOS/ovos-stt-bench-pt-PT"

    def test_empty_queue(self, tmp_path):
        from runner.queue_config import load_queue
        q = tmp_path / "queue.yaml"
        q.write_text("jobs: []\n")
        assert load_queue(q) == []

    def test_multiple_jobs(self, tmp_path):
        from runner.queue_config import load_queue
        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - plugin:
                  plugin_name: plugin-a
                  model_name: m1
                  lang: pt-PT
                dataset:
                  hf_repo: ds/a
              - plugin:
                  plugin_name: plugin-b
                  model_name: m2
                  lang: en-US
                dataset:
                  hf_repo: ds/b
        """))
        jobs = load_queue(q)
        assert len(jobs) == 2
        assert jobs[1].plugin.lang == "en-US"
