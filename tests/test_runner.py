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


class TestPluginFromCompetitor:
    """Regression: fighters are complete mycroft.conf snippets — the plugin's
    settings (including its model) live nested under config.stt.<module>. A
    loader that only reads config["model"] hands the plugin
    model=competitor_id and vosk-style fighters explode with
    "Invalid model: vosk-small-it" on the runner (ser9, 2026-08-11)."""

    def _write_fighter(self, root, competitor_id, config, langs):
        import json
        d = root / "competitors" / "stt"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{competitor_id}.json").write_text(json.dumps({
            "competitor_id": competitor_id,
            "modality": "stt",
            "plugin": config["stt"]["module"],
            "config": config,
            "langs": langs,
        }))

    def test_nested_plugin_section_supplies_model_and_settings(self, tmp_path):
        from runner.queue_config import _plugin_from_competitor
        url = "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip"
        self._write_fighter(tmp_path, "vosk-small-it", {
            "lang": "it",
            "stt": {
                "module": "ovos-stt-plugin-vosk",
                "ovos-stt-plugin-vosk": {"model": url, "verbose": False},
            },
        }, ["it"])
        spec = _plugin_from_competitor("vosk-small-it", registry_root=tmp_path)
        assert spec.plugin_name == "ovos-stt-plugin-vosk"
        assert spec.model_name == url  # NOT the competitor_id
        assert spec.extra_config.get("verbose") is False
        # the raw mycroft.conf "stt" wrapper must not leak into the flat
        # plugin config
        assert "stt" not in spec.extra_config

    def test_top_level_model_still_wins_when_no_nested_model(self, tmp_path):
        from runner.queue_config import _plugin_from_competitor
        self._write_fighter(tmp_path, "fw-base", {
            "lang": "de",
            "model": "base",
            "stt": {
                "module": "ovos-stt-plugin-fasterwhisper",
                "ovos-stt-plugin-fasterwhisper": {"model": "base",
                                                  "compute_type": "int8"},
            },
        }, ["de"])
        spec = _plugin_from_competitor("fw-base", registry_root=tmp_path)
        assert spec.model_name == "base"
        assert spec.extra_config.get("compute_type") == "int8"

    def test_fighter_without_nested_section_falls_back_to_competitor_id(self, tmp_path):
        import json
        from runner.queue_config import _plugin_from_competitor
        d = tmp_path / "competitors" / "stt"
        d.mkdir(parents=True)
        (d / "bare.json").write_text(json.dumps({
            "competitor_id": "bare", "modality": "stt",
            "plugin": "ovos-stt-plugin-x", "config": {}, "langs": ["en-US"],
        }))
        spec = _plugin_from_competitor("bare", registry_root=tmp_path)
        assert spec.model_name == "bare"

    def test_lang_override_beats_fighter_config_lang(self, tmp_path):
        """The fighter's top-level config.lang must not leak into
        extra_config: _load_plugin builds {"lang": spec.lang, **extra_config},
        so a leaked key silently discards a queue-supplied lang override."""
        from runner.queue_config import _plugin_from_competitor
        self._write_fighter(tmp_path, "vosk-small-it", {
            "lang": "it",
            "stt": {
                "module": "ovos-stt-plugin-vosk",
                "ovos-stt-plugin-vosk": {"model": "http://x/y.zip"},
            },
        }, ["it"])
        spec = _plugin_from_competitor(
            "vosk-small-it", registry_root=tmp_path, lang_override="en-US")
        assert spec.lang == "en-US"
        assert "lang" not in spec.extra_config
        # without an override, the fighter's own config.lang still wins
        spec2 = _plugin_from_competitor("vosk-small-it", registry_root=tmp_path)
        assert spec2.lang == "it"


class TestRunJobAttribution:
    """Regression: rows from competitor-referenced jobs must carry an explicit
    competitor_id — ingestion's plugin_id alias fallback returns the FIRST
    fighter matching the plugin name, so every multi-fighter plugin (8
    fasterwhisper tiers, 5 vosk models) would collapse onto one competitor
    and mis-score its board."""

    def _run(self, tmp_path, monkeypatch, competitor_id):
        import json
        import numpy as np
        import runner.plugin_runner as pr
        from runner.queue_config import DatasetSpec, JobSpec, PluginSpec

        class FakeSTT:
            def execute(self, audio, language=None):
                return "hello world"

        monkeypatch.setattr(pr, "_load_plugin", lambda p: FakeSTT())
        monkeypatch.setattr(
            pr, "_stream_dataset",
            lambda d: iter([("s1", "hello world",
                             np.zeros(16000, dtype=np.float32), 16000)]))
        monkeypatch.setattr(
            pr, "_transcribe",
            lambda stt, a, sr, lang: ("hello world", 0.9))
        plugin = PluginSpec(
            plugin_name="ovos-stt-plugin-vosk",
            model_name="http://x/model.zip",
            lang="en-US",
            competitor_id=competitor_id,
        )
        dataset = DatasetSpec(hf_repo="fake/ds", subset="en-US", split="train",
                              ground_truth_key="transcription")
        out = pr.run_job(JobSpec(plugin=plugin, dataset=dataset,
                                 hf_output_dataset="fake/out"),
                         base_dir=tmp_path)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        return out, rows

    def test_competitor_job_rows_carry_competitor_id(self, tmp_path, monkeypatch):
        out, rows = self._run(tmp_path, monkeypatch, "vosk-small-it")
        assert rows and rows[0]["competitor_id"] == "vosk-small-it"
        # filename keyed by competitor_id, not the URL-mangled model name
        assert out.name == "stt_en-US_vosk-small-it.jsonl"

    def test_inline_job_rows_keep_legacy_shape(self, tmp_path, monkeypatch):
        out, rows = self._run(tmp_path, monkeypatch, None)
        assert rows and "competitor_id" not in rows[0]
        assert "vosk" in out.name and "model.zip" in out.name
