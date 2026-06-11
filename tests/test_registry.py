"""Tests for the declarative evaluation registry (schemas + loaders)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure registry/ is importable
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registry.schemas import (
    CompetitorDef,
    DatasetDef,
    HuggingFaceSource,
    Modality,
    PathSource,
)
from registry.loaders import (
    load_competitor,
    load_dataset,
    list_competitors,
    list_datasets,
    get_competitor_by_alias,
    REGISTRY_ROOT,
)


# ---------------------------------------------------------------------------
# CompetitorDef schema
# ---------------------------------------------------------------------------


class TestCompetitorDefSchema:
    def _valid(self, **kw):
        defaults = {
            "competitor_id": "fasterwhisper-small-pt",
            "modality": "stt",
            "plugin": "ovos-stt-plugin-fasterwhisper",
            "config": {"model": "small"},
            "langs": ["pt-PT"],
        }
        defaults.update(kw)
        return CompetitorDef(**defaults)

    def test_basic_valid(self):
        c = self._valid()
        assert c.competitor_id == "fasterwhisper-small-pt"
        assert c.modality == Modality.STT
        assert c.plugin == "ovos-stt-plugin-fasterwhisper"

    def test_alias_auto_includes_plugin(self):
        c = self._valid()
        assert c.alias is not None
        assert "ovos-stt-plugin-fasterwhisper" in c.alias

    def test_explicit_alias_preserved(self):
        c = self._valid(alias=["old-plugin-name"])
        assert "old-plugin-name" in c.alias
        assert "ovos-stt-plugin-fasterwhisper" in c.alias  # still added

    def test_missing_competitor_id_raises(self):
        with pytest.raises(Exception):
            CompetitorDef(
                modality="stt",
                plugin="x",
                langs=["en-US"],
            )

    def test_invalid_modality_raises(self):
        with pytest.raises(Exception):
            self._valid(modality="nonexistent")

    def test_notes_optional(self):
        c = self._valid(notes="some notes")
        assert c.notes == "some notes"
        c2 = self._valid()
        assert c2.notes is None


# ---------------------------------------------------------------------------
# DatasetDef schema
# ---------------------------------------------------------------------------


class TestDatasetDefSchema:
    def _hf_source(self, **kw):
        defaults = {
            "type": "huggingface",
            "hf_id": "PolyAI/minds14",
            "split": "train",
            "subset": "pt-PT",
        }
        defaults.update(kw)
        return defaults

    def _valid(self, **kw):
        defaults = {
            "dataset_id": "minds14-pt-PT",
            "modality": "stt",
            "source": self._hf_source(),
            "reference_fields": {"audio": "audio", "ground_truth": "transcription"},
            "lang": "pt-PT",
            "license": "cc-by-4.0",
            "role": "eval",
        }
        defaults.update(kw)
        return DatasetDef(**defaults)

    def test_basic_valid(self):
        d = self._valid()
        assert d.dataset_id == "minds14-pt-PT"
        assert d.lang == "pt-PT"
        assert d.role == "eval"

    def test_hf_source_discriminated(self):
        d = self._valid()
        assert isinstance(d.source, HuggingFaceSource)
        assert d.source.hf_id == "PolyAI/minds14"

    def test_path_source(self):
        d = DatasetDef(
            dataset_id="local-test",
            modality="intent",
            source={"type": "path", "path": "/data/test.jsonl"},
            reference_fields={"utterance": "text", "intent": "label"},
            lang="en-US",
        )
        assert isinstance(d.source, PathSource)

    def test_role_unrestricted(self):
        d = self._valid(role="unrestricted")
        assert d.role == "unrestricted"

    def test_invalid_role_raises(self):
        with pytest.raises(Exception):
            self._valid(role="invalid_role")

    def test_hf_source_dataset_id_str(self):
        d = self._valid()
        # HuggingFaceSource.dataset_id_str should produce a stable string
        assert d.source.dataset_id_str == "PolyAI/minds14/pt-PT/train"


# ---------------------------------------------------------------------------
# File-based loaders (use actual registry/ files in the repo)
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_load_competitor_stt(self):
        comp = load_competitor("stt", "fasterwhisper-small-pt")
        assert comp.competitor_id == "fasterwhisper-small-pt"
        assert comp.modality == Modality.STT
        assert comp.plugin == "ovos-stt-plugin-fasterwhisper"

    def test_load_competitor_stt_base(self):
        comp = load_competitor("stt", "fasterwhisper-base-pt")
        assert comp.competitor_id == "fasterwhisper-base-pt"
        assert "base" in comp.config.get("model", "")

    def test_load_competitor_intent(self):
        comp = load_competitor("intent", "adapt-medium")
        assert comp.competitor_id == "adapt-medium"
        assert comp.modality == Modality.INTENT
        # config is a valid mycroft.conf fragment; plugin derived from it
        assert comp.pipeline == ["ovos-adapt-pipeline-plugin-medium"]
        assert comp.plugin == "ovos-adapt-pipeline-plugin"
        assert comp.pipeline_plugins == ["ovos-adapt-pipeline-plugin"]
        assert comp.plugin_config("ovos-adapt-pipeline-plugin", "adapt") == {}
        assert comp.species == "AdaptPipeline"
        assert "GOFAI" in comp.types

    def test_load_competitor_ensemble(self):
        comp = load_competitor("intent", "ovos-default-cascade")
        assert comp.plugin is None  # multi-engine pipeline
        assert comp.pipeline_plugins == [
            "ovos-padatious-pipeline-plugin",
            "ovos-adapt-pipeline-plugin",
        ]
        assert "ensemble" in comp.types

    def test_intent_competitor_requires_pipeline(self):
        import pytest as _pytest
        from registry.schemas import CompetitorDef
        with _pytest.raises(Exception):
            CompetitorDef(
                competitor_id="bad", modality="intent",
                config={"intents": {}},
            )

    def test_split_pipeline_stage(self):
        from registry.schemas import split_pipeline_stage
        assert split_pipeline_stage("ovos-adapt-pipeline-plugin-high") == (
            "ovos-adapt-pipeline-plugin", "high",
        )
        import pytest as _pytest
        with _pytest.raises(ValueError):
            split_pipeline_stage("ovos-converse-pipeline-plugin")

    def test_load_competitor_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_competitor("stt", "nonexistent-competitor-xyz")

    def test_load_dataset_stt(self):
        ds = load_dataset("stt", "minds14-pt-PT")
        assert ds.dataset_id == "minds14-pt-PT"
        assert ds.lang == "pt-PT"
        assert ds.role == "eval"

    def test_load_dataset_intent(self):
        ds = load_dataset("intent", "intents-for-eval")
        assert ds.dataset_id == "intents-for-eval"
        assert ds.modality == Modality.INTENT
        assert ds.lang == "multi"
        assert ds.langs and "en-US" in ds.langs and len(ds.langs) == 12
        assert ds.role == "eval"

    def test_load_dataset_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("stt", "totally-missing-dataset-xyz")

    def test_list_competitors_stt(self):
        comps = list_competitors("stt")
        ids = [c.competitor_id for c in comps]
        assert "fasterwhisper-small-pt" in ids
        assert "fasterwhisper-base-pt" in ids

    def test_list_competitors_all(self):
        all_comps = list_competitors()
        mods = {c.modality for c in all_comps}
        assert Modality.STT in mods
        assert Modality.INTENT in mods

    def test_list_datasets(self):
        dsets = list_datasets()
        ids = [d.dataset_id for d in dsets]
        assert "minds14-pt-PT" in ids
        assert "intents-for-eval" in ids

    def test_get_competitor_by_alias_plugin_name(self):
        comp = get_competitor_by_alias("stt", "ovos-stt-plugin-fasterwhisper")
        # should return one of the fasterwhisper competitors
        assert comp is not None
        assert comp.plugin == "ovos-stt-plugin-fasterwhisper"

    def test_get_competitor_by_alias_missing(self):
        result = get_competitor_by_alias("stt", "totally-unknown-plugin-xyz-123")
        assert result is None


# ---------------------------------------------------------------------------
# Round-trip: JSON file → schema → JSON
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_competitor_roundtrip(self, tmp_path):
        comp = CompetitorDef(
            competitor_id="test-comp-rt",
            modality="stt",
            plugin="ovos-stt-plugin-test",
            config={"model": "tiny", "lang": "en-US"},
            langs=["en-US"],
            notes="round-trip test",
        )
        p = tmp_path / "competitors" / "stt" / "test-comp-rt.json"
        p.parent.mkdir(parents=True)
        p.write_text(comp.model_dump_json())
        loaded = CompetitorDef.model_validate(json.loads(p.read_text()))
        assert loaded.competitor_id == comp.competitor_id
        assert loaded.config == comp.config

    def test_dataset_roundtrip(self, tmp_path):
        ds = DatasetDef(
            dataset_id="test-ds-rt",
            modality="intent",
            source={"type": "huggingface", "hf_id": "test/data", "split": "test"},
            reference_fields={"utterance": "text", "intent": "label"},
            lang="en-US",
            license="mit",
            role="unrestricted",
        )
        p = tmp_path / "datasets" / "intent" / "test-ds-rt.json"
        p.parent.mkdir(parents=True)
        p.write_text(ds.model_dump_json())
        loaded = DatasetDef.model_validate(json.loads(p.read_text()))
        assert loaded.dataset_id == ds.dataset_id
        assert loaded.role == "unrestricted"


# ---------------------------------------------------------------------------
# Queue loader backward-compat with registry
# ---------------------------------------------------------------------------


class TestQueueConfigRegistry:
    def test_inline_job_unchanged(self, tmp_path):
        """Original inline format still works."""
        import textwrap
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
        jobs = load_queue(q, registry_root=REGISTRY_ROOT)
        assert len(jobs) == 1
        assert jobs[0].plugin.plugin_name == "ovos-stt-plugin-fasterwhisper"
        assert jobs[0].competitor_id is None  # inline job has no competitor_id

    def test_competitor_ref_job(self, tmp_path):
        """Registry-reference format resolves to correct plugin/config."""
        import textwrap
        from runner.queue_config import load_queue

        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - competitor: fasterwhisper-small-pt
                dataset_ref: minds14-pt-PT
                hf_output_dataset: OpenVoiceOS/ovos-stt-bench-pt-PT
                max_samples: 50
        """))
        jobs = load_queue(q, registry_root=REGISTRY_ROOT)
        assert len(jobs) == 1
        j = jobs[0]
        assert j.competitor_id == "fasterwhisper-small-pt"
        assert j.dataset_registry_id == "minds14-pt-PT"
        assert j.plugin.plugin_name == "ovos-stt-plugin-fasterwhisper"
        assert j.plugin.model_name == "small"
        assert j.plugin.competitor_id == "fasterwhisper-small-pt"
        assert j.dataset.hf_repo == "PolyAI/minds14"
        assert j.dataset.max_samples == 50

    def test_mixed_competitor_inline_dataset(self, tmp_path):
        """Competitor + inline dataset block."""
        import textwrap
        from runner.queue_config import load_queue

        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - competitor: fasterwhisper-base-pt
                dataset:
                  hf_repo: PolyAI/minds14
                  subset: pt-PT
                  split: train
                hf_output_dataset: OpenVoiceOS/ovos-stt-bench-pt-PT
        """))
        jobs = load_queue(q, registry_root=REGISTRY_ROOT)
        assert jobs[0].plugin.plugin_name == "ovos-stt-plugin-fasterwhisper"
        assert jobs[0].plugin.model_name == "base"
        assert jobs[0].dataset.hf_repo == "PolyAI/minds14"
        assert jobs[0].dataset_registry_id is None  # inline dataset

    def test_competitor_not_found_raises(self, tmp_path):
        import textwrap
        from runner.queue_config import load_queue

        q = tmp_path / "queue.yaml"
        q.write_text(textwrap.dedent("""\
            jobs:
              - competitor: nonexistent-competitor-xyz
                dataset_ref: minds14-pt-PT
        """))
        with pytest.raises(FileNotFoundError):
            load_queue(q, registry_root=REGISTRY_ROOT)
