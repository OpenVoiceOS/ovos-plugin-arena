"""Tests for ingestion layer updates — competitor alias resolution and JSONL ingest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
REPO_ROOT = Path(__file__).parent.parent
for p in (str(BACKEND), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.arena import db as arena_db
from app.arena import ingestion
from app.arena.models import PluginFamily


# ---------------------------------------------------------------------------
# _resolve_competitor_id
# ---------------------------------------------------------------------------


class TestResolveCompetitorId:
    def test_known_plugin_resolves_to_competitor(self):
        rr = REPO_ROOT / "registry"
        # fasterwhisper is registered in registry/competitors/stt/
        result = ingestion._resolve_competitor_id(
            "ovos-stt-plugin-fasterwhisper", PluginFamily.STT, registry_root=rr
        )
        # Should resolve to one of the fasterwhisper competitors
        assert "fasterwhisper" in result

    def test_unknown_plugin_returns_unchanged(self):
        rr = REPO_ROOT / "registry"
        result = ingestion._resolve_competitor_id(
            "totally-unknown-plugin-xyz-abc", PluginFamily.STT, registry_root=rr
        )
        assert result == "totally-unknown-plugin-xyz-abc"

    def test_missing_registry_returns_unchanged(self, tmp_path):
        # Non-existent registry dir → fallback gracefully; plugin not found
        result = ingestion._resolve_competitor_id(
            "totally-unknown-plugin-xyz", PluginFamily.STT,
            registry_root=tmp_path / "nonexistent"
        )
        assert result == "totally-unknown-plugin-xyz"


# ---------------------------------------------------------------------------
# ingest_jsonl
# ---------------------------------------------------------------------------


def _make_intent_jsonl(path: Path, competitor_id: str, n: int = 5):
    rows = []
    for i in range(n):
        rows.append({
            "competitor_id": competitor_id,
            "sample_id": f"s_{i:04d}",
            "dataset_id": "clinc150-en",
            "lang": "en-US",
            "plugin_id": "ovos-adapt-pipeline-plugin",
            "plugin_version": "ovos-adapt-pipeline-plugin/default",
            "utterance": f"utterance {i}",
            "reference_intent": "greeting",
            "prediction": "greeting" if i % 2 == 0 else "",
            "exact_match": i % 2 == 0,
            "entity_f1": 0.0,
            "runner_version": "0.1.0",
            "created_at": "2026-01-01T00:00:00",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


class TestIngestJsonl:
    def test_ingests_all_rows(self, tmp_db, tmp_path):
        jsonl = tmp_path / "adapt-default-en.jsonl"
        _make_intent_jsonl(jsonl, "adapt-default-en", n=4)

        src = ingestion.ingest_jsonl(
            jsonl,
            modality=PluginFamily.INTENT,
            lang="en-US",
        )
        assert src.row_count == 4
        preds = arena_db.list_predictions_for_source(src.id)
        assert len(preds) == 4

    def test_registers_plugin(self, tmp_db, tmp_path):
        jsonl = tmp_path / "adapt-default-en.jsonl"
        _make_intent_jsonl(jsonl, "adapt-default-en", n=3)

        ingestion.ingest_jsonl(jsonl, PluginFamily.INTENT, "en-US")
        plugin = arena_db.get_plugin_by_name("ovos-adapt-pipeline-plugin")
        assert plugin is not None

    def test_idempotent(self, tmp_db, tmp_path):
        jsonl = tmp_path / "adapt-default-en.jsonl"
        _make_intent_jsonl(jsonl, "adapt-default-en", n=4)

        ingestion.ingest_jsonl(jsonl, PluginFamily.INTENT, "en-US")
        ingestion.ingest_jsonl(jsonl, PluginFamily.INTENT, "en-US")

        # source row_count reflects last run; distinct predictions should be 4
        src = arena_db.get_prediction_source_by_dataset(f"local:{jsonl}", "local")
        preds = arena_db.list_predictions_for_source(src.id)
        assert len(preds) == 4

    def test_empty_file_raises(self, tmp_db, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(ingestion.IngestionError):
            ingestion.ingest_jsonl(empty, PluginFamily.INTENT, "en-US")

    def test_exact_match_stored_in_metrics(self, tmp_db, tmp_path):
        jsonl = tmp_path / "adapt-default-en.jsonl"
        _make_intent_jsonl(jsonl, "adapt-default-en", n=2)

        src = ingestion.ingest_jsonl(jsonl, PluginFamily.INTENT, "en-US")
        preds = arena_db.list_predictions_for_source(src.id)
        # First row: exact_match=True (i=0, i%2==0)
        first = next((p for p in preds if p.sample_id == "s_0000"), None)
        assert first is not None
        assert first.metrics.get("exact_match") == 1
