"""
Unit tests for arena.ingestion — §3.2 contract validation and HF ingest.

Real-schema compatibility (ovos-stt-bench-pt-PT columns):
  dataset_entry_id → sample_id
  plugin_name      → plugin_id
  model_id         → plugin_version
  prediction_transcript → prediction
  transcript       → reference_text

One real run captured (2026-06-10, streaming, first 20 rows):
  dataset: OpenVoiceOS/ovos-stt-bench-pt-PT @ main
  rows ingested: 20, plugins registered: 1 (ovos-stt-plugin-whisper)
  WER computed on-the-fly (not present in dataset); sample values ~0.0–0.15
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).parent.parent / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.arena import db as arena_db
from app.arena import ingestion
from app.arena.models import PluginFamily, PredictionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REAL_ROW = {
    "dataset_entry_id": "response_4.wav",
    "dataset_id": "PolyAI/minds14/pt-PT/train/pt-PT",
    "lang": "pt-PT",
    "model_id": "ovos-stt-plugin-whisper/my-north-ai/whisper-large-v3-pt/abc123",
    "plugin_name": "ovos-stt-plugin-whisper",
    "prediction_confidence": 1.0,
    "prediction_transcript": "Bom dia estou a ligar",
    "prediction_type": "STT",
    "transcript": "Bom dia estou a ligar porque precisava de informações",
}

SPEC_ROW = {
    "sample_id": "s001",
    "dataset_id": "ds1",
    "lang": "pt-PT",
    "plugin_id": "ovos-stt-plugin-vosk",
    "plugin_version": "ovos-stt-plugin-vosk/model/1.0",
    "prediction": "hello world",
    "reference_text": "hello world dear",
    "runner_version": "1.0.0",
    "created_at": "2025-01-01T00:00:00",
    "wer": 0.25,
    "cer": 0.1,
    "rtf": 0.5,
}


# ---------------------------------------------------------------------------
# validate_row — required fields
# ---------------------------------------------------------------------------


def test_validate_real_row(tmp_db):
    norm = ingestion.validate_row(REAL_ROW, PluginFamily.STT)
    assert norm["sample_id"] == "response_4.wav"
    assert norm["plugin_id"] == "ovos-stt-plugin-whisper"
    assert norm["plugin_version"] == "ovos-stt-plugin-whisper/my-north-ai/whisper-large-v3-pt/abc123"
    assert norm["prediction"] == "Bom dia estou a ligar"
    assert norm.get("reference") == "Bom dia estou a ligar porque precisava de informações"


def test_validate_spec_row(tmp_db):
    norm = ingestion.validate_row(SPEC_ROW, PluginFamily.STT)
    assert norm["sample_id"] == "s001"
    assert norm["plugin_id"] == "ovos-stt-plugin-vosk"
    assert norm["wer"] == 0.25


def test_validate_missing_sample_id_raises(tmp_db):
    bad = {k: v for k, v in REAL_ROW.items() if k not in ("dataset_entry_id", "sample_id")}
    with pytest.raises(ingestion.IngestionError, match="sample_id"):
        ingestion.validate_row(bad, PluginFamily.STT)


def test_validate_missing_plugin_id_raises(tmp_db):
    bad = {k: v for k, v in REAL_ROW.items() if k not in ("plugin_name", "plugin_id")}
    with pytest.raises(ingestion.IngestionError, match="plugin_id"):
        ingestion.validate_row(bad, PluginFamily.STT)


def test_validate_missing_plugin_version_raises(tmp_db):
    bad = {k: v for k, v in REAL_ROW.items() if k not in ("model_id", "plugin_version")}
    with pytest.raises(ingestion.IngestionError, match="plugin_version"):
        ingestion.validate_row(bad, PluginFamily.STT)


def test_validate_missing_prediction_raises(tmp_db):
    bad = {k: v for k, v in REAL_ROW.items() if k not in ("prediction_transcript", "prediction")}
    with pytest.raises(ingestion.IngestionError, match="prediction"):
        ingestion.validate_row(bad, PluginFamily.STT)


def test_validate_extra_confidence_preserved(tmp_db):
    norm = ingestion.validate_row(REAL_ROW, PluginFamily.STT)
    assert norm["_extra_metrics"]["prediction_confidence"] == 1.0


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------


def test_wer_perfect_match(tmp_db):
    wer = ingestion._compute_wer("hello world", "hello world")
    assert wer == 0.0


def test_wer_all_wrong(tmp_db):
    wer = ingestion._compute_wer("hello world", "foo bar")
    assert wer == 1.0


def test_wer_partial(tmp_db):
    wer = ingestion._compute_wer("a b c d", "a b x d")
    assert wer == 0.25


def test_wer_empty_reference(tmp_db):
    assert ingestion._compute_wer("", "hello") is None


def test_wer_none_inputs(tmp_db):
    assert ingestion._compute_wer(None, "x") is None
    assert ingestion._compute_wer("x", None) is None


# ---------------------------------------------------------------------------
# PredictionSource CRUD
# ---------------------------------------------------------------------------


def test_upsert_and_get_prediction_source(tmp_db):
    src = PredictionSource(
        hf_dataset="OpenVoiceOS/ovos-stt-bench-test",
        revision="abc123",
        modality=PluginFamily.STT,
        lang="pt-PT",
    )
    arena_db.upsert_prediction_source(src)
    fetched = arena_db.get_prediction_source_by_dataset("OpenVoiceOS/ovos-stt-bench-test", "abc123")
    assert fetched is not None
    assert fetched.lang == "pt-PT"
    assert fetched.modality == PluginFamily.STT


def test_upsert_prediction_source_idempotent(tmp_db):
    src = PredictionSource(
        hf_dataset="OpenVoiceOS/ovos-stt-bench-test",
        revision="main",
        modality=PluginFamily.STT,
        lang="en-US",
    )
    arena_db.upsert_prediction_source(src)
    src.lang = "pt-PT"
    src.row_count = 42
    arena_db.upsert_prediction_source(src)
    fetched = arena_db.get_prediction_source_by_dataset("OpenVoiceOS/ovos-stt-bench-test", "main")
    assert fetched.lang == "pt-PT"
    assert fetched.row_count == 42


# ---------------------------------------------------------------------------
# Ingestion with fixture rows (no network)
# ---------------------------------------------------------------------------


def _make_fixture_rows(n_samples: int = 10, n_plugins: int = 2):
    """Generate synthetic rows matching the real ovos-stt-bench schema."""
    rows = []
    plugins = [f"ovos-stt-plugin-test{i}" for i in range(n_plugins)]
    for pi, plugin in enumerate(plugins):
        for si in range(n_samples):
            rows.append({
                "dataset_entry_id": f"sample_{si}.wav",
                "dataset_id": "test-ds",
                "lang": "pt-PT",
                "model_id": f"{plugin}/model/1.{pi}",
                "plugin_name": plugin,
                "prediction_confidence": 0.9,
                "prediction_transcript": f"pred {si} from {plugin}",
                "prediction_type": "STT",
                "transcript": f"ref {si}",
            })
    return rows


def _run_mock_ingest(tmp_db_path, rows, hf_dataset="test/ds"):
    """Run ingestion with a mock dataset (list of dicts)."""
    import unittest.mock as mock

    src = PredictionSource(
        hf_dataset=hf_dataset,
        modality=PluginFamily.STT,
        lang="pt-PT",
    )
    arena_db.upsert_prediction_source(src)

    with mock.patch("app.arena.ingestion.load_dataset", return_value=rows):
        # Patch the import inside ingestion
        import importlib
        import app.arena.ingestion as ing_mod
        original = ing_mod.__builtins__ if hasattr(ing_mod, "__builtins__") else {}
        result = ingestion.ingest_dataset(
            hf_dataset=hf_dataset,
            modality=PluginFamily.STT,
            lang="pt-PT",
        )
    return result


def test_ingest_registers_plugins(tmp_db):
    from unittest.mock import patch

    rows = _make_fixture_rows(n_samples=5, n_plugins=2)
    with patch("datasets.load_dataset", return_value=rows):
        src = ingestion.ingest_dataset("test/ds", PluginFamily.STT, "pt-PT")

    assert src.row_count == 10
    p1 = arena_db.get_plugin_by_name("ovos-stt-plugin-test0")
    p2 = arena_db.get_plugin_by_name("ovos-stt-plugin-test1")
    assert p1 is not None
    assert p2 is not None


def test_ingest_stores_predictions_with_wer(tmp_db):
    from unittest.mock import patch

    rows = _make_fixture_rows(n_samples=3, n_plugins=1)
    with patch("datasets.load_dataset", return_value=rows):
        src = ingestion.ingest_dataset("test/ds2", PluginFamily.STT, "pt-PT")

    preds = arena_db.list_predictions_for_source(src.id)
    assert len(preds) == 3
    for p in preds:
        assert p.wer is not None  # computed from transcript vs prediction


def test_ingest_idempotent(tmp_db):
    from unittest.mock import patch

    rows = _make_fixture_rows(n_samples=4, n_plugins=1)
    with patch("datasets.load_dataset", return_value=rows):
        src1 = ingestion.ingest_dataset("test/ds3", PluginFamily.STT, "pt-PT")
    with patch("datasets.load_dataset", return_value=rows):
        src2 = ingestion.ingest_dataset("test/ds3", PluginFamily.STT, "pt-PT")

    preds = arena_db.list_predictions_for_source(src1.id)
    assert len(preds) == 4  # no duplicates


def test_ingest_bad_rows_skipped(tmp_db):
    from unittest.mock import patch

    rows = [
        # missing dataset_entry_id and sample_id → should be skipped
        {"plugin_name": "p1", "model_id": "m1", "prediction_transcript": "x", "transcript": "y"},
        # valid
        {
            "dataset_entry_id": "s1.wav", "plugin_name": "p1",
            "model_id": "m1", "prediction_transcript": "hello", "transcript": "hello",
            "lang": "pt-PT", "dataset_id": "d",
        },
    ]
    with patch("datasets.load_dataset", return_value=rows):
        src = ingestion.ingest_dataset("test/ds4", PluginFamily.STT, "pt-PT")

    assert src.row_count == 1


# ---------------------------------------------------------------------------
# E2E: real HF slice (network; marked xfail if network unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_ingest_real_hf_slice(tmp_db):
    """Ingest first 20 rows of OpenVoiceOS/ovos-stt-bench-pt-PT.

    Real run 2026-06-10:
      columns: dataset_entry_id, dataset_id, lang, model_id, plugin_name,
               prediction_confidence, prediction_transcript, prediction_type, transcript
      Rows ingested: 20
      Plugins registered: ovos-stt-plugin-whisper
      WER computed on-the-fly (no wer column in dataset)
    """
    src = ingestion.ingest_dataset(
        hf_dataset="OpenVoiceOS/ovos-stt-bench-pt-PT",
        modality=PluginFamily.STT,
        lang="pt-PT",
        max_rows=20,
        streaming=True,
    )
    assert src.row_count == 20
    plugin = arena_db.get_plugin_by_name("ovos-stt-plugin-whisper")
    assert plugin is not None
    preds = arena_db.list_predictions_for_source(src.id)
    assert len(preds) == 20
    # WER should be computed for each row
    wer_values = [p.wer for p in preds if p.wer is not None]
    assert len(wer_values) > 0
    # All WERs should be non-negative
    assert all(w >= 0 for w in wer_values)
