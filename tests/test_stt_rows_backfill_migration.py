"""Unit tests for runner.migrate.stt_rows_to_prediction_rows — pure-local,
no network. See docstring of the module under test for the migration plan.
"""
from __future__ import annotations

import json

from runner.migrate import stt_rows_to_prediction_rows as migrate_mod
from runner.migrate.stt_rows_to_prediction_rows import (
    MigrationResult,
    convert_row,
    is_already_migrated,
    migrate_dataset,
    migrate_file,
    write_jsonl,
)


def _legacy_row(**over) -> dict:
    row = {
        "dataset_entry_id": "sample_0001.wav",
        "plugin_name": "ovos-stt-plugin-fasterwhisper",
        "model_id": "ovos-stt-plugin-fasterwhisper/small",
        "prediction_transcript": "ola mundo",
        "transcript": "olá mundo",
        "prediction_confidence": 0.87,
        "prediction_type": "STT",
        "dataset_id": "PolyAI/minds14",
        "lang": "pt-PT",
    }
    row.update(over)
    return row


def _canonical_row(**over) -> dict:
    row = {
        "competitor_id": "fasterwhisper-small-pt",
        "sample_id": "sample_0001.wav",
        "dataset_id": "PolyAI/minds14",
        "lang": "pt-PT",
        "plugin_id": "ovos-stt-plugin-fasterwhisper",
        "modality": "stt",
        "prediction": "ola mundo",
        "reference_text": "olá mundo",
        "confidence": 0.87,
        "schema_version": 2,
        "extras": {"model_id": "ovos-stt-plugin-fasterwhisper/small"},
    }
    row.update(over)
    return row


class TestIsAlreadyMigrated:
    def test_legacy_row_is_not_migrated(self):
        assert is_already_migrated(_legacy_row()) is False

    def test_canonical_row_is_migrated(self):
        assert is_already_migrated(_canonical_row()) is True

    def test_canonical_row_without_explicit_schema_version_defaults_migrated(self):
        row = _canonical_row()
        del row["schema_version"]
        assert is_already_migrated(row) is True

    def test_schema_version_1_row_is_not_migrated(self):
        row = _canonical_row(schema_version=1)
        assert is_already_migrated(row) is False


class TestConvertRow:
    def test_field_for_field_round_trip(self):
        legacy = _legacy_row()
        out = convert_row(legacy, competitor_id_fallback="fasterwhisper-small-pt")

        assert out["sample_id"] == legacy["dataset_entry_id"]
        assert out["plugin_id"] == legacy["plugin_name"]
        assert out["prediction"] == legacy["prediction_transcript"]
        assert out["reference_text"] == legacy["transcript"]
        assert out["confidence"] == legacy["prediction_confidence"]
        assert out["dataset_id"] == legacy["dataset_id"]
        assert out["lang"] == legacy["lang"]
        assert out["modality"] == "stt"
        assert out["extras"]["model_id"] == legacy["model_id"]
        assert out["extras"]["legacy_schema"] == "STTRow"
        # stamped as source-of-truth, not the read-time-shim provenance tag
        assert out["schema_version"] == 2

    def test_unresolvable_competitor_falls_back_to_filename_stem(self):
        legacy = _legacy_row(plugin_name="totally-unknown-plugin")
        out = convert_row(legacy, competitor_id_fallback="some-file-stem")
        assert out["competitor_id"] == "some-file-stem"

    def test_already_canonical_row_stays_schema_version_2(self):
        canonical = _canonical_row(schema_version=1)  # simulate read-time shim tag
        out = convert_row(canonical, competitor_id_fallback="fasterwhisper-small-pt")
        assert out["schema_version"] == 2
        assert out["sample_id"] == canonical["sample_id"]
        assert out["prediction"] == canonical["prediction"]


class TestMigrateFile:
    def test_legacy_shard_is_converted(self):
        rows = [_legacy_row(), _legacy_row(dataset_entry_id="sample_0002.wav")]
        migrated, skipped = migrate_file("fasterwhisper-small-pt.jsonl", rows)
        assert skipped is False
        assert len(migrated) == 2
        assert {r["sample_id"] for r in migrated} == {
            "sample_0001.wav", "sample_0002.wav",
        }

    def test_already_migrated_shard_is_a_no_op(self):
        rows = [_canonical_row(), _canonical_row(sample_id="sample_0002.wav")]
        migrated, skipped = migrate_file("fasterwhisper-small-pt.jsonl", rows)
        assert skipped is True
        assert migrated == []

    def test_empty_shard_is_converted_not_skipped(self):
        # all() over an empty rows list is vacuously True; guard explicitly
        # excludes the empty case from "skip" so an empty shard still
        # produces an (empty) migrated output rather than silently vanishing.
        migrated, skipped = migrate_file("empty.jsonl", [])
        assert migrated == []
        assert skipped is False


class TestWriteJsonlStability:
    def test_sort_keys_byte_stable_across_key_order(self, tmp_path):
        row_a = _canonical_row()
        row_b = dict(reversed(list(row_a.items())))  # same content, different order

        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"
        write_jsonl([row_a], path_a)
        write_jsonl([row_b], path_b)

        assert path_a.read_text() == path_b.read_text()

    def test_one_row_per_line(self, tmp_path):
        rows = [_canonical_row(), _canonical_row(sample_id="s2")]
        path = tmp_path / "out.jsonl"
        write_jsonl(rows, path)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        for line, row in zip(lines, rows, strict=True):
            assert json.loads(line)["sample_id"] == row["sample_id"]


class TestMigrateDatasetIdempotency:
    """migrate_dataset drives download -> migrate_file -> (optional) push.
    All network entry points are monkeypatched so these stay pure-local."""

    def _patch_source(self, monkeypatch, shards: dict[str, list[dict]], sha="abc123"):
        monkeypatch.setattr(migrate_mod, "_source_revision", lambda repo_id, revision="main": sha)
        monkeypatch.setattr(
            migrate_mod, "download_legacy_files",
            lambda repo_id, revision="main": shards,
        )

    def test_second_run_on_migrated_data_is_a_no_op(self, monkeypatch, tmp_path):
        shards = {"fw.jsonl": [_canonical_row()]}
        self._patch_source(monkeypatch, shards)

        def _boom(*a, **kw):
            raise AssertionError("push_migrated must not be called")
        monkeypatch.setattr(migrate_mod, "push_migrated", _boom)

        result = migrate_dataset(
            "OpenVoiceOS/ovos-stt-bench-pt-PT", out_dir=tmp_path, apply=True,
        )
        assert result.files_skipped_idempotent == 1
        assert result.rows_migrated == 0
        assert result.output_files == []
        assert result.applied is False

    def test_dry_run_performs_zero_network_writes(self, monkeypatch, tmp_path):
        shards = {"fw.jsonl": [_legacy_row()]}
        self._patch_source(monkeypatch, shards)

        def _boom(*a, **kw):
            raise AssertionError("push_migrated must not be called on dry-run")
        monkeypatch.setattr(migrate_mod, "push_migrated", _boom)

        result = migrate_dataset(
            "OpenVoiceOS/ovos-stt-bench-pt-PT", out_dir=tmp_path, apply=False,
        )
        assert result.rows_migrated == 1
        assert result.applied is False
        assert result.new_revision == ""
        # output was written locally, but no network call happened (no raise)
        assert len(result.output_files) == 1

    def test_apply_pushes_and_stamps_new_revision(self, monkeypatch, tmp_path):
        shards = {"fw.jsonl": [_legacy_row()]}
        self._patch_source(monkeypatch, shards)
        monkeypatch.setattr(
            migrate_mod, "push_migrated",
            lambda repo_id, files, token=None: "def456",
        )

        result = migrate_dataset(
            "OpenVoiceOS/ovos-stt-bench-pt-PT", out_dir=tmp_path, apply=True,
        )
        assert result.applied is True
        assert result.new_revision == "def456"
        assert result.source_revision == "abc123"


def test_migration_result_summary_is_readable():
    result = MigrationResult(
        repo_id="OpenVoiceOS/ovos-stt-bench-pt-PT",
        source_revision="abc123",
        new_revision="def456",
        files_seen=2,
        files_skipped_idempotent=1,
        rows_seen=10,
        rows_migrated=5,
        applied=True,
    )
    text = result.summary()
    assert "OpenVoiceOS/ovos-stt-bench-pt-PT" in text
    assert "5/10" in text
    assert "abc123" in text
    assert "def456" in text
