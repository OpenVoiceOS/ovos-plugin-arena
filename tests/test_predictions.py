"""Unit tests for arena.predictions — JSONL loading and grouping."""
from __future__ import annotations

import json

from arena.predictions import (
    group_rows,
    infer_modality,
    load_predictions,
    load_predictions_dir,
    parse_row,
    read_jsonl,
)


def _intent_row(**over):
    row = {
        "competitor_id": "padatious-medium",
        "sample_id": "en-US/00001",
        "dataset_id": "intents-for-eval",
        "lang": "en-US",
        "plugin_id": "ovos-padatious-pipeline-plugin",
        "utterance": "play a song",
        "reference_intent": "media:play_song",
        "prediction": "media:play_song",
        "exact_match": True,
    }
    row.update(over)
    return row


class TestInferModality:
    def test_intent(self):
        assert infer_modality(_intent_row()) == "intent"

    def test_stt(self):
        assert infer_modality({"reference_text": "ola", "prediction": "ola"}) == "stt"

    def test_unknown(self):
        assert infer_modality({"prediction": "?"}) == "unknown"

    def test_explicit_modality_wins(self):
        assert infer_modality(_intent_row(modality="intent_template")) == (
            "intent_template"
        )
        assert infer_modality(
            {"label": "positive", "prediction": "detected", "modality": "vad"}
        ) == "vad"

    def test_wake_word(self):
        assert infer_modality(
            {"label": "positive", "prediction": "detected"}
        ) == "wake_word"
        assert infer_modality(
            {"label": "negative", "prediction": "not_detected"}
        ) == "wake_word"

    def test_vad_from_label_vocabulary(self):
        # runner.vad_bench rows: label speech/non_speech, prediction speech/silence
        assert infer_modality(
            {"label": "speech", "prediction": "silence"}
        ) == "vad"
        assert infer_modality(
            {"label": "non_speech", "prediction": "speech"}
        ) == "vad"

    def test_vad_from_prediction_only(self):
        # even a mislabelled row is caught by the decision vocabulary
        assert infer_modality(
            {"label": "positive", "prediction": "silence"}
        ) == "vad"


class TestParseRow:
    def test_known_fields_mapped(self):
        row = parse_row(_intent_row(), "fallback-id")
        assert row.competitor_id == "padatious-medium"
        assert row.reference_intent == "media:play_song"

    def test_competitor_falls_back_to_filename(self):
        raw = _intent_row()
        del raw["competitor_id"]
        row = parse_row(raw, "from-filename")
        assert row.competitor_id == "from-filename"

    def test_unknown_keys_preserved_in_extras(self):
        row = parse_row(_intent_row(dataset_revision="abc123"), "c")
        assert row.extras["dataset_revision"] == "abc123"


class TestReadJsonl:
    def test_reads_rows_and_skips_malformed(self, tmp_path):
        path = tmp_path / "competitor-x.jsonl"
        path.write_text(
            json.dumps(_intent_row()) + "\n"
            + "NOT JSON\n"
            + json.dumps(_intent_row(sample_id="en-US/00002")) + "\n"
        )
        rows = read_jsonl(path)
        assert len(rows) == 2
        assert rows[0].competitor_id == "padatious-medium"

    def test_dir_loader(self, tmp_path):
        for name in ("a.jsonl", "b.jsonl"):
            (tmp_path / name).write_text(json.dumps(_intent_row()) + "\n")
        assert len(load_predictions_dir(tmp_path)) == 2

    def test_load_predictions_local_path(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(json.dumps(_intent_row()) + "\n")
        assert len(load_predictions(str(tmp_path))) == 1


class TestGroupRows:
    def test_groups_by_modality_dataset_lang(self):
        rows = [
            parse_row(_intent_row(), "a"),
            parse_row(_intent_row(lang="pt-PT", sample_id="pt-PT/00001"), "a"),
            parse_row(_intent_row(competitor_id="other"), "other"),
        ]
        grouped = group_rows(rows)
        assert ("intent", "intents-for-eval", "en-US") in grouped
        assert ("intent", "intents-for-eval", "pt-PT") in grouped
        en = grouped[("intent", "intents-for-eval", "en-US")]
        assert set(en["en-US/00001"]) == {"padatious-medium", "other"}

    def test_duplicate_rows_keep_last(self):
        rows = [
            parse_row(_intent_row(prediction="first"), "a"),
            parse_row(_intent_row(prediction="second"), "a"),
        ]
        grouped = group_rows(rows)
        sample = grouped[("intent", "intents-for-eval", "en-US")]["en-US/00001"]
        assert sample["padatious-medium"].prediction == "second"

    def test_unknown_modality_dropped(self):
        rows = [parse_row({"competitor_id": "c", "sample_id": "s",
                           "dataset_id": "d", "lang": "x", "plugin_id": "p",
                           "prediction": "?"}, "c")]
        assert group_rows(rows) == {}
