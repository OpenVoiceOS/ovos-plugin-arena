"""Unit tests for runner.intent_bench — pure helpers, no engines needed."""
from __future__ import annotations

from unittest.mock import patch

from registry.loaders import load_dataset as load_dataset_def
from registry.loaders import load_competitor
from runner.intent_bench import (
    fetch_hf_classification_rows,
    fetch_rows,
    make_row,
    needed_paradigms,
    results_repo_for,
    split_name,
)


class TestNaming:
    def test_repo_per_benchmark_modality(self):
        assert results_repo_for("intent_template", "intents-for-eval") == (
            "OpenVoiceOS/ovos-intent-template-bench-intents-for-eval"
        )
        assert results_repo_for("intent", "massive-templates") == (
            "OpenVoiceOS/ovos-intent-bench-massive-templates"
        )

    def test_split_name_word_chars_only(self):
        assert split_name("pt-PT") == "pt_PT"
        assert split_name("en-US") == "en_US"


class TestEligibility:
    def test_single_engine_paradigm(self):
        comp = load_competitor("intent_keyword", "adapt-medium")
        assert needed_paradigms(comp) == {"keyword"}

    def test_fusion_needs_both(self):
        comp = load_competitor("intent", "padapt")
        assert needed_paradigms(comp) == {"template", "keyword"}

    def test_template_pure_fusion(self):
        comp = load_competitor("intent", "nebulatious")
        assert needed_paradigms(comp) == {"template"}


class TestMakeRow:
    def _row(self, prediction, reference="media:play_song"):
        comp = load_competitor("intent_template", "padacioso-medium")
        return make_row(
            comp, "intents-for-eval", "en-US", 7,
            {"utterance": "play a song", "expected_intent": reference,
             "expected_slots": {"song": "a song"}, "split": "template"},
            prediction, {}, None, 1.5,
            "ovos-padacioso-pipeline-plugin-medium", "rev123",
        )

    def test_row_contract(self):
        row = self._row("media:play_song")
        assert row["sample_id"] == "en-US/00007"
        assert row["modality"] == "intent_template"
        assert row["dataset_revision"] == "rev123"
        assert row["stage"] == "ovos-padacioso-pipeline-plugin-medium"
        assert row["exact_match"] is True

    def test_wrong_prediction(self):
        assert self._row("media:stop")["exact_match"] is False

    def test_ood_correct_rejection(self):
        row = self._row(None, reference=None)
        assert row["exact_match"] is True
        assert row["reference_intent"] is None


class _FakeClassLabel:
    """Mimics ``datasets.ClassLabel`` well enough for int2str()."""

    def __init__(self, names):
        self.names = names

    def int2str(self, value):
        return self.names[value]


class _FakeHFDataset:
    """Minimal stand-in for a ``datasets.Dataset`` — iterable + .features."""

    def __init__(self, rows, features):
        self._rows = rows
        self.features = features

    def __iter__(self):
        return iter(self._rows)


class TestFetchHFClassificationRows:
    """runner.intent_bench.fetch_hf_classification_rows — the path absorbed
    text-classification datasets (SNIPS/BANKING77/CLINC150/MASSIVE) go
    through instead of the file_pattern JSONL path."""

    def test_eval_decodes_int_labels_via_features(self):
        ds_def = load_dataset_def("intent", "banking77")
        rows = [
            {"text": "why was my card declined", "label": 0},
            {"text": "how do I top up", "label": 1},
        ]
        feat = _FakeClassLabel(["card_declined", "top_up"])
        fake_ds = _FakeHFDataset(rows, {"label": feat})
        # banking77's reference_fields point at 'label_text' (already a
        # string) in the real mirror; point the fake at the int column
        # 'label' instead, purely to exercise ClassLabel decoding.
        ds_def.reference_fields = {"utterance": "text", "intent": "label"}
        with patch("datasets.load_dataset", return_value=fake_ds) as mocked:
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        mocked.assert_called_once_with(
            "mteb/banking77", name=None, split="test", revision="rev123",
        )
        assert out == [
            {"utterance": "why was my card declined",
             "expected_intent": "card_declined", "split": "test"},
            {"utterance": "how do I top up",
             "expected_intent": "top_up", "split": "test"},
        ]

    def test_eval_accepts_plain_string_labels(self):
        ds_def = load_dataset_def("intent", "snips")
        rows = [{"text": "play some jazz", "category": "PlayMusic"}]
        fake_ds = _FakeHFDataset(rows, {"category": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"utterance": "play some jazz", "expected_intent": "PlayMusic",
             "split": "test"},
        ]

    def test_train_role_emits_intent_id_and_template(self):
        ds_def = load_dataset_def("intent", "snips-train")
        rows = [{"text": "play some jazz", "category": "PlayMusic"}]
        fake_ds = _FakeHFDataset(rows, {"category": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [{"intent_id": "PlayMusic", "template": "play some jazz"}]

    def test_oos_label_routed_to_far_ood_on_eval(self):
        ds_def = load_dataset_def("intent", "clinc150")
        rows = [
            {"text": "in scope question", "intent": 0},
            {"text": "totally unrelated nonsense", "intent": 1},
        ]
        feat = _FakeClassLabel(["book_flight", "oos"])
        fake_ds = _FakeHFDataset(rows, {"intent": feat})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"utterance": "in scope question",
             "expected_intent": "book_flight", "split": "test"},
            {"utterance": "totally unrelated nonsense",
             "expected_intent": None, "split": "far_ood"},
        ]

    def test_oos_label_dropped_on_train(self):
        ds_def = load_dataset_def("intent", "clinc150-train")
        rows = [
            {"text": "in scope question", "intent": 0},
            {"text": "totally unrelated nonsense", "intent": 1},
        ]
        feat = _FakeClassLabel(["book_flight", "oos"])
        fake_ds = _FakeHFDataset(rows, {"intent": feat})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"intent_id": "book_flight", "template": "in scope question"},
        ]

    def test_fetch_rows_dispatches_to_classification_path(self):
        """fetch_rows() routes file_pattern-less sources to the classification
        loader instead of hf_hub_download (the JSONL path)."""
        ds_def = load_dataset_def("intent", "snips")
        with patch(
            "runner.intent_bench.fetch_hf_classification_rows",
            return_value=[{"utterance": "x", "expected_intent": "y",
                            "split": "test"}],
        ) as mocked:
            out = fetch_rows(ds_def, "en-US", "rev123")
        mocked.assert_called_once_with(ds_def, "en-US", "rev123")
        assert out == [{"utterance": "x", "expected_intent": "y", "split": "test"}]

    def test_missing_reference_fields_raise(self):
        ds_def = load_dataset_def("intent", "snips")
        ds_def.reference_fields = {}
        with patch("datasets.load_dataset"):
            try:
                fetch_hf_classification_rows(ds_def, "en-US", "rev123")
            except ValueError as exc:
                assert "reference_fields" in str(exc)
            else:
                raise AssertionError("expected ValueError")

    def test_id_field_dedupes_duplicated_rows(self):
        """source.id_field is general hygiene, not a MASSIVE special case —
        no currently-registered dataset needs it (their sources ship each
        row once), but a source that turns it on must have duplicates
        collapsed to one row per id, keeping first occurrence. Built off
        a synthetic source rather than a registered one, since none of
        the absorbed datasets (SNIPS/BANKING77/CLINC150) require dedup."""
        ds_def = load_dataset_def("intent", "banking77").model_copy(deep=True)
        ds_def.source.id_field = "id"
        base_rows = [
            {"id": "1", "text": "why was my card declined", "label_text": "card_declined"},
            {"id": "2", "text": "how do I top up", "label_text": "top_up"},
        ]
        duplicated = base_rows * 3  # e.g. a mirror shipping every row 3x
        fake_ds = _FakeHFDataset(duplicated, {"label_text": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"utterance": "why was my card declined",
             "expected_intent": "card_declined", "split": "test"},
            {"utterance": "how do I top up",
             "expected_intent": "top_up", "split": "test"},
        ]

    def test_id_field_dedup_is_noop_on_clean_sources(self):
        """Sources without id_field set (SNIPS/BANKING77/CLINC150 as
        registered) are unaffected — dedup only fires when id_field is set."""
        ds_def = load_dataset_def("intent", "banking77")
        assert ds_def.source.id_field is None
        rows = [
            {"text": "why was my card declined", "label_text": "card_declined"},
            {"text": "how do I top up", "label_text": "top_up"},
        ]
        fake_ds = _FakeHFDataset(rows, {"label_text": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert len(out) == 2

    def test_missing_id_field_treated_as_a_dedup_key(self):
        """Rows genuinely missing the id column collapse to a single
        None-keyed row rather than raising — a defensive edge case, not
        the expected shape for any registered dataset."""
        ds_def = load_dataset_def("intent", "banking77").model_copy(deep=True)
        ds_def.source.id_field = "id"
        rows = [
            {"text": "a", "label_text": "x"},
            {"text": "b", "label_text": "y"},
        ]
        fake_ds = _FakeHFDataset(rows, {"label_text": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert len(out) == 1
