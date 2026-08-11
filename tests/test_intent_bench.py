"""Unit tests for runner.intent_bench — pure helpers, no engines needed."""
from __future__ import annotations

from registry.loaders import load_competitor
from runner.intent_bench import (
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
