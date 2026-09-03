"""Tests for the pure normalization logic in
``scripts/build_golden_utterances.py`` (no network involved).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODULE_PATH = ROOT / "scripts" / "build_golden_utterances.py"

_spec = importlib.util.spec_from_file_location("build_golden_utterances", MODULE_PATH)
build_golden_utterances = importlib.util.module_from_spec(_spec)
sys.modules["build_golden_utterances"] = build_golden_utterances
_spec.loader.exec_module(build_golden_utterances)

normalize_locale = build_golden_utterances.normalize_locale
normalize_row = build_golden_utterances.normalize_row
file_locale_from_path = build_golden_utterances.file_locale_from_path
build_dataset = build_golden_utterances.build_dataset
validate_dataset = build_golden_utterances.validate_dataset


def test_normalize_locale_casing():
    assert normalize_locale("en-us") == "en-US"
    assert normalize_locale("PT-pt") == "pt-PT"
    assert normalize_locale("en-US") == "en-US"


def test_normalize_locale_leaves_non_two_part_alone():
    assert normalize_locale("multi") == "multi"


def test_file_locale_from_path():
    assert file_locale_from_path("test/end2end/golden_utterances_pt-pt.jsonl") == "pt-PT"
    assert file_locale_from_path("test/end2end/golden_utterances.jsonl") is None


def test_normalize_row_strips_single_trailing_intent_suffix():
    row = {
        "skill_id": "skill-weather.openvoiceos",
        "utterance": "what's the weather",
        "intent_label": "weather.intent",
    }
    out = normalize_row(row, "ovos-skill-weather", "test/end2end/golden_utterances.jsonl", None)
    assert out["expected_intent"] == "skill-weather.openvoiceos:weather"
    assert out["intent_label_original"] == "weather.intent"


def test_normalize_row_strips_only_one_trailing_suffix():
    row = {
        "skill_id": "skill-x",
        "utterance": "u",
        "intent_label": "weird.intent.intent",
    }
    out = normalize_row(row, "repo", "path", None)
    # only ONE trailing ".intent" is stripped
    assert out["expected_intent"] == "skill-x:weird.intent"


def test_normalize_row_adapt_intent_unchanged():
    row = {"skill_id": "skill-x", "utterance": "u", "intent_label": "some.adapt.intent.name"}
    out = normalize_row(row, "repo", "path", None)
    # only a trailing ".intent" suffix is special-cased; this label does not end in ".intent"
    assert out["expected_intent"] == "skill-x:some.adapt.intent.name"


def test_normalize_row_defaults_to_en_us_with_no_lang():
    row = {"skill_id": "skill-x", "utterance": "u", "intent_label": "foo"}
    out = normalize_row(row, "repo", "path", None)
    assert out["lang"] == "en-US"


def test_normalize_row_uses_file_locale_when_row_has_none():
    row = {"skill_id": "skill-x", "utterance": "u", "intent_label": "foo"}
    out = normalize_row(row, "repo", "path", "pt-PT")
    assert out["lang"] == "pt-PT"


def test_normalize_row_row_lang_wins_over_file_locale():
    row = {"skill_id": "skill-x", "utterance": "u", "intent_label": "foo", "lang": "de-de"}
    out = normalize_row(row, "repo", "path", "pt-PT")
    assert out["lang"] == "de-DE"


def test_normalize_row_skips_missing_required_fields():
    assert normalize_row({"skill_id": "s", "utterance": "u"}, "r", "p", None) is None
    assert normalize_row({"skill_id": "s", "intent_label": "i"}, "r", "p", None) is None
    assert normalize_row({"utterance": "u", "intent_label": "i"}, "r", "p", None) is None


def test_normalize_row_keeps_provenance_columns():
    row = {
        "skill_id": "skill-x",
        "utterance": "u",
        "intent_label": "foo.intent",
        "intent_type": "padatious",
        "intent_method": "handle_foo",
        "needs_manual": True,
        "machine_generated": False,
        "required_vocab": ["a"],
        "expected_messages": ["mycroft.foo"],
    }
    out = normalize_row(row, "ovos-skill-x", "test/end2end/golden_utterances.jsonl", None)
    assert out["source_repo"] == "ovos-skill-x"
    assert out["source_file"] == "test/end2end/golden_utterances.jsonl"
    assert out["needs_manual"] is True
    assert out["machine_generated"] is False
    assert out["required_vocab"] == ["a"]
    assert out["expected_messages"] == ["mycroft.foo"]


def test_build_dataset_skips_malformed_and_dialog_shaped_rows():
    files = [
        (
            "ovos-skill-fallback-unknown",
            "test/end2end/golden_utterances.jsonl",
            "\n".join(
                [
                    '{"skill_id": "skill-fallback-unknown", "utterance": "asdkjh", "dialog": "sorry"}',
                    '{"skill_id": "skill-x", "utterance": "hi", "intent_label": "greet.intent"}',
                    "not json at all",
                ]
            ),
        )
    ]
    out_rows, stats = build_dataset(files)
    assert stats["total"] == 1
    assert stats["skipped_count"] == 2
    assert out_rows["en-US"][0]["expected_intent"] == "skill-x:greet"


def test_build_dataset_partitions_by_locale_file():
    files = [
        (
            "ovos-skill-x",
            "test/end2end/golden_utterances_pt-pt.jsonl",
            '{"skill_id": "skill-x", "utterance": "ola", "intent_label": "greet.intent"}',
        ),
        (
            "ovos-skill-x",
            "test/end2end/golden_utterances.jsonl",
            '{"skill_id": "skill-x", "utterance": "hi", "intent_label": "greet.intent"}',
        ),
    ]
    out_rows, stats = build_dataset(files)
    assert set(out_rows) == {"en-US", "pt-PT"}
    assert stats["total"] == 2


def test_validate_dataset_flags_bad_lang_and_dotintent_suffix():
    out_rows = {
        "en-US": [{"utterance": "hi", "expected_intent": "skill-x:greet"}],
        "en": [{"utterance": "hi", "expected_intent": "skill-x:greet"}],
        "de-DE": [{"utterance": "", "expected_intent": "skill-x:greet.intent"}],
    }
    errors = validate_dataset(out_rows)
    assert any("invalid lang tag" in e for e in errors)
    assert any("empty utterance" in e for e in errors)
    assert any(".intent suffix" in e for e in errors)


def test_validate_dataset_clean_dataset_has_no_errors():
    out_rows = {"en-US": [{"utterance": "hi", "expected_intent": "skill-x:greet"}]}
    assert validate_dataset(out_rows) == []
