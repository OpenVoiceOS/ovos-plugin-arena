"""Tests for the strict registry validation gate (``registry.loaders.validate_registry``).

Closed schemas (``extra="forbid"``) mean an unknown/typo'd key in any
``registry/**/*.json`` file MUST be caught here rather than silently
ignored or degrading to a runtime warning.
"""
from __future__ import annotations

import json

from registry.loaders import REGISTRY_ROOT, validate_registry


class TestRealRegistry:
    def test_real_registry_validates_cleanly(self):
        """Guards every committed competitor/dataset file (~236 as of writing)."""
        errors = validate_registry()
        assert errors == []

    def test_real_registry_has_files(self):
        # Sanity check the glob patterns actually found something, so an
        # empty-directory false negative doesn't slip past the assertion above.
        competitors = list((REGISTRY_ROOT / "competitors").glob("**/*.json"))
        datasets = list((REGISTRY_ROOT / "datasets").glob("**/*.json"))
        assert len(competitors) > 50
        assert len(datasets) > 10


class TestMalformedRegistry:
    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def test_extra_key_on_competitor_is_caught(self, tmp_path):
        self._write(
            tmp_path / "competitors" / "stt" / "bad.json",
            {
                "competitor_id": "bad",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-x",
                "langs": ["en-US"],
                "revison": "not-a-real-field",  # typo of "revision" — no such field
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "bad.json" in errors[0]
        assert "revison" in errors[0] or "extra" in errors[0].lower()

    def test_extra_key_on_dataset_is_caught(self, tmp_path):
        self._write(
            tmp_path / "datasets" / "stt" / "bad-ds.json",
            {
                "dataset_id": "bad-ds",
                "modality": "stt",
                "source": {"type": "path", "path": "/x.jsonl"},
                "lang": "en-US",
                "unexpected_field": True,
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "bad-ds.json" in errors[0]

    def test_eval_dataset_without_summary_is_caught(self, tmp_path):
        """An eval corpus reaches the leaderboard as a bare code name unless
        it carries a human name and a plain-language summary."""
        self._write(
            tmp_path / "datasets" / "stt" / "no-summary.json",
            {
                "dataset_id": "no-summary",
                "display_name": "Some corpus (English)",
                "modality": "stt",
                "source": {"type": "path", "path": "/x.jsonl"},
                "lang": "en-US",
                "role": "eval",
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "no-summary.json" in errors[0]
        assert "summary is required" in errors[0]

    def test_dataset_without_display_name_is_caught(self, tmp_path):
        self._write(
            tmp_path / "datasets" / "intent_template" / "no-name.json",
            {
                "dataset_id": "no-name",
                "modality": "intent_template",
                "source": {"type": "path", "path": "/x.jsonl"},
                "lang": "en-US",
                "role": "train",
                "paradigm": "template",
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "no-name.json" in errors[0]
        assert "display_name is required" in errors[0]

    def test_train_dataset_needs_no_summary(self, tmp_path):
        self._write(
            tmp_path / "datasets" / "intent_template" / "trainer.json",
            {
                "dataset_id": "trainer",
                "display_name": "Some corpus, training templates (English)",
                "modality": "intent_template",
                "source": {"type": "path", "path": "/x.jsonl"},
                "lang": "en-US",
                "role": "train",
                "paradigm": "template",
            },
        )
        assert validate_registry(registry_root=tmp_path) == []

    def test_bad_type_is_caught(self, tmp_path):
        self._write(
            tmp_path / "competitors" / "stt" / "bad-type.json",
            {
                "competitor_id": "bad-type",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-x",
                "langs": ["en-US"],
                "config": "this-should-be-a-dict-not-a-string",
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "bad-type.json" in errors[0]

    def test_valid_entry_alongside_malformed_only_flags_the_bad_one(self, tmp_path):
        self._write(
            tmp_path / "competitors" / "stt" / "good.json",
            {
                "competitor_id": "good",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-x",
                "langs": ["en-US"],
            },
        )
        self._write(
            tmp_path / "competitors" / "stt" / "bad.json",
            {
                "competitor_id": "bad",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-x",
                "langs": ["en-US"],
                "typo_field": 1,
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "bad.json" in errors[0]

    def test_empty_registry_root_has_no_errors(self, tmp_path):
        assert validate_registry(registry_root=tmp_path) == []

class TestNegativesDatasetIds:
    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def _ww_dataset(self, dataset_id, lang="en-US", negatives_dataset_ids=None):
        payload = {
            "dataset_id": dataset_id,
            "display_name": f"Fake wake word corpus ({dataset_id})",
            "summary": "A stand-in corpus used by these tests.",
            "modality": "wake_word",
            "source": {
                "type": "huggingface",
                "hf_id": "TigreGotico/fake-wakeword-corpus",
                "revision": "main",
            },
            "wakeword": "hey_fake",
            "lang": lang,
            "role": "eval",
        }
        if negatives_dataset_ids is not None:
            payload["negatives_dataset_ids"] = negatives_dataset_ids
        return payload

    def test_unknown_negatives_dataset_id_is_caught(self, tmp_path):
        self._write(
            tmp_path / "datasets" / "wake_word" / "fighter.json",
            self._ww_dataset("fighter", negatives_dataset_ids=["does-not-exist"]),
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "fighter.json" in errors[0]
        assert "does-not-exist" in errors[0]

    def test_wrong_modality_negatives_dataset_id_is_caught(self, tmp_path):
        self._write(
            tmp_path / "datasets" / "wake_word" / "fighter.json",
            self._ww_dataset("fighter", negatives_dataset_ids=["some-vad-set"]),
        )
        self._write(
            tmp_path / "datasets" / "vad" / "some-vad-set.json",
            {
                "dataset_id": "some-vad-set",
                "display_name": "A stand-in VAD corpus",
                "summary": "A stand-in corpus used by these tests.",
                "modality": "vad",
                "source": {"type": "path", "path": "/x.jsonl"},
                "lang": "en-US",
            },
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "fighter.json" in errors[0]
        assert "some-vad-set" in errors[0]
        assert "modality" in errors[0].lower()

    def test_cross_lingual_negatives_dataset_id_is_only_a_warning(self, tmp_path):
        self._write(
            tmp_path / "datasets" / "wake_word" / "fighter.json",
            self._ww_dataset(
                "fighter", lang="en-US", negatives_dataset_ids=["other-lang-negs"]
            ),
        )
        self._write(
            tmp_path / "datasets" / "wake_word" / "other-lang-negs.json",
            self._ww_dataset("other-lang-negs", lang="pt-PT"),
        )
        import warnings as warnings_mod

        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            errors = validate_registry(registry_root=tmp_path)

        assert errors == []
        assert any(
            "other-lang-negs" in str(w.message) and "lang" in str(w.message)
            for w in caught
        )


class TestTrainedOn:
    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def _ww_dataset(self, dataset_id, modality="wake_word", lang="en-US"):
        return {
            "dataset_id": dataset_id,
            "display_name": f"Fake corpus ({dataset_id})",
            "summary": "A stand-in corpus used by these tests.",
            "modality": modality,
            "source": {
                "type": "huggingface",
                "hf_id": "TigreGotico/fake-wakeword-corpus",
                "revision": "main",
            },
            "wakeword": "hey_fake",
            "lang": lang,
            "role": "eval",
        }

    def _ww_competitor(self, competitor_id, trained_on=None):
        payload = {
            "competitor_id": competitor_id,
            "modality": "wake_word",
            "plugin": "ovos-ww-plugin-x",
            "langs": ["en-US"],
        }
        if trained_on is not None:
            payload["trained_on"] = trained_on
        return payload

    def test_unknown_trained_on_dataset_id_is_caught(self, tmp_path):
        self._write(
            tmp_path / "competitors" / "wake_word" / "fighter.json",
            self._ww_competitor("fighter", trained_on=["does-not-exist"]),
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "fighter.json" in errors[0]
        assert "does-not-exist" in errors[0]

    def test_wrong_modality_trained_on_dataset_id_is_caught(self, tmp_path):
        self._write(
            tmp_path / "competitors" / "wake_word" / "fighter.json",
            self._ww_competitor("fighter", trained_on=["some-stt-set"]),
        )
        self._write(
            tmp_path / "datasets" / "stt" / "some-stt-set.json",
            self._ww_dataset("some-stt-set", modality="stt"),
        )
        errors = validate_registry(registry_root=tmp_path)
        assert len(errors) == 1
        assert "fighter.json" in errors[0]
        assert "some-stt-set" in errors[0]
        assert "modality" in errors[0].lower()

    def test_valid_trained_on_dataset_id_has_no_errors(self, tmp_path):
        self._write(
            tmp_path / "competitors" / "wake_word" / "fighter.json",
            self._ww_competitor("fighter", trained_on=["community-fake"]),
        )
        self._write(
            tmp_path / "datasets" / "wake_word" / "community-fake.json",
            self._ww_dataset("community-fake"),
        )
        assert validate_registry(registry_root=tmp_path) == []
