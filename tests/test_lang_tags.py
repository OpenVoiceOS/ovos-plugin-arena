"""Tests for BCP-47 language-tag validation on registry schemas.

The registry's ``lang``/``langs`` fields must carry full BCP-47 tags
(``lang-REGION``, optionally with a script or private-use subtag) — a bare
primary subtag like ``"en"`` is a defect, not a shorthand. See
``registry/schemas.py::validate_lang_tag``.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from registry.loaders import REGISTRY_ROOT, validate_registry
from registry.schemas import CompetitorDef, DatasetDef, validate_lang_tag

ACCEPTED = [
    "en-US",
    "pt-PT",
    "es-419",  # UN M.49 region code
    "azj-Cyrl",  # script subtag, no region
    "ug-Arab",
    "crk-Cans",
    "mwl-x-ifanes",  # private-use
    "mwl-x-sendim",
    "cac-x-sanmateo-ixtatan",  # multi-part private-use
    "multi",
    "abi-ZZ",  # tts placeholder region — valid for now
    "sw-KE",
    "sl-SI",
    # "sl-SL" is syntactically well-formed BCP-47 (SL is a real ISO 3166
    # alpha-2 code, for Sierra Leone) even though it is the wrong region
    # for Slovene — the pattern-based validator cannot and does not catch
    # semantically-wrong-but-syntactically-valid region codes; that defect
    # is fixed in the registry data itself (sl-SL -> sl-SI), not detected
    # by validate_lang_tag.
    "sl-SL",
]

REJECTED = [
    "en",  # bare primary subtag
    "pt",
    "en-us",  # region must be uppercase
    "EN-US",  # primary subtag must be lowercase
    "abi-zz",  # placeholder region must be normalized to uppercase
    "en_US",  # wrong separator
    "e-US",  # primary subtag too short
]


class TestValidateLangTag:
    @pytest.mark.parametrize("tag", ACCEPTED)
    def test_accepted(self, tag):
        assert validate_lang_tag(tag) == tag

    @pytest.mark.parametrize("tag", REJECTED)
    def test_rejected(self, tag):
        with pytest.raises(ValueError):
            validate_lang_tag(tag)


class TestDatasetDefValidation:
    def test_bare_lang_rejected(self):
        with pytest.raises(ValidationError):
            DatasetDef(
                dataset_id="x",
                modality="stt",
                source={"type": "path", "path": "/x.jsonl"},
                lang="en",
            )

    def test_full_tag_accepted(self):
        d = DatasetDef(
            dataset_id="x",
            modality="stt",
            source={"type": "path", "path": "/x.jsonl"},
            lang="en-US",
        )
        assert d.lang == "en-US"

    def test_bad_lang_in_langs_list_rejected(self):
        with pytest.raises(ValidationError):
            DatasetDef(
                dataset_id="x",
                modality="stt",
                source={"type": "path", "path": "/x.jsonl"},
                lang="multi",
                langs=["en-US", "pt"],
            )


class TestCompetitorDefValidation:
    def test_bare_lang_rejected(self):
        with pytest.raises(ValidationError):
            CompetitorDef(
                competitor_id="x",
                modality="stt",
                plugin="some-plugin",
                langs=["en"],
            )

    def test_full_tag_accepted(self):
        c = CompetitorDef(
            competitor_id="x",
            modality="stt",
            plugin="some-plugin",
            langs=["en-US"],
        )
        assert c.langs == ["en-US"]


class TestRegistryWideLangTags:
    """Every committed competitor/dataset file must carry valid BCP-47 tags.

    This is a strict superset of ``test_registry_validation.py``'s generic
    schema gate: it fails loudly on a lang-tag regression even if someone
    weakens the field_validator, because it independently re-derives the
    set of lang tags straight from the JSON files.
    """

    def test_no_bare_or_malformed_lang_tags_in_registry(self):
        offenders = []
        for path in sorted(REGISTRY_ROOT.glob("**/*.json")):
            data = json.loads(path.read_text())
            tags = []
            if data.get("lang"):
                tags.append(data["lang"])
            if data.get("langs"):
                tags.extend(data["langs"])
            for tag in tags:
                try:
                    validate_lang_tag(tag)
                except ValueError as exc:
                    offenders.append(f"{path}: {tag!r} ({exc})")
        assert offenders == [], "\n".join(offenders)

    def test_real_registry_validates_via_pydantic(self):
        """The generic registry gate must also be lang-tag clean."""
        errors = validate_registry()
        assert errors == []
