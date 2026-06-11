"""Pydantic schemas for the declarative evaluation registry.

§3.1 revision — Competitors as .json
--------------------------------------
Each file ``registry/competitors/<modality>/<competitor_id>.json`` describes a
single plugin under one configuration.  The same underlying plugin entry point
with a different model or config is a *different competitor*.

Battles, ELO scores, and leaderboards are keyed on ``competitor_id``.  An
``alias`` field provides backward-compatibility when legacy data was produced
under a plain ``plugin_id`` key (e.g. from the old ``plugin_name`` column in
``ovos-stt-bench-*`` datasets before the registry existed).

§3.1 revision — Datasets as .json
--------------------------------------
Each file ``registry/datasets/<modality>/<dataset_id>.json`` describes one
benchmark corpus.  ``role: eval`` marks held-out sets that gate leaderboard
metrics; ``role: unrestricted`` marks openly-available training/development data.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Modality
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    STT = "stt"
    TTS = "tts"
    WAKE_WORD = "wake_word"
    INTENT = "intent"


# ---------------------------------------------------------------------------
# DatasetSource — where the corpus lives
# ---------------------------------------------------------------------------


class HuggingFaceSource(BaseModel):
    type: Literal["huggingface"] = "huggingface"
    hf_id: str = Field(..., description="HuggingFace dataset identifier, e.g. PolyAI/minds14")
    revision: str = "main"
    split: str = "train"
    subset: Optional[str] = None

    @property
    def dataset_id_str(self) -> str:
        """Stable dataset_id for use in prediction rows."""
        parts = [self.hf_id]
        if self.subset:
            parts.append(self.subset)
        parts.append(self.split)
        return "/".join(parts)


class PathSource(BaseModel):
    type: Literal["path"] = "path"
    path: str = Field(..., description="Local filesystem path to the dataset")
    format: str = "jsonl"  # jsonl | csv | parquet


DatasetSource = Union[HuggingFaceSource, PathSource]


# ---------------------------------------------------------------------------
# DatasetDef
# ---------------------------------------------------------------------------


class DatasetDef(BaseModel):
    """Definition of one benchmark corpus (``registry/datasets/<mod>/<id>.json``)."""

    dataset_id: str = Field(..., description="Stable unique identifier for this dataset")
    modality: Modality
    source: DatasetSource = Field(..., discriminator="type")
    reference_fields: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map from semantic role to column name in the source corpus. "
            "e.g. {'audio': 'audio', 'ground_truth': 'transcription'} for STT "
            "or {'utterance': 'text', 'intent': 'intent'} for intent."
        ),
    )
    lang: str = Field(
        ...,
        description="BCP-47 language tag, or 'multi' for multilingual corpora",
    )
    langs: Optional[List[str]] = Field(
        None,
        description="Language list for multilingual corpora (lang='multi')",
    )
    license: Optional[str] = None
    role: Literal["eval", "unrestricted"] = "eval"
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# CompetitorDef
# ---------------------------------------------------------------------------


class CompetitorDef(BaseModel):
    """Definition of one competitor (``registry/competitors/<mod>/<id>.json``).

    A competitor is a specific plugin entry point under a specific configuration.
    The same plugin + different model/config = different competitor.

    ``alias`` lets the ingestion layer accept legacy ``plugin_id`` values from
    prediction rows produced before the registry existed (e.g. ``plugin_name``
    from ``ovos-stt-bench-*`` datasets).  Any match on alias is re-keyed to
    ``competitor_id`` on ingestion.
    """

    competitor_id: str = Field(
        ..., description="Stable unique identifier for this competitor"
    )
    modality: Modality
    plugin: str = Field(..., description="OPM plugin entry-point name")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="mycroft.conf-equivalent config passed to the plugin",
    )
    langs: List[str] = Field(
        default_factory=list,
        description="BCP-47 language tags this competitor supports",
    )
    alias: Optional[List[str]] = Field(
        None,
        description=(
            "Legacy plugin_id values that map to this competitor in ingested rows. "
            "Enables backward-compat without re-running old prediction jobs."
        ),
    )
    # Pokedex card fields (fighter-browser UI)
    display_name: Optional[str] = Field(
        None, description="Human-friendly fighter name shown in the UI"
    )
    species: Optional[str] = Field(
        None,
        description=(
            "Parent plugin class this fighter is an instance of, "
            "e.g. 'PadatiousPipeline'"
        ),
    )
    types: List[str] = Field(
        default_factory=list,
        description=(
            "Architecture tags, e.g. 'GOFAI', 'fuzzy-match', 'embedding', "
            "'neural-net', 'LLM'"
        ),
    )
    description: Optional[str] = Field(
        None, description="Short blurb about how this fighter works"
    )
    model: Optional[str] = Field(
        None, description="Underlying model identifier, when one exists"
    )
    links: Dict[str, str] = Field(
        default_factory=dict,
        description="Named URLs: source, pypi, paper, …",
    )
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _alias_includes_plugin(self) -> "CompetitorDef":
        """Ensure the plugin entry-point is always an alias of itself."""
        aliases = list(self.alias or [])
        if self.plugin not in aliases:
            aliases.append(self.plugin)
        self.alias = aliases
        return self
