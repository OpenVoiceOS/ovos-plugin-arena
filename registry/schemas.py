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
    """Arena leagues. Keyword-paradigm and template-paradigm intent engines
    consume different supervision, so they compete in separate leagues; the
    open ``intent`` league hosts mixed-paradigm pipeline fusions."""

    STT = "stt"
    TTS = "tts"
    WAKE_WORD = "wake_word"
    INTENT = "intent"  # open league — mixed-paradigm fusions
    INTENT_TEMPLATE = "intent_template"
    INTENT_KEYWORD = "intent_keyword"


INTENT_MODALITIES = (
    Modality.INTENT, Modality.INTENT_TEMPLATE, Modality.INTENT_KEYWORD,
)


# ---------------------------------------------------------------------------
# DatasetSource — where the corpus lives
# ---------------------------------------------------------------------------


class HuggingFaceSource(BaseModel):
    type: Literal["huggingface"] = "huggingface"
    hf_id: str = Field(..., description="HuggingFace dataset identifier, e.g. PolyAI/minds14")
    revision: str = "main"
    split: str = "train"
    subset: Optional[str] = None
    file_pattern: Optional[str] = Field(
        None,
        description=(
            "Raw repo file path per language, e.g. '{lang}/test.jsonl'. "
            "Used for datasets stored as plain files instead of HF splits."
        ),
    )

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
    """Definition of one benchmark corpus (``registry/datasets/<mod>/<id>.json``).

    Keyword-paradigm and template-paradigm training corpora are *different
    datasets with different datashapes* — each gets its own registry entry
    (``role: train`` + ``paradigm``) with ``reference_fields`` describing its
    row shape.  An eval corpus links its paradigm-specific training sets via
    ``train_datasets``.
    """

    dataset_id: str = Field(..., description="Stable unique identifier for this dataset")
    modality: Modality
    source: DatasetSource = Field(..., discriminator="type")
    reference_fields: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map from semantic role to column name in the source corpus — "
            "the datashape contract. e.g. {'utterance': 'utterance', "
            "'intent': 'expected_intent'} for an intent eval set, "
            "{'intent': 'intent_id', 'template': 'template'} for a "
            "template-paradigm train set."
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
    role: Literal["eval", "train", "unrestricted"] = "eval"
    paradigm: Optional[Literal["template", "keyword"]] = Field(
        None,
        description=(
            "For role=train intent corpora: which engine paradigm this "
            "datashape feeds (template engines vs keyword engines)."
        ),
    )
    train_datasets: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "For role=eval corpora: paradigm → dataset_id of the matching "
            "training corpus, e.g. {'template': 'intents-for-eval-templates'}."
        ),
    )
    wakeword: Optional[str] = Field(
        None,
        description=(
            "Wake-word audiofolder corpora: the top-level folder holding "
            "positive clips for this benchmark's phrase (e.g. 'hey_mycroft'). "
            "Clips in other folders are negatives."
        ),
    )
    negative_dirs: Optional[List[str]] = Field(
        None,
        description=(
            "Wake-word audiofolder corpora: which top-level folders to draw "
            "negatives from (default: every folder except ``wakeword``)."
        ),
    )
    negatives_hf: Optional[str] = Field(
        None,
        description=(
            "Wake word: a separate HF dataset to draw negatives from — a "
            "'not-wake-word' corpus of general speech/noise that must never "
            "trigger detection (the proper false-accept test). Overrides "
            "same-corpus negatives."
        ),
    )
    negatives_dir: Optional[str] = Field(
        None,
        description="Folder within ``negatives_hf`` holding the negative clips.",
    )
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# CompetitorDef
# ---------------------------------------------------------------------------


PIPELINE_TIERS = ("high", "medium", "low")


def split_pipeline_stage(stage: str) -> tuple:
    """Split a pipeline stage name into (plugin_id, tier).

    ``ovos-padatious-pipeline-plugin-high`` → ``("ovos-padatious-pipeline-plugin", "high")``.
    Raises ValueError for stages without a known tier suffix.
    """
    for tier in PIPELINE_TIERS:
        suffix = f"-{tier}"
        if stage.endswith(suffix):
            return stage[: -len(suffix)], tier
    raise ValueError(
        f"Pipeline stage {stage!r} has no -high/-medium/-low tier suffix"
    )


class CompetitorDef(BaseModel):
    """Definition of one competitor (``registry/competitors/<mod>/<id>.json``).

    A competitor is a *configuration you could ship*: for the intent
    modality, ``config`` is a valid ``mycroft.conf`` fragment — an
    ``intents`` section with an ordered ``pipeline`` list of
    ``<plugin>-<tier>`` stages plus per-plugin config blocks.  A
    single-stage pipeline benchmarks one engine; a multi-stage pipeline is
    an ensemble fighter in its own right.  The same plugin under a
    different config = a different competitor.

    ``alias`` lets the ingestion layer accept legacy ``plugin_id`` values from
    prediction rows produced before the registry existed (e.g. ``plugin_name``
    from ``ovos-stt-bench-*`` datasets).  Any match on alias is re-keyed to
    ``competitor_id`` on ingestion.
    """

    competitor_id: str = Field(
        ..., description="Stable unique identifier for this competitor"
    )
    modality: Modality
    plugin: Optional[str] = Field(
        None,
        description=(
            "OPM plugin entry-point name. Optional for intent fighters — "
            "derived from the pipeline (None for ensembles)."
        ),
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Valid mycroft.conf fragment. Intent fighters carry an 'intents' "
            "section: {'pipeline': ['<plugin>-<tier>', …], '<plugin>': {…}}."
        ),
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
    # Bestiary card fields (fighter-browser UI)
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
    size: Optional[Literal[
        "micro", "tiny", "small", "base", "medium",
        "large", "x-large", "giant", "titan",
    ]] = Field(
        None,
        description=(
            "Installed footprint class (package + models): "
            "micro <5MB · tiny 5-50MB · small 50-200MB · base 200-500MB · "
            "medium 500MB-2GB · large 2-8GB · x-large 8-20GB · "
            "giant 20-80GB · titan >80GB (LLM-class)"
        ),
    )
    links: Dict[str, str] = Field(
        default_factory=dict,
        description="Named URLs: source, pypi, paper, …",
    )
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_pipeline_and_alias(self) -> "CompetitorDef":
        """Validate the intents pipeline and derive plugin/alias fields.

        Intent fighters MUST carry ``config.intents.pipeline`` (non-empty,
        tier-suffixed stage names).  ``plugin`` is derived when the pipeline
        uses a single engine; ensembles keep ``plugin = None``.
        """
        if self.modality in INTENT_MODALITIES:
            intents = self.config.get("intents") or {}
            pipeline = intents.get("pipeline") or []
            if not pipeline or not isinstance(pipeline, list):
                raise ValueError(
                    f"{self.competitor_id}: config.intents.pipeline must be a "
                    "non-empty list of '<plugin>-<tier>' stage names"
                )
            plugins = []
            for stage in pipeline:
                plugin_id, _tier = split_pipeline_stage(stage)
                if plugin_id not in plugins:
                    plugins.append(plugin_id)
            if self.plugin is None and len(plugins) == 1:
                self.plugin = plugins[0]

        aliases = list(self.alias or [])
        if self.plugin and self.plugin not in aliases:
            aliases.append(self.plugin)
        self.alias = aliases
        return self

    @property
    def pipeline(self) -> List[str]:
        """The ordered pipeline stage names (empty for non-intent fighters)."""
        return list((self.config.get("intents") or {}).get("pipeline") or [])

    @property
    def pipeline_plugins(self) -> List[str]:
        """Unique plugin ids referenced by the pipeline, in stage order."""
        plugins: List[str] = []
        for stage in self.pipeline:
            plugin_id, _tier = split_pipeline_stage(stage)
            if plugin_id not in plugins:
                plugins.append(plugin_id)
        return plugins

    def plugin_config(self, plugin_id: str, short_name: str = "") -> Dict[str, Any]:
        """Per-plugin config block from the intents section.

        Accepts both the full entry-point key and the legacy short key
        (``"adapt"``, ``"padatious"`` …), mirroring how the plugins
        themselves resolve their config from mycroft.conf.
        """
        intents = self.config.get("intents") or {}
        cfg = intents.get(plugin_id)
        if cfg is None and short_name:
            cfg = intents.get(short_name)
        return dict(cfg or {})
