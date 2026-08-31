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

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# BCP-47 language tag validation
# ---------------------------------------------------------------------------

# Full lang-REGION tags only — a bare primary subtag (e.g. "en") is a defect,
# not a shorthand. Accepts:
#   - a 2-3 letter primary subtag
#   - an optional 4-letter script subtag (Title-case, e.g. "Cyrl", "Arab")
#   - an optional region: ISO 3166 alpha-2 (upper-case) or UN M.49 3-digit
#     (e.g. "419"), including the tts placeholder region "ZZ"
#   - an optional private-use subtag chain ("-x-...")
_BCP47_RE = re.compile(
    r"^[a-z]{2,3}"
    r"(-[A-Z][a-z]{3})?"
    r"(-([A-Z]{2}|[0-9]{3}))?"
    r"(-x-[a-z0-9]+(-[a-z0-9]+)*)?$"
)


def validate_lang_tag(tag: str) -> str:
    """Validate a single BCP-47 language tag (or the literal ``"multi"``).

    Raises ``ValueError`` on a bare primary subtag or any other malformed
    tag. Full lang-REGION tags are required everywhere in the registry —
    see the project's HARD rule on this.
    """
    if tag in ("multi", ""):
        # "" is the resolved_dataset_lang() sentinel for "unset — fall back
        # to parsing the dataset id's trailing locale suffix instead".
        return tag
    if "-" not in tag or not _BCP47_RE.match(tag):
        raise ValueError(
            f"{tag!r} is not a valid BCP-47 language tag — full lang-REGION "
            "tags are required (e.g. 'en-US', not bare 'en')"
        )
    return tag

# ---------------------------------------------------------------------------
# Family aliases — collapsing config-variant siblings of the same engine
# ---------------------------------------------------------------------------

# A handful of intent-template engines ship a "domain-scoped" and/or
# "hierarchical" wrapper alongside the plain pipeline plugin — same
# underlying engine, different `species` string (e.g. AdaptPipeline /
# DomainAdaptPipeline / HierarchicalAdaptPipeline) because each wrapper is
# its own OPM plugin entry point. That's a real config distinction worth
# keeping as separate *competitors*, but it is NOT a distinct *family* for
# ladder/bestiary grouping purposes — the owner wants one collapsed card per
# engine, not one per wrapper. This table folds each wrapper's species onto
# its base engine's species; anything not listed here (including engines
# that are genuinely distinct products, e.g. PadatiousPipeline vs
# PadaciosoPipeline, or HierarchicalKNNIntentPipeline — "hierarchical" is
# part of that engine's own name, not a variant qualifier on some other
# engine) is left alone: `family` just falls back to `species`.
FAMILY_ALIASES: dict[str, str] = {
    "DomainAdaptPipeline": "AdaptPipeline",
    "HierarchicalAdaptPipeline": "AdaptPipeline",
    "DomainLinhaFinaPipeline": "LinhaFinaPipeline",
    "HierarchicalLinhaFinaPipeline": "LinhaFinaPipeline",
    "DomainMarkovPipeline": "MarkovPipeline",
    "HierarchicalNebulentoPipeline": "NebulentoPipeline",
    "HierarchicalPalavreadoPipeline": "PalavreadoPipeline",
}

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
    VAD = "vad"
    INTENT = "intent"  # open league — mixed-paradigm fusions
    INTENT_TEMPLATE = "intent_template"
    INTENT_KEYWORD = "intent_keyword"
    # Streaming wake-word league (§A3.2 / R15) — same fighters as WAKE_WORD,
    # a distinct board scored from continuous-audio detection events rather
    # than isolated clips. See registry/datasets/ww_stream/*.json.
    WW_STREAM = "ww_stream"


INTENT_MODALITIES = (
    Modality.INTENT, Modality.INTENT_TEMPLATE, Modality.INTENT_KEYWORD,
)


# ---------------------------------------------------------------------------
# DatasetSource — where the corpus lives
# ---------------------------------------------------------------------------


class HuggingFaceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["huggingface"] = "huggingface"
    hf_id: str = Field(..., description="HuggingFace dataset identifier, e.g. PolyAI/minds14")
    revision: str = "main"
    split: str = "train"
    subset: str | None = None
    file_pattern: str | None = Field(
        None,
        description=(
            "Raw repo file path per language, e.g. '{lang}/test.jsonl'. "
            "Used for datasets stored as plain files instead of HF splits."
        ),
    )
    id_field: str | None = Field(
        None,
        description=(
            "Column name holding a row's stable source id, for plain HF "
            "classification datasets. When set, fetch_hf_classification_rows "
            "deduplicates on this column, keeping the first occurrence — "
            "guards against HF mirrors that ship the same row multiple "
            "times (observed on AmazonScience/massive's "
            "refs/convert/parquet conversion, which triplicates every row; "
            "kept as general hygiene even though no currently-registered "
            "dataset needs it). "
            "None disables dedup for sources with no reliable id column."
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
    model_config = ConfigDict(extra="forbid")

    type: Literal["path"] = "path"
    path: str = Field(..., description="Local filesystem path to the dataset")
    format: str = "jsonl"  # jsonl | csv | parquet


DatasetSource = HuggingFaceSource | PathSource


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

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Stable unique identifier for this dataset")
    # Schema revision of this registry entry's shape — 1 is the only shape
    # defined so far; bump when the DatasetDef contract changes.
    schema_version: int = 1
    modality: Modality
    source: DatasetSource = Field(..., discriminator="type")
    reference_fields: dict[str, str] = Field(
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
    langs: list[str] | None = Field(
        None,
        description="Language list for multilingual corpora (lang='multi')",
    )

    @field_validator("lang")
    @classmethod
    def _validate_lang(cls, v: str) -> str:
        return validate_lang_tag(v)

    @field_validator("langs")
    @classmethod
    def _validate_langs(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        return [validate_lang_tag(tag) for tag in v]

    license: str | None = None
    role: Literal["eval", "train", "unrestricted"] = "eval"
    paradigm: Literal["template", "keyword"] | None = Field(
        None,
        description=(
            "For role=train intent corpora: which engine paradigm this "
            "datashape feeds (template engines vs keyword engines)."
        ),
    )
    train_datasets: dict[str, str] | None = Field(
        None,
        description=(
            "For role=eval corpora: paradigm → dataset_id of the matching "
            "training corpus, e.g. {'template': 'intents-for-eval-templates'}."
        ),
    )
    wakeword: str | None = Field(
        None,
        description=(
            "Wake-word audiofolder corpora: the top-level folder holding "
            "positive clips for this benchmark's phrase (e.g. 'hey_mycroft'). "
            "Clips in other folders are negatives."
        ),
    )
    negative_dirs: list[str] | None = Field(
        None,
        description=(
            "Wake-word audiofolder corpora: which top-level folders to draw "
            "negatives from (default: every folder except ``wakeword``)."
        ),
    )
    negatives_hf: str | None = Field(
        None,
        description=(
            "Wake word: a separate HF dataset to draw negatives from — a "
            "'not-wake-word' corpus of general speech/noise that must never "
            "trigger detection (the proper false-accept test). Overrides "
            "same-corpus negatives."
        ),
    )
    negatives_dir: str | None = Field(
        None,
        description="Folder within ``negatives_hf`` holding the negative clips.",
    )
    negatives_sources: list[str] | None = Field(
        None,
        description=(
            "Wake word: several not-wake-word corpora to pool negatives from, "
            "each ``hf_id`` or ``hf_id/subdir`` — speech, music, ambient noise, "
            "household sounds — so false-accept rate reflects many scenarios. "
            "Overrides ``negatives_hf``."
        ),
    )
    predictions_hf: str | None = Field(
        None,
        description=(
            "HF dataset repo holding the arena's prediction rows for this "
            "benchmark corpus, following the runner convention "
            "``<owner>/ovos-<modality>-bench-<dataset_id>`` (modality with "
            "'_' replaced by '-'). The ``assemble`` step pulls predictions "
            "from these repos when no explicit source list is given."
        ),
    )
    # ``ww_stream`` (§A3.2 / R15) — continuous-audio ground-truth-event
    # corpora. Isolated-clip benchmarking structurally favors clip-shaped
    # detectors and can't exercise streaming false-accept behaviour, which
    # needs hours of continuous negative audio, not seconds-long clips.
    sample_rate_hz: int | None = Field(
        None,
        description=(
            "Pinned sample rate (Hz) for a streaming/event corpus, e.g. "
            "16000 — every clip MUST be resampled to this rate before "
            "onset timestamps are meaningful."
        ),
    )
    event_manifest_fields: dict[str, str] | None = Field(
        None,
        description=(
            "Column names for a streaming ground-truth-event manifest: "
            "{'audio': <path column>, 'onsets': <wake-word onset "
            "timestamps column, seconds>, 'duration_s': <clip length "
            "column>}. Each manifest row is one long continuous clip; "
            "'onsets' is empty for negative-hours-only clips."
        ),
    )
    min_negative_hours: float | None = Field(
        None,
        description=(
            "Target/minimum hours of negative (non-wake) continuous audio "
            "the corpus must provide for a stable false-accept-per-hour "
            "estimate (a handful of clips is not enough)."
        ),
    )
    event_tolerance_s: float | None = Field(
        None,
        description=(
            "± tolerance window (seconds) for matching a detection event "
            "to a ground-truth onset when scoring (see "
            "``arena.metrics.EVENT_TOLERANCE_S``)."
        ),
    )
    notes: str | None = None
    predictions_revision: str | None = Field(
        None,
        description=(
            "Immutable HF commit SHA to pin the predictions dataset "
            "(``predictions_hf``) to for reproducible board assembly. "
            "None means the assemble step tracks whatever revision it is "
            "invoked with (e.g. a floating branch) instead of a fixed SHA."
        ),
    )
    oos_label: str | None = Field(
        None,
        description=(
            "Plain HF classification datasets only: the decoded label "
            "value that marks an out-of-scope/negative row (e.g. CLINC150's "
            "'oos'). Train corpora drop rows with this label; eval corpora "
            "keep them with expected_intent=None and bucket='far_ood' so "
            "the scorer treats a non-fire as the correct answer."
        ),
    )
    reference_granularity: Literal["intent", "domain"] = Field(
        "intent",
        description=(
            "What a prediction is scored against. 'intent' (default) "
            "compares the full prediction string to expected_intent "
            "verbatim. 'domain' compares only the text before the first "
            "':' on both sides — for corpora that only carry a domain-level "
            "label (e.g. a Catalan weather-query set with no per-intent "
            "annotation), so a 'weather:current_conditions' prediction from "
            "a domain/hierarchical fighter still scores correct against a "
            "bare 'weather' reference. See ``arena.metrics.domain_of``."
        ),
    )
    domain_label: str | None = Field(
        None,
        description=(
            "Plain HF classification datasets only, paired with "
            "reference_granularity='domain': a constant reference label "
            "applied to every row when the source corpus has no label "
            "column at all (e.g. a single-domain query set). Takes over "
            "for reference_fields['intent'] when the latter is absent."
        ),
    )
    @model_validator(mode="after")
    def _validate_paradigm_directory(self) -> DatasetDef:
        """A role=train intent corpus with a ``paradigm`` lives under
        ``registry/datasets/intent_<paradigm>/`` — its ``modality`` field
        must say so. This is what makes the directory the league: a file
        physically filed under ``intent_template/`` cannot silently claim
        to be a keyword corpus (or the open ``intent`` league)."""
        if self.role == "train" and self.paradigm is not None:
            expected = Modality(f"intent_{self.paradigm}")
            if self.modality != expected:
                raise ValueError(
                    f"{self.dataset_id}: role=train paradigm={self.paradigm!r} "
                    f"corpus must set modality={expected.value!r} "
                    f"(got {self.modality.value!r})"
                )
        return self


# ---------------------------------------------------------------------------
# CompetitorDef
# ---------------------------------------------------------------------------


PIPELINE_TIERS = ("high", "medium", "low")

Capability = Literal["clip", "stream"]


def _default_capabilities() -> list[Capability]:
    return ["clip"]


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

    model_config = ConfigDict(extra="forbid")

    competitor_id: str = Field(
        ..., description="Stable unique identifier for this competitor"
    )
    # Schema revision of this registry entry's shape — 1 is the only shape
    # defined so far; bump when the CompetitorDef contract changes.
    schema_version: int = 1
    modality: Modality
    plugin: str | None = Field(
        None,
        description=(
            "OPM plugin entry-point name. Optional for intent fighters — "
            "derived from the pipeline (None for ensembles)."
        ),
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Valid mycroft.conf fragment. Intent fighters carry an 'intents' "
            "section: {'pipeline': ['<plugin>-<tier>', …], '<plugin>': {…}}."
        ),
    )
    langs: list[str] = Field(
        default_factory=list,
        description="BCP-47 language tags this competitor supports",
    )

    @field_validator("langs")
    @classmethod
    def _validate_langs(cls, v: list[str]) -> list[str]:
        return [validate_lang_tag(tag) for tag in v]

    alias: list[str] | None = Field(
        None,
        description=(
            "Legacy plugin_id values that map to this competitor in ingested rows. "
            "Enables backward-compat without re-running old prediction jobs."
        ),
    )
    capabilities: list[Capability] = Field(
        default_factory=_default_capabilities,
        description=(
            "Wake-word fighters only (§A3.2 / R15): which benchmarking shapes "
            "this engine genuinely supports. 'clip' — scored on isolated "
            "labelled clips (every wake-word fighter). 'stream' — the "
            "underlying detector is designed to run continuously over "
            "streamed audio and accumulate false accepts across hours of "
            "negative audio (openWakeWord, microWakeWord, precise-onnx); "
            "only these compete on the `ww_stream` board "
            "(`runner.ww_bench.WakeWordStreamBench.filter_competitors`). "
            "Defaults to clip-only for back-compat and for genuinely "
            "clip-shaped or unverified engines."
        ),
    )
    # Bestiary card fields (fighter-browser UI)
    display_name: str | None = Field(
        None, description="Human-friendly fighter name shown in the UI"
    )
    species: str | None = Field(
        None,
        description=(
            "Parent plugin class this fighter is an instance of, "
            "e.g. 'PadatiousPipeline'"
        ),
    )
    family: str | None = Field(
        None,
        description=(
            "Collapsed ladder/bestiary grouping key — coarser than `species`: "
            "folds config-variant wrappers of the same underlying engine "
            "(domain-scoped, hierarchical-scoped, …) into one family so the "
            "frontend shows one card per engine, not one per wrapper. "
            "Derived from `species` via FAMILY_ALIASES; equals `species` "
            "unchanged when no folding applies. Always set once the model "
            "is validated — never author this by hand."
        ),
    )
    types: list[str] = Field(
        default_factory=list,
        description=(
            "Architecture tags, e.g. 'GOFAI', 'fuzzy-match', 'embedding', "
            "'neural-net', 'LLM'"
        ),
    )
    description: str | None = Field(
        None, description="Short blurb about how this fighter works"
    )
    model: str | None = Field(
        None, description="Underlying model identifier, when one exists"
    )
    model_hf_repo: str | None = Field(
        None,
        description=(
            "HuggingFace model repo id this fighter's weights ship from "
            "(e.g. 'openai/whisper-tiny'), used ONCE per build to look up "
            "the model's total download size in MB (arena.model_size, M2 "
            "performance boards). None for fighters with no distinct "
            "downloadable model repo (rule-based engines, sklearn "
            "pipelines shipped as plugin code, …) — they simply get no "
            "model-size column value, never a fabricated 0."
        ),
    )
    size: Literal[
        "micro", "tiny", "small", "base", "medium", "large", "x-large", "giant", "titan",
    ] | None = Field(
        None,
        description=(
            "Installed footprint class (package + models): "
            "micro <5MB · tiny 5-50MB · small 50-200MB · base 200-500MB · "
            "medium 500MB-2GB · large 2-8GB · x-large 8-20GB · "
            "giant 20-80GB · titan >80GB (LLM-class)"
        ),
    )
    links: dict[str, str] = Field(
        default_factory=dict,
        description="Named URLs: source, pypi, paper, …",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_pipeline_and_alias(self) -> CompetitorDef:
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

        # `family` grouping is an intent-league concept only: intent
        # paradigms field genuine config-variant *wrappers* of the same
        # underlying engine (domain-scoped, hierarchical-scoped, …), where
        # collapsing wrapper siblings onto one card is the desired ladder/
        # bestiary presentation — see FAMILY_ALIASES above. TTS/STT/wake-word/
        # VAD leagues are different: each competitor there is its own model
        # (e.g. each Phoonnx voice, each onnx-asr checkpoint) and earns its
        # own entry, never collapsed under the shared plugin id. So outside
        # the intent leagues `family` always equals the competitor's own id
        # — every non-intent fighter is a singleton family by construction.
        if self.family is None:
            if self.modality in INTENT_MODALITIES:
                if self.species is not None:
                    self.family = FAMILY_ALIASES.get(self.species, self.species)
            else:
                self.family = self.competitor_id
        return self

    @property
    def pipeline(self) -> list[str]:
        """The ordered pipeline stage names (empty for non-intent fighters)."""
        return list((self.config.get("intents") or {}).get("pipeline") or [])

    @property
    def pipeline_plugins(self) -> list[str]:
        """Unique plugin ids referenced by the pipeline, in stage order."""
        plugins: list[str] = []
        for stage in self.pipeline:
            plugin_id, _tier = split_pipeline_stage(stage)
            if plugin_id not in plugins:
                plugins.append(plugin_id)
        return plugins

    def plugin_config(self, plugin_id: str, short_name: str = "") -> dict[str, Any]:
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
