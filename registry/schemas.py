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

# Full lang-REGION tags, plus the handful of bare ISO 639-3 primary subtags in
# BARE_PRIMARY_SUBTAGS below. Any other bare subtag — two-letter ("en") or
# three-letter ("eng") — is a defect, not a shorthand. Accepts:
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


#: Languages the corpora carry with no region, because none of the regions
#: they are spoken in is a dialect the data distinguishes. An addition here is
#: an owner ruling about a language, never a convenience.
BARE_PRIMARY_SUBTAGS = frozenset({"arb", "kab"})


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
    if (tag not in BARE_PRIMARY_SUBTAGS and "-" not in tag) or not _BCP47_RE.match(tag):
        raise ValueError(
            f"{tag!r} is not a valid BCP-47 language tag — full lang-REGION "
            "tags are required (e.g. 'en-US', not bare 'en'). The only bare "
            f"subtags allowed are {sorted(BARE_PRIMARY_SUBTAGS)}, languages "
            "spoken across no country dialect the corpora distinguish."
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
# PadaciosoPipeline, where "hierarchical"-style qualifiers would be part of
# that engine's own name rather than a variant qualifier on some other
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
    """Arena leagues, plus the two intent training-corpus namespaces.

    Intent fighters are split by *training regime* — what a fighter has to be
    given before it can answer at all. A zero-shot fighter is handed the
    skill's templates and answers immediately, an online fighter trains on
    them when the device boots, an offline fighter ships a pretrained
    artefact and never sees them. Ranking those against each other compares
    different things, so each is its own league.

    Keyword-supervised engines stay in their own league: hand-written
    vocabulary rules are a different kind of supervision from phrase
    templates, and they need their own corpora.

    ``intent`` names the shared eval-corpus namespace (``registry/datasets/
    intent/``) rather than a fighter league; ``intent_template`` names the
    template-paradigm training-corpus namespace. No competitor is filed
    under either.
    """

    STT = "stt"
    TTS = "tts"
    WAKE_WORD = "wake_word"
    VAD = "vad"
    INTENT = "intent"  # eval-corpus namespace
    INTENT_TEMPLATE = "intent_template"  # template-paradigm training corpora
    INTENT_ZERO_SHOT = "intent_zero_shot"
    INTENT_ONLINE = "intent_online"
    INTENT_OFFLINE = "intent_offline"
    INTENT_KEYWORD = "intent_keyword"
    # Streaming wake-word league (§A3.2 / R15) — same fighters as WAKE_WORD,
    # a distinct board scored from continuous-audio detection events rather
    # than isolated clips. See registry/datasets/ww_stream/*.json.
    WW_STREAM = "ww_stream"


INTENT_MODALITIES = (
    Modality.INTENT, Modality.INTENT_TEMPLATE, Modality.INTENT_ZERO_SHOT,
    Modality.INTENT_ONLINE, Modality.INTENT_OFFLINE, Modality.INTENT_KEYWORD,
)

#: The intent leagues fighters are filed under, in board order.
INTENT_LEAGUES = (
    Modality.INTENT_ZERO_SHOT, Modality.INTENT_ONLINE,
    Modality.INTENT_OFFLINE, Modality.INTENT_KEYWORD,
)


def dataset_namespace(modality: str) -> str:
    """Registry dataset directory a league's eval corpora live in.

    Intent leagues share one set of eval corpora under
    ``registry/datasets/intent/``: the corpus does not care which regime
    answered it.
    """
    if modality in {league.value for league in INTENT_LEAGUES}:
        return Modality.INTENT.value
    return modality


# ---------------------------------------------------------------------------
# Training regimes and engine traits
# ---------------------------------------------------------------------------


class TrainingRegime(str, Enum):
    """What an intent engine needs before it can answer.

    ``zero_shot``  — nothing but the registered templates, consumed as they
                     arrive (prototype embeddings, string matching).
    ``online``     — a training pass over those templates when the device
                     boots.
    ``offline``    — a pretrained artefact, shipped as a model id, that never
                     sees the device's own templates.
    """

    ZERO_SHOT = "zero_shot"
    ONLINE = "online"
    OFFLINE = "offline"


#: Ascending "how much does this cost before it answers" order. A fusion's
#: regime is its heaviest stage.
REGIME_ORDER: dict[TrainingRegime, int] = {
    TrainingRegime.ZERO_SHOT: 0,
    TrainingRegime.ONLINE: 1,
    TrainingRegime.OFFLINE: 2,
}


class EngineTraits(BaseModel):
    """Supervision and training regime of one OPM intent pipeline plugin.

    ``paradigm`` decides which training corpus an engine consumes (see
    ``DatasetDef.paradigm``); ``regime`` decides which league a fighter built
    on it competes in. ``runner.intent_pipeline`` reads both from here.
    """

    model_config = ConfigDict(frozen=True)

    paradigm: Literal["template", "keyword"]
    regime: TrainingRegime
    #: Legacy mycroft.conf key an engine also accepts for its config block
    #: (``"adapt"`` beside ``"ovos-adapt-pipeline-plugin"``), mirroring how
    #: the plugins resolve their own config.
    short_name: str = ""


ENGINE_TRAITS: dict[str, EngineTraits] = {
    "ovos-padatious-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="padatious"),
    "ovos-padacioso-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="padacioso"),
    "ovos-nebulento-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="nebulento"),
    "ovos-jurebes-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="jurebes"),
    "ovos-linha-fina-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="linha_fina"),
    "ovos-markov-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="markov"),
    "ovos-nebulento-hierarchical-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="nebulento_hierarchical"),
    "ovos-linha-fina-domain-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="linha_fina_domain"),
    "ovos-linha-fina-hierarchical-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="linha_fina_hierarchical"),
    "ovos-markov-domain-pipeline-plugin": EngineTraits(
        paradigm="template", regime=TrainingRegime.ONLINE,
        short_name="markov_domain"),
    "ovos-adapt-pipeline-plugin": EngineTraits(
        paradigm="keyword", regime=TrainingRegime.ONLINE,
        short_name="adapt"),
    "ovos-adapt-domain-pipeline-plugin": EngineTraits(
        paradigm="keyword", regime=TrainingRegime.ONLINE,
        short_name="adapt_domain"),
    "ovos-adapt-hierarchical-pipeline-plugin": EngineTraits(
        paradigm="keyword", regime=TrainingRegime.ONLINE,
        short_name="adapt_hierarchical"),
    "ovos-palavreado-pipeline-plugin": EngineTraits(
        paradigm="keyword", regime=TrainingRegime.ONLINE,
        short_name="palavreado"),
    "ovos-palavreado-hierarchical-pipeline": EngineTraits(
        paradigm="keyword", regime=TrainingRegime.ONLINE,
        short_name="palavreado_hierarchical"),
    # The m2v pipeline is two engines behind one entry point: its default
    # classifier mode loads a pretrained head, its prototype mode embeds the
    # registered templates on the spot. ``stage_regime`` reads the mode.
    "ovos-m2v-pipeline": EngineTraits(
        paradigm="template", regime=TrainingRegime.OFFLINE,
        short_name="m2v"),
    "ovos-m2v-prototype-pipeline": EngineTraits(
        paradigm="template", regime=TrainingRegime.ZERO_SHOT,
        short_name="m2v_prototype"),
}


def engine_paradigm(plugin_id: str) -> str:
    """Which training-corpus datashape *plugin_id* consumes."""
    return ENGINE_TRAITS[plugin_id].paradigm


def engine_short_name(plugin_id: str) -> str:
    """Legacy mycroft.conf config key *plugin_id* also answers to."""
    return ENGINE_TRAITS[plugin_id].short_name


def stage_config(plugin_id: str, intents_config: dict[str, Any] | None = None) -> dict:
    """One stage's config block, resolved the way the plugins resolve it:
    under the full entry-point id, or under the legacy short key."""
    intents = intents_config or {}
    block = intents.get(plugin_id)
    if block is None:
        short = ENGINE_TRAITS[plugin_id].short_name if plugin_id in ENGINE_TRAITS else ""
        block = intents.get(short) if short else None
    return dict(block or {})


def stage_regime(plugin_id: str, intents_config: dict[str, Any] | None = None) -> TrainingRegime:
    """Training regime of one pipeline stage as configured.

    *intents_config* is the fighter's whole ``intents`` section, so the mode a
    dual-mode engine runs in is read from whichever key the fighter used —
    the same resolution the runner does.
    """
    if plugin_id == "ovos-m2v-pipeline":
        mode = stage_config(plugin_id, intents_config).get("mode", "classifier")
        return (TrainingRegime.ZERO_SHOT if mode == "prototype"
                else TrainingRegime.OFFLINE)
    return ENGINE_TRAITS[plugin_id].regime


def pipeline_regime(
    plugins: list[str], intents_config: dict[str, Any] | None = None
) -> TrainingRegime:
    """Regime of a whole pipeline: its heaviest stage."""
    return max(
        (stage_regime(p, intents_config) for p in plugins if p in ENGINE_TRAITS),
        key=lambda r: REGIME_ORDER[r],
        default=TrainingRegime.ONLINE,
    )


def expected_league(
    plugins: list[str], intents_config: dict[str, Any] | None = None
) -> Modality:
    """The league a fighter built from *plugins* belongs in.

    Keyword supervision wins over the regime: an all-keyword fighter needs
    hand-written vocabulary corpora nothing else consumes, so it competes in
    the keyword league whatever its regime. Everything else is filed by
    regime.
    """
    if plugins and all(
        p in ENGINE_TRAITS and ENGINE_TRAITS[p].paradigm == "keyword"
        for p in plugins
    ):
        return Modality.INTENT_KEYWORD
    return Modality(f"intent_{pipeline_regime(plugins, intents_config).value}")


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
    lang_field: str | None = Field(
        None,
        description=(
            "Column name holding a per-row language tag, for plain HF "
            "classification datasets that ship every language mixed into "
            "one split instead of one config/split per language (e.g. "
            "tasksource/mtop's 'lang' column: 'de_XX', 'en_XX', ...). "
            "Paired with 'lang_value' — rows whose lang_field value does "
            "not equal lang_value are dropped. None means the split is "
            "already single-language (the common case)."
        ),
    )
    lang_value: str | None = Field(
        None,
        description=(
            "Raw source-column value to keep when 'lang_field' is set "
            "(e.g. 'de_XX' for a dataset entry registered as lang='de-DE'). "
            "Ignored when lang_field is None."
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


class SamplePolicy(BaseModel):
    """Registry-owned sampling cap for a dataset (§ sampling policy).

    A sweep's effective sample set used to depend on whatever ``--max-samples``
    the operator typed, which made cross-fighter and cross-run comparisons
    unreliable and let an unbounded sweep stream a whole corpus by accident.
    ``sample_policy`` moves that decision into the registry: ``max_samples``
    caps how many rows the streamer draws per language (``None`` means no
    cap — small curated eval sets stay uncapped), and ``seed`` pins the
    deterministic subset selection (sorted corpus order, seeded shuffle, head
    ``max_samples``) so every fighter and every run see the SAME rows. An
    operator ``--max-samples`` smaller than the policy's cap still wins, for
    smoke runs.
    """

    model_config = ConfigDict(extra="forbid")

    max_samples: int | None = Field(
        None, description="Cap on rows drawn per language; None = all rows.",
    )
    seed: int = Field(
        1337, description="Seed for the deterministic sorted-shuffle-head subset selection.",
    )


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
    display_name: str | None = Field(
        None,
        description=(
            "Human-readable corpus name for the leaderboard, e.g. "
            "'MTOP (English)' — what a visitor reads instead of the "
            "dataset_id code name."
        ),
    )
    summary: str | None = Field(
        None,
        description=(
            "Two to four plain sentences a non-expert understands: where "
            "the utterances or clips come from, what one row is, how big "
            "the corpus is, what a fighter has to do with it, and any "
            "caveat that changes how to read the score. Required on every "
            "role=eval corpus; ``notes`` stays the technical footnote."
        ),
    )
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
    negatives_dataset_ids: list[str] | None = Field(
        None,
        description=(
            "Wake word: registry dataset_ids of other datasets (usually "
            "role=unrestricted parquet corpora, e.g. ml_spoken_words "
            "negatives) to pool as additional false-accept negatives, "
            "resolved via the registry loader and streamed through the "
            "generic dataset-audio path (unlike ``negatives_sources``, which "
            "lists raw HF repo files and only works for audiofolder "
            "corpora); with no --max-samples cap this streams the full "
            "referenced corpus each sweep."
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
    sample_policy: SamplePolicy | None = Field(
        None,
        description=(
            "Registry-owned deterministic sampling cap for this dataset — "
            "see SamplePolicy. Applied by runner.audio_io.stream_audio_dataset "
            "and by pooled wake-word negatives (each negatives_dataset_ids "
            "entry uses ITS OWN sample_policy). Left unset for small curated "
            "sets that are meant to stream in full."
        ),
    )
    predictions_revision: str | None = Field(
        None,
        description=(
            "Immutable HF commit SHA to pin the predictions dataset "
            "(``predictions_hf``) to for reproducible board assembly. "
            "None means the assemble step tracks whatever revision it is "
            "invoked with (e.g. a floating branch) instead of a fixed SHA."
        ),
    )
    bucket_roles: dict[str, Literal["in_distribution", "generalization"]] | None = Field(
        None,
        description=(
            "Intent eval corpora: what each value of the bucket column means "
            "for the ranked metric. 'in_distribution' rows are phrasings the "
            "fighter's training data already covers (held-out templates from "
            "the same generator, in-domain paraphrase-adjacent utterances) "
            "and are excluded from generalization_accuracy; 'generalization' "
            "rows are the phrasings that metric is about. A corpus that "
            "declares nothing falls back to arena.metrics's default bucket "
            "names, so a corpus whose bucket names are its own "
            "(ovos-intents-v5's id_test/ood) MUST declare them or its ranked "
            "column would not mean the same thing as everyone else's."
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
    # Spoken-intent eval corpora (intent leagues, audio-input eval sets —
    # e.g. FBK-MT/Speech-MASSIVE-test). Intent leagues stay keyed by
    # TRAINING-DATA FORMAT (template/keyword/fusion); audio-vs-text input is
    # a property of the EVAL dataset, never a new league. ``input="audio"``
    # marks a dataset whose eval rows are audio clips, not typed text: the
    # runner transcribes each clip ONCE per (dataset, lang) with the pinned
    # STT below, caches the transcript, then feeds that SAME transcript to
    # every intent fighter — isolating intent-ranking from STT variance.
    # v1 does not fan out combinatorial STT×intent fighters; see
    # docs/SPECIFICATION.md §3.2 and runner/intent_bench.py.
    input: Literal["text", "audio"] = Field(
        "text",
        description=(
            "'text' (default): reference_fields['utterance'] is already "
            "the utterance text a fighter consumes. 'audio': eval rows are "
            "audio clips (reference_fields['audio'] names the audio "
            "column); the runner MUST transcribe with stt_plugin/"
            "stt_config before any intent fighter sees the row."
        ),
    )
    stt_plugin: str | None = Field(
        None,
        description=(
            "input='audio' only (required): OVOS STT plugin id used to "
            "produce the ONE pinned transcript per (dataset, lang) that "
            "every intent fighter is scored against, e.g. "
            "'ovos-stt-plugin-onnx-asr'."
        ),
    )
    stt_config: dict[str, Any] | None = Field(
        None,
        description=(
            "input='audio' only (required): the plugin's config block "
            "(e.g. {'model': 'nemo-parakeet-tdt-0.6b-v3'}) pinning exactly "
            "which model/settings produced the cached transcript — stamped "
            "on every prediction row's stt_config for provenance."
        ),
    )

    @model_validator(mode="after")
    def _validate_audio_input(self) -> DatasetDef:
        """input='audio' eval datasets MUST carry a full STT pin and an
        'audio' reference field — the cached transcript's provenance MUST
        be reconstructible from the dataset def alone."""
        if self.input != "audio":
            return self
        if self.modality not in INTENT_MODALITIES:
            raise ValueError(
                f"{self.dataset_id}: input='audio' is only defined for "
                "intent-league datasets"
            )
        if not self.stt_plugin or not self.stt_config:
            raise ValueError(
                f"{self.dataset_id}: input='audio' requires both "
                "stt_plugin and stt_config (the pinned per-language "
                "default STT that produces the cached transcript)"
            )
        if "audio" not in self.reference_fields:
            raise ValueError(
                f"{self.dataset_id}: input='audio' requires "
                "reference_fields['audio'] naming the audio column"
            )
        return self

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
    training_regime: TrainingRegime | None = Field(
        None,
        description=(
            "Intent fighters (required): what this fighter needs before it "
            "can answer — the league it competes in. A fusion carries the "
            "regime of its heaviest stage; declaring a lighter one is a "
            "validation error."
        ),
    )
    label_set: list[str] | None = Field(
        None,
        description=(
            "Offline intent fighters (required): the dataset_ids whose label "
            "set this fighter's pretrained artefact was trained on. It is "
            "only benchmarked on those datasets — anywhere else it cannot "
            "emit the corpus's labels at all. An empty list means no "
            "registered corpus matches, so the fighter is unranked until one "
            "does."
        ),
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
    trained_on: list[str] = Field(
        default_factory=list,
        description=(
            "dataset_id values whose recordings were in this fighter's "
            "training data. A fighter is never scored on a dataset it "
            "lists here — every scoring point (media/wake-word bench, the "
            "assembler, the autorun scheduler) skips the (fighter, "
            "dataset) pair instead. The deny-list mirror of the offline "
            "intent league's training-regime keying: that one narrows "
            "which corpora a fighter IS eligible for, this one excludes "
            "the ones it must never be graded on. Registry-validated: "
            "every id must resolve to a registered dataset for this "
            "competitor's modality."
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
    model_revision: str | None = Field(
        None,
        description=(
            "Immutable commit sha of the model this fighter runs, for "
            "fighters whose weights come from a model repo. The runner "
            "downloads exactly this commit and hands the plugin the local "
            "path (runner.intent_bench.resolve_model_pin), because no OVOS "
            "pipeline plugin takes a revision today; the sha the snapshot "
            "resolved to is stamped on every prediction row, so a published "
            "row names the artefact that produced it — the same standard the "
            "registry already holds datasets to. It is also written into the "
            "plugin's config block under 'revision' for plugins that grow "
            "support for it."
        ),
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
            self._validate_league(plugins)

        if self.model_revision:
            self._pin_model_revision()

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

    def _pin_model_revision(self) -> None:
        """Carry ``model_revision`` into the plugin config the runner passes
        through, so the pin reaches the code that downloads the weights
        instead of only describing them."""
        intents = self.config.get("intents")
        if not isinstance(intents, dict):
            return
        for key, block in intents.items():
            if key != "pipeline" and isinstance(block, dict) and block.get("model"):
                block["revision"] = self.model_revision

    def _validate_league(self, plugins: list[str]) -> None:
        """League, regime and label set of an intent fighter must agree.

        Every intent fighter declares ``training_regime``; it must match the
        regime derived from its stages (a fusion's heaviest one), and the
        league it is filed under must follow from that regime — except for
        all-keyword fighters, which compete in the keyword league whatever
        their regime. Offline fighters additionally declare the label sets
        their pretrained artefact can emit.
        """
        if self.training_regime is None:
            raise ValueError(
                f"{self.competitor_id}: intent fighters must declare "
                "training_regime (zero_shot | online | offline)"
            )
        known = [p for p in plugins if p in ENGINE_TRAITS]
        if known:
            intents = self.config.get("intents") or {}
            derived = pipeline_regime(known, intents)
            if derived != self.training_regime:
                raise ValueError(
                    f"{self.competitor_id}: declares "
                    f"training_regime={self.training_regime.value!r} but its "
                    f"stages need {derived.value!r}"
                )
            league = expected_league(plugins, intents)
            if self.modality != league:
                raise ValueError(
                    f"{self.competitor_id}: belongs in the "
                    f"{league.value!r} league, not {self.modality.value!r}"
                )

        if self.training_regime is TrainingRegime.OFFLINE:
            if self.label_set is None:
                raise ValueError(
                    f"{self.competitor_id}: offline fighters must declare "
                    "label_set — the dataset_ids their pretrained artefact "
                    "was trained on"
                )
        elif self.label_set is not None:
            raise ValueError(
                f"{self.competitor_id}: label_set is only meaningful for "
                "offline fighters"
            )

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
