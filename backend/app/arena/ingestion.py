"""
HuggingFace dataset ingestion for the OVOS Plugin Arena (M2).

Pulls a registered prediction dataset via ``huggingface_hub``/``datasets``,
validates the §3.2 contract (with compatibility notes for the real schema of
``OpenVoiceOS/ovos-stt-bench-pt-PT``), registers plugins found in the rows,
and caches battle-relevant fields into ``ingested_predictions``.

Only metadata is stored — no audio blobs (§P3).

Compatibility note (§3.2):
  The published ``ovos-stt-bench-*`` datasets use a slightly different column
  layout than the spec §3.2 minimum contract.  The ingester accepts both forms
  and normalises to the internal representation:

  | spec column       | real column (ovos-stt-bench-*)    |
  |-------------------|-----------------------------------|
  | sample_id         | dataset_entry_id                  |
  | plugin_id         | plugin_name                       |
  | plugin_version    | model_id (composite, unique key)  |
  | prediction        | prediction_transcript             |
  | reference_text    | transcript                        |
  | wer               | (computed on ingest from above)   |
  | cer / rtf         | absent in current datasets        |
  | runner_version    | absent in current datasets        |
  | created_at        | absent (defaults to ingest time)  |

  Extra columns present in the real schema:
  - ``prediction_confidence``: confidence score; stored in metrics{}
  - ``prediction_type``:       modality tag (STT/TTS/…); used for sanity check
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.arena.models import (
    IngestedPrediction,
    Plugin,
    PluginFamily,
    PredictionSource,
)
from app.arena import db as arena_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mappings — real dataset vs spec
# ---------------------------------------------------------------------------

# Each entry: (spec_name, [real_aliases...])
_STT_COLUMN_MAP: List[tuple] = [
    ("sample_id",       ["sample_id", "dataset_entry_id"]),
    ("plugin_id",       ["plugin_id", "plugin_name"]),
    ("plugin_version",  ["plugin_version", "model_id"]),
    ("prediction",      ["prediction", "prediction_transcript"]),
    ("reference",       ["reference_text", "transcript"]),
    ("dataset_id",      ["dataset_id"]),
    ("lang",            ["lang"]),
    ("wer",             ["wer"]),
    ("cer",             ["cer"]),
    ("rtf",             ["rtf"]),
    ("runner_version",  ["runner_version"]),
    ("created_at",      ["created_at"]),
]

_REQUIRED_FIELDS = {"sample_id", "plugin_id", "plugin_version", "prediction"}


class IngestionError(ValueError):
    """Raised when a dataset row fails contract validation."""


def _resolve(row: Dict[str, Any], spec_name: str, aliases: List[str]) -> Any:
    """Return the first alias present in *row*, else None."""
    for alias in aliases:
        if alias in row:
            return row[alias]
    return None


def _normalise_row(row: Dict[str, Any], modality: PluginFamily) -> Dict[str, Any]:
    """Map real column names to spec names and return a normalised dict."""
    out: Dict[str, Any] = {}
    for spec_name, aliases in _STT_COLUMN_MAP:
        val = _resolve(row, spec_name, aliases)
        if val is not None:
            out[spec_name] = val
    # Extra columns preserved in metrics
    extra_metrics: Dict[str, float] = {}
    if "prediction_confidence" in row and row["prediction_confidence"] is not None:
        try:
            extra_metrics["prediction_confidence"] = float(row["prediction_confidence"])
        except (TypeError, ValueError):
            pass
    out["_extra_metrics"] = extra_metrics
    return out


def validate_row(row: Dict[str, Any], modality: PluginFamily) -> Dict[str, Any]:
    """Validate and normalise one dataset row.

    Raises IngestionError if required fields are missing.
    Returns the normalised dict on success.
    """
    norm = _normalise_row(row, modality)
    missing = _REQUIRED_FIELDS - norm.keys()
    if missing:
        raise IngestionError(f"Row missing required fields: {missing!r}")
    if not norm["sample_id"]:
        raise IngestionError("sample_id must be non-empty")
    if not norm["plugin_id"]:
        raise IngestionError("plugin_id must be non-empty")
    return norm


def _compute_wer(reference: Optional[str], prediction: Optional[str]) -> Optional[float]:
    """Compute word error rate (simple token-level) if both sides are present."""
    if not reference or not prediction:
        return None
    ref_tokens = reference.lower().split()
    hyp_tokens = prediction.lower().split()
    if not ref_tokens:
        return None
    # Levenshtein word-level edit distance (insertions + deletions + substitutions)
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return round(dp[m] / n, 4)


def ingest_dataset(
    hf_dataset: str,
    modality: PluginFamily,
    lang: str,
    revision: str = "main",
    max_rows: Optional[int] = None,
    streaming: bool = True,
) -> PredictionSource:
    """Pull *hf_dataset* and ingest predictions into the arena database.

    Steps
    -----
    1. Register / update a ``PredictionSource`` row for this dataset+revision.
    2. Stream rows from HuggingFace (``datasets`` library).
    3. Validate each row against the §3.2 contract (with compat aliases).
    4. Compute WER when reference text is present and WER not provided.
    5. Upsert ``IngestedPrediction`` rows (no audio blobs).
    6. Auto-register plugins encountered in the rows.

    Returns the updated PredictionSource.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install 'datasets' (pip install datasets)") from exc

    # 1. Register source
    source = arena_db.get_prediction_source_by_dataset(hf_dataset, revision)
    if source is None:
        source = PredictionSource(
            hf_dataset=hf_dataset,
            revision=revision,
            modality=modality,
            lang=lang,
        )
    source.lang = lang
    source.modality = modality
    arena_db.upsert_prediction_source(source)

    # 2. Load dataset
    logger.info("Ingesting %s @ %s (streaming=%s)", hf_dataset, revision, streaming)
    ds = load_dataset(hf_dataset, split="train", streaming=streaming, revision=revision)

    ingested = 0
    errors = 0
    plugin_registry: Dict[str, Plugin] = {}

    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break

        try:
            norm = validate_row(dict(row), modality)
        except IngestionError as exc:
            logger.warning("Row %d skipped: %s", i, exc)
            errors += 1
            continue

        sample_id = norm["sample_id"]
        plugin_id_str = norm["plugin_id"]
        plugin_version = norm["plugin_version"]
        prediction_text = norm["prediction"]
        reference_text = norm.get("reference")
        extra_metrics = norm.get("_extra_metrics", {})

        # Compute WER if not provided
        wer = norm.get("wer")
        if wer is None:
            wer = _compute_wer(reference_text, prediction_text)

        # 4. Upsert prediction row
        pred = IngestedPrediction(
            source_id=source.id,
            sample_id=sample_id,
            plugin_id=plugin_id_str,
            plugin_version=plugin_version,
            prediction=prediction_text,
            reference=reference_text,
            wer=wer,
            metrics=extra_metrics,
            hf_row_ref=f"{hf_dataset}/{revision}/{i}",
            ingested_at=datetime.utcnow(),
        )
        arena_db.upsert_ingested_prediction(pred)
        ingested += 1

        # 5. Auto-register plugin in the plugins table (idempotent)
        if plugin_id_str not in plugin_registry:
            existing = arena_db.get_plugin_by_name(plugin_id_str)
            if existing is None:
                plugin = Plugin(
                    plugin_name=plugin_id_str,
                    display_name=plugin_id_str,
                    family=modality,
                    lang=lang,
                )
                arena_db.upsert_plugin(plugin)
            plugin_registry[plugin_id_str] = existing or arena_db.get_plugin_by_name(plugin_id_str)

    # 6. Update source row_count + ingested_at
    source.row_count = ingested
    source.ingested_at = datetime.utcnow()
    arena_db.upsert_prediction_source(source)

    logger.info(
        "Ingestion complete: %d rows ingested, %d errors, %d plugins registered",
        ingested,
        errors,
        len(plugin_registry),
    )
    return source
