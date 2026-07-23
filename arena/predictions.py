"""
Prediction loading for the OVOS Plugin Arena.

Predictions live in HuggingFace dataset repos (§P2 — HF is the artifact
layer) as per-competitor JSON-lines files::

    predictions/<competitor_id>.jsonl

Each row follows the §3.2 contract (see ``arena.models.PredictionRow``).
This module fetches those files (or reads a local directory with the same
layout) and groups rows for the assembler and metrics builders.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from arena.models import PredictionRow

logger = logging.getLogger(__name__)

# §4 A2 schema convergence — memoized plugin_id -> competitor_id re-keying
# (registry.loaders.get_competitor_by_alias scans every registry JSON file;
# doing that per-row for a large legacy dataset would be far too slow).
# Cleared implicitly per process; registry content doesn't change mid-run.
_alias_cache: dict[tuple[str, str], str | None] = {}


def _resolve_competitor_id(modality: str, plugin_id: str) -> str | None:
    key = (modality, plugin_id)
    if key not in _alias_cache:
        try:
            from registry.loaders import get_competitor_by_alias
            comp = get_competitor_by_alias(modality, plugin_id)
            _alias_cache[key] = comp.competitor_id if comp else None
        except Exception as exc:
            logger.warning("Alias re-keying unavailable (%s): %s", plugin_id, exc)
            _alias_cache[key] = None
    return _alias_cache[key]

# Modality is inferred per row from the §3.2 payload fields.
_INTENT_FIELDS = {"reference_intent", "exact_match"}
_STT_FIELDS = {"reference_text", "wer"}
# VAD rows label clips speech vs non-speech and decide speech vs silence
# (see runner.vad_bench); wake-word rows use positive/detected vocabulary.
_VAD_VALUES = {"speech", "silence", "non_speech"}


def infer_modality(row: dict) -> str:
    """League of one row — the explicit ``modality`` field wins, payload
    field sniffing is the fallback for legacy rows."""
    if row.get("modality"):
        return row["modality"]
    if _INTENT_FIELDS & row.keys():
        return "intent"
    if _STT_FIELDS & row.keys():
        return "stt"
    if "label" in row:
        values = {
            str(row.get(key)).strip().lower()
            for key in ("label", "prediction")
            if row.get(key) is not None
        }
        if values & _VAD_VALUES:
            return "vad"
        return "wake_word"
    return "unknown"


def parse_row(raw: dict, competitor_id: str) -> PredictionRow:
    """Validate one raw JSONL row into a PredictionRow.

    §4 A2 schema convergence: rows in the legacy ``STTRow`` column layout
    (``dataset_entry_id``/``plugin_name``, no ``sample_id`` — already
    published to ``ovos-stt-bench-*`` before the runner switched to writing
    the canonical shape directly) are converted first via
    ``STTRow.to_prediction_row_dict``.

    Unknown keys are preserved in ``extras``. ``competitor_id`` resolution,
    in order: the row's own value → registry alias re-keying from
    ``plugin_id`` (canonical rows written by ``runner/plugin_runner.py``
    carry ``plugin_id`` but not ``competitor_id`` — the runner has no
    registry dependency by design) → the filename stem (the canonical
    per-competitor-file layout, §3.2).
    """
    if raw.get("dataset_entry_id") and not raw.get("sample_id"):
        from runner.schema import STTRow
        legacy = STTRow.from_dict(raw)
        resolved = _resolve_competitor_id("stt", legacy.plugin_name)
        raw = legacy.to_prediction_row_dict(resolved or "")
        if not resolved:
            del raw["competitor_id"]  # let the fallback chain below decide
        raw["schema_version"] = 1  # provenance: converted from the legacy layout

    known = set(PredictionRow.model_fields)
    data = {k: v for k, v in raw.items() if k in known}
    extras = {k: v for k, v in raw.items() if k not in known}
    data.setdefault("extras", {})
    data["extras"] = {**extras, **data.get("extras", {})}

    if not data.get("competitor_id") and data.get("plugin_id") and data.get("modality"):
        resolved = _resolve_competitor_id(data["modality"], data["plugin_id"])
        if resolved:
            data["competitor_id"] = resolved
    data.setdefault("competitor_id", competitor_id)
    return PredictionRow(**data)


def read_jsonl(path: Path) -> list[PredictionRow]:
    """Read one per-competitor prediction file, skipping malformed lines."""
    rows: list[PredictionRow] = []
    competitor_id = path.stem
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(parse_row(json.loads(line), competitor_id))
            except Exception as exc:
                logger.warning("%s:%d skipped: %s", path.name, lineno, exc)
    return rows


def load_predictions_dir(predictions_dir: Path) -> list[PredictionRow]:
    """Load every ``*.jsonl`` under *predictions_dir* (nested per-lang dirs
    and the flat legacy layout alike)."""
    rows: list[PredictionRow] = []
    for path in sorted(predictions_dir.glob("**/*.jsonl")):
        file_rows = read_jsonl(path)
        logger.info("Loaded %d rows from %s",
                    len(file_rows), path.relative_to(predictions_dir))
        rows.extend(file_rows)
    return rows


def resolve_predictions_revision(repo_id: str, revision: str = "main") -> str:
    """Resolve *revision* (a branch, tag, or SHA) to an immutable commit SHA.

    Used by ``assemble`` (§C — pinned predictions revision) so a benchmark
    board's provenance is a fixed commit, not a floating ref that could
    change under it after the board is published.
    """
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision)
    if not info.sha:
        raise ValueError(f"HF did not return a commit sha for {repo_id}@{revision}")
    return info.sha


def fetch_hf_predictions(repo_id: str, revision: str = "main") -> Path:
    """Download the ``predictions/`` folder of an HF dataset repo.

    Returns the local path of the downloaded ``predictions`` directory.
    Public datasets need no token; CI therefore runs unauthenticated.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["predictions/**/*.jsonl", "predictions/*.jsonl"],
    )
    return Path(local) / "predictions"


def load_predictions(source: str, revision: str = "main") -> list[PredictionRow]:
    """Load predictions from a local directory or an HF dataset repo id."""
    path = Path(source)
    if path.is_dir():
        return load_predictions_dir(path)
    return load_predictions_dir(fetch_hf_predictions(source, revision))


def group_rows(
    rows: list[PredictionRow],
) -> dict[tuple[str, str, str], dict[str, dict[str, PredictionRow]]]:
    """Group rows as (modality, dataset_id, lang) → sample_id → competitor → row.

    Rows whose modality cannot be inferred are dropped (with a warning).
    Duplicate (sample, competitor) rows keep the last occurrence.
    """
    grouped: dict[tuple[str, str, str], dict[str, dict[str, PredictionRow]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    dropped = 0
    for row in rows:
        modality = infer_modality(row.model_dump(exclude_none=True))
        if modality == "unknown":
            dropped += 1
            continue
        key = (modality, row.dataset_id, row.lang)
        grouped[key][row.sample_id][row.competitor_id] = row
    if dropped:
        logger.warning("Dropped %d rows with undetectable modality", dropped)
    return {k: dict(v) for k, v in grouped.items()}
