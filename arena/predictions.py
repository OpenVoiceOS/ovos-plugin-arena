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
from typing import Dict, List, Tuple

from arena.models import PredictionRow

logger = logging.getLogger(__name__)

# Modality is inferred per row from the §3.2 payload fields.
_INTENT_FIELDS = {"reference_intent", "exact_match"}
_STT_FIELDS = {"reference_text", "wer"}


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
        return "wake_word"
    return "unknown"


def parse_row(raw: dict, competitor_id: str) -> PredictionRow:
    """Validate one raw JSONL row into a PredictionRow.

    Unknown keys are preserved in ``extras``; the ``competitor_id`` falls
    back to the filename stem when absent from the row.
    """
    known = set(PredictionRow.model_fields)
    data = {k: v for k, v in raw.items() if k in known}
    extras = {k: v for k, v in raw.items() if k not in known}
    data.setdefault("competitor_id", competitor_id)
    data["extras"] = extras
    return PredictionRow(**data)


def read_jsonl(path: Path) -> List[PredictionRow]:
    """Read one per-competitor prediction file, skipping malformed lines."""
    rows: List[PredictionRow] = []
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


def load_predictions_dir(predictions_dir: Path) -> List[PredictionRow]:
    """Load every ``*.jsonl`` under *predictions_dir* (nested per-lang dirs
    and the flat legacy layout alike)."""
    rows: List[PredictionRow] = []
    for path in sorted(predictions_dir.glob("**/*.jsonl")):
        file_rows = read_jsonl(path)
        logger.info("Loaded %d rows from %s",
                    len(file_rows), path.relative_to(predictions_dir))
        rows.extend(file_rows)
    return rows


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


def load_predictions(source: str, revision: str = "main") -> List[PredictionRow]:
    """Load predictions from a local directory or an HF dataset repo id."""
    path = Path(source)
    if path.is_dir():
        return load_predictions_dir(path)
    return load_predictions_dir(fetch_hf_predictions(source, revision))


def group_rows(
    rows: List[PredictionRow],
) -> Dict[Tuple[str, str, str], Dict[str, Dict[str, PredictionRow]]]:
    """Group rows as (modality, dataset_id, lang) → sample_id → competitor → row.

    Rows whose modality cannot be inferred are dropped (with a warning).
    Duplicate (sample, competitor) rows keep the last occurrence.
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[str, PredictionRow]]] = (
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
