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


def load_predictions_dir(
    predictions_dir: Path, lang: str | None = None
) -> list[PredictionRow]:
    """Load ``*.jsonl`` files under *predictions_dir*.

    *lang*, when given, restricts loading to the dataset's own language
    shard: ``predictions/<lang>/*.jsonl`` plus the flat legacy
    ``predictions/*.jsonl`` root files — mirroring the matching policy in
    ``runner.queue_tools.find_missing_pairs`` (post-#54). Without it (the
    default, and always for genuinely multi/unknown-lang datasets), every
    ``*.jsonl`` under *predictions_dir* is loaded, nested per-lang dirs and
    the flat layout alike.

    Restricting by lang matters: a prediction repo commonly accumulates
    orphaned shards from other lang runs (e.g. an English-forced decode of
    German audio published under ``predictions/en/``) alongside a
    concrete-lang dataset's own dir. Merging those into the same
    competitor pool silently poisons that dataset's scores with
    wrong-language predictions.
    """
    if lang:
        paths = sorted(predictions_dir.glob("*.jsonl"))
        lang_dir = predictions_dir / lang
        if lang_dir.is_dir():
            paths = sorted(set(paths) | set(lang_dir.glob("*.jsonl")))
    else:
        paths = sorted(predictions_dir.glob("**/*.jsonl"))

    rows: list[PredictionRow] = []
    for path in paths:
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


def load_predictions(
    source: str, revision: str = "main", lang: str | None = None
) -> list[PredictionRow]:
    """Load predictions from a local directory or an HF dataset repo id.

    *lang* is forwarded to :func:`load_predictions_dir` — pass the
    dataset's own concrete lang to exclude other-lang shards published to
    the same prediction repo; omit it (or pass ``None``) for a genuinely
    multi/unknown-lang dataset, which must keep loading every lang dir.
    """
    path = Path(source)
    if path.is_dir():
        return load_predictions_dir(path, lang=lang)
    return load_predictions_dir(fetch_hf_predictions(source, revision), lang=lang)


def group_rows(
    rows: list[PredictionRow],
    unregistered: dict[str, int] | None = None,
) -> dict[tuple[str, str, str], dict[str, dict[str, PredictionRow]]]:
    """Group rows as (modality, dataset_id, lang) → sample_id → competitor → row.

    Rows whose modality cannot be inferred are dropped (with a warning).
    Duplicate (sample, competitor) rows keep the last occurrence.

    This is the single choke point every board (benchmark, battles, ELO)
    flows through, so it is also where board truth is enforced: rows whose
    ``competitor_id`` is not present in the current registry for that
    modality are dropped — a fighter removed from the registry (e.g. its
    definition deleted) must not keep appearing on published boards just
    because its orphaned HF prediction shards are still fetched. Dropped
    (competitor_id → row count) is aggregated into *unregistered* when
    given, so callers can surface it in the assemble output.
    """
    from registry.loaders import list_competitors

    registered_by_modality: dict[str, set[str]] = {}

    def _is_registered(modality: str, competitor_id: str) -> bool:
        if modality not in registered_by_modality:
            try:
                registered_by_modality[modality] = {
                    c.competitor_id for c in list_competitors(modality)
                }
            except Exception as exc:
                logger.warning(
                    "Could not load registry for modality %s: %s", modality, exc
                )
                registered_by_modality[modality] = set()
        return competitor_id in registered_by_modality[modality]

    grouped: dict[tuple[str, str, str], dict[str, dict[str, PredictionRow]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    dropped = 0
    unregistered_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        modality = infer_modality(row.model_dump(exclude_none=True))
        if modality == "unknown":
            dropped += 1
            continue
        if not _is_registered(modality, row.competitor_id):
            unregistered_counts[row.competitor_id] += 1
            continue
        key = (modality, row.dataset_id, row.lang)
        grouped[key][row.sample_id][row.competitor_id] = row
    if dropped:
        logger.warning("Dropped %d rows with undetectable modality", dropped)
    for competitor_id, count in sorted(unregistered_counts.items()):
        logger.warning(
            "Excluded %d prediction row(s) for unregistered competitor_id "
            "%r (not in the current registry — orphaned shard?)",
            count, competitor_id,
        )
    if unregistered is not None:
        for competitor_id, count in unregistered_counts.items():
            unregistered[competitor_id] = unregistered.get(competitor_id, 0) + count
    return {k: dict(v) for k, v in grouped.items()}
