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
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from arena.models import PredictionRow

logger = logging.getLogger(__name__)

# Pause before each retry of a Hub download; the final ``None`` marks the
# last attempt, after which the error is re-raised.
HF_FETCH_BACKOFF_SECONDS: tuple[float | None, ...] = (5.0, 15.0, None)

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


def _load_paths(predictions_dir: Path, paths: list[Path]) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for path in paths:
        file_rows = read_jsonl(path)
        logger.info("Loaded %d rows from %s",
                    len(file_rows), path.relative_to(predictions_dir))
        rows.extend(file_rows)
    return rows


def iter_predictions_dir(
    predictions_dir: Path, lang: str | None = None
) -> Iterator[list[PredictionRow]]:
    """Like :func:`load_predictions_dir`, but yields one chunk of rows at a
    time instead of building one list for the whole *predictions_dir* —
    so a caller can group each chunk and release its raw rows before the
    next one loads (see ``arena.cli.cmd_assemble``).

    A concrete *lang* (one dataset's own shard, the common case) is small
    enough to stay one chunk. A genuinely multi/unknown-lang dataset
    (``lang=None``) is chunked per top-level lang subdirectory instead —
    that is what bounds memory for a dataset repo that bundles every
    language's predictions together (e.g. the intents-for-eval repo's 989
    files across ~15 langs, ~1.7M rows total if loaded as one list: the
    §assemble memory hosted-runner OOM was this exact dataset, loaded
    whole, under a 7GB cap). Flat legacy root-level files (no lang
    subdirs at all) are still yielded as a single chunk.
    """
    if lang:
        paths = sorted(predictions_dir.glob("*.jsonl"))
        lang_dir = predictions_dir / lang
        if lang_dir.is_dir():
            paths = sorted(set(paths) | set(lang_dir.glob("*.jsonl")))
        yield _load_paths(predictions_dir, paths)
        return

    root_files = sorted(predictions_dir.glob("*.jsonl"))
    if root_files:
        yield _load_paths(predictions_dir, root_files)
    for sub in sorted(p for p in predictions_dir.iterdir() if p.is_dir()):
        chunk = _load_paths(predictions_dir, sorted(sub.glob("**/*.jsonl")))
        if chunk:
            yield chunk


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

    Whole-list convenience wrapper over :func:`iter_predictions_dir` — use
    that directly (as ``arena.cli.cmd_assemble`` does) when *predictions_dir*
    may be large enough that holding every row at once matters.
    """
    rows: list[PredictionRow] = []
    for chunk in iter_predictions_dir(predictions_dir, lang=lang):
        rows.extend(chunk)
    return rows


#: Build-process-lifetime cache: "repo_id@revision" -> resolved commit sha.
#: An assemble run calls this once per source repo already (``sources`` in
#: ``arena.cli.cmd_assemble`` is a deduped set), but the cache still matters:
#: it makes the "at most once per repo per run" contract hold even if a
#: future caller resolves the same repo from more than one place (e.g. a
#: dataset's own board plus a paradigm sub-league sharing an owner), and it
#: is what a test can assert against without mocking every call site.
_revision_cache: dict[str, str] = {}


def _is_revision_resolution_fatal(exc: Exception) -> bool:
    """True when *exc* means ``dataset_info`` will never resolve *revision*,
    no matter how many times it is retried: the repo does not exist, is
    gated, is private to credentials this run does not hold, or the
    revision string itself does not exist on the repo. Anything else (a
    429, a 5xx, a timeout, a dropped connection) is a transient Hub hiccup
    that a retry can plausibly clear.
    """
    if isinstance(exc, (GatedRepoError, RevisionNotFoundError, RepositoryNotFoundError)):
        return True
    return (isinstance(exc, HfHubHTTPError) and exc.response is not None
            and exc.response.status_code in (401, 403, 404))


def resolve_predictions_revision(repo_id: str, revision: str = "main") -> str:
    """Resolve *revision* (a branch, tag, or SHA) to an immutable commit SHA.

    Used by ``assemble`` (§C — pinned predictions revision) so a benchmark
    board's provenance is a fixed commit, not a floating ref that could
    change under it after the board is published. Memoized per
    ``repo_id@revision`` for the life of the process (see
    ``_revision_cache``) — a fresh process re-resolves, since a floating
    ref like ``main`` can move between runs.

    A transient Hub failure (rate limiting, a 5xx, a timeout) is retried
    with the same bounded backoff ``fetch_hf_predictions`` uses
    (``HF_FETCH_BACKOFF_SECONDS``) rather than handed to the caller after a
    single attempt — an unauthenticated daily assemble walks the same
    ~120 repos this way and trips the same rate limiter. A fatal failure
    (see ``_is_revision_resolution_fatal``) propagates immediately,
    unretried, so it stays visible.
    """
    key = f"{repo_id}@{revision}"
    if key in _revision_cache:
        return _revision_cache[key]

    from huggingface_hub import HfApi

    api = HfApi()
    last: Exception | None = None
    for attempt, pause in enumerate(HF_FETCH_BACKOFF_SECONDS, 1):
        try:
            info = api.dataset_info(repo_id, revision=revision)
            if not info.sha:
                raise ValueError(f"HF did not return a commit sha for {repo_id}@{revision}")
            _revision_cache[key] = info.sha
            return info.sha
        except Exception as exc:
            if _is_revision_resolution_fatal(exc):
                raise
            last = exc
            if pause is None:
                break
            logger.warning(
                "Resolving %s@%s failed (attempt %d/%d): %s — retrying in %ss",
                repo_id, revision, attempt, len(HF_FETCH_BACKOFF_SECONDS), exc, pause)
            time.sleep(pause)
    raise last


def reset_revision_cache() -> None:
    """Clear the module-level revision-resolution cache (tests only)."""
    _revision_cache.clear()


def fetch_hf_predictions(repo_id: str, revision: str = "main") -> Path | None:
    """Download the ``predictions/`` folder of an HF dataset repo.

    Returns the local path of the downloaded ``predictions`` directory, or
    ``None`` when the repo (or its ``predictions/`` folder) does not exist
    — a dataset registered but never swept has no prediction repo yet, and
    the arena already renders such fighters as upcoming. That is missing
    data, not a failed source, so it neither retries nor fails the run.

    Public datasets need no token; CI therefore runs unauthenticated.

    An unauthenticated daily assemble walks ~120 prediction repos back to
    back and routinely trips the Hub's rate limiter, so a transient
    failure is retried a bounded number of times with a growing pause
    before it is allowed to propagate. Bounded on purpose: a repo that
    stays unreachable must still fail the run rather than stall it. A repo
    that is gated, private to credentials the run does hold, or pinned to a
    revision that no longer exists is a registry pointing at something the
    arena cannot read — that propagates immediately, unretried, so it stays
    visible.
    """
    from huggingface_hub import snapshot_download

    last: Exception | None = None
    for attempt, pause in enumerate(HF_FETCH_BACKOFF_SECONDS, 1):
        try:
            local = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["predictions/**/*.jsonl", "predictions/*.jsonl"],
            )
            predictions = Path(local) / "predictions"
            if not predictions.is_dir():
                logger.info("No predictions published for %s yet", repo_id)
                return None
            return predictions
        except Exception as exc:
            if _is_missing(exc):
                logger.info("No predictions published for %s yet", repo_id)
                return None
            if _is_unreadable(exc):
                logger.warning("Cannot read predictions repo %s: %s", repo_id, exc)
                raise
            last = exc
            if pause is None:
                break
            logger.warning("Fetching %s failed (attempt %d/%d): %s — retrying in %ss",
                        repo_id, attempt, len(HF_FETCH_BACKOFF_SECONDS), exc, pause)
            time.sleep(pause)
    raise last


def _is_missing(exc: Exception) -> bool:
    """True when *exc* says the repo simply is not there. Unauthenticated,
    the Hub answers 401 for both a private and a nonexistent repo;
    ``huggingface_hub`` raises ``RepositoryNotFoundError`` for that pair,
    and the arena reads it as nonexistent because every registered
    prediction repo is meant to be public.

    ``GatedRepoError`` subclasses ``RepositoryNotFoundError``, so it has to
    be excluded before the isinstance check rather than after it.

    A repo that exists but holds no ``predictions/`` tree is missing data
    too, but no exception reports that: ``snapshot_download`` with
    ``allow_patterns`` that match nothing succeeds, downloads zero files
    and returns a snapshot path it never created. That case is caught by
    the caller looking at the returned path instead."""
    if isinstance(exc, GatedRepoError):
        return False
    return isinstance(exc, RepositoryNotFoundError)


def _is_unreadable(exc: Exception) -> bool:
    """True when the registry points at something this run cannot read: a
    gated repo, a repo private to the credentials in hand (403), or a
    pinned revision that no longer resolves. A vanished revision is not
    absent data — the repo is there and its default ref may well carry
    live rows — so it must stay visible rather than pass for an
    unpublished dataset."""
    if isinstance(exc, GatedRepoError):
        return True
    if isinstance(exc, RevisionNotFoundError):
        return True
    return (isinstance(exc, HfHubHTTPError) and exc.response is not None
            and exc.response.status_code == 403)


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
    fetched = fetch_hf_predictions(source, revision)
    return load_predictions_dir(fetched, lang=lang) if fetched else []


def iter_predictions(
    source: str, revision: str = "main", lang: str | None = None
) -> Iterator[list[PredictionRow]]:
    """Like :func:`load_predictions`, but yields one chunk of rows at a
    time via :func:`iter_predictions_dir` instead of returning one list
    for the whole *source*. Used by ``arena.cli.cmd_assemble`` so a large
    multi-lang predictions source never has to sit fully in memory before
    grouping starts."""
    path = Path(source)
    if path.is_dir():
        yield from iter_predictions_dir(path, lang=lang)
        return
    fetched = fetch_hf_predictions(source, revision)
    if fetched:
        yield from iter_predictions_dir(fetched, lang=lang)


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
