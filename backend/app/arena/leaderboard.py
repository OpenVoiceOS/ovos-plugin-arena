"""
Auto-metric leaderboard builder (§3 "Auto metric leaderboards").

Reads ``predictions/<competitor_id>.jsonl`` files (one per competitor, §3 §3.2
contract) and computes per-modality, per-dataset, per-language leaderboard
tables from the automatic metrics:

- **STT**: WER (word error rate) — computed inline if absent; lower is better.
- **Intent**: accuracy (exact_match rate) and macro F1 — higher is better.
- **WW**: FAR / FRR when ww-bench rows carry ``label`` and ``prediction`` booleans.
- **TTS**: skipped (no objective metric available yet).

The output is a list of ``LeaderboardRow`` dicts that can be serialised to
JSON for the static frontend.

Designed to run standalone (no arena DB required) so the GitHub Actions
``tally.yml`` workflow can call it after pulling fresh prediction JSONLs.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row contract — matches §3.2 prediction JSONL layout
# ---------------------------------------------------------------------------

# Required fields shared by all modalities
_COMMON_REQUIRED = {"competitor_id", "sample_id", "dataset_id", "lang", "plugin_id"}

# Per-modality required fields
_MODALITY_REQUIRED: Dict[str, set] = {
    "stt": {"utterance", "reference_intent"} - {"utterance"},  # empty - just common
    "intent": {"utterance", "reference_intent", "exact_match"},
    "wake_word": {"label", "prediction"},
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _wer(reference: str, hypothesis: str) -> float:
    """Simple word error rate."""
    if not reference:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            dp[j] = (
                prev[j - 1]
                if ref_tokens[i - 1] == hyp_tokens[j - 1]
                else 1 + min(prev[j - 1], prev[j], dp[j - 1])
            )
    return round(dp[m] / n, 4)


def _f1(tp: int, fp: int, fn: int) -> float:
    """Binary F1."""
    if tp + fp + fn == 0:
        return 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


# ---------------------------------------------------------------------------
# Per-modality aggregation
# ---------------------------------------------------------------------------


def _aggregate_stt(rows: List[dict]) -> Dict[str, float]:
    """Aggregate STT rows into WER / sample count metrics."""
    wers = []
    for row in rows:
        wer = row.get("wer")
        if wer is None:
            ref = row.get("reference_text") or row.get("reference") or row.get("transcript")
            pred = row.get("prediction") or row.get("prediction_transcript", "")
            if ref:
                wer = _wer(ref, pred)
        if wer is not None:
            wers.append(wer)
    if not wers:
        return {"samples": len(rows)}
    return {
        "samples": len(rows),
        "wer_mean": round(sum(wers) / len(wers), 4),
        "wer_median": round(sorted(wers)[len(wers) // 2], 4),
    }


def _aggregate_intent(rows: List[dict]) -> Dict[str, float]:
    """Aggregate intent rows into accuracy and macro-F1 metrics."""
    # Per-class true-pos / false-pos / false-neg for macro-F1
    tp_per: Dict[str, int] = defaultdict(int)
    fp_per: Dict[str, int] = defaultdict(int)
    fn_per: Dict[str, int] = defaultdict(int)

    correct = 0
    total = 0
    for row in rows:
        ref = row.get("reference_intent", "")
        pred = row.get("prediction") or ""
        exact = row.get("exact_match")
        if exact is None:
            exact = ref == pred
        if exact:
            correct += 1
            tp_per[ref] += 1
        else:
            fn_per[ref] += 1
            if pred:
                fp_per[pred] += 1
        total += 1

    accuracy = round(correct / total, 4) if total else 0.0

    # Macro F1
    all_classes = set(tp_per) | set(fn_per) | set(fp_per)
    class_f1s = [_f1(tp_per[c], fp_per[c], fn_per[c]) for c in all_classes]
    macro_f1 = round(sum(class_f1s) / len(class_f1s), 4) if class_f1s else 0.0

    return {
        "samples": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "correct": correct,
    }


def _aggregate_ww(rows: List[dict]) -> Dict[str, float]:
    """Aggregate wake-word rows into FAR / FRR metrics."""
    fa = 0  # false activations (predicted True when label False)
    fr = 0  # false rejects (predicted False when label True)
    pos = 0  # total positive samples (label True)
    neg = 0  # total negative samples (label False)

    for row in rows:
        label = row.get("label")
        pred = row.get("prediction")
        if label is None or pred is None:
            continue
        label_bool = bool(label) if not isinstance(label, bool) else label
        pred_bool = bool(pred) if not isinstance(pred, bool) else pred
        if label_bool:
            pos += 1
            if not pred_bool:
                fr += 1
        else:
            neg += 1
            if pred_bool:
                fa += 1

    far = round(fa / neg, 4) if neg else 0.0
    frr = round(fr / pos, 4) if pos else 0.0

    return {
        "samples": len(rows),
        "far": far,
        "frr": frr,
    }


_AGGREGATORS = {
    "stt": _aggregate_stt,
    "intent": _aggregate_intent,
    "wake_word": _aggregate_ww,
}


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", i + 1, path, exc)
    return rows


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_leaderboard(
    predictions_dir: Path,
    modality: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build leaderboard rows from prediction JSONL files.

    Parameters
    ----------
    predictions_dir:
        Directory containing ``<competitor_id>.jsonl`` files (one per competitor).
    modality:
        If given, only process files whose rows belong to this modality.
        Auto-detected from the ``competitor_id`` field if ``None``.

    Returns
    -------
    List of dicts, one per (competitor_id, dataset_id, lang) triple, sorted by
    the primary metric (lower-is-better for STT/WW, higher-is-better for intent).
    """
    results: List[Dict[str, Any]] = []

    for jsonl_path in sorted(predictions_dir.glob("*.jsonl")):
        competitor_id = jsonl_path.stem
        rows = _read_jsonl(jsonl_path)
        if not rows:
            logger.debug("Empty file: %s", jsonl_path)
            continue

        # Group by (dataset_id, lang, modality)
        groups: Dict[tuple, List[dict]] = defaultdict(list)
        for row in rows:
            ds = row.get("dataset_id", "unknown")
            lang = row.get("lang", "unknown")
            # Infer modality from row fields
            if "exact_match" in row:
                mod = "intent"
            elif "wer" in row or "reference_text" in row or "transcript" in row or "reference" in row:
                mod = "stt"
            elif "label" in row:
                mod = "wake_word"
            else:
                mod = "unknown"
            if modality and mod != modality:
                continue
            groups[(ds, lang, mod)].append(row)

        for (ds, lang, mod), grp_rows in groups.items():
            aggregator = _AGGREGATORS.get(mod)
            if aggregator is None:
                logger.debug("No aggregator for modality %s — skipping %s", mod, competitor_id)
                continue

            metrics = aggregator(grp_rows)
            results.append(
                {
                    "competitor_id": competitor_id,
                    "plugin_id": grp_rows[0].get("plugin_id", ""),
                    "dataset_id": ds,
                    "lang": lang,
                    "modality": mod,
                    **metrics,
                }
            )

    # Sort: for STT/WW lower is better (wer_mean / far); for intent higher is better
    def _sort_key(r: dict) -> tuple:
        mod = r["modality"]
        if mod == "stt":
            return (r.get("wer_mean", 1.0),)
        if mod == "intent":
            return (-r.get("accuracy", 0.0),)
        if mod == "wake_word":
            return (r.get("far", 1.0) + r.get("frr", 1.0),)
        return (0.0,)

    results.sort(key=_sort_key)

    # Add rank per (dataset_id, lang, modality) group
    rank_counters: Dict[tuple, int] = defaultdict(int)
    for row in results:
        key = (row["dataset_id"], row["lang"], row["modality"])
        rank_counters[key] += 1
        row["rank"] = rank_counters[key]

    return results


def write_leaderboard_json(
    predictions_dir: Path,
    output_dir: Path,
    modality: Optional[str] = None,
) -> List[Path]:
    """Build leaderboards and write per-(modality, lang) JSON files.

    Files are written to ``output_dir/leaderboard-<modality>-<lang>.json``,
    matching the naming convention used by the static frontend in Mode B/C.

    Returns the list of written file paths.
    """
    rows = build_leaderboard(predictions_dir, modality=modality)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by (modality, lang)
    by_key: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["modality"], row["lang"])].append(row)

    written: List[Path] = []
    for (mod, lang), entries in sorted(by_key.items()):
        fname = f"leaderboard-{mod}-{lang}.json"
        out = output_dir / fname
        out.write_text(
            json.dumps(
                {
                    "modality": mod,
                    "lang": lang,
                    "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
                    "entries": entries,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        written.append(out)
        logger.info("Wrote %s (%d entries)", out, len(entries))

    return written
