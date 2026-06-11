"""
Auto-metric benchmark boards (§3.1 "Auto-metric leaderboards").

Computes per-(modality, dataset, lang) benchmark tables straight from
prediction rows — no votes involved.  These boards sit beside the ELO board
in the UI and also drive the ELO seeding (§4 R5).

Intent scoring conventions
--------------------------
- A row with ``reference_intent = null`` is an out-of-scope (OOD) sample:
  the correct behaviour is to predict nothing.  Predicting any intent there
  is a false positive (counted in ``ood_fpr`` and penalising that intent's
  precision in macro-F1).
- ``accuracy`` covers ALL samples, counting correct OOD rejections.
- ``slot_f1`` is exact-match over the gold slot dict, evaluated only on
  rows where the intent was correct and gold slots exist.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Dict, List

from arena.models import BenchmarkBoard, BenchmarkEntry, PredictionRow

PRIMARY_METRIC = {
    "intent": "accuracy",
    "stt": "wer_mean",
    "wake_word": "error_rate",
}

# Higher is better for these primary metrics; lower for the rest.
_HIGHER_BETTER = {"accuracy"}


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def row_is_correct(row: PredictionRow) -> bool:
    """Whether one intent prediction row is correct (incl. OOD rejection)."""
    if row.reference_intent is None:
        return row.prediction is None
    if row.exact_match is not None:
        return bool(row.exact_match)
    return row.prediction == row.reference_intent


def score_intent(rows: List[PredictionRow]) -> Dict[str, float]:
    """Aggregate intent rows into accuracy / macro-F1 / OOD-FPR / slot metrics."""
    per_intent: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    per_bucket: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    correct = 0
    ood_fp = 0
    ood_n = 0
    slot_correct = 0
    slot_total = 0
    latencies: List[float] = []

    for row in rows:
        bucket = row.bucket or "test"
        per_bucket[bucket]["total"] += 1
        if row.latency_ms is not None:
            latencies.append(row.latency_ms)

        if row.reference_intent is None:
            ood_n += 1
            if row.prediction is None:
                per_bucket[bucket]["correct"] += 1
                correct += 1
            else:
                ood_fp += 1
                per_intent[row.prediction]["fp"] += 1
            continue

        if row_is_correct(row):
            per_intent[row.reference_intent]["tp"] += 1
            per_bucket[bucket]["correct"] += 1
            correct += 1
            gold_slots = row.reference_slots or {}
            if gold_slots:
                slot_total += 1
                pred_slots = row.predicted_slots or {}
                if all(
                    str(pred_slots.get(k, "")).strip().lower()
                    == str(v).strip().lower()
                    for k, v in gold_slots.items()
                ):
                    slot_correct += 1
        else:
            per_intent[row.reference_intent]["fn"] += 1
            if row.prediction is not None:
                per_intent[row.prediction]["fp"] += 1

    n = len(rows)
    f1s = [_f1(v["tp"], v["fp"], v["fn"]) for v in per_intent.values()]
    metrics: Dict[str, float] = {
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
    }
    if ood_n:
        metrics["ood_fpr"] = round(ood_fp / ood_n, 4)
    if slot_total:
        metrics["slot_exact_match"] = round(slot_correct / slot_total, 4)
    if latencies:
        metrics["latency_ms_median"] = round(median(latencies), 2)
    for bucket, v in sorted(per_bucket.items()):
        if v["total"]:
            metrics[f"acc_{bucket}"] = round(v["correct"] / v["total"], 4)
    return metrics


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate via word-level Levenshtein distance."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens:
        return 0.0
    dp = list(range(len(hyp_tokens) + 1))
    for i in range(1, len(ref_tokens) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(hyp_tokens) + 1):
            dp[j] = (
                prev[j - 1]
                if ref_tokens[i - 1] == hyp_tokens[j - 1]
                else 1 + min(prev[j - 1], prev[j], dp[j - 1])
            )
    return round(dp[-1] / len(ref_tokens), 4)


def row_wer(row: PredictionRow) -> float | None:
    if row.wer is not None:
        return row.wer
    if row.reference_text is not None and row.prediction is not None:
        return _wer(row.reference_text, row.prediction)
    return None


def score_stt(rows: List[PredictionRow]) -> Dict[str, float]:
    wers = [w for w in (row_wer(r) for r in rows) if w is not None]
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    metrics: Dict[str, float] = {}
    if wers:
        metrics["wer_mean"] = round(sum(wers) / len(wers), 4)
        metrics["wer_median"] = round(median(wers), 4)
    if latencies:
        metrics["latency_ms_median"] = round(median(latencies), 2)
    return metrics


_SCORERS = {
    "intent": score_intent,
    "stt": score_stt,
}


def build_benchmark_board(
    modality: str,
    dataset_id: str,
    lang: str,
    by_competitor: Dict[str, List[PredictionRow]],
    generated_at: str,
) -> BenchmarkBoard:
    """Build one benchmark board from per-competitor row lists."""
    scorer = _SCORERS.get(modality)
    primary = PRIMARY_METRIC.get(modality, "accuracy")
    entries: List[BenchmarkEntry] = []
    if scorer is not None:
        for competitor_id, rows in by_competitor.items():
            entries.append(
                BenchmarkEntry(
                    competitor_id=competitor_id,
                    plugin_id=rows[0].plugin_id if rows else "",
                    samples=len(rows),
                    metrics=scorer(rows),
                )
            )

    reverse = primary in _HIGHER_BETTER
    worst = 0.0 if reverse else float("inf")
    entries.sort(key=lambda e: e.metrics.get(primary, worst), reverse=reverse)
    for i, entry in enumerate(entries, 1):
        entry.rank = i

    return BenchmarkBoard(
        modality=modality,
        dataset_id=dataset_id,
        lang=lang,
        generated_at=generated_at,
        primary_metric=primary,
        entries=entries,
    )
