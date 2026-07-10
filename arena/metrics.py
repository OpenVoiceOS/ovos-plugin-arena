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

import random
from collections import defaultdict
from statistics import median

from arena.models import BenchmarkBoard, BenchmarkEntry, PredictionRow
from arena.rating import percentile

BOOTSTRAP_ROUNDS = 1000
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5
BOOTSTRAP_SEED = 0

PRIMARY_METRIC = {
    "intent": "accuracy",
    "intent_template": "accuracy",
    "intent_keyword": "accuracy",
    "stt": "wer_mean",
    "wake_word": "error_rate",
    "vad": "error_rate",
}

# Higher is better for these primary metrics; lower for the rest (error rates,
# WER, latency).  ``accuracy`` is the only intent/wake-word board ranked
# descending; every other primary metric ranks ascending.
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


def score_intent(rows: list[PredictionRow]) -> dict[str, float]:
    """Aggregate intent rows into accuracy / macro-F1 / OOD-FPR / slot metrics."""
    per_intent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    per_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    correct = 0
    ood_fp = 0
    ood_n = 0
    slot_correct = 0
    slot_total = 0
    latencies: list[float] = []

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
    metrics: dict[str, float] = {
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


def _wer_components(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word edit distance, reference word count) via word-level Levenshtein."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens:
        return 0, 0
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
    return dp[-1], len(ref_tokens)


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate via word-level Levenshtein distance."""
    errors, ref_words = _wer_components(reference, hypothesis)
    return round(errors / ref_words, 4) if ref_words else 0.0


def row_wer(row: PredictionRow) -> float | None:
    if row.wer is not None:
        return row.wer
    if row.reference_text is not None and row.prediction is not None:
        return _wer(row.reference_text, row.prediction)
    return None


def row_wer_components(row: PredictionRow) -> tuple[float, float] | None:
    """(word errors, reference word count) for one row, for WER bootstrap CIs.

    Word-level errors are aggregated as ``sum(errors) / sum(ref_words)``
    across a resample (§4 A1.2) rather than averaging per-row WER, since
    per-utterance WER is not comparable across utterances of different
    length — a single error in a 2-word command is not the same signal as
    one error in a 20-word sentence.

    Falls back to unit weight (``errors=row.wer, ref_words=1.0``) when only
    a precomputed ``row.wer`` is available with no ``reference_text`` to
    recover the true word count from — that degrades gracefully to
    per-row-equal-weight averaging for that row, rather than raising.
    """
    if row.reference_text is not None and row.prediction is not None:
        errors, ref_words = _wer_components(row.reference_text, row.prediction)
        return float(errors), float(ref_words)
    if row.wer is not None:
        return row.wer, 1.0
    return None


def score_stt(rows: list[PredictionRow]) -> dict[str, float]:
    wers = [w for w in (row_wer(r) for r in rows) if w is not None]
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    metrics: dict[str, float] = {}
    if wers:
        metrics["wer_mean"] = round(sum(wers) / len(wers), 4)
        metrics["wer_median"] = round(median(wers), 4)
    if latencies:
        metrics["latency_ms_median"] = round(median(latencies), 2)
    return metrics


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------

# Tokens that mean "the wake word is present / was detected" on either the
# reference ``label`` or the predicted ``prediction`` side of a row. The same
# normaliser scores the wake-word and VAD leagues — both are binary detection,
# so the VAD speech/voice tokens live here too (positive = speech present /
# detected).
_WW_POSITIVE = {"positive", "wake", "wakeword", "detected", "hit",
                "true", "yes", "1", "speech", "voice"}


def _ww_is_positive(value: object) -> bool | None:
    """Normalise a wake-word label/decision to a boolean, or None if unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if not token:
        return None
    if token in _WW_POSITIVE:
        return True
    return False


def ww_reference(row: PredictionRow) -> bool | None:
    """Ground truth for a wake-word row: was the wake word actually present?"""
    return _ww_is_positive(row.label)


def ww_detected(row: PredictionRow) -> bool | None:
    """Detector decision for a wake-word row: did it fire?"""
    return _ww_is_positive(row.prediction)


def ww_row_correct(row: PredictionRow) -> bool | None:
    """Whether one wake-word decision matches the ground-truth label."""
    ref = ww_reference(row)
    pred = ww_detected(row)
    if ref is None or pred is None:
        return None
    return ref == pred


def score_wake_word(rows: list[PredictionRow]) -> dict[str, float]:
    """Detection metrics for a wake-word competitor.

    ``error_rate`` (primary, lower is better) is the share of all samples
    decided wrong; ``false_accept_rate`` is firing on negatives (noise that
    triggers the assistant), ``false_reject_rate`` is missing positives (the
    user says the wake word and nothing happens).
    """
    positives = negatives = 0
    false_accepts = false_rejects = 0
    scored = 0
    latencies: list[float] = []
    for row in rows:
        if row.latency_ms is not None:
            latencies.append(row.latency_ms)
        ref = ww_reference(row)
        pred = ww_detected(row)
        if ref is None or pred is None:
            continue
        scored += 1
        if ref:
            positives += 1
            if not pred:
                false_rejects += 1
        else:
            negatives += 1
            if pred:
                false_accepts += 1

    metrics: dict[str, float] = {}
    if scored:
        errors = false_accepts + false_rejects
        metrics["error_rate"] = round(errors / scored, 4)
        metrics["accuracy"] = round((scored - errors) / scored, 4)
    if negatives:
        metrics["false_accept_rate"] = round(false_accepts / negatives, 4)
    if positives:
        metrics["false_reject_rate"] = round(false_rejects / positives, 4)
    if latencies:
        metrics["latency_ms_median"] = round(median(latencies), 2)
    return metrics


# VAD is binary speech/non-speech detection — the same FP (fires on non-speech)
# / FN (misses speech) scoring as wake word, so it reuses the scorer.
score_vad = score_wake_word

_SCORERS = {
    "intent": score_intent,
    "intent_template": score_intent,
    "intent_keyword": score_intent,
    "stt": score_stt,
    "wake_word": score_wake_word,
    "vad": score_vad,
}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals on the primary metric (§4 A1.2)
# ---------------------------------------------------------------------------
#
# A point-estimate gap on a few hundred samples is often noise. Every
# BenchmarkEntry carries a seeded bootstrap 95% CI on its primary metric so
# the frontend can show "≈ tied with #1" instead of implying false
# precision. Two extraction strategies:
#
# - "mean": the metric is the mean of a per-row 0/1 indicator (accuracy,
#   error_rate) — bootstrap the indicator list directly.
# - "ratio": the metric is sum(numerator)/sum(denominator) over rows (WER —
#   errors/reference_words). Per-utterance WER is not directly comparable
#   across utterances of different length, so the bootstrap resamples
#   (errors, ref_words) *pairs* and recomputes the ratio each round, rather
#   than averaging per-row WER values as if they were i.i.d.


def _intent_correct_indicators(rows: list[PredictionRow]) -> list[float]:
    return [1.0 if row_is_correct(r) else 0.0 for r in rows]


def _ww_error_indicators(rows: list[PredictionRow]) -> list[float]:
    out = []
    for r in rows:
        correct = ww_row_correct(r)
        if correct is None:
            continue
        out.append(0.0 if correct else 1.0)
    return out


def _stt_wer_pairs(rows: list[PredictionRow]) -> list[tuple[float, float]]:
    pairs = (row_wer_components(r) for r in rows)
    return [p for p in pairs if p is not None]


# modality -> "mean" extractor (returns per-row 0/1 indicators)
_CI_MEAN_EXTRACTORS = {
    "intent": _intent_correct_indicators,
    "intent_template": _intent_correct_indicators,
    "intent_keyword": _intent_correct_indicators,
    "wake_word": _ww_error_indicators,
    "vad": _ww_error_indicators,
}

# modality -> "ratio" extractor (returns (numerator, denominator) pairs)
_CI_RATIO_EXTRACTORS = {
    "stt": _stt_wer_pairs,
}


def bootstrap_mean_ci(
    values: list[float], seed: int = BOOTSTRAP_SEED, rounds: int = BOOTSTRAP_ROUNDS
) -> tuple[float, float] | None:
    """Seeded bootstrap 95% CI for the mean of *values* (e.g. 0/1 indicators)."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(rounds):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (percentile(means, CI_LOWER_PCT), percentile(means, CI_UPPER_PCT))


def bootstrap_ratio_ci(
    pairs: list[tuple[float, float]], seed: int = BOOTSTRAP_SEED, rounds: int = BOOTSTRAP_ROUNDS
) -> tuple[float, float] | None:
    """Seeded bootstrap 95% CI for sum(numerator)/sum(denominator) over *pairs*."""
    pairs = [(n, d) for n, d in pairs if d > 0]
    if not pairs:
        return None
    if len(pairs) == 1:
        num, den = pairs[0]
        v = num / den
        return (v, v)
    rng = random.Random(seed)
    n = len(pairs)
    ratios = []
    for _ in range(rounds):
        num_sum = den_sum = 0.0
        for _ in range(n):
            num, den = pairs[rng.randrange(n)]
            num_sum += num
            den_sum += den
        ratios.append(num_sum / den_sum if den_sum else 0.0)
    ratios.sort()
    return (percentile(ratios, CI_LOWER_PCT), percentile(ratios, CI_UPPER_PCT))


def primary_metric_ci(modality: str, rows: list[PredictionRow]) -> tuple[float, float] | None:
    """Bootstrap 95% CI for *rows*' primary metric under *modality*, or None
    when the modality has no CI strategy or too few scoreable rows."""
    mean_extractor = _CI_MEAN_EXTRACTORS.get(modality)
    if mean_extractor is not None:
        return bootstrap_mean_ci(mean_extractor(rows))
    ratio_extractor = _CI_RATIO_EXTRACTORS.get(modality)
    if ratio_extractor is not None:
        return bootstrap_ratio_ci(ratio_extractor(rows))
    return None


def _ci_overlaps(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    if a is None or b is None:
        return False
    return a[0] <= b[1] and b[0] <= a[1]


def pair_metric_significant(
    modality: str, rows_a: list[PredictionRow], rows_b: list[PredictionRow]
) -> bool:
    """True when two competitors' primary-metric CIs (§4 A1.2, same dataset)
    do not overlap — i.e. there is a statistically meaningful difference to
    seed a rating with, rather than benchmark noise (§4 seed-battle bias
    audit). Modalities without a CI strategy, or with too few scoreable
    rows for either competitor to fit one, default to significant (no
    gate) — that preserves per-sample auto-battle behavior for anything not
    covered by a bootstrap strategy.
    """
    ci_a = primary_metric_ci(modality, rows_a)
    ci_b = primary_metric_ci(modality, rows_b)
    if ci_a is None or ci_b is None:
        return True
    return not _ci_overlaps(ci_a, ci_b)


def build_benchmark_board(
    modality: str,
    dataset_id: str,
    lang: str,
    by_competitor: dict[str, list[PredictionRow]],
    generated_at: str,
) -> BenchmarkBoard:
    """Build one benchmark board from per-competitor row lists."""
    scorer = _SCORERS.get(modality)
    primary = PRIMARY_METRIC.get(modality, "accuracy")
    entries: list[BenchmarkEntry] = []
    cis: dict[str, tuple[float, float] | None] = {}
    if scorer is not None:
        for competitor_id, rows in by_competitor.items():
            ci = primary_metric_ci(modality, rows)
            cis[competitor_id] = ci
            entries.append(
                BenchmarkEntry(
                    competitor_id=competitor_id,
                    plugin_id=rows[0].plugin_id if rows else "",
                    samples=len(rows),
                    metrics=scorer(rows),
                    primary_metric_ci_lower=ci[0] if ci else None,
                    primary_metric_ci_upper=ci[1] if ci else None,
                )
            )

    reverse = primary in _HIGHER_BETTER
    worst = 0.0 if reverse else float("inf")
    entries.sort(key=lambda e: e.metrics.get(primary, worst), reverse=reverse)
    for i, entry in enumerate(entries, 1):
        entry.rank = i

    if entries:
        leader_ci = cis.get(entries[0].competitor_id)
        for entry in entries:
            entry.tied_with_leader = _ci_overlaps(
                leader_ci, cis.get(entry.competitor_id)
            )

    return BenchmarkBoard(
        modality=modality,
        dataset_id=dataset_id,
        lang=lang,
        generated_at=generated_at,
        primary_metric=primary,
        entries=entries,
    )
