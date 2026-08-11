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

import logging
import random
import re
import unicodedata
from collections import defaultdict
from statistics import median
from typing import Any

from arena.models import BenchmarkBoard, BenchmarkEntry, PredictionRow
from arena.rating import percentile

log = logging.getLogger(__name__)

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
    "tts": "utmos",
    # §A3.2 / R17 — FRR at a fixed 2-false-accepts/hour operating point, not
    # raw threshold-0.5 FRR alone: a streaming detector's usable operating
    # point is a trade-off curve, and ranking on FRR alone would reward a
    # fighter that only "wins" by firing so eagerly its FA/hour is unusable.
    "ww_stream": "error_at_2fa_per_hour",
}

# Higher is better for these primary metrics; lower for the rest (error rates,
# WER, latency).  ``accuracy`` is the only intent/wake-word board ranked
# descending; ``utmos`` (TTS naturalness MOS, 1-5) is likewise ascending-bad —
# every other primary metric ranks ascending.
_HIGHER_BETTER = {"accuracy", "utmos"}


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def domain_of(value: str | None) -> str | None:
    """Domain part of a ``skill_id:intent_name``-shaped prediction string.

    Everything before the first ``:``; a value with no ``:`` is its own
    domain (bare domain-only references, e.g. ``"weather"``, pass through
    unchanged). ``None`` stays ``None``.
    """
    if value is None:
        return None
    return value.split(":", 1)[0]


def row_is_correct(row: PredictionRow) -> bool:
    """Whether one intent prediction row is correct (incl. OOD rejection).

    ``row.exact_match`` is computed once at bench time (``runner.intent_bench
    .make_row``), honouring the dataset's ``reference_granularity`` — this is
    only a fallback for rows persisted before that field existed.
    """
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


# ---------------------------------------------------------------------------
# Canonical transcript normalization for WER scoring (§E)
# ---------------------------------------------------------------------------
#
# WER is only comparable across runners/competitors when both sides of the
# comparison go through the *same* normalization before tokenizing. This is
# a versioned, documented, dependency-free convention:
#
#   1. Unicode NFKC normalization (canonicalizes compatibility characters,
#      e.g. full-width digits/letters, ligatures).
#   2. Casefold (Unicode-aware lowercasing, stronger than ``str.lower()``).
#   3. Strip punctuation — anything that is not a Unicode letter, digit,
#      or whitespace is dropped (so "hello." and "hello" compare equal,
#      and "don't" becomes "dont" — apostrophes are punctuation here).
#   4. Numeral policy: every run of ASCII digits is spelled out digit-by-digit
#      as English number words joined by spaces (e.g. "123" -> "one two
#      three", "7" -> "seven"). This keeps "7" and "seven" comparable without
#      guessing at magnitude words (hundred/thousand) or locale-specific
#      grouping, and needs no numeral-to-word dependency.
#   5. Collapse all whitespace runs to a single space and strip the ends.
#
# ``WER_NORMALIZER_VERSION`` is bumped whenever this convention changes, and
# recorded on generated STT benchmark boards so historical scores stay
# distinguishable from scores produced under a different convention.

WER_NORMALIZER_VERSION = 1

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

_DIGIT_RUN_RE = re.compile(r"\d+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^\w\s]", re.UNICODE)
_UNDERSCORE_RE = re.compile(r"_")
_WHITESPACE_RE = re.compile(r"\s+")


def _spell_digits(match: re.Match[str]) -> str:
    return " ".join(_DIGIT_WORDS[d] for d in match.group(0))


def normalize_transcript(text: str) -> str:
    """Canonical WER-scoring normalization (§E, version ``WER_NORMALIZER_VERSION``).

    Applies NFKC normalization, casefolding, punctuation stripping, the
    digit-by-digit numeral spelling policy, and whitespace collapsing — see
    the module-level convention notes above. Both the reference and the
    hypothesis MUST be passed through this before tokenizing so WER is
    computed on a consistent surface form.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _DIGIT_RUN_RE.sub(_spell_digits, text)
    text = _UNDERSCORE_RE.sub(" ", text)
    text = _NON_ALNUM_SPACE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _wer_components(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word edit distance, reference word count) via word-level Levenshtein.

    Both sides are passed through ``normalize_transcript`` (§E) before
    tokenizing, so WER is comparable across runners regardless of raw
    punctuation/casing/numeral-formatting differences in the source text.
    """
    ref_tokens = normalize_transcript(reference).split()
    hyp_tokens = normalize_transcript(hypothesis).split()
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


def _cer_components(reference: str, hypothesis: str) -> tuple[int, int]:
    """(character edit distance, reference char count) after normalization.

    Same §E normalization as WER, then Levenshtein over characters instead
    of whitespace-split words — CER is comparable across languages where
    word segmentation is not meaningful (e.g. some low-resource or
    non-whitespace-delimited orthographies) in a way word-level WER isn't.
    """
    ref_chars = list(normalize_transcript(reference).replace(" ", ""))
    hyp_chars = list(normalize_transcript(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0, 0
    dp = list(range(len(hyp_chars) + 1))
    for i in range(1, len(ref_chars) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(hyp_chars) + 1):
            dp[j] = (
                prev[j - 1]
                if ref_chars[i - 1] == hyp_chars[j - 1]
                else 1 + min(prev[j - 1], prev[j], dp[j - 1])
            )
    return dp[-1], len(ref_chars)


def _cer(reference: str, hypothesis: str) -> float:
    """Character error rate via character-level Levenshtein distance."""
    errors, ref_chars = _cer_components(reference, hypothesis)
    return round(errors / ref_chars, 4) if ref_chars else 0.0


def intelligibility_scores(reference: str, hypothesis: str) -> tuple[float, float]:
    """(WER, CER) for one STT round-trip transcript against its prompt text
    (§4 R16) — reuses the canonical §E normalizer so intelligibility scores
    are computed the same way as every other WER in the arena."""
    return _wer(reference, hypothesis), _cer(reference, hypothesis)


def row_wer(row: PredictionRow) -> float | None:
    """WER for one row, preferring canonical recomputation (§E) over a
    stored precomputed ``row.wer`` when the raw reference/prediction text is
    available — so scores are comparable across runners that may have
    normalized differently before persisting their own WER. Falls back to
    the stored value only when the raw text is unavailable."""
    if row.reference_text is not None and row.prediction is not None:
        return _wer(row.reference_text, row.prediction)
    if row.wer is not None:
        return row.wer
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


# ---------------------------------------------------------------------------
# TTS — objective UTMOS board (§4 R14)
# ---------------------------------------------------------------------------
#
# TTS has no ground-truth reference (no single "correct" waveform for a
# prompt), so human preference votes stay primary (spec §2.1, §3.2). UTMOS —
# a reference-free naturalness MOS predictor scored per-clip by
# ``runner/tts_bench.py`` and stashed in ``row.extras["utmos"]`` — gives the
# league an objective, reproducible *secondary* board and feeds the same
# benchmark-seeded-ELO machinery every other league gets (§4 R5).


def row_utmos(row: PredictionRow) -> float | None:
    """Per-row UTMOS score, or None when the row wasn't scored (or is NaN)."""
    value = row.extras.get("utmos")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN guard
        return None
    return value


def row_intelligibility_wer(row: PredictionRow) -> float | None:
    """Per-row STT round-trip WER (§4 R16), or None when unscored/non-finite."""
    value = row.extras.get("intelligibility_wer")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN guard
        return None
    return value


def score_tts(rows: list[PredictionRow]) -> dict[str, float]:
    """Aggregate TTS rows into a mean UTMOS board metric.

    Rows missing a ``utmos`` score (or carrying a non-finite one) are
    excluded from the mean rather than crashing the whole board; ``n_scored``
    reports how many rows actually contributed.

    UTMOS stays the board's primary metric (§4 R14). Alongside it, the mean
    STT round-trip intelligibility WER (§4 R16) is reported as a secondary
    metric with its own bootstrap 95% CI — a synthesis crash or nonsense
    render still gets a row (WER forced to 1.0, §4 R16), so this mean is
    never silently computed over a partial, best-case-only subset of rows.
    """
    scores = [u for u in (row_utmos(r) for r in rows) if u is not None]
    wers = [w for w in (row_intelligibility_wer(r) for r in rows) if w is not None]
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    metrics: dict[str, float] = {"n_scored": float(len(scores))}
    if scores:
        metrics["utmos"] = round(sum(scores) / len(scores), 4)
    if wers:
        metrics["intelligibility_wer"] = round(sum(wers) / len(wers), 4)
        metrics["intelligibility_n_scored"] = float(len(wers))
        ci = bootstrap_mean_ci(wers)
        if ci:
            metrics["intelligibility_wer_ci_lower"] = round(ci[0], 4)
            metrics["intelligibility_wer_ci_upper"] = round(ci[1], 4)
    if latencies:
        metrics["latency_ms_median"] = round(median(latencies), 2)
    return metrics


# ---------------------------------------------------------------------------
# Streaming wake word (§A3.2 / R17)
# ---------------------------------------------------------------------------
#
# Isolated-clip benchmarking (score_wake_word above) structurally favors
# clip-shaped detectors: a streaming detector never gets to fire the way it
# does against a live mic, and a false-accept rate needs hours of continuous
# negative audio, not seconds-long clips. This is a genuinely separate
# league (`ww_stream`), scored from continuous-audio detection *events*
# rather than per-clip binary decisions — it does NOT change score_wake_word
# or the `wake_word` board above. Only competitors whose registry entry
# declares `"stream"` in `capabilities` are ever benchmarked here
# (`runner.ww_bench.WakeWordStreamBench.filter_competitors`).
#
# Row shape (one row per long continuous clip, `prediction == "WW_STREAM"`):
#   extras.events        — [[timestamp_s, score], ...] every candidate
#                           activation the engine produced over the clip.
#                           Score defaults to 1.0 for engines that only
#                           expose a binary latch (the arena does not invent
#                           a confidence the plugin doesn't provide — P2:
#                           the plugin owns its own threshold).
#   extras.truth_onsets   — [timestamp_s, ...] ground-truth wake-word onsets
#                           in that clip (empty for negative-only clips).
#   extras.duration_s     — total clip length, for the FA/hour denominator.

EVENT_TOLERANCE_S = 1.5  # ± window for matching an event to a truth onset
TARGET_FA_PER_HOUR = 2.0  # operating point for the primary metric
_DET_THRESHOLDS = [round(0.1 * i, 2) for i in range(1, 10)]  # 0.1 .. 0.9


def _stream_clips(
    rows: list[PredictionRow],
) -> list[tuple[list[float], list[tuple[float, float]], float]]:
    """(truth_onsets, events, duration_s) per streaming row, others ignored."""
    clips = []
    for row in rows:
        if row.prediction != "WW_STREAM":
            continue
        extras = row.extras or {}
        truth = [float(t) for t in (extras.get("truth_onsets") or [])]
        events = [(float(t), float(s)) for t, s in (extras.get("events") or [])]
        duration = float(extras.get("duration_s") or 0.0)
        clips.append((truth, events, duration))
    return clips


def _stream_stats_at_threshold(
    clips: list[tuple[list[float], list[tuple[float, float]], float]],
    threshold: float,
) -> dict[str, Any]:
    """FRR / FA-per-hour / latencies at one detection threshold.

    Each truth onset matches at most one fired event within
    ``EVENT_TOLERANCE_S`` (closest in time); every fired event left
    unmatched is a false accept.
    """
    total_onsets = 0
    missed = 0
    false_accepts = 0
    total_duration = 0.0
    latencies: list[float] = []
    for truth, events, duration in clips:
        total_duration += duration
        fired = [(t, s) for t, s in events if s >= threshold]
        matched_events: set[int] = set()
        for onset in truth:
            best_idx, best_gap = None, None
            for idx, (t, _s) in enumerate(fired):
                if idx in matched_events:
                    continue
                gap = abs(t - onset)
                if gap <= EVENT_TOLERANCE_S and (best_gap is None or gap < best_gap):
                    best_idx, best_gap = idx, gap
            total_onsets += 1
            if best_idx is None:
                missed += 1
            else:
                matched_events.add(best_idx)
                latencies.append(fired[best_idx][0] - onset)
        false_accepts += len(fired) - len(matched_events)

    hours = total_duration / 3600.0
    return {
        "frr": round(missed / total_onsets, 4) if total_onsets else 0.0,
        "fa_per_hour": round(false_accepts / hours, 4) if hours else 0.0,
        "latencies": latencies,
        "total_onsets": total_onsets,
        "false_accepts": false_accepts,
        "hours": hours,
    }


def score_ww_stream(rows: list[PredictionRow]) -> dict[str, float]:
    """Continuous-stream detection metrics for a `ww_stream` competitor.

    ``error_at_2fa_per_hour`` (primary, lower is better) is the FRR at the
    lowest scanned threshold whose FA/hour stays within
    ``TARGET_FA_PER_HOUR`` — the FRR a deployer would actually get while
    respecting a 2-false-accepts-per-hour budget, rather than a raw
    threshold-0.5 number that hides the trade-off. ``det_frr@<thr>`` /
    ``det_fa_per_hour@<thr>`` flatten a small DET curve into float metrics
    (``BenchmarkEntry.metrics`` is ``dict[str, float]``, so the curve can't
    be nested).
    """
    clips = _stream_clips(rows)
    if not clips:
        return {}

    base = _stream_stats_at_threshold(clips, 0.5)
    metrics: dict[str, float] = {
        "frr": base["frr"],
        "fa_per_hour": base["fa_per_hour"],
        "negative_hours": round(base["hours"], 3),
        "n_onsets": float(base["total_onsets"]),
    }
    if base["latencies"]:
        metrics["latency_s_median"] = round(median(base["latencies"]), 3)

    det_points = []
    for thr in _DET_THRESHOLDS:
        stats = _stream_stats_at_threshold(clips, thr)
        det_points.append((thr, stats["frr"], stats["fa_per_hour"]))
        metrics[f"det_frr@{thr}"] = stats["frr"]
        metrics[f"det_fa_per_hour@{thr}"] = stats["fa_per_hour"]

    within_budget = [p for p in det_points if p[2] <= TARGET_FA_PER_HOUR]
    # Ascending threshold order: the first (lowest) threshold within budget
    # gives the best (lowest) FRR achievable at that FA/hour ceiling.
    chosen = within_budget[0] if within_budget else det_points[-1]
    metrics["error_at_2fa_per_hour"] = chosen[1]
    return metrics


_SCORERS = {
    "intent": score_intent,
    "intent_template": score_intent,
    "intent_keyword": score_intent,
    "stt": score_stt,
    "wake_word": score_wake_word,
    "vad": score_vad,
    "tts": score_tts,
    "ww_stream": score_ww_stream,
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


def _tts_utmos_values(rows: list[PredictionRow]) -> list[float]:
    return [u for u in (row_utmos(r) for r in rows) if u is not None]


# modality -> "mean" extractor (returns per-row 0/1 indicators, or raw values
# for tts's UTMOS mean)
_CI_MEAN_EXTRACTORS = {
    "intent": _intent_correct_indicators,
    "intent_template": _intent_correct_indicators,
    "intent_keyword": _intent_correct_indicators,
    "wake_word": _ww_error_indicators,
    "vad": _ww_error_indicators,
    "tts": _tts_utmos_values,
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
            # §G version-blend guard: flag (never silently aggregate) when a
            # competitor's rows span more than one distinct plugin_version.
            versions = sorted({r.plugin_version for r in rows if r.plugin_version})
            version_blended = len(versions) > 1
            if version_blended:
                log.warning(
                    "%s/%s/%s: competitor %r blends metrics across plugin "
                    "versions %s — scores are not attributable to a single "
                    "version",
                    modality, dataset_id, lang, competitor_id, versions,
                )
            entries.append(
                BenchmarkEntry(
                    competitor_id=competitor_id,
                    plugin_id=rows[0].plugin_id if rows else "",
                    samples=len(rows),
                    metrics=scorer(rows),
                    primary_metric_ci_lower=ci[0] if ci else None,
                    primary_metric_ci_upper=ci[1] if ci else None,
                    plugin_versions=versions,
                    version_blended=version_blended,
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
        wer_normalizer_version=WER_NORMALIZER_VERSION if modality == "stt" else None,
        entries=entries,
    )
