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
- ``generalization_accuracy`` covers only the samples an engine cannot have
  seen during training (see ``CONTAMINATED_BUCKETS``) and is what the intent
  boards rank on.
- ``slot_f1`` is exact-match over the gold slot dict, evaluated only on
  rows where the intent was correct and gold slots exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import defaultdict
from statistics import median
from typing import Any

from arena.models import BenchmarkBoard, BenchmarkEntry, JudgeAgreement, PredictionRow
from arena.rating import percentile

log = logging.getLogger(__name__)

BOOTSTRAP_ROUNDS = 1000
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5
BOOTSTRAP_SEED = 0

# Bumped whenever scoring/CI logic that affects a benchmark board's OUTPUT
# changes (a new/changed scorer, BOOTSTRAP_ROUNDS, the CI percentiles, …).
# Folded into ``benchmark_board_input_signature`` so a stale on-disk
# ``input_hash`` never survives a scoring-logic change — the whole point of
# the signature is "same input AND same logic produces the same output",
# not just "same input".
BOARD_LOGIC_VERSION = 2

#: Test buckets whose utterances also reach the engines as training data.
#: A template-paradigm fighter is trained on every phrase template AND on
#: ``runner.intent_pipeline.expand_template``'s slot-filled expansions of it,
#: and ``intents-for-eval``'s ``template`` and ``near_ood`` test rows are
#: largely those same expansions verbatim — measured at 95% and 83% of the two
#: buckets on en-US, 46% of the whole test set. Accuracy inside them is a
#: memorization lookup, not evidence an engine understands anything, and it
#: systematically favours exact-string matchers. They stay published as their
#: own ``acc_*`` columns; they are excluded from the ranked metric.
CONTAMINATED_BUCKETS = frozenset({"template", "near_ood"})

#: Published column suffix for a raw dataset bucket name, where the dataset's
#: own name misdescribes the rows. ``near_ood`` is not out-of-distribution: no
#: row in it has a null expected intent, and the overwhelming majority repeat a
#: training utterance under the same gold label.
BUCKET_METRIC_NAMES = {"near_ood": "in_distribution"}

PRIMARY_METRIC = {
    "intent": "generalization_accuracy",
    "intent_template": "generalization_accuracy",
    "intent_keyword": "generalization_accuracy",
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
_HIGHER_BETTER = {
    "accuracy", "generalization_accuracy", "utmos", "slot_exact_match",
}

#: Polarity of the flattened SIGMOS/DNSMOS/NISQA quality-dimension columns
#: (§4 R14 extension) — NOT primary-metric ranking keys (UTMOS stays
#: primary), but secondary board metrics whose "higher/lower is better"
#: direction still has to be known by callers (e.g. the frontend). SIGMOS
#: (P.804, MIT-licensed Microsoft weights) provides the headline dimensions
#: surfaced as board columns; NISQA-v2 is recorded as a complementary
#: predictor alongside it — this arena is a non-commercial OVOS project, so
#: NISQA's CC BY-NC-SA 4.0 weights are usable here (see
#: docs/methodology.md); NISQA is not surfaced as its own board column, only
#: aggregated into the row/board JSON. VERIFIED against speechonnxmetrics'
#: own docs/metric-guides (sigmos.md, dnsmos.md, nisqa.md) and source
#: (``SIGMOS.higher_is_better`` / ``DNSMOS.higher_is_better`` /
#: ``NISQA.higher_is_better`` are all ``True``): every dimension of all
#: three metrics is a 1-5 MOS-style scale where HIGHER is always better —
#: "noise"/"noi" and "disc"/"dis" (discontinuity) are scored as perceived-
#: quality-along-that-axis, not as raw noise/glitch amount, so a quiet,
#: continuous clip scores HIGH on both. Do not assume the metric name
#: implies "lower is better" without checking this.
TTS_QUALITY_DIMENSION_KEYS = (
    "sigmos.col", "sigmos.disc", "sigmos.loud", "sigmos.noise",
    "sigmos.reverb", "sigmos.sig", "sigmos.ovrl",
    "dnsmos.sig", "dnsmos.bak", "dnsmos.ovrl",
    "nisqa.mos", "nisqa.noi", "nisqa.dis", "nisqa.col", "nisqa.loud",
)
_HIGHER_BETTER.update(TTS_QUALITY_DIMENSION_KEYS)


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


ECE_N_BINS = 10


def expected_calibration_error(rows: list[PredictionRow]) -> float | None:
    """Expected Calibration Error (ECE) over rows carrying a confidence.

    Standard equal-width-binning ECE (10 bins over ``[0, 1]``):
    ``sum_bin (n_bin / N) * |accuracy_bin - mean_confidence_bin|``. Rows
    with ``confidence is None`` (a fighter that never emits one) are
    excluded rather than treated as 0 — if *no* row in the competitor's
    prediction set carries a confidence, ECE is undefined for that fighter
    and this returns ``None`` (never 0, which would misleadingly read as
    "perfectly calibrated").
    """
    scored = [r for r in rows if r.confidence is not None]
    if not scored:
        return None
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for row in scored:
        conf = max(0.0, min(1.0, float(row.confidence)))
        # confidence == 1.0 falls in the last bin (10 equal-width bins over
        # [0, 1] means the top edge belongs to bin index 9, not a stray 10th).
        idx = min(int(conf * ECE_N_BINS), ECE_N_BINS - 1)
        bins[idx].append((conf, row_is_correct(row)))

    n = len(scored)
    ece = 0.0
    for bucket in bins.values():
        n_bin = len(bucket)
        mean_conf = sum(c for c, _ in bucket) / n_bin
        acc_bin = sum(1.0 for _, ok in bucket if ok) / n_bin
        ece += (n_bin / n) * abs(acc_bin - mean_conf)
    return round(ece, 4)


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
        "n_scored": float(n),
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
    }
    if ood_n:
        metrics["ood_fpr"] = round(ood_fp / ood_n, 4)
    if slot_total:
        metrics["slot_exact_match"] = round(slot_correct / slot_total, 4)
    if latencies:
        metrics["latency_ms_median"] = round(median(latencies), 2)
    ece = expected_calibration_error(rows)
    if ece is not None:
        metrics["ece"] = ece
    clean_total = sum(
        v["total"] for b, v in per_bucket.items() if b not in CONTAMINATED_BUCKETS
    )
    if clean_total:
        clean_correct = sum(
            v["correct"] for b, v in per_bucket.items()
            if b not in CONTAMINATED_BUCKETS
        )
        metrics["generalization_accuracy"] = round(clean_correct / clean_total, 4)
    for bucket, v in sorted(per_bucket.items()):
        if v["total"]:
            name = BUCKET_METRIC_NAMES.get(bucket, bucket)
            metrics[f"acc_{name}"] = round(v["correct"] / v["total"], 4)
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


# §M5 — false-accepts-per-hour (FA/h) on the isolated-clip wake_word board.
#
# false_accept_rate (share of negative *clips* that fire) is the wrong signal
# for real-world annoyance: a fighter scored on a pool of many short clips can
# hold a low FAR while still firing constantly against a live mic, because FAR
# says nothing about how much negative *time* backs it. FA/h — false
# activations per hour of negative audio — is the number that actually
# predicts how often a deployed assistant wakes up on nothing.
#
# Denominator honesty: audio_secs was only added to prediction rows by the
# perf-capture campaign (PR #90) — rows benched before that carry
# ``audio_secs is None`` and are silently excluded from BOTH the FA/h
# numerator and denominator, rather than treated as zero-duration. The
# ``fa_per_hour_hours`` field reports exactly how much negative audio backed
# the number, so a board never implies more coverage than it has.
#
# Coverage floor: below MIN_FA_HOURS of covered negative audio the ratio is
# noise (a single 3-second clip firing once would already read as an
# absurd number of FA/h) — the metric is omitted (None) rather than printed.
MIN_FA_HOURS = 0.25  # 15 minutes of covered negative audio


def score_wake_word(rows: list[PredictionRow]) -> dict[str, float]:
    """Detection metrics for a wake-word competitor.

    ``error_rate`` (primary, lower is better) is the share of all samples
    decided wrong; ``false_accept_rate`` is firing on negatives (noise that
    triggers the assistant), ``false_reject_rate`` is missing positives (the
    user says the wake word and nothing happens). ``fa_per_hour`` (§M5) is
    false activations per hour of *covered* negative audio — see the
    module note above for the ``audio_secs``-only denominator and the
    ``MIN_FA_HOURS`` coverage floor.
    """
    positives = negatives = 0
    false_accepts = false_rejects = 0
    scored = 0
    latencies: list[float] = []
    fa_negative_secs = 0.0
    fa_false_accepts = 0
    fa_covered_negatives = 0
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
            if row.audio_secs is not None:
                fa_covered_negatives += 1
                fa_negative_secs += row.audio_secs
                if pred:
                    fa_false_accepts += 1

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
    if fa_covered_negatives:
        fa_hours = fa_negative_secs / 3600.0
        metrics["fa_per_hour_hours"] = round(fa_hours, 4)
        if fa_hours >= MIN_FA_HOURS:
            metrics["fa_per_hour"] = round(fa_false_accepts / fa_hours, 4)
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


def row_intelligibility_cer(row: PredictionRow) -> float | None:
    """Per-row ROVER-consensus STT round-trip CER (§4 R16), or None when
    unscored/non-finite. Same shape as :func:`row_intelligibility_wer` —
    CJK/no-space languages have no meaningful WER, so the ELO seed formula
    below always uses CER, never WER."""
    value = row.extras.get("intelligibility_cer")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN guard
        return None
    return value


def row_intelligibility_agreement(row: PredictionRow) -> float | None:
    """Per-row inter-judge ROVER agreement (§4 R16 extension) — the mean
    per-slot vote share of the ROVER consensus (``arena.rover``): how much
    the judge panel agreed on what the clip said. ``1.0`` for legacy rows
    scored before panels existed (a single judge trivially "agrees with
    itself") — see :func:`tts_seed_score`, which relies on that default."""
    value = row.extras.get("intelligibility_agreement")
    if value is None:
        return 1.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if value != value:  # NaN guard
        return 1.0
    return value


def tts_seed_score(row: PredictionRow) -> float | None:
    """Composite TTS ELO-seed score: naturalness x intelligibility x
    inter-judge agreement.

    ``score = (utmos / 5.0) * (1.0 - min(cer, 1.0)) * agreement``

    A voice that is judged natural (high UTMOS) but unintelligible (high
    ROVER-consensus CER, §4 R16) is not a good voice, and vice versa — the
    two dimensions are multiplied, not averaged or ladder-compared, because
    either one being bad should tank the composite: a beautifully-voiced
    clip nobody can understand and a perfectly-transcribable robotic
    monotone are both bad TTS. ``agreement`` (``arena.rover``'s mean
    per-slot ROVER vote share, §4 R16 extension) discounts the
    intelligibility term further when the judge panel itself could not
    agree on what was said — panel disagreement is evidence the speech was
    unclear, independent of what any single judge's CER says, so a
    genuinely garbled clip is penalised twice by design: once through a
    high CER and again through low agreement. Rows scored before judge
    panels existed (or scored by a single judge) default ``agreement`` to
    ``1.0`` (nothing to disagree with), so this reduces to the two-factor
    formula for them.

    UTMOS is capped into ``[0, 5]`` before dividing (a pathological model
    output outside that range must not invert the product), and CER is
    capped into ``[0, 1]`` before subtracting so a CER above 100% cannot
    make the intelligibility factor negative. Returns ``None`` when either
    UTMOS or CER is missing — no partial score, so callers can tell "not
    computable" apart from "computed to near-zero".
    """
    utmos = row_utmos(row)
    cer = row_intelligibility_cer(row)
    if utmos is None or cer is None:
        return None
    utmos = max(0.0, min(utmos, 5.0))
    cer = max(0.0, min(cer, 1.0))
    agreement = row_intelligibility_agreement(row)
    return (utmos / 5.0) * (1.0 - cer) * agreement


def row_slot_match(row: PredictionRow) -> float | None:
    """Per-row exact slot match (1.0/0.0), or None when not applicable.

    Only defined when the row's intent prediction was correct AND the
    sample carries gold slots — same eligibility ``score_intent`` uses for
    the aggregate ``slot_exact_match`` board metric (§ intent scoring
    conventions above). Rows outside that eligibility (wrong intent, no
    gold slots) have no slot signal, so this returns ``None`` rather than
    0.0 — a 0.0 there would misleadingly count as "wrong slots" instead of
    "not applicable".
    """
    if not row_is_correct(row):
        return None
    gold_slots = row.reference_slots or {}
    if not gold_slots:
        return None
    pred_slots = row.predicted_slots or {}
    match = all(
        str(pred_slots.get(k, "")).strip().lower() == str(v).strip().lower()
        for k, v in gold_slots.items()
    )
    return 1.0 if match else 0.0


def row_quality_dimension(row: PredictionRow, key: str) -> float | None:
    """Per-row SIGMOS/DNSMOS/NISQA dimension score (e.g. ``"sigmos.noise"``), or None
    when the row wasn't scored for that dimension (or carries a non-finite
    value) — old rows benched before this metric existed simply lack the
    key, so this must tolerate absence rather than crash (§4 R14 ext.)."""
    value = row.extras.get(key)
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
    # SIGMOS/DNSMOS/NISQA quality dimensions (§4 R14 extension) — mean-aggregated
    # exactly like utmos above, one dimension at a time, missing values
    # excluded rather than fatal (old rows benched before this metric
    # existed simply have no key for it).
    for key in TTS_QUALITY_DIMENSION_KEYS:
        values = [v for v in (row_quality_dimension(r, key) for r in rows) if v is not None]
        if values:
            metrics[key] = round(sum(values) / len(values), 4)
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


def _intent_generalization_indicators(rows: list[PredictionRow]) -> list[float]:
    return [
        1.0 if row_is_correct(r) else 0.0
        for r in rows
        if (r.bucket or "test") not in CONTAMINATED_BUCKETS
    ]


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
    "intent": _intent_generalization_indicators,
    "intent_template": _intent_generalization_indicators,
    "intent_keyword": _intent_generalization_indicators,
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


# R11 benchmark boards carry confidence intervals
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


def significant_from_cis(
    ci_a: tuple[float, float] | None, ci_b: tuple[float, float] | None
) -> bool:
    """True when two already-computed primary-metric CIs (§4 A1.2) do not
    overlap. Split out of ``pair_metric_significant`` so a caller iterating
    every competitor pair (O(fighters²)) can bootstrap each competitor's CI
    once (O(fighters)) and reuse it here, instead of recomputing the same
    competitor's CI on every pair it appears in.
    """
    if ci_a is None or ci_b is None:
        return True
    return not _ci_overlaps(ci_a, ci_b)


def pair_metric_significant(
    modality: str, rows_a: list[PredictionRow], rows_b: list[PredictionRow]
) -> bool:
    """R5a significance gate. True when two competitors' primary-metric CIs (§4 A1.2, same dataset)
    do not overlap — i.e. there is a statistically meaningful difference to
    seed a rating with, rather than benchmark noise (§4 seed-battle bias
    audit). Modalities without a CI strategy, or with too few scoreable
    rows for either competitor to fit one, default to significant (no
    gate) — that preserves per-sample auto-battle behavior for anything not
    covered by a bootstrap strategy.
    """
    ci_a = primary_metric_ci(modality, rows_a)
    ci_b = primary_metric_ci(modality, rows_b)
    return significant_from_cis(ci_a, ci_b)


def benchmark_board_input_signature(
    by_competitor: dict[str, list[PredictionRow]],
    sample_set_ids: set[str] | None = None,
) -> str:
    """Stable hash of everything that determines a benchmark board's
    scored output: every competitor's rows, ``BOARD_LOGIC_VERSION``, and
    (when applicable) the published sample-set manifest's id set.

    Two calls with the same rows (regardless of dict/list ORDER — both are
    sorted before hashing) and the same scoring logic hash identically.
    Lets a caller skip the O(rounds * samples) bootstrap CI entirely when
    the on-disk board already carries this exact signature — the recompute
    would just reproduce the same numbers (see cli.py's board-write loop,
    which already no-ops the file WRITE on unchanged content via
    ``_unchanged``; this skips the expensive COMPUTE that precedes it).

    ``sample_set_ids`` MUST be included here: publishing a new/changed
    manifest for a policy-capped dataset doesn't touch any prediction row,
    so without this the unchanged-input skip above would never notice a
    republished manifest and the board would keep stale coverage/ranking
    forever instead of "republish and the board re-filters automatically".
    """
    # Sort each competitor's own rows by sample_id/plugin_id, so row ORDER
    # never affects the hash — only row CONTENT does — then sort competitors.
    rows_repr = sorted(
        (
            competitor_id,
            sorted(
                (row.model_dump(mode="json") for row in rows),
                key=lambda d: (d.get("sample_id", ""), d.get("plugin_id", "")),
            ),
        )
        for competitor_id, rows in by_competitor.items()
    )
    payload = json.dumps(
        {
            "logic_version": BOARD_LOGIC_VERSION,
            "rows": rows_repr,
            "sample_set_ids": sorted(sample_set_ids) if sample_set_ids is not None else None,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# TTS — judge agreement (UTMOS vs SIGMOS/DNSMOS/NISQA/Intelligibility)
# ---------------------------------------------------------------------------
#
# UTMOS stays the board's primary metric, but a reader has no way to tell a
# robust ranking from one judge picking favorites when several objective
# judges score the same fighters. This computes the pairwise Spearman rank
# correlation between every pair of judges (over the fighters both scored)
# plus each judge's own top-5, straight from the per-fighter mean scores
# already aggregated into ``BenchmarkEntry.metrics`` by ``score_tts`` above
# — no separate row-level pass.

#: judge display name -> the ``BenchmarkEntry.metrics`` key holding that
#: judge's per-fighter mean score. Matches the metric keys ``score_tts``
#: already writes (arena.metrics.TTS_QUALITY_DIMENSION_KEYS uses the
#: overall/mos dimension per judge; per-dimension SIGMOS/NISQA breakdowns
#: are not separate "judges" for this panel).
_TTS_JUDGE_METRIC_KEYS = {
    "UTMOS": "utmos",
    "SIGMOS": "sigmos.ovrl",
    "DNSMOS": "dnsmos.ovrl",
    "NISQA": "nisqa.mos",
    "Intelligibility": "intelligibility_wer",
}
#: Judges whose metric key is lower-is-better (WER) — negated before
#: ranking/correlating so every judge's "score" here is oriented
#: higher-is-better, matching the sense of the other judges.
_TTS_JUDGE_LOWER_BETTER = {"Intelligibility"}

TOP_N_JUDGE_AGREEMENT = 5


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation of *x* and *y*, implemented directly with
    numpy (no scipy dependency): rank each array with average ranks for
    ties, then Pearson-correlate the two rank vectors.

    Returns ``None`` when fewer than 2 paired values are given, or when
    either ranked vector has zero variance (all-tied — correlation is
    undefined, not 0.0).

    numpy is imported lazily here (not at module level) — ``arena.metrics``
    is on the base install's import path (``ovos-arena --help``), while
    numpy is only pulled in by the ``audio``/``test`` extras, never core
    ``dependencies``.
    """
    import numpy as np

    if len(x) != len(y) or len(x) < 2:
        return None
    xr = _average_ranks(np.asarray(x, dtype=float))
    yr = _average_ranks(np.asarray(y, dtype=float))
    if xr.std() == 0.0 or yr.std() == 0.0:
        return None
    rho = float(np.corrcoef(xr, yr)[0, 1])
    if rho != rho:  # NaN guard
        return None
    return rho


def _average_ranks(values):
    """1-based ranks of *values* (a numpy array), ascending, ties given their
    average rank. Local ``import numpy`` for the same reason as
    ``spearman_rho`` above (kept out of arena.metrics' module-level
    imports)."""
    import numpy as np

    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    for v in np.unique(values):
        mask = values == v
        ranks[mask] = ranks[mask].mean()
    return ranks


def _tts_judge_agreement(entries: list[BenchmarkEntry]) -> JudgeAgreement:
    """Per-judge mean scores -> pairwise Spearman matrix + top-N, robust to
    missing judges/fighters (never raises on 0/1-fighter or 0-judge boards)."""
    scores: dict[str, dict[str, float]] = {}
    for judge, key in _TTS_JUDGE_METRIC_KEYS.items():
        per_fighter = {
            e.competitor_id: e.metrics[key] for e in entries if key in e.metrics
        }
        if not per_fighter:
            continue
        if judge in _TTS_JUDGE_LOWER_BETTER:
            per_fighter = {c: -v for c, v in per_fighter.items()}
        scores[judge] = per_fighter

    fighters: set[str] = set()
    for per_fighter in scores.values():
        fighters.update(per_fighter)

    top5 = {
        judge: [
            c for c, _ in sorted(
                per_fighter.items(), key=lambda kv: kv[1], reverse=True,
            )[:TOP_N_JUDGE_AGREEMENT]
        ]
        for judge, per_fighter in scores.items()
    }

    matrix: dict[str, dict[str, float]] = {}
    judges = sorted(scores)
    for j1 in judges:
        for j2 in judges:
            if j1 == j2:
                matrix.setdefault(j1, {})[j2] = 1.0
                continue
            common = sorted(set(scores[j1]) & set(scores[j2]))
            if len(common) < 2:
                continue
            rho = spearman_rho(
                [scores[j1][c] for c in common], [scores[j2][c] for c in common],
            )
            if rho is not None:
                matrix.setdefault(j1, {})[j2] = round(rho, 4)

    return JudgeAgreement(n_fighters=len(fighters), matrix=matrix, top5=top5)


def build_benchmark_board(
    modality: str,
    dataset_id: str,
    lang: str,
    by_competitor: dict[str, list[PredictionRow]],
    generated_at: str,
    input_hash: str | None = None,
    sample_set_ids: set[str] | None = None,
) -> BenchmarkBoard:
    """Build one benchmark board from per-competitor row lists.

    *input_hash*, when given, is stamped onto the returned board as-is
    (normally ``benchmark_board_input_signature(by_competitor)`` — the
    caller computes it once and reuses it both here and for the
    unchanged-input skip check, rather than hashing twice).

    *sample_set_ids*, when given, is a published ``sample_sets/<lang>.json``
    manifest's id set (see ``runner.publish_sample_set``) for a
    ``sample_policy``-capped dataset. Every competitor's rows are filtered
    to it BEFORE scoring, so fighters that were swept on different sample
    populations (a full-corpus sweep predating the policy, a smaller ad hoc
    ``--max-samples``, the current policy subset) are compared on the exact
    same rows rather than each on however many rows it happens to have.
    ``None`` means no manifest applied — either the dataset carries no
    ``sample_policy``, or one exists but its manifest hasn't been published
    yet; entries are scored unfiltered and marked ``sample_set="unmanaged"``.
    """
    scorer = _SCORERS.get(modality)
    primary = PRIMARY_METRIC.get(modality, "accuracy")
    entries: list[BenchmarkEntry] = []
    cis: dict[str, tuple[float, float] | None] = {}
    if scorer is not None:
        for competitor_id, rows in by_competitor.items():
            plugin_id = rows[0].plugin_id if rows else ""
            sample_set = "unmanaged"
            coverage = None
            if sample_set_ids is not None:
                sample_set = "manifest"
                kept = [r for r in rows if r.sample_id in sample_set_ids]
                coverage = (len(kept) / len(sample_set_ids)) if sample_set_ids else 0.0
                rows = kept
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
                    plugin_id=plugin_id,
                    samples=len(rows),
                    metrics=scorer(rows),
                    primary_metric_ci_lower=ci[0] if ci else None,
                    primary_metric_ci_upper=ci[1] if ci else None,
                    plugin_versions=versions,
                    version_blended=version_blended,
                    perf=perf_metrics_by_tier(rows) or None,
                    sample_set=sample_set,
                    sample_set_coverage=coverage,
                )
            )

    # A run with zero scored samples (or that never computed the primary
    # metric at all — e.g. a synthesis crash on every row) has no signal to
    # rank on. Sorting it in with a placeholder "worst" value would still let
    # it land at #1 whenever it's the board's only entry or every entry
    # happens to share that placeholder, so these are pulled out and marked
    # unranked instead of taking a rank.
    # A run that scored rows but produced no primary metric is a different
    # animal: nothing crashed, the rows simply all fall outside the ranked
    # population (an intent fighter whose every row sits in a
    # ``CONTAMINATED_BUCKETS`` bucket has no generalization_accuracy). It is
    # equally unrankable, but calling it a failed run misdescribes it.
    def _scored_any(entry: BenchmarkEntry) -> bool:
        return bool(entry.metrics) and entry.metrics.get("n_scored", 1.0) != 0.0

    def _has_signal(entry: BenchmarkEntry) -> bool:
        return primary in entry.metrics and _scored_any(entry)

    # A fighter whose rows barely intersect the published manifest (e.g. a
    # sweep that predates the sample_policy, or an ad hoc small
    # --max-samples) is not comparable to a fighter that covers it fully —
    # ranking them together would silently mix sample populations again,
    # exactly the gap sample_set_ids exists to close. Below-threshold
    # coverage is unranked the same way a failed run is, with its own
    # reason so a frontend/operator can tell the two apart.
    _MIN_COVERAGE = 0.9

    def _partial_coverage(entry: BenchmarkEntry) -> bool:
        return (entry.sample_set_coverage is not None
                and entry.sample_set_coverage < _MIN_COVERAGE)

    ranked = [e for e in entries if _has_signal(e) and not _partial_coverage(e)]
    partial = [e for e in entries if _has_signal(e) and _partial_coverage(e)]
    off_metric = [e for e in entries if not _has_signal(e) and _scored_any(e)]
    failed = [e for e in entries if not _scored_any(e)]

    reverse = primary in _HIGHER_BETTER
    ranked.sort(key=lambda e: e.metrics.get(primary), reverse=reverse)
    for i, entry in enumerate(ranked, 1):
        entry.rank = i
    for entry in partial:
        entry.rank = 0
        entry.unranked = True
        entry.unranked_reason = (
            f"sample_set_partial — covers {entry.sample_set_coverage:.0%} "
            "of the published manifest"
        )
    for entry in off_metric:
        entry.rank = 0
        entry.unranked = True
        entry.unranked_reason = (
            f"no {primary} — the run scored samples, but none of them fall "
            "inside the ranked metric's population"
        )
    for entry in failed:
        entry.rank = 0
        entry.unranked = True
        entry.unranked_reason = "run failed — no scored samples"
    entries = ranked + partial + off_metric + failed

    if ranked:
        leader_ci = cis.get(ranked[0].competitor_id)
        for entry in ranked:
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
        input_hash=input_hash,
        judge_agreement=_tts_judge_agreement(entries) if modality == "tts" else None,
    )


# ---------------------------------------------------------------------------
# Per-metric ladders — browsable BT ratings for every row-level metric
# ---------------------------------------------------------------------------
#
# The primary-metric ladder (elo-seed / leaderboard) is the only one that
# ever carries human votes. Every OTHER metric a league scores per-ROW (not
# just aggregate-only, dataset-level) is "ladderable": a pairwise auto-only
# BT rating can be fit from same-sample comparisons on that metric, exactly
# like the primary-metric auto-battle seed, just never touched by tally.
#
# Deliberately excluded: dataset-aggregate-only metrics with no single
# per-row value to compare (ECE, macro_f1, latency percentiles) — there is
# no sample-level "A beat B on this metric" signal to derive a battle from.

#: modality -> {metric_key: row-level accessor(row) -> float | None}.
#: Only metrics with a genuine per-row value belong here.
ROW_METRIC_ACCESSORS: dict[str, dict[str, Any]] = {
    "stt": {
        "wer_mean": row_wer,
    },
    "tts": {
        "utmos": row_utmos,
        **{
            key: (lambda row, _key=key: row_quality_dimension(row, _key))
            for key in TTS_QUALITY_DIMENSION_KEYS
        },
    },
}
for _intent_modality in ("intent", "intent_template", "intent_keyword"):
    ROW_METRIC_ACCESSORS[_intent_modality] = {
        "accuracy": lambda row: 1.0 if row_is_correct(row) else 0.0,
        "slot_exact_match": row_slot_match,
    }


def ladder_metrics_for(modality: str) -> list[str]:
    """Every ladderable metric key for *modality*, primary metric first."""
    accessors = ROW_METRIC_ACCESSORS.get(modality)
    if not accessors:
        return []
    primary = PRIMARY_METRIC.get(modality)
    keys = list(accessors)
    if primary in keys:
        keys.remove(primary)
        keys.insert(0, primary)
    return keys


def secondary_ladder_metrics_for(modality: str) -> list[str]:
    """Ladderable metrics for *modality* excluding the primary metric —
    these are the auto-only secondary ladders (§ per-metric ladders)."""
    primary = PRIMARY_METRIC.get(modality)
    return [m for m in ladder_metrics_for(modality) if m != primary]


def row_metric_value(row: PredictionRow, modality: str, metric: str) -> float | None:
    """Per-row value of *metric* for *row*, or None when not scoreable."""
    accessor = ROW_METRIC_ACCESSORS.get(modality, {}).get(metric)
    if accessor is None:
        return None
    return accessor(row)


def metric_higher_is_better(metric: str) -> bool:
    """Whether a higher value of *metric* ranks better (§ polarity table)."""
    return metric in _HIGHER_BETTER


# ---------------------------------------------------------------------------
# Performance metrics — RTF, peak memory, model size (M2, per hardware tier)
# ---------------------------------------------------------------------------
#
# Rows carry elapsed_ms/peak_rss_mb/audio_secs/hw since #90 (M1). This section
# turns them into board-ready per-hardware-tier aggregates:
#
# - rtf (real-time factor, lower is better) = elapsed_ms / 1000 / audio_secs,
#   only defined for a row that has both elapsed_ms and a positive
#   audio_secs (stt/tts/wake_word — the modalities with an audio duration).
#   Aggregated as the MEDIAN across a tier's rows, not the mean: a cold
#   model-load or a stalled CI runner produces occasional huge outliers on
#   the very first call, and a median is robust to those in a way a mean
#   isn't — a single 30s cold start shouldn't drag a fighter's whole board
#   number away from its steady-state speed.
# - peak_rss_mb (lower is better) is aggregated as MAX across a tier's rows —
#   the number a deployer cares about is the worst-case memory the process
#   ever needed, not an average that hides the peak.
# - Model download size (model_mb) is NOT a per-row aggregate at all: it is
#   looked up ONCE per fighter from HF repo metadata (arena.model_size) and
#   attached in export-bestiary/assemble, not computed here.
#
# Hardware tiers are NEVER blended: rows are grouped by ``hw["host_class"]``
# (e.g. "cpu-x86", "gpu") and every aggregate is computed independently per
# tier. Old rows without ``hw`` or ``audio_secs`` are simply excluded from
# these aggregates (a blank/missing tier entry), never coerced to 0 — a
# missing measurement is not the same claim as "instant and free".


def row_rtf(row: PredictionRow) -> float | None:
    """Real-time factor for one row: ``elapsed_ms / 1000 / audio_secs``.

    None unless both ``elapsed_ms`` and a strictly positive ``audio_secs``
    are present — old rows (pre-#90) and non-audio modalities never have
    this data, and a zero/negative duration would make RTF meaningless.
    """
    if row.elapsed_ms is None or row.audio_secs is None or row.audio_secs <= 0:
        return None
    return row.elapsed_ms / 1000.0 / row.audio_secs


# rtf is ladderable (per-row, same-sample comparable) wherever a row carries
# a real audio duration — stt, tts, and wake_word all bench against a
# labelled audio clip with a duration; VAD/intent do not.
for _rtf_modality in ("stt", "tts", "wake_word"):
    ROW_METRIC_ACCESSORS.setdefault(_rtf_modality, {})["rtf"] = row_rtf


def hw_tier_of(row: PredictionRow) -> str | None:
    """The hardware tier (``hw["host_class"]``) a row's perf numbers belong
    to, or None when the row predates hw fingerprinting (#90)."""
    if not row.hw:
        return None
    tier = row.hw.get("host_class")
    return str(tier) if tier else None


def perf_metrics_by_tier(rows: list[PredictionRow]) -> dict[str, dict[str, float]]:
    """Per-hardware-tier perf aggregates for one competitor's rows.

    Returns ``{host_class: {"rtf": median, "rtf_n": count,
    "peak_rss_mb": max, "peak_rss_mb_n": count}}``. Tiers are never mixed —
    a fighter benched on both "cpu-x86" and "gpu" gets two independent
    entries here, never one blended number. Rows missing ``hw`` (old, pre-
    #90 data) are excluded entirely; within a tier, rtf and peak_rss_mb are
    aggregated independently since a row can carry one without the other.
    """
    rtf_by_tier: dict[str, list[float]] = defaultdict(list)
    rss_by_tier: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        tier = hw_tier_of(row)
        if tier is None:
            continue
        rtf = row_rtf(row)
        if rtf is not None:
            rtf_by_tier[tier].append(rtf)
        if row.peak_rss_mb is not None:
            rss_by_tier[tier].append(row.peak_rss_mb)

    tiers = set(rtf_by_tier) | set(rss_by_tier)
    out: dict[str, dict[str, float]] = {}
    for tier in tiers:
        entry: dict[str, float] = {}
        rtfs = rtf_by_tier.get(tier)
        if rtfs:
            entry["rtf"] = round(median(rtfs), 4)
            entry["rtf_n"] = float(len(rtfs))
        rss = rss_by_tier.get(tier)
        if rss:
            entry["peak_rss_mb"] = round(max(rss), 2)
            entry["peak_rss_mb_n"] = float(len(rss))
        if entry:
            out[tier] = entry
    return out


def primary_perf_tier(perf: dict[str, dict[str, float]]) -> str | None:
    """The tier with the most rtf samples (falling back to most peak_rss_mb
    samples) — the tier a board CELL should show, with the tier name as a
    badge, per M2 board-column convention (§ board columns)."""
    if not perf:
        return None

    def _n(tier: str) -> float:
        entry = perf[tier]
        return entry.get("rtf_n", 0.0) or entry.get("peak_rss_mb_n", 0.0)

    return max(perf, key=_n)


# ---------------------------------------------------------------------------
# Pareto / efficiency view — quality vs. compute (RTF)
# ---------------------------------------------------------------------------
#
# A fighter A dominates fighter B when A is at least as good on BOTH quality
# and RTF, and strictly better on at least one — i.e. there is no reason to
# ever pick B over A. The Pareto frontier is every fighter nobody dominates:
# "best quality per unit of compute" along the whole curve, not just the
# single best-quality or single-fastest fighter.


def pareto_frontier(
    points: list[tuple[str, float, float]], quality_higher_is_better: bool = True
) -> tuple[list[str], int]:
    """Pareto-optimal competitor ids over (quality, rtf) points.

    *points* is ``[(competitor_id, quality, rtf), ...]``; rtf is always
    lower-is-better (real time factor). Returns ``(frontier_ids, dominated)``
    — frontier ids in the input's relative order, and how many input points
    were dominated (excluded). Ties (identical quality AND identical rtf)
    do not dominate each other — both stay on the frontier.
    """
    def _quality_at_least_as_good(a: float, b: float) -> bool:
        return a >= b if quality_higher_is_better else a <= b

    def _quality_strictly_better(a: float, b: float) -> bool:
        return a > b if quality_higher_is_better else a < b

    frontier: list[str] = []
    dominated = 0
    for cid, q, r in points:
        is_dominated = False
        for cid2, q2, r2 in points:
            if cid2 == cid:
                continue
            at_least_as_good = _quality_at_least_as_good(q2, q) and r2 <= r
            strictly_better = _quality_strictly_better(q2, q) or r2 < r
            if at_least_as_good and strictly_better:
                is_dominated = True
                break
        if is_dominated:
            dominated += 1
        else:
            frontier.append(cid)
    return frontier, dominated
