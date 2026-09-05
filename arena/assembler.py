"""
Battle assembly and ELO seeding for the OVOS Plugin Arena.

Everything here is deterministic (§P5): given the same prediction rows the
assembler always emits the same battles (stable ``battle_id``s, so votes on
GitHub issues stay valid across re-runs) and the same ELO seed.

Matchmaking (§4):

- **R1** — a battle pairs two predictions for the *same* sample.
- **R3** — samples where both competitors erred are preferred; disagreements
  where exactly one erred come next; identical outputs are never battled.
- **R5** — auto-battles derive an outcome from the reference metric
  (intent: exact match; STT: lower WER; TTS: higher UTMOS) and seed the ELO
  ledger at reduced K.  Auto votes are never attributed to users.

Candidate A/B order is blind: derived from the battle-id hash, not from
competitor names, so the voter cannot infer identity from position.
"""

from __future__ import annotations

import itertools
import logging

from arena.elo import EloLedger
from arena.metrics import (
    CONTAMINATED_BUCKETS,
    metric_higher_is_better,
    primary_metric_ci,
    row_is_correct,
    row_metric_value,
    row_intelligibility_cer,
    row_intelligibility_judge,
    row_utmos,
    row_wer,
    tts_seed_score,
    secondary_ladder_metrics_for,
    significant_from_cis,
    ww_row_correct,
)
from arena.models import (
    Battle,
    EloSeed,
    PredictionRow,
    SecondaryMetricSeed,
    VoteOutcome,
    battle_id_for,
    is_intent_modality,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_BATTLES = 200

# §4 seed-battle bias audit — a benchmark dataset can have thousands of
# samples; without a cap its auto-battles would dwarf a modest number of
# real human votes in the Bradley-Terry fit regardless of BT_AUTO_WEIGHT.
# Capped at the weighted-game-total level (arena/rating.py PairwiseGames),
# in human-vote-equivalent units, per (competitor_a, competitor_b) pair.
MAX_AUTO_WEIGHT_PER_PAIR = 5.0  # R5b weight cap


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------


def _is_correct(row: PredictionRow, modality: str) -> bool | None:
    """Reference-metric correctness for one row, if computable."""
    if is_intent_modality(modality):
        return row_is_correct(row)
    if modality == "stt":
        wer = row_wer(row)
        return None if wer is None else wer == 0.0
    if modality in ("wake_word", "vad"):
        return ww_row_correct(row)
    return None


def seeds_elo(row: PredictionRow, modality: str) -> bool:
    """Whether *row* may contribute an auto-battle to the seeded ratings.

    The seeded ladder has to be built from the same population as the board's
    primary metric, or a rating labelled "generalization" would still be a
    memorization score. Intent rows in a ``CONTAMINATED_BUCKETS`` bucket are
    training data the fighters have seen, and are excluded from
    ``generalization_accuracy``; they are excluded here too. Every other
    modality battles on all of its rows.
    """
    return not (
        is_intent_modality(modality)
        and (row.bucket or "test") in CONTAMINATED_BUCKETS
    )


def auto_outcome(
    row_a: PredictionRow, row_b: PredictionRow, modality: str
) -> VoteOutcome | None:
    """Benchmark-derived outcome of A vs B, or None when there is no signal."""
    if modality == "stt":
        wer_a, wer_b = row_wer(row_a), row_wer(row_b)
        if wer_a is None or wer_b is None or wer_a == wer_b:
            return None
        return VoteOutcome.CANDIDATE_A if wer_a < wer_b else VoteOutcome.CANDIDATE_B

    if modality == "tts":
        # No ground-truth reference to call "correct" against — the
        # composite ``tts_seed_score`` (UTMOS naturalness x ROVER-consensus
        # intelligibility x inter-judge agreement, §4 R14/R16) picks the
        # winner when both rows carry an intelligibility CER. Rows scored
        # before intelligibility judging existed (both missing CER) fall
        # back to a UTMOS-only comparison so old shards still seed;  a row
        # with CER against one without is a mixed signal — no auto-vote,
        # rather than silently comparing an apples score to an oranges one.
        cer_a, cer_b = row_intelligibility_cer(row_a), row_intelligibility_cer(row_b)
        if cer_a is not None and cer_b is not None:
            # A WER from one ASR is not comparable with a WER from another
            # (§4 R16), so two rows judged by different models are the same
            # mixed signal as one row having no CER at all — no auto-vote.
            judge_a, judge_b = row_intelligibility_judge(row_a), row_intelligibility_judge(row_b)
            if judge_a is not None and judge_b is not None and judge_a != judge_b:
                return None
            score_a, score_b = tts_seed_score(row_a), tts_seed_score(row_b)
            if score_a is None or score_b is None or score_a == score_b:
                return None
            return VoteOutcome.CANDIDATE_A if score_a > score_b else VoteOutcome.CANDIDATE_B
        if cer_a is None and cer_b is None:
            utmos_a, utmos_b = row_utmos(row_a), row_utmos(row_b)
            if utmos_a is None or utmos_b is None or utmos_a == utmos_b:
                return None
            return VoteOutcome.CANDIDATE_A if utmos_a > utmos_b else VoteOutcome.CANDIDATE_B
        return None  # exactly one row carries a CER — mixed signal, no vote

    correct_a = _is_correct(row_a, modality)
    correct_b = _is_correct(row_b, modality)
    if correct_a is None or correct_b is None or correct_a == correct_b:
        return None
    return VoteOutcome.CANDIDATE_A if correct_a else VoteOutcome.CANDIDATE_B


def _battle_priority(
    row_a: PredictionRow, row_b: PredictionRow, modality: str
) -> int:
    """§4 R3 — lower sorts first: 0 both-wrong, 1 one-wrong, 2 rest."""
    correct_a = _is_correct(row_a, modality)
    correct_b = _is_correct(row_b, modality)
    if correct_a is False and correct_b is False:
        return 0
    if correct_a is not None and correct_b is not None and correct_a != correct_b:
        return 1
    return 2


def _payload(row: PredictionRow, modality: str):
    """What the voter sees for one candidate."""
    if is_intent_modality(modality):
        out: dict[str, object] = {"intent": row.prediction}
        if row.predicted_slots:
            out["slots"] = row.predicted_slots
        return out
    return row.prediction


def _stimulus(row: PredictionRow, modality: str):
    """The shared battle stimulus: ``(input_text, audio_url, reference)``.

    What the voter is shown and asked to judge differs per modality: intent
    voters read the utterance, STT/wake-word voters listen to the source clip,
    TTS voters read the prompt and listen to each candidate's synthesis.
    """
    if is_intent_modality(modality):
        return row.utterance, None, row.reference_intent
    if modality == "stt":
        return None, row.audio_url, row.reference_text
    if modality == "tts":
        return (row.input_text or row.utterance
                or row.extras.get("input_text")), None, None
    if modality in ("wake_word", "vad"):
        return None, row.audio_url, row.label
    return row.utterance or row.extras.get("input_text"), row.audio_url, None


# ---------------------------------------------------------------------------
# Battles
# ---------------------------------------------------------------------------


def _reference_for_identity(row: PredictionRow, modality: str) -> str | None:
    """The ground-truth text that identifies *which stimulus* a row answers.

    Two rows sharing a ``sample_id`` are only a valid battle pair when they
    were scored against the *same* underlying content. Legacy pre-#70 shards
    can collide on ``sample_id`` while carrying different ``reference_text``
    (or ``reference_intent``) — pairing those produces a battle where the
    voter is silently asked to judge two unrelated stimuli.
    """
    if is_intent_modality(modality):
        return row.reference_intent
    if modality == "stt":
        return row.reference_text
    if modality in ("wake_word", "vad"):
        return row.label
    return None


def _normalize_ref(text: str | None) -> str | None:
    if text is None:
        return None
    return " ".join(text.split())


def assemble_battles(
    modality: str,
    dataset_id: str,
    lang: str,
    samples: dict[str, dict[str, PredictionRow]],
    max_battles: int = DEFAULT_MAX_BATTLES,
    stats: dict[str, int] | None = None,
) -> list[Battle]:
    """Assemble a deterministic battle pool from grouped prediction rows.

    *samples* maps ``sample_id`` → ``competitor_id`` → row.  Pairs whose
    predictions are identical are skipped (R2 — no signal for a voter).  Pairs
    whose reference text disagrees (colliding ``sample_id`` across legacy
    shards) are skipped too — see ``_reference_for_identity`` — and counted
    into ``stats["skipped_reference_mismatches"]`` when *stats* is given, so
    the mismatch is surfaced instead of silently producing garbage battles.
    The pool interleaves competitor pairs so no single pair dominates, and
    prefers discriminative samples within each pair.
    """
    candidates: dict[tuple[str, str], list[tuple[int, str, Battle]]] = {}
    skipped_reference_mismatches = 0

    # §assemble memory — no single pair's queue can ever contribute more
    # than `max_battles` battles to the final pool (the round-robin below
    # stops the instant `len(battles) == max_battles`, regardless of how
    # many pairs are still queued), so nothing is lost by capping each
    # pair's candidate list to its `max_battles` best (lowest
    # (priority, battle_id)) entries as they stream in. Without this, a
    # dataset with many near-duplicate competitors builds
    # O(C(competitors, 2) * samples) candidate Battle objects before the
    # cap is ever applied — for intent_template's intents-for-eval/en-US
    # pool (~90 competitors, ~1400 samples) that is millions of Battle
    # objects held in memory just to keep 200, and it OOM-killed the
    # hosted runner's 7GB assemble matrix leg. `_PAIR_RESORT_MULTIPLE`
    # trades a little extra sorting for not re-sorting on every single
    # insert.
    pair_cap = max(max_battles, 1)
    _PAIR_RESORT_MULTIPLE = 4

    def _trim_pair(bucket: list[tuple[int, str, Battle]]) -> None:
        bucket.sort(key=lambda c: (c[0], c[1]))
        del bucket[pair_cap:]

    for sample_id in sorted(samples):
        rows = samples[sample_id]
        for comp_a, comp_b in itertools.combinations(sorted(rows), 2):
            row_a, row_b = rows[comp_a], rows[comp_b]
            if row_a.prediction == row_b.prediction and (
                (row_a.predicted_slots or {}) == (row_b.predicted_slots or {})
            ):
                continue

            ref_a = _normalize_ref(_reference_for_identity(row_a, modality))
            ref_b = _normalize_ref(_reference_for_identity(row_b, modality))
            if ref_a is not None and ref_b is not None and ref_a != ref_b:
                skipped_reference_mismatches += 1
                logger.warning(
                    "Skipping battle %s/%s/%s sample %r: %s vs %s reference "
                    "text mismatch (%r != %r) — likely colliding sample_id "
                    "across shards",
                    modality, dataset_id, lang, sample_id, comp_a, comp_b,
                    ref_a, ref_b,
                )
                continue

            bid = battle_id_for(modality, dataset_id, lang, sample_id, comp_a, comp_b)
            # Blind A/B order from the hash, decoupled from competitor names
            if int(bid, 16) % 2:
                comp_a, comp_b = comp_b, comp_a
                row_a, row_b = row_b, row_a

            input_text, audio_url, reference = _stimulus(row_a, modality)
            battle = Battle(
                battle_id=bid,
                modality=modality,
                dataset_id=dataset_id,
                lang=lang,
                sample_id=sample_id,
                input_text=input_text,
                audio_url=audio_url,
                reference=reference,
                prediction_a=_payload(row_a, modality),
                prediction_b=_payload(row_b, modality),
                competitor_a=comp_a,
                competitor_b=comp_b,
                plugin_a=row_a.plugin_id,
                plugin_b=row_b.plugin_id,
            )
            priority = _battle_priority(row_a, row_b, modality)
            pair_a, pair_b = sorted((comp_a, comp_b))
            bucket = candidates.setdefault((pair_a, pair_b), [])
            bucket.append((priority, bid, battle))
            if len(bucket) > pair_cap * _PAIR_RESORT_MULTIPLE:
                _trim_pair(bucket)

    # Sort within each pair by (priority, battle_id) — deterministic —
    # and apply the final trim (a no-op for a pair the loop above already
    # trimmed down to `pair_cap`).
    for pair_candidates in candidates.values():
        _trim_pair(pair_candidates)

    # Round-robin across pairs (sorted) until the cap is hit
    battles: list[Battle] = []
    queues = [candidates[pair] for pair in sorted(candidates)]
    while queues and len(battles) < max_battles:
        next_queues = []
        for queue in queues:
            if len(battles) >= max_battles:
                break
            battles.append(queue.pop(0)[2])
            if queue:
                next_queues.append(queue)
        queues = next_queues

    logger.info(
        "Assembled %d battles for %s/%s/%s (%d pairs, %d reference "
        "mismatches skipped)",
        len(battles), modality, dataset_id, lang, len(candidates),
        skipped_reference_mismatches,
    )
    if stats is not None:
        stats["skipped_reference_mismatches"] = (
            stats.get("skipped_reference_mismatches", 0)
            + skipped_reference_mismatches
        )
    return battles


# ---------------------------------------------------------------------------
# Free-form matchups (direct subjective preference, no stimulus)
# ---------------------------------------------------------------------------


def freeform_battles(
    group: str, lang: str, competitor_plugin: dict[str, str],
    subgroups: dict[str, str] | None = None,
) -> list[Battle]:
    """Every competitor pair as a stimulus-less matchup for direct voting.

    A free-form vote is a subjective head-to-head ("which plugin do you
    prefer?") cast by someone who tested the plugins out of band — no sample,
    no blind masking.  Each pair gets a stable ``battle_id`` so ``tally``
    dedupes one vote per (author, pair) and replays it into the same
    (group, lang) ELO ladder as the blind battles.

    *subgroups* (competitor → key) restricts pairing to within a subgroup —
    used for wake word, where only fighters for the *same phrase* are
    comparable (you cannot prefer a 'hey jarvis' detector over a 'computer'
    detector).
    """
    battles: list[Battle] = []
    for comp_a, comp_b in itertools.combinations(sorted(competitor_plugin), 2):
        if subgroups and subgroups.get(comp_a) != subgroups.get(comp_b):
            continue
        bid = battle_id_for(group, "freeform", lang, "freeform", comp_a, comp_b)
        battles.append(Battle(
            battle_id=bid,
            modality=group,
            dataset_id="freeform",
            lang=lang,
            sample_id="freeform",
            competitor_a=comp_a,
            competitor_b=comp_b,
            plugin_a=competitor_plugin.get(comp_a, ""),
            plugin_b=competitor_plugin.get(comp_b, ""),
        ))
    return battles


# ---------------------------------------------------------------------------
# ELO seeding (§4 R5)
# ---------------------------------------------------------------------------


def _rows_by_competitor(
    samples: dict[str, dict[str, PredictionRow]],
) -> dict[str, list[PredictionRow]]:
    """Flatten a dataset's ``{sample_id: {competitor: row}}`` map into
    ``{competitor: [row, ...]}`` — the shape ``primary_metric_ci`` needs."""
    out: dict[str, list[PredictionRow]] = {}
    for sample_id in sorted(samples):
        for competitor, row in samples[sample_id].items():
            out.setdefault(competitor, []).append(row)
    return out


def _cap_auto_pairwise_weight(
    ledger: EloLedger, max_weight: float = MAX_AUTO_WEIGHT_PER_PAIR
) -> None:
    """Scale down each pair's total auto-battle weight to at most
    *max_weight* (in human-vote-equivalent BT weight units), preserving the
    observed win rate. Every entry in ``ledger.pairwise_wins/games`` at seed
    time is purely auto-vote weight (seeding runs before any human vote is
    known), so this caps the whole ledger in place.
    """
    for i in list(ledger.pairwise_games):
        for j, g in list(ledger.pairwise_games[i].items()):
            if g <= max_weight or g <= 0:
                continue
            scale = max_weight / g
            ledger.pairwise_games[i][j] = g * scale
            ledger.pairwise_wins[i][j] = ledger.pairwise_wins[i].get(j, 0.0) * scale


def seed_elo(
    modality: str,
    lang: str,
    samples_by_dataset: dict[str, dict[str, dict[str, PredictionRow]]],
    generated_at: str,
) -> EloSeed:
    """Derive the initial ELO ledger from benchmark metrics.

    Replays an auto-battle for every (sample, competitor-pair) where the
    reference metric picks a winner and the pair's *aggregate* metrics are
    statistically distinguishable (§4 seed-battle bias audit —
    ``pair_metric_significant``; a pair whose overall CIs overlap
    contributes no auto-battles, even if individual samples happen to
    disagree), in deterministic order, at reduced K. The resulting
    Bradley-Terry pairwise weight is further capped per pair
    (``MAX_AUTO_WEIGHT_PER_PAIR``) so a large benchmark dataset can never
    outweigh a modest number of real human votes.
    """
    ledger = EloLedger()
    competitor_plugin: dict[str, str] = {}
    auto_votes = 0

    for dataset_id in sorted(samples_by_dataset):
        samples = samples_by_dataset[dataset_id]
        rows_by_competitor = _rows_by_competitor(samples)
        # Bootstrap a CI per *competitor* once (O(fighters)) instead of once
        # per (comp_a, comp_b) *pair* (O(fighters²)) — a jurebes-scale roster
        # (60+ fighters) turns ~1000-round bootstraps over full-size row
        # lists into the dominant cost of assemble otherwise.
        ci_cache: dict[str, tuple[float, float] | None] = {}

        def _ci(competitor: str) -> tuple[float, float] | None:
            if competitor not in ci_cache:
                ci_cache[competitor] = primary_metric_ci(
                    modality, rows_by_competitor[competitor]
                )
            return ci_cache[competitor]

        significant: dict[tuple[str, str], bool] = {}

        for sample_id in sorted(samples):
            rows = samples[sample_id]
            # every competitor that ran is listed on the board, even with no
            # auto-battle signal (e.g. a competitor whose rows never got a
            # UTMOS score) — start them at the baseline rating so the board
            # shows who is competing.
            for competitor, row in rows.items():
                competitor_plugin.setdefault(competitor, row.plugin_id)
                ledger.ensure(competitor)
            for comp_a, comp_b in itertools.combinations(sorted(rows), 2):
                pair = (comp_a, comp_b)
                if pair not in significant:
                    significant[pair] = significant_from_cis(
                        _ci(comp_a), _ci(comp_b)
                    )
                if not significant[pair]:
                    continue
                row_a, row_b = rows[comp_a], rows[comp_b]
                if not (seeds_elo(row_a, modality) and seeds_elo(row_b, modality)):
                    continue
                outcome = auto_outcome(row_a, row_b, modality)
                if outcome is None:
                    continue
                ledger.apply(comp_a, comp_b, outcome, auto=True)
                auto_votes += 1

    _cap_auto_pairwise_weight(ledger)

    return EloSeed(
        modality=modality,
        lang=lang,
        generated_at=generated_at,
        auto_vote_count=auto_votes,
        ratings={k: round(v, 2) for k, v in ledger.ratings.items()},
        battles=ledger.battles,
        wins=ledger.wins,
        losses=ledger.losses,
        ties=ledger.ties,
        competitor_plugin=competitor_plugin,
        pairwise_wins=ledger.pairwise_wins,
        pairwise_games=ledger.pairwise_games,
    )


def seed_secondary_metrics(
    modality: str,
    samples_by_dataset: dict[str, dict[str, dict[str, PredictionRow]]],
) -> dict[str, SecondaryMetricSeed]:
    """Auto-only BT seeds for every row-level secondary metric of *modality*.

    Perf note (§ per-metric ladders campaign): this is a SINGLE extra pass
    over ``samples_by_dataset`` shared across every secondary metric —
    each (sample, competitor-pair) is visited once and compared on every
    ladderable metric in the same inner loop, rather than re-looping the
    whole dataset once per metric. Unlike ``seed_elo`` (the primary-metric
    seed), this skips the per-pair significance-CI gate: secondary ladders
    are explicitly documented as "auto-battles only, lower rigor" (§ human
    votes only attach to the primary ladder), so the extra bootstrap-CI
    pass per metric — the dominant cost of ``seed_elo`` on a large roster —
    is not worth paying N times over.
    """
    metric_keys = secondary_ladder_metrics_for(modality)
    if not metric_keys:
        return {}

    ledgers = {key: EloLedger() for key in metric_keys}
    auto_votes = dict.fromkeys(metric_keys, 0)

    for dataset_id in sorted(samples_by_dataset):
        samples = samples_by_dataset[dataset_id]
        for sample_id in sorted(samples):
            rows = samples[sample_id]
            for competitor in rows:
                for ledger in ledgers.values():
                    ledger.ensure(competitor)
            for comp_a, comp_b in itertools.combinations(sorted(rows), 2):
                row_a, row_b = rows[comp_a], rows[comp_b]
                for metric in metric_keys:
                    val_a = row_metric_value(row_a, modality, metric)
                    val_b = row_metric_value(row_b, modality, metric)
                    if val_a is None or val_b is None or val_a == val_b:
                        continue
                    higher_better = metric_higher_is_better(metric)
                    a_wins = (val_a > val_b) if higher_better else (val_a < val_b)
                    outcome = (
                        VoteOutcome.CANDIDATE_A if a_wins else VoteOutcome.CANDIDATE_B
                    )
                    ledgers[metric].apply_pairwise_only(comp_a, comp_b, outcome, auto=True)
                    auto_votes[metric] += 1

    out: dict[str, SecondaryMetricSeed] = {}
    for metric, ledger in ledgers.items():
        _cap_auto_pairwise_weight(ledger)
        out[metric] = SecondaryMetricSeed(
            higher_is_better=metric_higher_is_better(metric),
            auto_vote_count=auto_votes[metric],
            battles=ledger.battles,
            wins=ledger.wins,
            losses=ledger.losses,
            ties=ledger.ties,
            pairwise_wins=ledger.pairwise_wins,
            pairwise_games=ledger.pairwise_games,
        )
    return out
