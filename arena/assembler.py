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
  (intent: exact match; STT: lower WER) and seed the ELO ledger at reduced
  K.  Auto votes are never attributed to users.

Candidate A/B order is blind: derived from the battle-id hash, not from
competitor names, so the voter cannot infer identity from position.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional, Tuple

from arena.elo import EloLedger
from arena.metrics import row_is_correct, row_wer, ww_row_correct
from arena.models import (
    Battle,
    EloSeed,
    PredictionRow,
    VoteOutcome,
    battle_id_for,
    is_intent_modality,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_BATTLES = 200


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------


def _is_correct(row: PredictionRow, modality: str) -> Optional[bool]:
    """Reference-metric correctness for one row, if computable."""
    if is_intent_modality(modality):
        return row_is_correct(row)
    if modality == "stt":
        wer = row_wer(row)
        return None if wer is None else wer == 0.0
    if modality in ("wake_word", "vad"):
        return ww_row_correct(row)
    return None


def auto_outcome(
    row_a: PredictionRow, row_b: PredictionRow, modality: str
) -> Optional[VoteOutcome]:
    """Benchmark-derived outcome of A vs B, or None when there is no signal."""
    if modality == "stt":
        wer_a, wer_b = row_wer(row_a), row_wer(row_b)
        if wer_a is None or wer_b is None or wer_a == wer_b:
            return None
        return VoteOutcome.CANDIDATE_A if wer_a < wer_b else VoteOutcome.CANDIDATE_B

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
        out = {"intent": row.prediction}
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


def assemble_battles(
    modality: str,
    dataset_id: str,
    lang: str,
    samples: Dict[str, Dict[str, PredictionRow]],
    max_battles: int = DEFAULT_MAX_BATTLES,
) -> List[Battle]:
    """Assemble a deterministic battle pool from grouped prediction rows.

    *samples* maps ``sample_id`` → ``competitor_id`` → row.  Pairs whose
    predictions are identical are skipped (no signal for a voter).  The
    pool interleaves competitor pairs so no single pair dominates, and
    prefers discriminative samples within each pair.
    """
    candidates: Dict[Tuple[str, str], List[Tuple[int, str, Battle]]] = {}

    for sample_id in sorted(samples):
        rows = samples[sample_id]
        for comp_a, comp_b in itertools.combinations(sorted(rows), 2):
            row_a, row_b = rows[comp_a], rows[comp_b]
            if row_a.prediction == row_b.prediction and (
                (row_a.predicted_slots or {}) == (row_b.predicted_slots or {})
            ):
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
            pair = tuple(sorted((comp_a, comp_b)))
            candidates.setdefault(pair, []).append((priority, bid, battle))

    # Sort within each pair by (priority, battle_id) — deterministic
    for pair_candidates in candidates.values():
        pair_candidates.sort(key=lambda c: (c[0], c[1]))

    # Round-robin across pairs (sorted) until the cap is hit
    battles: List[Battle] = []
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
        "Assembled %d battles for %s/%s/%s (%d pairs)",
        len(battles), modality, dataset_id, lang, len(candidates),
    )
    return battles


# ---------------------------------------------------------------------------
# Free-form matchups (direct subjective preference, no stimulus)
# ---------------------------------------------------------------------------


def freeform_battles(
    group: str, lang: str, competitor_plugin: Dict[str, str],
    subgroups: Optional[Dict[str, str]] = None,
) -> List[Battle]:
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
    battles: List[Battle] = []
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


def seed_elo(
    modality: str,
    lang: str,
    samples_by_dataset: Dict[str, Dict[str, Dict[str, PredictionRow]]],
    generated_at: str,
) -> EloSeed:
    """Derive the initial ELO ledger from benchmark metrics.

    Replays an auto-battle for every (sample, competitor-pair) where the
    reference metric picks a winner, in deterministic order, at reduced K.
    """
    ledger = EloLedger()
    competitor_plugin: Dict[str, str] = {}
    auto_votes = 0

    for dataset_id in sorted(samples_by_dataset):
        samples = samples_by_dataset[dataset_id]
        for sample_id in sorted(samples):
            rows = samples[sample_id]
            # every competitor that ran is listed on the board, even with no
            # auto-battle signal (e.g. TTS, which is human-vote-only) — start
            # them at the baseline rating so the board shows who is competing.
            for competitor, row in rows.items():
                competitor_plugin.setdefault(competitor, row.plugin_id)
                ledger.ensure(competitor)
            for comp_a, comp_b in itertools.combinations(sorted(rows), 2):
                row_a, row_b = rows[comp_a], rows[comp_b]
                outcome = auto_outcome(row_a, row_b, modality)
                if outcome is None:
                    continue
                ledger.apply(comp_a, comp_b, outcome, auto=True)
                auto_votes += 1

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
    )
