"""
Battle assembler for the OVOS Plugin Arena (M3 — §4 R1–R3, R5).

Assembles READY battles from ingested prediction rows (no live plugin execution —
§P1).  Implements the matchmaking rules from the spec:

    R1 — Same stimulus: both predictions share the same sample_id.
    R2 — ELO proximity: prefer opponents within a configurable ELO window.
    R3 — Prefer discriminative samples: both WER > 0 (both plugins erred).
    R5 — Auto-battles: seed initial ELO via WER-judged votes before humans
         vote; stored with voter_source=system:wer, reduced K-factor (K/4).

The assembler only creates ``Matchup`` rows; it does not execute plugins or
fetch audio — those live in the HF prediction datasets.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.arena.models import (
    IngestedPrediction,
    Matchup,
    Plugin,
    PluginFamily,
    PredictionSource,
    Vote,
    VoteOutcome,
    VoteSource,
)
from app.arena import db as arena_db
from app.arena.elo import K_FACTOR, process_vote, INITIAL_ELO

logger = logging.getLogger(__name__)

# Auto-battle K-factor weight (spec §6 — suggested K/4)
K_AUTO_FACTOR: float = K_FACTOR / 4.0

# Default configurable assembler parameters
DEFAULT_ELO_WINDOW: float = 200.0       # R2 — max ELO difference for close match
DEFAULT_ALL_SAMPLES_FRACTION: float = 0.2  # R3 — fraction from all samples (not just both-wrong)
DEFAULT_MAX_BATTLES: int = 200          # max battles to assemble per call


@dataclass
class AssemblerConfig:
    elo_window: float = DEFAULT_ELO_WINDOW
    all_samples_fraction: float = DEFAULT_ALL_SAMPLES_FRACTION
    max_battles: int = DEFAULT_MAX_BATTLES
    seed: Optional[int] = None  # for deterministic tests


@dataclass
class AssembledBattle:
    """Result of one assembled battle (before it is committed to DB)."""

    sample_id: str
    plugin_a: str    # plugin_id (OPM name)
    plugin_b: str
    pred_a: IngestedPrediction
    pred_b: IngestedPrediction
    wer_a: Optional[float]
    wer_b: Optional[float]
    both_wrong: bool   # R3 — both WER > 0


def _elo_for_plugin(plugin_name: str) -> float:
    """Return current ELO for a plugin by name, or INITIAL_ELO if unknown."""
    existing = arena_db.get_plugin_by_name(plugin_name)
    if existing is None:
        return INITIAL_ELO
    stats = arena_db.get_elo_stats(existing.id)
    return stats.get("elo", INITIAL_ELO)


def _group_predictions_by_sample(
    predictions: List[IngestedPrediction],
) -> Dict[str, List[IngestedPrediction]]:
    """Group predictions by sample_id; keep only samples with ≥2 plugin_versions."""
    groups: Dict[str, List[IngestedPrediction]] = {}
    for p in predictions:
        groups.setdefault(p.sample_id, []).append(p)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _pick_pair(
    preds: List[IngestedPrediction],
    elo_window: float,
) -> Optional[Tuple[IngestedPrediction, IngestedPrediction]]:
    """Pick two predictions from the same sample respecting ELO proximity (R2).

    Tries ELO-window first; falls back to any pair if no close pair found.
    """
    if len(preds) < 2:
        return None

    # Build all pairs, sorted by ELO difference
    scored: List[Tuple[float, IngestedPrediction, IngestedPrediction]] = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i], preds[j]
            if a.plugin_version == b.plugin_version:
                continue  # same model, skip
            elo_a = _elo_for_plugin(a.plugin_id)
            elo_b = _elo_for_plugin(b.plugin_id)
            diff = abs(elo_a - elo_b)
            scored.append((diff, a, b))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    # Prefer pairs within ELO window
    close = [(a, b) for diff, a, b in scored if diff <= elo_window]
    if close:
        return random.choice(close)
    # Fall back to closest available pair
    return scored[0][1], scored[0][2]


def assemble_battles(
    source: PredictionSource,
    cfg: Optional[AssemblerConfig] = None,
    auto_seed: bool = True,
) -> List[Matchup]:
    """Assemble READY battles from *source* and persist them.

    Parameters
    ----------
    source:     the PredictionSource to draw from
    cfg:        assembler configuration; defaults applied if None
    auto_seed:  if True, emit auto-battle votes (R5) for assembled pairs
                where WER comparison gives a clear winner

    Returns the list of newly created Matchup rows.
    """
    if cfg is None:
        cfg = AssemblerConfig()

    rng = random.Random(cfg.seed)

    predictions = arena_db.list_predictions_for_source(source.id)
    if not predictions:
        logger.warning("No predictions found for source %s", source.hf_dataset)
        return []

    groups = _group_predictions_by_sample(predictions)
    if not groups:
        logger.warning(
            "No samples with ≥2 distinct plugin_versions in source %s; "
            "battles require predictions from multiple plugins for the same sample_id",
            source.hf_dataset,
        )
        return []

    # R3 — split into discriminative (both wrong) and all-samples pools
    both_wrong_samples = []
    other_samples = []
    for sid, preds in groups.items():
        wers = [p.wer for p in preds if p.wer is not None]
        if len(wers) >= 2 and all(w > 0 for w in wers):
            both_wrong_samples.append(sid)
        else:
            other_samples.append(sid)

    # Determine how many come from each pool (R3)
    # n_all = fraction from all-samples (default 20%); n_bw = remainder from both-wrong
    n_all = max(0, int(cfg.max_battles * cfg.all_samples_fraction))
    n_bw = cfg.max_battles - n_all

    actual_bw = min(n_bw, len(both_wrong_samples))
    candidates_bw = rng.sample(both_wrong_samples, actual_bw)

    # Fill remaining quota from all-samples pool (including shortfall from bw pool)
    shortfall = n_bw - actual_bw
    need_all = min(n_all + shortfall, len(other_samples))
    candidates_all = rng.sample(other_samples, need_all) if need_all > 0 else []

    selected_samples = candidates_bw + candidates_all
    rng.shuffle(selected_samples)

    matchups: List[Matchup] = []
    auto_votes: List[Vote] = []

    for sample_id in selected_samples:
        preds = groups[sample_id]
        pair = _pick_pair(preds, cfg.elo_window)
        if pair is None:
            continue
        pred_a, pred_b = pair

        plugin_a = arena_db.get_plugin_by_name(pred_a.plugin_id)
        plugin_b = arena_db.get_plugin_by_name(pred_b.plugin_id)
        if plugin_a is None or plugin_b is None:
            logger.debug("Skipping sample %s — plugins not registered", sample_id)
            continue

        # Use source dataset as the input_ref for the matchup
        matchup = Matchup(
            family=source.modality,
            input_ref=f"hf://{source.hf_dataset}/{sample_id}",
            sample_a_id=pred_a.id,
            sample_b_id=pred_b.id,
            plugin_a_id=plugin_a.id,
            plugin_b_id=plugin_b.id,
            status="ready",
        )
        arena_db.create_matchup(matchup)
        matchups.append(matchup)

        # R5 — auto-seed with WER-judged vote when reference is available
        if auto_seed and pred_a.wer is not None and pred_b.wer is not None:
            auto_vote = _make_wer_vote(matchup, pred_a.wer, pred_b.wer)
            if auto_vote is not None:
                arena_db.create_vote(auto_vote)
                auto_votes.append(auto_vote)
                # Update ELO with reduced K
                _apply_auto_vote(auto_vote, matchup)

    logger.info(
        "Assembled %d battles from %s (%d auto-seeded)",
        len(matchups),
        source.hf_dataset,
        len(auto_votes),
    )
    return matchups


def _make_wer_vote(
    matchup: Matchup,
    wer_a: float,
    wer_b: float,
) -> Optional[Vote]:
    """Create an auto-battle vote judged by WER (§4 R5).

    Lower WER wins.  If both are equal, it's a tie.  The voter_id is
    'system:wer'; voter_source is VoteSource.AUTO_WER.
    """
    if wer_a < wer_b:
        outcome = VoteOutcome.CANDIDATE_A
    elif wer_b < wer_a:
        outcome = VoteOutcome.CANDIDATE_B
    else:
        outcome = VoteOutcome.TIE

    return Vote(
        matchup_id=matchup.id,
        outcome=outcome,
        voter_id="system:wer",
        voter_source=VoteSource.AUTO_WER,
        automated=True,
        note=f"auto: wer_a={wer_a:.4f} wer_b={wer_b:.4f}",
    )


def _apply_auto_vote(vote: Vote, matchup: Matchup) -> None:
    """Apply vote to ELO with reduced K-factor (§6 — K/4 for auto votes)."""
    pid_a = matchup.plugin_a_id
    pid_b = matchup.plugin_b_id

    stats_a = arena_db.get_elo_stats(pid_a)
    stats_b = arena_db.get_elo_stats(pid_b)
    r_a = stats_a.get("elo", INITIAL_ELO)
    r_b = stats_b.get("elo", INITIAL_ELO)
    b_a = stats_a.get("battles", 0)
    b_b = stats_b.get("battles", 0)

    from app.arena.elo import expected_score

    e_a = expected_score(r_a, r_b)
    e_b = 1.0 - e_a

    if vote.outcome == VoteOutcome.CANDIDATE_A:
        s_a, s_b = 1.0, 0.0
    elif vote.outcome == VoteOutcome.CANDIDATE_B:
        s_a, s_b = 0.0, 1.0
    else:
        s_a, s_b = 0.5, 0.5

    new_a = r_a + K_AUTO_FACTOR * (s_a - e_a)
    new_b = r_b + K_AUTO_FACTOR * (s_b - e_b)

    from app.arena.models import RatingSnapshot

    snap_a = RatingSnapshot(
        vote_id=vote.id,
        plugin_id=pid_a,
        elo_before=r_a,
        elo_after=new_a,
        delta=new_a - r_a,
    )
    snap_b = RatingSnapshot(
        vote_id=vote.id,
        plugin_id=pid_b,
        elo_before=r_b,
        elo_after=new_b,
        delta=new_b - r_b,
    )
    arena_db.create_rating_snapshot(snap_a)
    arena_db.create_rating_snapshot(snap_b)

    arena_db.update_elo_stats(pid_a, new_a, won=(vote.outcome == VoteOutcome.CANDIDATE_A), tied=(vote.outcome == VoteOutcome.TIE))
    arena_db.update_elo_stats(pid_b, new_b, won=(vote.outcome == VoteOutcome.CANDIDATE_B), tied=(vote.outcome == VoteOutcome.TIE))
    arena_db.mark_matchup_voted(matchup.id)
