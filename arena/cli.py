"""
CLI for the GitHub-native OVOS Plugin Arena.

Commands (each one maps to a GitHub Actions workflow step):

assemble
    Pull prediction JSONLs (HF dataset repos or local dirs), then write the
    static data artifacts: ``battles-<mod>-<lang>.json`` (blind A/B pool),
    ``benchmark-<mod>-<lang>.json`` (auto-metric boards) and
    ``elo-seed-<mod>-<lang>.json`` (benchmark-derived initial ELO).

tally
    Read GitHub vote issues, dedupe (one vote per author per battle),
    replay human votes on top of the ELO seed in issue-number order, write
    ``leaderboard-<mod>-<lang>.json`` and close processed issues.

export-index
    Regenerate ``index.json`` describing every data artifact.

export-bestiary
    Flatten the competitor registry into ``competitors.json`` for the
    fighter-browser UI, tagging each competitor with ``has_predictions``
    and the ``(dataset_id, lang)`` pairs it has at least one prediction row
    for, read from the assembled benchmark boards.

export-evidence
    Regenerate ``evidence.json`` — per-league completeness counts (fighters,
    registered datasets, datasets with published predictions, benchmark
    boards, ELO leaderboards) plus global totals, for the evidence page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena.assembler import (
    MAX_AUTO_WEIGHT_PER_PAIR,
    assemble_battles,
    freeform_battles,
    seed_elo,
    seed_secondary_metrics,
)
from arena.badges import emit_badges
from arena.elo import BT_AUTO_WEIGHT, EloLedger
from arena.fraud import resolve_vote_weights
from arena.metrics import (
    PRIMARY_METRIC,
    benchmark_board_input_signature,
    build_benchmark_board,
    metric_higher_is_better,
    row_on_pinned_revision,
)
from arena.models import (
    VOTELESS_MODALITIES,
    BattlesPool,
    EloBoard,
    EloEntry,
    EloSeed,
    MetricLadder,
    MetricLadderEntry,
    SecondaryMetricSeed,
    VoteOutcome,
    battle_group,
    leagues,
)
from arena.patch_notes import build_patch_notes, diff_board, load_board
from arena.rating import (
    PROVISIONAL_MIN_HUMAN_VOTES,
    PairResult,
    bootstrap_confidence_intervals,
    fit_bradley_terry,
    to_rating_scale,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

VOTE_TITLE_RE = re.compile(
    r"^vote\|(?P<battle_id>[^|]+)\|(?P<choice>a|b|tie|both_wrong)$",
    re.IGNORECASE,
)

CHOICE_TO_OUTCOME = {
    "a": VoteOutcome.CANDIDATE_A,
    "b": VoteOutcome.CANDIDATE_B,
    "tie": VoteOutcome.TIE,
    "both_wrong": VoteOutcome.BOTH_WRONG,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Vote parsing / deduplication (pure helpers, unit-tested)
# ---------------------------------------------------------------------------


def parse_vote_title(title: str) -> tuple[str, str] | None:
    """Parse ``vote|<battle_id>|<choice>`` — returns (battle_id, choice) or None."""
    m = VOTE_TITLE_RE.match(title.strip())
    if not m:
        return None
    return m.group("battle_id"), m.group("choice").lower()


def dedupe_votes(raw_votes: list[dict]) -> list[dict]:
    """Keep the first vote per (author, battle_id), ordered by issue number."""
    seen: set = set()
    out: list[dict] = []
    ordered = sorted(
        raw_votes, key=lambda v: (v["issue_number"], v.get("created_at", ""))
    )
    for vote in ordered:
        key = (vote["author"], vote["battle_id"])
        if key not in seen:
            seen.add(key)
            out.append(vote)
    return out


# ---------------------------------------------------------------------------
# Data dir helpers
# ---------------------------------------------------------------------------


def load_battles_pools(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Map battle_id → battle dict from every ``battles-*.json`` pool."""
    battles: dict[str, dict[str, Any]] = {}
    for path in sorted(data_dir.glob("battles-*.json")):
        try:
            payload = json.loads(path.read_text())
            for battle in payload.get("battles", []):
                battles[battle["battle_id"]] = battle
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
    return battles


def load_elo_seeds(data_dir: Path) -> dict[tuple[str, str], EloSeed]:
    """Map (modality, lang) → EloSeed from every ``elo-seed-*.json``."""
    seeds: dict[tuple[str, str], EloSeed] = {}
    for path in sorted(data_dir.glob("elo-seed-*.json")):
        try:
            seed = EloSeed(**json.loads(path.read_text()))
            seeds[(seed.modality.value, seed.lang)] = seed
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
    return seeds


def _ledger_from_seed(seed: EloSeed) -> EloLedger:
    ledger = EloLedger()
    for competitor, rating in seed.ratings.items():
        ledger.ensure(competitor)
        ledger.ratings[competitor] = rating
        ledger.battles[competitor] = seed.battles.get(competitor, 0)
        ledger.wins[competitor] = seed.wins.get(competitor, 0)
        ledger.losses[competitor] = seed.losses.get(competitor, 0)
        ledger.ties[competitor] = seed.ties.get(competitor, 0)
        ledger.auto_votes[competitor] = seed.battles.get(competitor, 0)
    ledger.pairwise_wins = {i: dict(js) for i, js in seed.pairwise_wins.items()}
    ledger.pairwise_games = {i: dict(js) for i, js in seed.pairwise_games.items()}
    return ledger


_CHOICE_TO_SCORE_A = {"a": 1.0, "b": 0.0, "tie": 0.5, "both_wrong": 0.5}


def build_elo_board(
    modality: str,
    lang: str,
    seed: EloSeed | None,
    human_votes: list[dict],
    battles_pool: dict[str, dict[str, Any]],
) -> EloBoard:
    """Replay *human_votes* (ordered) on top of *seed* and rank the result.

    Two ratings are computed from the same replayed vote log: the legacy
    sequential ELO (``EloEntry.elo``, order-dependent, kept for continuity,
    always applied at full strength) and a batch Bradley-Terry fit with
    bootstrap confidence intervals (``EloEntry.bt_rating`` / ``ci_lower`` /
    ``ci_upper``, order-independent — see ``arena/rating.py``). Ranking is
    by ``bt_rating``.

    Each vote dict may carry an optional ``"weight"`` key (§4 A1.4 vote
    fraud rules, default 1.0 when absent) that scales only its
    Bradley-Terry pairwise contribution — a down-weighted vote (e.g. a
    one-sided voter) still shows up in the legacy ELO column and vote
    counts, but its influence on the statistically-rigorous rating is
    reduced. Fully discarded votes (weight 0) should be filtered out of
    *human_votes* by the caller before this function ever sees them.
    """
    ledger = _ledger_from_seed(seed) if seed else EloLedger()
    competitor_plugin = dict(seed.competitor_plugin) if seed else {}
    fixed_wins = {i: dict(js) for i, js in seed.pairwise_wins.items()} if seed else {}
    fixed_games = {i: dict(js) for i, js in seed.pairwise_games.items()} if seed else {}

    counted = 0
    human_results: list[PairResult] = []
    for vote in human_votes:
        battle = battles_pool.get(vote["battle_id"])
        if not battle:
            continue
        comp_a = battle["competitor_a"]
        comp_b = battle["competitor_b"]
        weight = vote.get("weight", 1.0)
        competitor_plugin.setdefault(comp_a, battle.get("plugin_a", ""))
        competitor_plugin.setdefault(comp_b, battle.get("plugin_b", ""))
        ledger.apply(comp_a, comp_b, CHOICE_TO_OUTCOME[vote["choice"]], bt_weight=weight)
        human_results.append(
            PairResult(comp_a, comp_b, _CHOICE_TO_SCORE_A[vote["choice"]], weight=weight)
        )
        counted += 1

    competitors = sorted(ledger.ratings)
    bt_strengths = fit_bradley_terry(ledger.pairwise_wins, ledger.pairwise_games, competitors)
    bt_ratings = to_rating_scale(bt_strengths)
    cis = bootstrap_confidence_intervals(
        human_results, fixed_wins, fixed_games, competitors,
    )

    entries = []
    for competitor, rating in ledger.ratings.items():
        battles = ledger.battles[competitor]
        wins = ledger.wins[competitor]
        ci_lower, ci_upper = cis.get(competitor, (None, None))
        entries.append(
            EloEntry(
                competitor_id=competitor,
                plugin_id=competitor_plugin.get(competitor, ""),
                elo=round(rating, 2),
                battles=battles,
                wins=wins,
                losses=ledger.losses[competitor],
                ties=ledger.ties[competitor],
                win_rate=round(wins / battles, 4) if battles else 0.0,
                human_votes=ledger.human_votes[competitor],
                auto_votes=ledger.auto_votes[competitor],
                bt_rating=round(bt_ratings.get(competitor, 1200.0), 2),
                ci_lower=round(ci_lower, 2) if ci_lower is not None else None,
                ci_upper=round(ci_upper, 2) if ci_upper is not None else None,
            )
        )
    entries.sort(key=lambda e: (-(e.bt_rating or 0.0), e.competitor_id))
    for i, entry in enumerate(entries, 1):
        entry.rank = i

    metric_ladders: dict[str, MetricLadder] = {}
    primary_metric = PRIMARY_METRIC.get(modality)
    if primary_metric:
        metric_ladders[primary_metric] = MetricLadder(
            metric=primary_metric,
            higher_is_better=metric_higher_is_better(primary_metric),
            auto_only=False,
            entries=[
                MetricLadderEntry(
                    rank=e.rank, competitor_id=e.competitor_id,
                    plugin_id=e.plugin_id, bt_rating=e.bt_rating or 0.0,
                    battles=e.battles, wins=e.wins, losses=e.losses, ties=e.ties,
                )
                for e in entries
            ],
        )
    if seed is not None:
        for metric, sec in seed.secondary_metrics.items():
            metric_ladders[metric] = _build_secondary_ladder(
                metric, sec, competitor_plugin
            )

    return EloBoard(
        modality=modality,
        lang=lang,
        generated_at=_now_iso(),
        vote_count=(seed.auto_vote_count if seed else 0) + counted,
        human_vote_count=counted,
        provisional=counted < PROVISIONAL_MIN_HUMAN_VOTES,
        entries=entries,
        metric_ladders=metric_ladders,
    )


def _build_secondary_ladder(
    metric: str, sec: SecondaryMetricSeed, competitor_plugin: dict[str, str]
) -> MetricLadder:
    """Fit a fresh, auto-only BT ladder from one metric's stored pairwise
    seed totals (§ per-metric ladders) — cheap, deterministic, no bootstrap
    (there is no human vote log to resample for this metric)."""
    competitors = sorted(
        set(sec.battles) | set(sec.pairwise_wins) | set(sec.pairwise_games)
    )
    strengths = fit_bradley_terry(sec.pairwise_wins, sec.pairwise_games, competitors)
    ratings = to_rating_scale(strengths)
    ladder_entries = [
        MetricLadderEntry(
            competitor_id=c,
            plugin_id=competitor_plugin.get(c, ""),
            bt_rating=round(ratings.get(c, 1200.0), 2),
            battles=sec.battles.get(c, 0),
            wins=sec.wins.get(c, 0),
            losses=sec.losses.get(c, 0),
            ties=sec.ties.get(c, 0),
        )
        for c in competitors
    ]
    ladder_entries.sort(key=lambda e: (-e.bt_rating, e.competitor_id))
    for i, entry in enumerate(ladder_entries, 1):
        entry.rank = i
    return MetricLadder(
        metric=metric,
        higher_is_better=sec.higher_is_better,
        auto_only=True,
        entries=ladder_entries,
    )


def _sync_leaderboard_with_seed(
    board_path: Path, modality: str, lang: str, seed: EloSeed
) -> None:
    """Make sure every seeded fighter appears on an already-existing board.

    ``cmd_assemble`` only ever *creates* ``leaderboard-<mod>-<lang>.json``
    once — after that, ``tally`` is normally the one rewriting it from a
    full vote replay. But ``tally`` only rewrites boards on a run where at
    least one vote was counted *anywhere*, so a fighter onboarded after the
    board already existed can be permanently missing if that (modality,
    lang) board never collects a human vote.

    When the on-disk board has no human votes yet, it is safe to fully
    regenerate it from the current seed (identical to what the bootstrap
    path above would produce) — there is no replayed vote state to lose.
    When it already carries real human votes, ``assemble`` cannot safely
    reconstruct the full Bradley-Terry replay (it has no battles pool or
    vote log here), so missing fighters are appended at their seed rating
    instead, leaving every existing entry untouched.
    """
    try:
        payload = json.loads(board_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", board_path, exc)
        return

    existing_ids = {e["competitor_id"] for e in payload.get("entries", [])}
    missing = sorted(set(seed.ratings) - existing_ids)

    if not payload.get("entries") or payload.get("human_vote_count", 0) == 0:
        # A vote-free board is a pure function of the seed, so it is
        # resynced whenever EITHER the roster or the seed's numbers moved.
        # Keying this on missing fighters alone left a board describing a
        # superseded seed with the same roster — and `verify-replay`
        # rebuilds the published board from the committed seed, so that
        # board goes red until the next vote tally happens to rewrite it.
        _write_json(board_path, build_elo_board(modality, lang, seed, [], {}))
        log.info("Resynced %s from seed (%d previously-missing fighter(s)%s)",
                 board_path.name, len(missing),
                 ": " + ", ".join(missing) if missing else "")
        return

    if not missing:
        return

    entries = payload.setdefault("entries", [])
    for competitor in missing:
        rating = seed.ratings.get(competitor, 1200.0)
        battles = seed.battles.get(competitor, 0)
        wins = seed.wins.get(competitor, 0)
        losses = seed.losses.get(competitor, 0)
        ties = seed.ties.get(competitor, 0)
        entries.append({
            "rank": 0,
            "competitor_id": competitor,
            "plugin_id": seed.competitor_plugin.get(competitor, ""),
            "elo": round(rating, 2),
            "battles": battles,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(wins / battles, 4) if battles else 0.0,
            "human_votes": 0,
            "auto_votes": battles,
            "bt_rating": round(rating, 2),
            # An appended fighter has cast-in no human votes, and with zero
            # human votes a bootstrap interval collapses to the seed-only
            # point estimate (see arena.rating.bootstrap_confidence_
            # intervals), so that is the honest value here. It is not the
            # interval a full replay would give the fighter once votes
            # arrive — this branch cannot run one, since a leaderboard
            # stores per-fighter aggregates and not the pairwise matrices
            # the fit needs, and the vote log is not available to
            # `assemble`. The next `tally` recomputes it properly. What
            # matters is that the shape stays the one `build_elo_board`
            # produces: null CIs made the whole board unreplayable.
            "ci_lower": round(rating, 2),
            "ci_upper": round(rating, 2),
        })
    entries.sort(key=lambda e: (-(e.get("bt_rating") or 0.0), e["competitor_id"]))
    for i, entry in enumerate(entries, 1):
        entry["rank"] = i
    _write_json_payload(board_path, payload)
    log.info("Appended %d previously-missing fighter(s) to %s: %s",
              len(missing), board_path.name, ", ".join(missing))


def _unchanged(path: Path, payload: dict[str, Any]) -> bool:
    """True when *payload* matches the file on disk apart from ``generated_at``.

    Keeps artifact timestamps stable: an identical regeneration is not
    rewritten, so the workflows' ``git diff --cached --quiet`` guards skip
    the commit instead of churning ``generated_at``-only diffs.
    """
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(existing, dict):
        return False
    drop = "generated_at"
    return ({k: v for k, v in existing.items() if k != drop}
            == {k: v for k, v in payload.items() if k != drop})


def _board_disk_input_hash(path: Path) -> str | None:
    """The ``input_hash`` already on disk for a benchmark board — but ONLY
    when the file is also internally self-consistent, so a rerun stays
    self-healing (§TestTimestampStability.test_changed_content_still_
    rewrites — a board hand-edited/corrupted on disk without going through
    this tool must still get fixed by the next assemble, never trusted
    just because its ``input_hash`` field happens to still read as valid).
    None when the file is missing/unreadable/pre-dates the field/was
    tampered with. A cheap ``json.load`` + rehash of the file's OWN
    ``entries`` — no pydantic validation, no bootstrap CI.
    """
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(existing, dict):
        return None
    value = existing.get("input_hash")
    if not isinstance(value, str):
        return None
    entries_hash = existing.get("entries_hash")
    if not isinstance(entries_hash, str):
        return None
    actual = hashlib.sha256(
        json.dumps(existing.get("entries"), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if actual != entries_hash:
        return None
    return value


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    if _unchanged(path, payload):
        log.info("Unchanged %s", path)
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    log.info("Wrote %s", path)


def _write_json(path: Path, model) -> None:
    _write_json_payload(path, model.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def _prediction_source_langs(prediction_sources: list[str]) -> dict[str, str | None]:
    """Map each prediction source (HF repo id or local dir) to the single
    concrete lang it must be loaded under, or ``None`` for a genuinely
    multi/unknown-lang dataset (every lang dir stays in scope for those).

    Mirrors the repo-naming convention in
    ``registry.loaders.list_prediction_repos``: a dataset's own
    ``predictions_hf`` repo, plus (for intent modalities) one
    ``ovos-intent-<paradigm>-bench-<dataset_id>`` repo per paradigm the
    corpus feeds. All of them carry that dataset's lang.
    """
    langs: dict[str, str | None] = {}
    try:
        from registry.loaders import list_datasets
        from registry.schemas import INTENT_MODALITIES
        from runner.queue_tools import resolved_dataset_lang
    except ImportError:
        return {s: None for s in prediction_sources}

    for dataset in list_datasets():
        if not dataset.predictions_hf:
            continue
        lang = resolved_dataset_lang(dataset) if dataset.lang != "multi" else None
        langs[dataset.predictions_hf] = lang
        if dataset.modality in INTENT_MODALITIES:
            from registry.loaders import paradigm_league_repo
            for paradigm in dataset.train_datasets or {}:
                langs[paradigm_league_repo(dataset, paradigm)] = lang
    return {s: langs.get(s) for s in prediction_sources}


_SAMPLE_SET_CACHE: dict[tuple[str, str, str], set[str] | None] = {}


def _load_sample_set(modality: str, dataset_id: str, lang: str) -> set[str] | None:
    """Load a published ``sample_sets/<lang>.json`` manifest's id set for
    one (modality, dataset, lang), or ``None`` when the dataset carries no
    ``sample_policy`` or the manifest hasn't been published yet — in which
    case the caller falls back to unfiltered scoring (§comparability gap,
    board build stays best-effort rather than failing outright) and
    ``build_benchmark_board`` marks every entry ``sample_set="unmanaged"``.
    """
    key = (modality, dataset_id, lang)
    if key in _SAMPLE_SET_CACHE:
        return _SAMPLE_SET_CACHE[key]

    result: set[str] | None = None
    try:
        from registry.loaders import load_dataset

        dataset = load_dataset(modality, dataset_id)
        if dataset.sample_policy is not None:
            from huggingface_hub import hf_hub_download

            from runner.intent_bench import results_repo_for

            repo = dataset.predictions_hf or results_repo_for(modality, dataset_id)
            lang_file = lang.replace("-", "_")
            local = hf_hub_download(
                repo, f"sample_sets/{lang_file}.json",
                repo_type="dataset", revision="main",
            )
            with open(local, encoding="utf-8") as fh:
                manifest = json.load(fh)
            result = set(manifest["sample_ids"])
    except Exception as exc:
        log.warning(
            "%s/%s/%s: sample_policy is set but no sample_sets manifest is "
            "published yet (%s) — scoring unfiltered this run; publish one "
            "with runner.publish_sample_set for a comparable board",
            modality, dataset_id, lang, exc,
        )
        result = None

    _SAMPLE_SET_CACHE[key] = result
    return result


def _dataset_info_lookup(prediction_sources: list[str]) -> dict[str, dict[str, Any]]:
    """Registry metadata per dataset_id, for the benchmark board UI."""
    info: dict[str, dict[str, Any]] = {}
    try:
        from registry.loaders import list_datasets
    except ImportError:
        return info
    hf_repos = [s for s in prediction_sources if not Path(s).is_dir()]
    datasets = list_datasets()
    by_id = {d.dataset_id: d for d in datasets}
    for dataset in datasets:
        entry: dict[str, Any] = {}
        hf_id = getattr(dataset.source, "hf_id", None)
        if hf_id:
            entry["url"] = f"https://huggingface.co/datasets/{hf_id}"
        if dataset.license:
            entry["license"] = dataset.license
        if dataset.notes:
            entry["notes"] = dataset.notes
        own_repos = [r for r in hf_repos
                     if r.endswith(f"-bench-{dataset.dataset_id}")]
        if dataset.predictions_hf:
            own_repos = sorted(set(own_repos) | {dataset.predictions_hf})
        if own_repos:
            entry["predictions"] = [
                f"https://huggingface.co/datasets/{r}" for r in own_repos]
        if dataset.train_datasets:
            trains: dict[str, Any] = {}
            for paradigm, train_id in dataset.train_datasets.items():
                train_def = by_id.get(train_id)
                train_hf = (getattr(train_def.source, "hf_id", None)
                            if train_def else None)
                trains[paradigm] = {
                    "dataset_id": train_id,
                    **({"url": f"https://huggingface.co/datasets/{train_hf}"}
                       if train_hf else {}),
                }
            entry["train_datasets"] = trains
        info[dataset.dataset_id] = entry
    return info


def _predictions_revision_for(source: str, default_revision: str) -> tuple[str, dict[str, Any]]:
    """The revision to pin *source* to for reproducible assembly (§C).

    Uses the source dataset's registry ``predictions_revision`` pin when one
    exists, else *default_revision* (typically ``--revision``, often
    ``"main"``). Resolves whichever revision is chosen to an immutable HF
    commit SHA via ``arena.predictions.resolve_predictions_revision``, so
    the board's provenance is a fixed commit rather than a floating ref.

    Returns ``(revision_to_fetch, meta)`` where *meta* carries the resolved
    SHA (``meta["resolved_sha"]``) when resolution succeeded, or is empty
    when the source is a local directory or resolution failed (in which
    case the caller falls back to fetching at *revision_to_fetch* as-is).

    A registry pin that no longer resolves gets no such fallback: falling
    back would fetch the repo's default ref under a provenance claiming the
    pinned one, or — once the pin string reaches the fetch — drop a source
    with live rows as if it held none. ``RevisionNotFoundError`` therefore
    propagates so the caller records the source as failed and the operator
    sees which pin went stale.

    A transient resolution failure (rate limiting, a 5xx, a timeout) is
    retried inside ``resolve_predictions_revision`` before it ever reaches
    here (see its docstring). Anything that still escapes that retry is
    classified exactly as ``fetch_hf_predictions`` classifies a fetch
    failure (``arena.predictions._is_missing``/``_is_unreadable``): a repo
    that simply does not exist yet (``RepositoryNotFoundError``, or the
    matching unauthenticated 401/404) is missing data, not a failed
    source, so resolution falls back to fetching *default_revision* as-is
    — the later fetch call independently reaches the same "no predictions
    published yet" conclusion and yields no rows, exactly as if this
    function had never run. Everything else (gated, private, or a failure
    with no known non-transient, non-fatal cause) propagates: returning
    the unresolved ref for it would silently degrade the board's
    provenance from a pinned commit to a floating ref with only a log
    line to show for it, so the caller records the source as failed
    instead of fetching it unpinned.
    """
    if Path(source).is_dir():
        return default_revision, {}

    from huggingface_hub.utils import RevisionNotFoundError

    revision = default_revision
    pinned = False
    try:
        from registry.loaders import list_datasets

        for dataset in list_datasets():
            if dataset.predictions_hf == source and dataset.predictions_revision:
                revision = dataset.predictions_revision
                pinned = True
                break
    except Exception as exc:
        log.warning("Could not consult registry for %s pin: %s", source, exc)

    try:
        from arena.predictions import _is_missing, resolve_predictions_revision

        sha = resolve_predictions_revision(source, revision=revision)
        return sha, {"resolved_sha": sha}
    except RevisionNotFoundError:
        if pinned:
            log.warning(
                "Pinned predictions revision %s@%s no longer exists on the Hub",
                source, revision,
            )
            raise
        log.warning("Could not resolve %s@%s to a commit SHA", source, revision)
        return revision, {}
    except Exception as exc:
        if _is_missing(exc):
            log.info(
                "No predictions repo yet for %s — resolving as absent data, "
                "not a failed source", source,
            )
            return revision, {}
        log.warning(
            "Could not resolve %s@%s to a commit SHA after retrying: %s "
            "— refusing to fetch it unpinned", source, revision, exc,
        )
        raise


def cmd_assemble(args: argparse.Namespace) -> int:
    from arena.predictions import (
        fetch_hf_predictions,
        group_rows,
        iter_predictions,
        iter_predictions_dir,
        read_jsonl,
    )

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.predictions.split(",") if s.strip()]
    explicit_local_dirs = [s for s in sources if Path(s).is_dir()]
    if not sources:
        # Registry-driven default: every eval dataset's predictions_hf repo.
        # Scoped to --modality when given, so a modality-scoped assemble
        # resolves+downloads only the repos it will actually write boards
        # for, instead of every repo in the registry (§assemble
        # scalability — an unscoped default here was the dominant cost of
        # a full assemble, and made even a single-modality run pay for
        # every OTHER modality's HF round trips too).
        from registry.loaders import list_prediction_repos

        sources = list_prediction_repos(modality=args.modality or None)
        log.info("No --predictions given — using %d registry prediction "
                 "repos%s", len(sources),
                 f" for modality {args.modality!r}" if args.modality else "")
    dataset_info = _dataset_info_lookup(sources)
    source_langs = _prediction_source_langs(sources)
    written_files: set[str] = set()

    # §C reproducibility — pin every HF predictions source to an immutable
    # commit SHA (registry ``predictions_revision`` when set, else
    # ``--revision``) and record the resolved mapping for the boards.
    resolved_revisions: dict[str, str] = {}
    fetch_revisions: dict[str, str] = {}
    # A source whose data could not be reached at all. It contributes
    # nothing, and every lang it could have contributed to refuses to
    # publish rather than ship a silently thinner board — except a
    # concrete-lang source (``source_langs[source]`` set), whose scope is
    # known statically from the registry with no network, so only ITS lang
    # is affected (``lang_failed_sources`` below), never every other lang
    # too.
    unreadable_sources: list[str] = []
    for source in sources:
        try:
            fetch_revision, meta = _predictions_revision_for(source, args.revision)
        except Exception:
            # A stale pin (``RevisionNotFoundError``) and an exhausted
            # transient-retry both mean this source's data could not be
            # reached at all, so both are treated the same: recorded
            # failed rather than fetched unpinned (see
            # ``_predictions_revision_for``).
            unreadable_sources.append(source)
            continue
        fetch_revisions[source] = fetch_revision
        if meta.get("resolved_sha"):
            resolved_revisions[source] = meta["resolved_sha"]

    # §assemble memory — the whole rest of this function processes ONE LANG
    # AT A TIME instead of loading every source's (or even one large
    # multi-lang source's) raw rows into memory before grouping/writing
    # anything. A large multi-dataset league (e.g. intent_template's 17
    # predictions repos) — and, within it, a single multi-lang dataset repo
    # bundling every lang's shards together (intents-for-eval: 989 files
    # across ~15 langs, ~1.7M rows if loaded as one structure) — held every
    # raw ``PredictionRow`` object for the ENTIRE modality (or, even just
    # loading per-source/per-lang-chunk with a single shared ``grouped``
    # kept across the whole run, the entire multi-lang dataset) in memory
    # at once. That is what OOM-killed the hosted runner's 7GB matrix leg
    # (proved locally: capped at 7GB via a cgroup, the process is
    # SIGKILLed by the kernel OOM killer — first during the initial
    # unchunked load, and still during the per-source-chunked load,
    # because ``grouped`` itself kept accumulating every lang of
    # intents-for-eval before any board/battle/ELO artifact was written
    # and its rows released).
    #
    # ``elo-seed``/``leaderboard`` artifacts are the only ones that merge
    # data ACROSS datasets, and they are scoped to (battle_group, lang) —
    # never across langs. So the true minimum working set for correctness
    # is one lang's rows across every source, not a whole source's (or the
    # whole modality's) rows. Each source's local predictions dir is
    # resolved once (a cached HF snapshot download, or the local dir
    # itself — no repeat network fetches) and its lang scope is
    # discovered: a concrete-lang source (``source_langs[source]`` set)
    # contributes just that one lang; a multi-lang source contributes
    # every lang subdirectory it actually publishes. The lang loop below
    # then loads + groups + builds + writes + discards one lang's data
    # before moving to the next, so peak memory is bounded by one lang's
    # rows (across every source that publishes it) plus that lang's
    # board/battle/ELO artifacts — never the whole modality's.
    source_dirs: dict[str, Path] = {}
    target_langs: set[str] = set()
    # A source that could not even be listed has an unknown lang scope, so
    # it potentially contributes to every lang — see the partial-input
    # guard in the lang loop below. A concrete-lang source's scope is known
    # statically (``source_langs``), so its pre-load failure is scoped to
    # just that lang via ``lang_failed_sources`` instead.
    undiscovered_sources: list[str] = [
        s for s in unreadable_sources if not source_langs.get(s)]
    lang_failed_sources: dict[str, list[str]] = {}
    for source in unreadable_sources:
        lang = source_langs.get(source)
        if lang:
            target_langs.add(lang)
            lang_failed_sources.setdefault(lang, []).append(source)
    for source in sources:
        if source in unreadable_sources:
            continue
        lang = source_langs.get(source)
        if lang:
            target_langs.add(lang)
            continue
        try:
            path = Path(source)
            if not path.is_dir():
                fetched = fetch_hf_predictions(source, fetch_revisions[source])
                if fetched is None:
                    # Registered but never swept — no predictions repo
                    # exists yet. Absent data, not a failed source.
                    continue
                path = fetched
            source_dirs[source] = path
            subdirs = [p.name for p in path.iterdir() if p.is_dir()]
            if subdirs:
                target_langs.update(subdirs)
            else:
                # Flat legacy layout (no per-lang subdirs, e.g. an old
                # repo or a test fixture) — the lang scope can't be read
                # off directory names, so peek the rows themselves. Safe
                # to read fully here: this layout is always small (it is
                # never the large per-lang-sharded layout the memory
                # bound above targets — that one always ships as
                # ``predictions/<lang>/*.jsonl``).
                for jsonl_path in sorted(path.glob("*.jsonl")):
                    for row in read_jsonl(jsonl_path):
                        if row.lang:
                            target_langs.add(row.lang)
        except Exception as exc:
            log.error("Skipping %s: %s", source, exc)
            # Reached only for sources with no concrete registry lang (the
            # ``if lang:`` branch above already ``continue``d those), so
            # this failure's scope really is unknown/multi.
            undiscovered_sources.append(source)

    now = _now_iso()
    registry_dataset_langs = _registry_dataset_langs()
    registry_dataset_revisions = _registry_dataset_revisions()
    ww_phrases = _wakeword_phrases()
    unregistered_competitors: dict[str, int] = {}
    all_seen_modalities: set[str] = set()
    any_data = False

    degraded_langs: dict[str, list[str]] = {}

    for target_lang in sorted(target_langs):
        grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        failed_sources: list[str] = (
            list(undiscovered_sources) + lang_failed_sources.get(target_lang, []))
        for source in sources:
            if source in unreadable_sources:
                continue
            src_lang = source_langs.get(source)
            if src_lang and src_lang != target_lang:
                continue
            try:
                if src_lang == target_lang:
                    chunks = iter_predictions(
                        source, revision=fetch_revisions[source], lang=target_lang)
                else:
                    src_dir = source_dirs.get(source)
                    if src_dir is None:
                        continue
                    chunks = iter_predictions_dir(src_dir, lang=target_lang)
                for chunk in chunks:
                    if not chunk:
                        continue
                    chunk_grouped = group_rows(chunk, unregistered=unregistered_competitors)
                    del chunk
                    for key, samples in chunk_grouped.items():
                        dest = grouped.setdefault(key, {})
                        for sample_id, comp_rows in samples.items():
                            dest.setdefault(sample_id, {}).update(comp_rows)
                    del chunk_grouped
            except Exception as exc:
                log.error("Skipping %s (%s): %s", source, target_lang, exc)
                failed_sources.append(source)
                continue

        # Partial input must never be published. Every artifact this loop
        # writes for a lang — the ELO seed above all — is a merge across
        # ALL of that lang's sources, so building it from the survivors of
        # a failed fetch silently rewrites public ratings downwards (run
        # 33927515324: one 429 on the snips predictions repo dropped
        # en-US's auto vote count from 8,521,627 to 7,639,078 and took
        # battles away from all 74 fighters). Leave the committed
        # artifacts alone and fail the run instead — same posture as the
        # "nothing to assemble" branch below, which also declines to
        # write rather than write something wrong.
        if failed_sources:
            log.error(
                "Refusing to publish %s artifacts — %d source(s) failed to "
                "load: %s. Existing committed artifacts left untouched.",
                target_lang, len(failed_sources), ", ".join(sorted(set(failed_sources))),
            )
            degraded_langs[target_lang] = sorted(set(failed_sources))
            del grouped
            continue

        if not grouped:
            continue

        # Legacy daemon rows keyed by the raw hf path (e.g.
        # FBK-MT/Speech-MASSIVE-test/de-DE/test) instead of the canonical
        # registry id: a board or battle filename built from such an id
        # explodes into nonexistent directories. Filter ONCE, before every
        # loop that embeds dataset_id in a filename; the runner-side fix
        # re-keys new rows, and stale rows are re-run/replaced, not shimmed.
        for modality, dataset_id, lang in [k for k in grouped
                                           if "/" in k[1] or "\\" in k[1]]:
            log.warning("skipping non-canonical dataset_id %r (%s/%s): "
                        "contains a path separator", dataset_id, modality, lang)
            del grouped[(modality, dataset_id, lang)]

        # Legacy predictions rows carrying a pre-normalization short lang
        # code (e.g. ``lang: "fr"`` published to an HF predictions repo
        # before the dataset's own registry entry settled on ``fr-FR``)
        # would otherwise keep regenerating a short-code board every
        # single run — no amount of post-hoc pruning survives that, since
        # the very next assemble run recreates the exact key it just
        # deleted. Drop rows whose lang the dataset's registry entry
        # doesn't recognize *at the source*, before any board/battle/seed
        # is built from them. Datasets absent from the registry entirely
        # (predictions dir passed via --predictions with no matching
        # registry entry, e.g. some test fixtures) are left alone — there
        # is nothing canonical to check them against.
        for modality, dataset_id, lang in list(grouped):
            valid_langs = registry_dataset_langs.get(dataset_id)
            if valid_langs is not None and lang not in valid_langs:
                log.warning(
                    "dropping %d sample(s) for %s/%s/%s: %r is not a lang the "
                    "registry's %r entry produces (%s) — pre-normalization "
                    "legacy predictions row(s), re-published under the wrong "
                    "lang at the source",
                    len(grouped[(modality, dataset_id, lang)]), modality,
                    dataset_id, lang, lang, dataset_id, sorted(valid_langs),
                )
                del grouped[(modality, dataset_id, lang)]

        if not grouped:
            continue
        any_data = True
        all_seen_modalities.update(m for (m, _, _) in grouped)

        # Benchmark boards stay per (modality, dataset, lang) — paradigm-pure, so a
        # template engine is never ranked against a keyword engine on metrics.
        for (modality, dataset_id, lang), samples in sorted(grouped.items()):
            if args.modality and modality != args.modality:
                continue
            by_competitor: dict[str, list] = {}
            for sample_rows in samples.values():
                for competitor_id, row in sample_rows.items():
                    by_competitor.setdefault(competitor_id, []).append(row)
            board_file = f"benchmark-{modality}-{dataset_id}-{lang}.json"
            board_path = data_dir / board_file
            sample_set_ids = _load_sample_set(modality, dataset_id, lang)
            input_hash = benchmark_board_input_signature(by_competitor, sample_set_ids)
            # Skip the O(rounds * samples) bootstrap CI entirely when this
            # board's prediction rows (+ scoring logic) are byte-identical to
            # the last assemble run — same input, same logic, same output, so
            # recomputing would just reproduce the file _write_json_payload's
            # _unchanged() would no-op anyway. Only dataset_info/predictions_
            # revisions/generated_at can legitimately differ without a row
            # change, and none of those affect the ranked entries, so reusing
            # the on-disk entries verbatim is safe. sample_set_ids is baked
            # into input_hash above, so a republished manifest also busts
            # this cache — no separate resweep needed.
            if _board_disk_input_hash(board_path) == input_hash:
                written_files.add(board_file)
                log.info("Unchanged %s (input identical — skipped bootstrap)", board_path)
                continue
            board = build_benchmark_board(
                modality, dataset_id, lang, by_competitor, now, input_hash=input_hash,
                sample_set_ids=sample_set_ids,
                dataset_revision=registry_dataset_revisions.get(dataset_id),
            )
            _attach_model_sizes(board, modality)
            board.dataset_info = dataset_info.get(dataset_id)
            own_revisions = {
                src: sha for src, sha in resolved_revisions.items()
                if src.endswith(f"-bench-{dataset_id}")
            }
            board.predictions_revisions = own_revisions or None
            board.entries_hash = hashlib.sha256(
                json.dumps(
                    [e.model_dump(mode="json") for e in board.entries],
                    sort_keys=True, default=str,
                ).encode("utf-8")
            ).hexdigest()
            _write_json(board_path, board)
            written_files.add(board_file)

        # Battles + ELO pool by battle group: every plugin that answered the same
        # stimulus in a language competes, so the intent paradigm leagues merge into
        # one open arena (battles across all plugins, same language).
        battle_samples: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        elo_samples: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = {}
        for (modality, dataset_id, lang), samples in grouped.items():
            if args.modality and modality != args.modality:
                continue
            if modality in VOTELESS_MODALITIES:
                # Vote-less leagues get benchmark boards only, no
                # battles/elo-seed/leaderboard artifacts at all.
                continue
            group = battle_group(modality)
            bs = battle_samples.setdefault((group, dataset_id, lang), {})
            es = elo_samples.setdefault((group, lang), {}).setdefault(dataset_id, {})
            # Same comparability fix as the benchmark board: when this
            # dataset has a published sample-set manifest, battles/ELO are
            # restricted to it too, so a battle is never assembled between
            # two fighters' rows drawn from different sample populations.
            sample_set_ids = _load_sample_set(modality, dataset_id, lang)
            # A row swept against another revision of the corpus is dropped
            # from the board, so it must not reach a battle or seed a rating
            # either — the seeded ladder has to battle over the same
            # population the board's primary metric is computed from.
            pinned_revision = registry_dataset_revisions.get(dataset_id)
            for sample_id, comp_rows in samples.items():
                if sample_set_ids is not None and sample_id not in sample_set_ids:
                    continue
                on_pin = {
                    competitor_id: row
                    for competitor_id, row in comp_rows.items()
                    if row_on_pinned_revision(row, pinned_revision)
                }
                if not on_pin:
                    continue
                bs.setdefault(sample_id, {}).update(on_pin)
                es.setdefault(sample_id, {}).update(on_pin)

        for (group, dataset_id, lang), samples in sorted(battle_samples.items()):
            stats: dict[str, int] = {}
            battles = assemble_battles(
                group, dataset_id, lang, samples,
                max_battles=args.max_battles, stats=stats,
            )
            pool = BattlesPool(
                modality=group,
                dataset_id=dataset_id,
                lang=lang,
                generated_at=now,
                dataset_info=dataset_info.get(dataset_id),
                battles=battles,
                skipped_reference_mismatches=stats.get(
                    "skipped_reference_mismatches", 0
                ),
            )
            battles_file = f"battles-{group}-{dataset_id}-{lang}.json"
            _write_json(data_dir / battles_file, pool)
            written_files.add(battles_file)

        for (group, lang), samples_by_dataset in sorted(elo_samples.items()):
            seed = seed_elo(group, lang, samples_by_dataset, now)
            seed.secondary_metrics = seed_secondary_metrics(group, samples_by_dataset)
            elo_seed_file = f"elo-seed-{group}-{lang}.json"
            _write_json(data_dir / elo_seed_file, seed)
            written_files.add(elo_seed_file)

            # Free-form matchup pool: every competitor pair, for direct subjective
            # votes that replay into this same ELO ladder. For wake word, restrict
            # pairs to the same phrase (a 'hey jarvis' detector is not comparable
            # to a 'computer' one).
            subgroups = ww_phrases if group == "wake_word" else None
            pool = BattlesPool(
                modality=group,
                dataset_id="freeform",
                lang=lang,
                generated_at=now,
                battles=freeform_battles(group, lang, seed.competitor_plugin,
                                         subgroups=subgroups),
            )
            freeform_file = f"battles-{group}-freeform-{lang}.json"
            _write_json(data_dir / freeform_file, pool)
            written_files.add(freeform_file)

            # Bootstrap the ELO board when none exists yet; `tally` owns it after
            # — but `tally` only rewrites a board when at least one vote is
            # counted *anywhere* in that run (`cmd_tally`'s `if
            # counted_decisions:` guard), so a fighter that only ever gets
            # prediction rows (never a human vote) could sit off a leaderboard
            # that already existed before it was onboarded, forever. Every
            # assemble run resyncs the board with the current seed to close
            # that gap.
            leaderboard_file = f"leaderboard-{group}-{lang}.json"
            board_path = data_dir / leaderboard_file
            written_files.add(leaderboard_file)
            if not board_path.exists():
                elo_board = build_elo_board(group, lang, seed, [], {})
                _write_json(board_path, elo_board)
            else:
                _sync_leaderboard_with_seed(board_path, group, lang, seed)

        del grouped, battle_samples, elo_samples

    if degraded_langs:
        # Non-zero exit so the workflow step fails. In assemble.yml the
        # per-modality leg's "Upload data delta" step runs only on a
        # successful assemble step, so failing the leg withholds this
        # modality's entire artifact from the commit job — the other
        # modalities' legs still commit their own output, which is the
        # intended sharded behaviour. Stale-artifact pruning is skipped
        # too: written_files is incomplete on a degraded run and pruning
        # against it would delete the very boards this guard preserved.
        log.error("assemble degraded — no artifacts written for %d lang(s): %s",
                  len(degraded_langs), ", ".join(sorted(degraded_langs)))
        return 1

    if not any_data:
        # A league with zero fighters/predictions registered (e.g. a
        # modality nobody has submitted a competitor for yet, like
        # ww_stream) is not a build failure — there is genuinely nothing
        # to assemble, and the matrix leg should complete cleanly instead
        # of failing the whole workflow run every single day.
        scope = f" for modality {args.modality!r}" if args.modality else ""
        if explicit_local_dirs:
            # An explicit --predictions local dir yielding zero rows is not
            # the same "genuinely nothing to assemble" case as the HF/
            # registry-driven path above: it almost always means the
            # directory doesn't match the expected layout
            # (<dir>/<lang-REGION>/<fighter>.jsonl, see
            # ``iter_predictions_dir``) rather than a real empty league, so
            # silently no-op'ing here just hides a wrong path. Fail loudly
            # instead.
            log.error(
                "nothing to assemble%s — 0 rows loaded from %s; expected "
                "<lang-REGION>/<fighter>.jsonl directly under each given "
                "directory (e.g. predictions/snips/intent_template/en-US/"
                "some-fighter.jsonl)",
                scope, ", ".join(explicit_local_dirs),
            )
            return 1
        log.info("nothing to assemble%s — no predictions loaded, "
                 "leaving existing data untouched", scope)
        return 0

    if unregistered_competitors:
        _write_json_payload(
            data_dir / "assemble-summary.json",
            {
                "generated_at": _now_iso(),
                "unregistered_competitors_excluded": dict(
                    sorted(unregistered_competitors.items())
                ),
            },
        )

    # Sub-leagues now merged into a group leave stale per-paradigm battle/ELO
    # files behind; drop them (benchmark boards are kept). Safe to run once
    # here, after every lang's artifacts are written: it only ever deletes
    # ``battles-<sub-league-modality>-*``/``elo-seed-<...>``/
    # ``leaderboard-<...>`` files, which never share a filename prefix with
    # the merged-group files this run just wrote (those are named
    # ``<...>-<battle_group>-*``, always a different string than the
    # sub-league modality unless the sub-league IS its own group, in which
    # case ``_clean_merged_artifacts`` is a no-op for it).
    _clean_merged_artifacts(data_dir, all_seen_modalities)

    modality_scope = {args.modality} if args.modality else None
    pruned = _prune_stale_artifacts(data_dir, written_files, modality_scope)
    if pruned:
        log.info("Pruned %d stale artifact(s): %s", len(pruned), ", ".join(pruned))

    return 0


def _attach_model_sizes(board, modality: str) -> None:
    """Fill in ``entry.model_mb`` for every entry on *board* (M2, §2 model
    size), looked up ONCE per fighter via ``arena.model_size`` (its own
    build-lifetime cache avoids re-fetching the same repo across every
    board that fighter appears on).

    Repo id source, in order: the dedicated ``model_hf_repo`` field when
    set (explicit, always trusted); otherwise the pre-existing ``model``
    field IF it has the ``owner/name`` shape of a HF repo id
    (``arena.model_size.likely_hf_repo_id``) — hundreds of registered
    fighters already carry their HF repo id there (e.g.
    ``"OpenVoiceOS/whisper-tiny-onnx"``) and requiring every one of them to
    be hand-edited with a second, redundant field before getting a model-
    size column would leave the column blank arena-wide. ``model`` values
    that aren't repo-id-shaped (voice ids, coqui model paths, free-text
    descriptions with spaces/parens, …) are never sent to HF as a lookup —
    ``likely_hf_repo_id`` filters those out before ``model_repo_size_mb``
    is even called, and that function separately tolerates a 404/lookup
    failure by returning None. Fighters with neither field set — or whose
    registry entry can't be resolved at all — get ``model_mb = None``,
    never a fabricated 0."""
    try:
        from registry.loaders import list_competitors
    except ImportError:
        return
    from arena.model_size import likely_hf_repo_id, model_repo_size_mb

    repos: dict[str, str | None] = {}
    try:
        for comp in list_competitors(modality):
            repo_id = getattr(comp, "model_hf_repo", None)
            if not repo_id:
                model = getattr(comp, "model", None)
                if likely_hf_repo_id(model):
                    repo_id = model
            repos[comp.competitor_id] = repo_id
    except Exception:
        log.warning("Could not consult registry for %s model_hf_repo", modality, exc_info=True)
        return

    for entry in board.entries:
        repo_id = repos.get(entry.competitor_id)
        if repo_id:
            entry.model_mb = model_repo_size_mb(repo_id)


def _wakeword_phrases() -> dict[str, str]:
    """Map each wake-word competitor id → its phrase (its hotword config key)."""
    phrases: dict[str, str] = {}
    try:
        from registry.loaders import list_competitors
    except ImportError:
        return phrases
    for comp in list_competitors("wake_word"):
        hotwords = (comp.config or {}).get("hotwords") or {}
        if hotwords:
            phrases[comp.competitor_id] = next(iter(hotwords))
    return phrases


def _clean_merged_artifacts(data_dir: Path, modalities: set) -> None:
    """Remove battle/ELO files of sub-leagues that now merge into a group.

    Benchmark boards are kept (paradigm-pure); only the per-sub-league
    ``battles`` / ``elo-seed`` / ``leaderboard`` artifacts are superseded by the
    merged group ones.
    """
    for modality in modalities:
        group = battle_group(modality)
        if group == modality:
            continue
        for prefix in ("battles", "elo-seed", "leaderboard"):
            for path in data_dir.glob(f"{prefix}-{modality}-*.json"):
                path.unlink()
                log.info("Removed superseded %s", path.name)


_PRUNABLE_PREFIXES = ("battles-", "benchmark-", "leaderboard-", "elo-seed-")


def _registry_dataset_langs() -> dict[str, set[str]]:
    """``dataset_id -> {canonical langs it can produce}`` across the whole
    registry (every modality), expanding ``lang == "multi"`` via ``langs``.

    Used only to decide whether a dataset-scoped pool this run's
    ``--modality`` filter (or a transient prediction-fetch failure) didn't
    happen to touch is still a *legitimate* artifact — as opposed to one
    keyed by a dataset/lang the registry can no longer produce at all.
    """
    from registry.loaders import list_datasets

    out: dict[str, set[str]] = {}
    for dataset in list_datasets():
        langs = set(dataset.langs) if dataset.lang == "multi" and dataset.langs \
            else {dataset.lang}
        out.setdefault(dataset.dataset_id, set()).update(langs)
    return out


def _registry_dataset_revisions() -> dict[str, str | None]:
    """``dataset_id -> source.revision`` across the whole registry.

    Board assembly scores a dataset's rows against the revision its registry
    entry pins; see ``arena.metrics.drop_rows_off_pinned_revision``.
    """
    from registry.loaders import list_datasets

    return {d.dataset_id: d.source.revision for d in list_datasets()}


def _registry_battle_groups() -> set[str]:
    """Every battle group the current registry's competitors can produce."""
    from registry.loaders import load_all_competitors

    return {battle_group(comp.modality) for comp in load_all_competitors()}


def _prune_stale_artifacts(
    data_dir: Path,
    written_files: set[str],
    modality_scope: set[str] | None,
) -> list[str]:
    """Delete ``battles-*``/``benchmark-*``/``leaderboard-*``/``elo-seed-*``
    files this run's registry can no longer produce.

    ``written_files`` is exactly the set of filenames the assemble loop
    above just wrote (or re-synced) this run — reusing that set instead of
    a second, independent key-generator means the two can never diverge.
    A file the loop didn't touch this run is still kept if it is a
    dataset-scoped pool (``battles-<group>-<dataset_id>-<lang>.json`` /
    ``benchmark-<modality>-<dataset_id>-<lang>.json``) whose
    ``(dataset_id, lang)`` pair still exists *somewhere* in the registry —
    a freeform/community one-shot pool this run's ``--modality`` filter (or
    a transient HF fetch failure) didn't happen to touch is not "stale",
    only untouched. Only keys the registry genuinely cannot produce
    (pre-normalization short lang codes, pre-#73 unguarded sub-league
    pools, empty dead pools from a since-renamed dataset id) get pruned.
    """
    dataset_langs = _registry_dataset_langs()
    pruned: list[str] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        name = path.name
        prefix = next((p for p in _PRUNABLE_PREFIXES if name.startswith(p)), None)
        if prefix is None:
            continue
        if name in written_files:
            continue
        stem = name[len(prefix):-len(".json")]
        parts = stem.split("-")
        if not parts:
            continue
        modality = parts[0]
        if modality_scope is not None and modality not in modality_scope:
            continue  # out of this run's --modality scope — leave it alone
        rest = parts[1:]
        if prefix == "battles-" and rest and rest[0] == "freeform":
            # Group-scoped freeform matchup pool: key is <group>-<lang>,
            # not dataset-scoped — same liveness check as leaderboard/
            # elo-seed below, just inlined here since the filename's
            # "freeform" segment isn't part of the lang.
            live_langs = {lang for langs in dataset_langs.values() for lang in langs}
            live_groups = _registry_battle_groups()
            lang = "-".join(rest[1:])
            if modality in live_groups and lang in live_langs:
                continue
        elif prefix == "battles-" and rest:
            # Dataset-scoped battle pool: try every dataset_id/lang split
            # point: kept if the registry still produces that pair.
            kept = any(
                "-".join(rest[i:]) in dataset_langs.get("-".join(rest[:i]), ())
                for i in range(1, len(rest))
            )
            if kept:
                continue
        if prefix == "benchmark-" and rest:
            kept = any(
                "-".join(rest[i:]) in dataset_langs.get("-".join(rest[:i]), ())
                for i in range(1, len(rest))
            )
            if kept:
                continue
        if prefix in ("leaderboard-", "elo-seed-"):
            # Board key is <group>-<lang>. Same transient-failure safeguard
            # as the dataset-scoped pools above: keep the file when both the
            # battle group and the lang are still live in the registry —
            # only dead groups (renamed sub-leagues) and dead langs
            # (pre-normalization short codes) are actually stale.
            live_langs = {lang for langs in dataset_langs.values() for lang in langs}
            live_groups = _registry_battle_groups()
            full = [modality, *rest]
            kept = any(
                "-".join(full[:i]) in live_groups
                and "-".join(full[i:]) in live_langs
                for i in range(1, len(full))
            )
            if kept:
                continue
        path.unlink()
        pruned.append(name)
        log.info("Pruned stale artifact %s (registry no longer produces this key)",
                  name)
    return pruned


def cmd_prune_data(args: argparse.Namespace) -> int:
    """Standalone entry point for ``_prune_stale_artifacts`` (§assemble
    scalability — the sharded assemble workflow's commit job).

    Each matrix leg's own ``assemble --modality <m>`` already prunes its
    own modality's stale artifacts, but only inside that leg's ephemeral
    runner workspace — a deletion there is invisible to the commit job,
    whose ``download-artifact --merge-multiple`` step only adds/overwrites
    files onto the dev checkout, never deletes. Run this in the commit job
    AFTER the artifact download and BEFORE the exports, once, over the
    merged data dir: with no ``written_files`` from a live assemble run
    to skip, every prunable file gets checked fresh against the current
    registry — a strict superset of what each leg's own in-run check does,
    so still correct — and it stays registry-derived, no network needed.
    """
    registry_root = Path(args.registry).resolve()
    if str(registry_root.parent) not in sys.path:
        sys.path.insert(0, str(registry_root.parent))

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        log.warning("prune-data: %s does not exist — nothing to prune", data_dir)
        return 0
    pruned = _prune_stale_artifacts(data_dir, written_files=set(), modality_scope=None)
    if pruned:
        log.info("Pruned %d stale artifact(s): %s", len(pruned), ", ".join(pruned))
    else:
        log.info("prune-data: nothing to prune")
    return 0


def _prune_stale_badges(out_dir: Path) -> list[str]:
    """Delete ``badges/<modality>/<competitor_id>.svg`` for competitors the
    current registry no longer registers under that modality.

    ``emit_badges`` only ever writes/overwrites the badge for a competitor
    still on a freshly-built board — it never removes one left behind by a
    fighter that was since archived/removed from the registry (e.g. the
    ``piper-*`` voices dropped in the phoonnx migration). Derived straight
    from ``registry.loaders.load_all_competitors`` — never a hardcoded list
    — so a badge is pruned only when the registry genuinely no longer
    produces that (modality, competitor_id) pair.
    """
    from registry.loaders import load_all_competitors

    live: dict[str, set[str]] = {}
    for comp in load_all_competitors():
        live.setdefault(comp.modality.value, set()).add(comp.competitor_id)

    badges_dir = out_dir / "badges"
    if not badges_dir.is_dir():
        return []
    pruned: list[str] = []
    for modality_dir in sorted(badges_dir.iterdir()):
        if not modality_dir.is_dir():
            continue
        valid_ids = live.get(modality_dir.name)
        if valid_ids is None:
            continue  # modality dir itself no longer registered — leave alone
        for svg in sorted(modality_dir.glob("*.svg")):
            competitor_id = svg.stem
            if competitor_id in valid_ids:
                continue
            svg.unlink()
            rel = f"badges/{modality_dir.name}/{svg.name}"
            pruned.append(rel)
            log.info("Pruned stale badge %s (registry no longer produces "
                      "this competitor under %s)", rel, modality_dir.name)
    return pruned


# ---------------------------------------------------------------------------
# tally
# ---------------------------------------------------------------------------


def fetch_vote_issues(repo: str) -> list[dict]:
    """List every ``vote``-labelled issue (open AND closed) via the gh CLI.

    §6/§P5: the vote log **is** the issue history, public and replayable —
    that only holds if every tally run sees the *complete* history, not
    just issues opened since the last run. Fetching ``--state all`` (not
    ``open``) means ``build_elo_board`` genuinely replays the full log
    every time, byte-reproducibly, rather than the leaderboard silently
    losing already-processed votes once their issues are closed.
    """
    result = subprocess.run(
        ["gh", "issue", "list",
         "--repo", repo,
         "--label", "vote",
         "--state", "all",
         "--limit", "5000",
         "--json", "number,title,author,createdAt,state,labels"],
        capture_output=True, timeout=90,
    )
    if result.returncode != 0:
        log.warning("gh issue list failed: %s", result.stderr.decode())
        return []
    return json.loads(result.stdout.decode())


def fetch_account_created_at(login: str) -> str | None:
    """GitHub account creation timestamp for *login*, or None on failure.

    Only called for authors missing from the persisted age cache (§4
    A1.4) — every subsequent tally run reuses the cached value instead of
    re-fetching, so replay stays network-free and deterministic.
    """
    result = subprocess.run(
        ["gh", "api", f"users/{login}", "--jq", ".created_at"],
        capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        log.warning("Could not fetch account age for %s: %s", login, result.stderr.decode())
        return None
    value = result.stdout.decode().strip()
    return value or None


def close_issue(repo: str, number: int, comment: str, add_label: str = "") -> None:
    try:
        if add_label:
            subprocess.run(
                ["gh", "issue", "edit", str(number), "--repo", repo,
                 "--add-label", add_label],
                capture_output=True, timeout=30,
            )
        subprocess.run(
            ["gh", "issue", "close", str(number), "--repo", repo,
             "--comment", comment],
            capture_output=True, timeout=30,
        )
    except Exception as exc:
        log.warning("Could not close issue #%d: %s", number, exc)


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _account_age_cache(data_dir: Path, authors: set[str]) -> dict[str, str]:
    """Load the persisted GitHub account-creation-date cache, fetching any
    missing author via the network exactly once and extending the cache
    (§4 A1.4 — ingest touches the network, replay never does)."""
    cache_path = data_dir / "voter-age-cache.json"
    cache = _load_json_dict(cache_path)
    missing = sorted(a for a in authors if a and a not in cache)
    for login in missing:
        created = fetch_account_created_at(login)
        if created:
            cache[login] = created
    if missing:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return cache


def cmd_tally(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    battles_pool = load_battles_pools(data_dir)
    seeds = load_elo_seeds(data_dir)
    log.info("Loaded %d battles, %d ELO seeds", len(battles_pool), len(seeds))

    issues: list[dict] = []
    if args.repo:
        log.info("Fetching vote issues from %s …", args.repo)
        issues = fetch_vote_issues(args.repo)
        log.info("  → %d vote issue(s) (open + closed)", len(issues))
    open_issue_numbers = {
        issue["number"] for issue in issues if issue.get("state") == "OPEN"
    }

    raw_votes: list[dict] = []
    invalid: list[tuple[int, str]] = []
    for issue in issues:
        number = issue["number"]
        author = (issue.get("author") or {}).get("login", "unknown")
        parsed = parse_vote_title(issue.get("title", ""))
        if parsed is None:
            invalid.append((number, "This issue does not match the vote title "
                                    "format `vote|<battle_id>|<choice>`."))
            continue
        battle_id, choice = parsed
        if battle_id not in battles_pool:
            invalid.append((number, f"Battle `{battle_id}` is not in the "
                                    "current battles pool."))
            continue
        raw_votes.append({
            "issue_number": number,
            "battle_id": battle_id,
            "choice": choice,
            "author": author,
            "created_at": issue.get("createdAt", ""),
        })

    votes = dedupe_votes(raw_votes)
    duplicates = {v["issue_number"] for v in raw_votes} - {
        v["issue_number"] for v in votes
    }
    log.info("  → %d deduped vote(s) (%d duplicates, %d invalid)",
             len(votes), len(duplicates), len(invalid))

    # §4 A1.4 vote fraud rules — daily cap, account-age gate, one-sided
    # downweight. Only the age-gate cache lookup touches the network
    # (ingest); resolve_vote_weights itself is pure and replay-deterministic.
    modality_by_battle = {
        bid: b["modality"] for bid, b in battles_pool.items()
    }
    account_created_at = (
        _account_age_cache(data_dir, {v["author"] for v in votes})
        if args.repo else {}
    )
    decisions = resolve_vote_weights(votes, modality_by_battle, account_created_at)
    counted_decisions = [d for d in decisions if d.weight > 0]
    discarded_decisions = [d for d in decisions if d.weight <= 0]
    log.info("  → %d counted vote(s), %d discarded by fraud rules",
             len(counted_decisions), len(discarded_decisions))

    # Group votes per (modality, lang) board, carrying each vote's
    # fraud-rule weight through to the rating.
    votes_by_board: dict[tuple[str, str], list[dict]] = {}
    for d in counted_decisions:
        battle = battles_pool[d.vote["battle_id"]]
        key = (battle["modality"], battle["lang"])
        votes_by_board.setdefault(key, []).append({**d.vote, "weight": d.weight})

    # Rebuild every board unconditionally — `build_elo_board` is a pure
    # function of (seed, votes, battles_pool), and the seed itself can
    # change between tally runs (a same-day `assemble` re-run loads fresh
    # predictions and regenerates `elo-seed-*.json` with a different
    # auto-battle tally) even when zero human votes were cast this run.
    # Gating the rebuild on `counted_decisions` left a stale published
    # board next to a fresh seed/battles pool whenever votes were zero,
    # which breaks R19 (verify-replay reproducibility) the moment the
    # underlying predictions grow — the workflow's git-diff guard already
    # skips the commit when nothing actually changed, so this costs
    # nothing when the seed is unchanged.
    boards = set(seeds) | set(votes_by_board)
    patch_note_entries: list[dict] = []
    for modality, lang in sorted(boards):
        board = build_elo_board(
            modality, lang,
            seeds.get((modality, lang)),
            votes_by_board.get((modality, lang), []),
            battles_pool,
        )
        board_path = out_dir / f"leaderboard-{modality}-{lang}.json"
        # Diff against the board currently on disk (git-tracked) before we
        # overwrite it, so patch notes reflect this run's movement (§A5.4).
        patch_note_entries.extend(diff_board(load_board(board_path), board))
        _write_json(board_path, board)
        # Emit embeddable rank badges (growth loop, §A5.3).
        emit_badges(board, out_dir)
    pruned_badges = _prune_stale_badges(out_dir)
    if pruned_badges:
        log.info("Pruned %d stale badge(s): %s", len(pruned_badges),
                  ", ".join(pruned_badges))
    _write_json_payload(
        out_dir / "patch-notes.json",
        build_patch_notes(patch_note_entries, _now_iso()),
    )
    if not counted_decisions:
        log.info("No counted votes this run — boards still rebuilt from "
                  "the current seed/battles pool (may be unchanged).")

    # Discards are recorded, never silently dropped (§4 A1.4) — this file
    # reflects the complete current vote log's audit trail every run.
    audit_path = out_dir / "vote-audit.json"
    audit_payload = {
        "generated_at": _now_iso(),
        "counted": len(counted_decisions),
        "discarded": [
            {"issue_number": d.vote["issue_number"], "author": d.vote["author"],
             "battle_id": d.vote["battle_id"], "reason": d.discarded_reason}
            for d in discarded_decisions
        ],
        "downweighted": [
            {"issue_number": d.vote["issue_number"], "author": d.vote["author"],
             "battle_id": d.vote["battle_id"], "weight": d.weight}
            for d in counted_decisions if d.weight < 1.0
        ],
    }
    _write_json_payload(audit_path, audit_payload)

    # Comment/close only issues not yet actioned (still open) — every prior
    # run's already-closed issues stay untouched even though they're
    # re-fetched every time for full-history replay (R12).
    if args.repo and not args.keep_issues_open:
        for d in counted_decisions:
            number = d.vote["issue_number"]
            if number not in open_issue_numbers:
                continue
            close_issue(
                args.repo, number,
                "Your vote has been counted — thank you! The leaderboard "
                "will reflect it once this run's commit deploys.",
                add_label="processed",
            )
        for d in discarded_decisions:
            number = d.vote["issue_number"]
            if number not in open_issue_numbers:
                continue
            close_issue(
                args.repo, number,
                "Your vote was recorded but did not count toward the "
                f"rating ({d.discarded_reason}).",
                add_label="processed",
            )
        for vote in raw_votes:
            number = vote["issue_number"]
            if number in duplicates and number in open_issue_numbers:
                close_issue(
                    args.repo, number,
                    "Duplicate vote on this battle — your earlier vote was "
                    "already counted.",
                    add_label="processed",
                )
        for number, reason in invalid:
            if number in open_issue_numbers:
                close_issue(args.repo, number, reason, add_label="processed")

    return 0


# ---------------------------------------------------------------------------
# verify-replay
# ---------------------------------------------------------------------------


_BOARD_ENTRY_FIELDS = (
    "rank", "elo", "bt_rating", "battles", "wins", "losses", "ties",
    "win_rate", "human_votes", "auto_votes", "ci_lower", "ci_upper",
)


def _diff_board(replayed: dict[str, Any], published: dict[str, Any]) -> dict[str, Any]:
    """Field-by-field diff between a freshly-replayed board and the
    published one on disk — ``generated_at`` (and any other timestamp) is
    ignored since a byte-identical timestamp is never the point (§P5); the
    *standings* (ratings, ranks, vote counts) are."""
    ignore = {"generated_at"}
    diff: dict[str, Any] = {}
    for key in set(replayed) | set(published):
        if key in ignore or key == "entries":
            continue
        if replayed.get(key) != published.get(key):
            diff[key] = {"replayed": replayed.get(key), "published": published.get(key)}

    r_entries = {e["competitor_id"]: e for e in replayed.get("entries", [])}
    p_entries = {e["competitor_id"]: e for e in published.get("entries", [])}
    entry_diff: dict[str, Any] = {}
    for cid in sorted(set(r_entries) | set(p_entries)):
        r, p = r_entries.get(cid), p_entries.get(cid)
        if r is None:
            entry_diff[cid] = {"missing_from": "replayed"}
            continue
        if p is None:
            entry_diff[cid] = {"missing_from": "published"}
            continue
        field_diffs = {
            f: {"replayed": r.get(f), "published": p.get(f)}
            for f in _BOARD_ENTRY_FIELDS if r.get(f) != p.get(f)
        }
        if field_diffs:
            entry_diff[cid] = field_diffs
    if entry_diff:
        diff["entries"] = entry_diff
    return diff


def cmd_verify_replay(args: argparse.Namespace) -> int:
    """Replay the public vote log from scratch and prove it reproduces the
    published leaderboards exactly (§P5, docs/operations.md "Replaying the
    arena from public logs").

    Reuses the *same* pure replay path as ``tally`` (``dedupe_votes`` →
    ``resolve_vote_weights`` → ``build_elo_board``) — this command never
    reimplements ELO, it only re-runs the existing functions against the
    current vote log and diffs the result against what is already
    committed, instead of overwriting it. Never touches the network for
    the account-age gate: it reads the already-committed
    ``voter-age-cache.json`` as-is, same as replay purity requires
    elsewhere in this module.
    """
    data_dir = Path(args.data_dir)
    battles_pool = load_battles_pools(data_dir)
    seeds = load_elo_seeds(data_dir)
    log.info("Loaded %d battles, %d ELO seeds", len(battles_pool), len(seeds))

    if args.votes_file:
        log.info("Reading vote issues from %s (offline)", args.votes_file)
        issues = json.loads(Path(args.votes_file).read_text())
    elif args.repo:
        log.info("Fetching vote issues from %s …", args.repo)
        issues = fetch_vote_issues(args.repo)
    else:
        log.error("verify-replay needs --votes-file (offline fixture/snapshot) "
                   "or --repo (live GitHub vote log)")
        return 2
    log.info("  → %d vote issue(s)", len(issues))

    raw_votes: list[dict] = []
    for issue in issues:
        parsed = parse_vote_title(issue.get("title", ""))
        if parsed is None:
            continue
        battle_id, choice = parsed
        if battle_id not in battles_pool:
            continue
        raw_votes.append({
            "issue_number": issue["number"],
            "battle_id": battle_id,
            "choice": choice,
            "author": (issue.get("author") or {}).get("login", "unknown"),
            "created_at": issue.get("createdAt", ""),
        })
    votes = dedupe_votes(raw_votes)
    log.info("  → %d deduped vote(s)", len(votes))

    modality_by_battle = {bid: b["modality"] for bid, b in battles_pool.items()}
    account_created_at = _load_json_dict(data_dir / "voter-age-cache.json")
    decisions = resolve_vote_weights(votes, modality_by_battle, account_created_at)
    counted_decisions = [d for d in decisions if d.weight > 0]
    log.info("  → %d counted vote(s), %d discarded by fraud rules",
             len(counted_decisions), len(decisions) - len(counted_decisions))

    votes_by_board: dict[tuple[str, str], list[dict]] = {}
    for d in counted_decisions:
        battle = battles_pool[d.vote["battle_id"]]
        key = (battle["modality"], battle["lang"])
        votes_by_board.setdefault(key, []).append({**d.vote, "weight": d.weight})

    boards = sorted(set(seeds) | set(votes_by_board))
    mismatches: list[tuple[str, dict]] = []
    missing_published: list[str] = []
    checked = 0
    for modality, lang in boards:
        replayed = build_elo_board(
            modality, lang, seeds.get((modality, lang)),
            votes_by_board.get((modality, lang), []), battles_pool,
        )
        published_path = data_dir / f"leaderboard-{modality}-{lang}.json"
        if not published_path.exists():
            missing_published.append(published_path.name)
            continue
        published = json.loads(published_path.read_text())
        diff = _diff_board(replayed.model_dump(mode="json"), published)
        checked += 1
        if diff:
            mismatches.append((published_path.name, diff))

    if missing_published:
        mismatches.append((
            "<not published>",
            {"missing_published_boards": sorted(missing_published)},
        ))

    # Cross-check the audit trail too — a tampered vote-audit.json (e.g. a
    # discard silently dropped or a weight edited by hand) is just as much
    # of an integrity break as a tampered rating, and this is the same
    # replay path that produces it in `tally`.
    audit_path = data_dir / "vote-audit.json"
    if audit_path.exists():
        replayed_audit = {
            "counted": len(counted_decisions),
            "discarded": sorted(
                (
                    {"issue_number": d.vote["issue_number"], "author": d.vote["author"],
                     "battle_id": d.vote["battle_id"], "reason": d.discarded_reason}
                    for d in decisions if d.weight <= 0
                ),
                key=lambda e: e["issue_number"],
            ),
            "downweighted": sorted(
                (
                    {"issue_number": d.vote["issue_number"], "author": d.vote["author"],
                     "battle_id": d.vote["battle_id"], "weight": d.weight}
                    for d in counted_decisions if d.weight < 1.0
                ),
                key=lambda e: e["issue_number"],
            ),
        }
        published_audit_raw = json.loads(audit_path.read_text())
        published_audit = {
            "counted": published_audit_raw.get("counted"),
            "discarded": sorted(
                published_audit_raw.get("discarded", []),
                key=lambda e: e.get("issue_number", 0),
            ),
            "downweighted": sorted(
                published_audit_raw.get("downweighted", []),
                key=lambda e: e.get("issue_number", 0),
            ),
        }
        if replayed_audit != published_audit:
            mismatches.append(("vote-audit.json", {
                "replayed": replayed_audit, "published": published_audit,
            }))

    if mismatches:
        print("REPLAY MISMATCH — replayed standings diverge from the "
              "published leaderboards:\n")
        for path, diff in mismatches:
            print(f"{path}:")
            print(json.dumps(diff, indent=2, sort_keys=True))
            print()
        log.error("verify-replay FAILED — %d board(s)/artifact(s) mismatched", len(mismatches))
        return 1

    log.info("verify-replay OK — %d published board(s) reproduced exactly "
              "by replaying the vote log", checked)
    return 0


# ---------------------------------------------------------------------------
# export-index / export-bestiary
# ---------------------------------------------------------------------------


def _index_entry(path: Path, count_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    counts = {
        "leaderboards": len(payload.get("entries", [])),
        "benchmarks": len(payload.get("entries", [])),
        "battles_pools": len(payload.get("battles", [])),
        "freeform_pools": len(payload.get("battles", [])),
    }
    entry = {
        "file": path.name,
        "modality": payload.get("modality"),
        "dataset_id": payload.get("dataset_id"),
        "lang": payload.get("lang"),
        "generated_at": payload.get("generated_at"),
        "count": counts[count_key],
    }
    if count_key == "leaderboards":
        # §provenance — carry each board's human/auto vote split into
        # index.json so the site can total "N human votes, M auto-judged
        # battles" without fetching every leaderboard-*.json file.
        entry["human_vote_count"] = payload.get("human_vote_count", 0)
        entry["vote_count"] = payload.get("vote_count", 0)
    return entry


def cmd_export_index(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    index: dict[str, Any] = {"generated_at": _now_iso()}
    for key in ("leaderboards", "benchmarks", "battles_pools", "freeform_pools"):
        index[key] = []
    for path in sorted(data_dir.glob("leaderboard-*.json")):
        index["leaderboards"].append(_index_entry(path, "leaderboards"))
    predictions_revisions: dict[str, str] = {}
    for path in sorted(data_dir.glob("benchmark-*.json")):
        index["benchmarks"].append(_index_entry(path, "benchmarks"))
        payload = json.loads(path.read_text())
        predictions_revisions.update(payload.get("predictions_revisions") or {})
    if predictions_revisions:
        # §C reproducibility — top-level {repo: resolved_commit_sha} map
        # gathered from every board, so a third party can re-fetch the
        # exact predictions that produced any row without hunting through
        # individual board files.
        index["predictions_revisions"] = predictions_revisions
    # blind sample battles vs free-form matchup pools (different voting UIs)
    for path in sorted(data_dir.glob("battles-*.json")):
        if "-freeform-" in path.name:
            index["freeform_pools"].append(_index_entry(path, "freeform_pools"))
        else:
            index["battles_pools"].append(_index_entry(path, "battles_pools"))
    index["has_bestiary"] = (data_dir / "competitors.json").exists()
    index["leagues"] = leagues()
    # §provenance — the pairwise weight an auto-judged battle carries
    # relative to a human vote (arena/elo.py BT_AUTO_WEIGHT), so the site
    # can state it without hardcoding a copy of the constant.
    index["auto_vote_weight"] = BT_AUTO_WEIGHT

    out_file = Path(args.output)
    if _unchanged(out_file, index):
        log.info("Unchanged %s", out_file)
        return 0
    out_file.write_text(json.dumps(index, indent=2) + "\n")
    log.info("Wrote %s (%d leaderboards, %d benchmarks, %d battle pools, "
             "%d freeform pools)", out_file, len(index["leaderboards"]),
             len(index["benchmarks"]), len(index["battles_pools"]),
             len(index["freeform_pools"]))
    return 0


def _competitor_prediction_presence(data_dir: Path) -> dict[str, list[dict[str, str]]]:
    """competitor_id -> sorted [{"dataset_id", "lang"}, ...] for every
    (dataset_id, lang) pair where that competitor has at least one row on a
    ``benchmark-<modality>-<dataset_id>-<lang>.json`` board.

    Reads the already-assembled board files instead of re-loading raw
    prediction rows — every board is only ever written from
    ``by_competitor`` (§cmd_assemble), which already required ≥1 row per
    competitor to appear, so board membership is an exact (not
    approximate) proxy for "has predictions" and costs zero extra HF round
    trips."""
    presence: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(data_dir.glob("benchmark-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        dataset_id = payload.get("dataset_id")
        lang = payload.get("lang")
        if not dataset_id or not lang:
            continue
        for entry in payload.get("entries", []):
            cid = entry.get("competitor_id")
            if cid:
                presence.setdefault(cid, set()).add((dataset_id, lang))
    return {
        cid: [
            {"dataset_id": dataset_id, "lang": lang}
            for dataset_id, lang in sorted(pairs)
        ]
        for cid, pairs in presence.items()
    }


def cmd_export_bestiary(args: argparse.Namespace) -> int:
    registry_root = Path(args.registry).resolve()
    if str(registry_root.parent) not in sys.path:
        sys.path.insert(0, str(registry_root.parent))
    from registry.loaders import load_all_competitors
    from registry.schemas import INTENT_MODALITIES

    loaded = load_all_competitors(registry_root=registry_root)
    presence = _competitor_prediction_presence(Path(args.data_dir))
    competitors = []
    for comp in loaded:
        record = comp.model_dump(mode="json", exclude_none=True)
        datasets = presence.get(comp.competitor_id, [])
        record["has_predictions"] = bool(datasets)
        record["prediction_datasets"] = datasets
        competitors.append(record)
    competitors.sort(key=lambda c: (c["modality"], c["competitor_id"]))

    # plugin_id -> family reverse lookup, so the frontend can resolve a
    # grouping key for "ghost" board/battle entries — a competitor_id that
    # appears in historical result data but has no current registry file
    # (e.g. retired by a later "one fighter per X" registry split). Every
    # board/leaderboard row still carries its own `plugin_id`, so a ghost
    # can be folded into its engine's collapsed family card without ever
    # needing a stub registry entry re-created for it.
    #
    # Restricted to intent leagues: intent `family` genuinely collapses
    # config-variant wrappers of one engine onto a shared plugin, so a
    # plugin -> family fallback is sound there. TTS/STT/wake-word/VAD never
    # collapse (each model is its own family == its own competitor_id), and
    # several such competitors legitimately share one `plugin` id (e.g. many
    # Phoonnx voices), so a plugin-keyed map would incorrectly fold distinct
    # per-model ghosts onto whichever one happened to be dumped last.
    plugin_families = {
        comp.plugin: comp.family
        for comp in loaded
        if comp.plugin and comp.family and comp.modality in INTENT_MODALITIES
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json_payload(
        out_file,
        {
            "generated_at": _now_iso(),
            "competitors": competitors,
            "plugin_families": plugin_families,
        },
    )
    log.info("Exported %d competitors", len(competitors))
    return 0


# Same definition regex ``tests/test_spec_coverage.py`` uses to enumerate
# every normative requirement the spec defines — kept in sync deliberately
# so the evidence page's "spec requirements" count always matches what that
# test treats as the authoritative R-number set.
_SPEC_REQUIREMENT_DEF_RE = re.compile(r"\*\*(R\d+[a-z]?)\b")

# Honest note attached to every league's ``fighter_coverage`` block: what
# "on_boards" actually measures (no network access here — this command only
# reads the local data dir + registry), and what "ghost" means so a reviewer
# doesn't mistake a pending sweep for a hidden failure.
_FIGHTER_COVERAGE_NOTE = (
    "on_boards = fighter's competitor_id appears in at least one "
    "benchmark-<modality>-*.json or battles-<modality>-*.json file in the "
    "local data dir (the only offline-derivable proxy for \"has results\"; "
    "this command makes no network calls, so it cannot check HuggingFace "
    "predictions directly). A ghost is a registered fighter with zero rows "
    "yet — most are pending benchmark/battle sweeps still running, not "
    "hidden failures."
)


def _fighters_on_boards(modality: str, data_dir: Path, board_paths: list[Path]) -> set[str]:
    """competitor_ids that appear in this league's benchmark boards or
    battles pools — the offline proxy for "has results" (see
    ``_FIGHTER_COVERAGE_NOTE``)."""
    ids: set[str] = set()
    for path in board_paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for entry in payload.get("entries", []):
            cid = entry.get("competitor_id")
            if cid:
                ids.add(cid)
    for path in sorted(data_dir.glob(f"battles-{modality}-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for battle in payload.get("battles", []):
            for key in ("competitor_a", "competitor_b"):
                cid = battle.get(key)
                if cid:
                    ids.add(cid)
    return ids


def cmd_export_evidence(args: argparse.Namespace) -> int:
    """Build ``evidence.json`` — per-league completeness counts, generated
    straight from the registry and the data dir so the numbers can never
    drift from what the site actually publishes (§P3, §P5).
    """
    registry_root = Path(args.registry).resolve()
    if str(registry_root.parent) not in sys.path:
        sys.path.insert(0, str(registry_root.parent))
    from registry.loaders import load_all_competitors, load_all_datasets

    data_dir = Path(args.data_dir)
    all_competitors = load_all_competitors(registry_root=registry_root)
    all_datasets = load_all_datasets(registry_root=registry_root)
    eval_datasets = [d for d in all_datasets if d.role == "eval"]

    league_rows: list[dict[str, Any]] = []
    for entry in leagues():
        modality = entry["id"]
        fighters = [c for c in all_competitors if c.modality == modality]

        # The two paradigm sub-leagues (§18) don't own eval corpora of their
        # own — they re-use the open "intent" league's eval corpora via
        # ``train_datasets``, publishing to the HF repo declared by the
        # matching ``intent_<paradigm>/`` training corpus (see
        # ``registry.loaders.paradigm_league_repo``).
        paradigm = (
            {"intent_template": "template", "intent_keyword": "keyword"}
            .get(modality)
        )
        league_datasets: list[tuple[str, str | None]]
        if paradigm:
            from registry.loaders import paradigm_league_repo
            league_datasets = []
            for d in eval_datasets:
                if d.modality != "intent" or paradigm not in (d.train_datasets or {}):
                    continue
                league_datasets.append((
                    d.dataset_id,
                    paradigm_league_repo(d, paradigm),
                ))
            league_datasets.sort()
        else:
            league_datasets = sorted(
                ((d.dataset_id, d.predictions_hf)
                 for d in eval_datasets if d.modality == modality),
            )

        board_paths = sorted(data_dir.glob(f"benchmark-{modality}-*.json"))
        board_dataset_ids: set[str] = set()
        for path in board_paths:
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Could not read %s: %s", path, exc)
                continue
            ds_id = payload.get("dataset_id")
            if ds_id:
                board_dataset_ids.add(ds_id)

        leaderboard_paths = sorted(data_dir.glob(f"leaderboard-{modality}-*.json"))

        on_board_ids = _fighters_on_boards(modality, data_dir, board_paths)
        fighter_ids = {c.competitor_id for c in fighters}
        on_boards = len(fighter_ids & on_board_ids)
        fighter_coverage = {
            "registered": len(fighters),
            "on_boards": on_boards,
            "ghosts": len(fighters) - on_boards,
        }

        predictions_links = [
            {
                "dataset_id": dataset_id,
                "predictions_hf": predictions_hf,
                "url": f"https://huggingface.co/datasets/{predictions_hf}",
                "has_predictions": dataset_id in board_dataset_ids,
            }
            for dataset_id, predictions_hf in league_datasets
            if predictions_hf
        ]

        league_rows.append({
            "id": modality,
            "label": entry["label"],
            "fighters": len(fighters),
            "datasets": len(league_datasets),
            "datasets_with_predictions": sum(
                1 for dataset_id, _ in league_datasets
                if dataset_id in board_dataset_ids
            ),
            "benchmark_boards": len(board_paths),
            "elo_leaderboards": len(leaderboard_paths),
            "predictions_links": predictions_links,
            "fighter_coverage": fighter_coverage,
            "spec_anchor": "docs/SPECIFICATION.md#21-leagues-modalities",
        })

    spec_path = Path(args.spec)
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    requirement_ids = sorted(set(_SPEC_REQUIREMENT_DEF_RE.findall(spec_text)))

    total_on_boards = sum(row["fighter_coverage"]["on_boards"] for row in league_rows)
    total_ghosts = sum(row["fighter_coverage"]["ghosts"] for row in league_rows)

    evidence = {
        "generated_at": _now_iso(),
        "leagues": league_rows,
        "totals": {
            "fighters": len(all_competitors),
            "datasets": len(eval_datasets),
            "spec_requirements": len(requirement_ids),
            "fighter_coverage": {
                "registered": sum(
                    row["fighter_coverage"]["registered"] for row in league_rows
                ),
                "on_boards": total_on_boards,
                "ghosts": total_ghosts,
            },
        },
        "notes": {
            "fighter_coverage": _FIGHTER_COVERAGE_NOTE,
        },
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json_payload(out_file, evidence)
    log.info("Exported evidence for %d league(s), %d fighter(s), %d dataset(s), "
              "%d spec requirement(s)", len(league_rows), len(all_competitors),
              len(eval_datasets), len(requirement_ids))
    return 0


def cmd_validate_registry(args: argparse.Namespace) -> int:
    """Strictly validate every registry JSON file; nonzero exit on any error."""
    registry_root = Path(args.registry).resolve()
    if str(registry_root.parent) not in sys.path:
        sys.path.insert(0, str(registry_root.parent))
    from registry.loaders import validate_registry

    errors = validate_registry(registry_root=registry_root)
    if errors:
        for error in errors:
            print(error)
        log.error("%d registry validation error(s)", len(errors))
        return 1
    log.info("Registry OK — every competitor/dataset file validated cleanly.")
    return 0


def cmd_audit_seeds(args: argparse.Namespace) -> int:
    """Diagnostic report on the seed-battle weight cap (§4, A1.3).

    For every ``elo-seed-*.json`` in *data_dir*, lists each competitor pair's
    Bradley-Terry weighted game total and flags pairs sitting at the cap
    (``MAX_AUTO_WEIGHT_PER_PAIR``) — i.e. a benchmark dataset large enough
    that the cap, not the dataset size, is now what bounds how much the
    auto-vote seed can move that pair's rating.
    """
    data_dir = Path(args.data_dir)
    seeds = load_elo_seeds(data_dir)
    if not seeds:
        log.warning("No elo-seed-*.json files found in %s", data_dir)
        return 0

    total_pairs = 0
    capped_pairs = 0
    for (modality, lang), seed in sorted(seeds.items()):
        seen: set[tuple[str, str]] = set()
        rows = []
        for i, games_i in sorted(seed.pairwise_games.items()):
            for j, weight in sorted(games_i.items()):
                a, b = sorted((i, j))
                pair = (a, b)
                if pair in seen:
                    continue
                seen.add(pair)
                is_capped = weight >= MAX_AUTO_WEIGHT_PER_PAIR - 1e-9
                rows.append((pair, weight, is_capped))

        print(f"\n{modality} / {lang} — {len(rows)} scored pair(s), "
              f"{seed.auto_vote_count} auto vote(s)")
        for (a, b), weight, is_capped in sorted(rows, key=lambda r: -r[1]):
            flag = " [CAPPED]" if is_capped else ""
            print(f"  {a} vs {b}: weight={weight:.2f}/{MAX_AUTO_WEIGHT_PER_PAIR:.2f}{flag}")
            total_pairs += 1
            capped_pairs += int(is_capped)

    print(f"\n{capped_pairs}/{total_pairs} pair(s) at the auto-vote weight cap.")
    return 0


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ovos-arena", description="OVOS Plugin Arena CLI"
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("assemble", help="Build battles, benchmark boards and ELO seeds")
    p.add_argument("--predictions", default="",
                   help="Comma-separated HF dataset repo ids or local predictions "
                        "dirs (default: every eval dataset's predictions_hf repo "
                        "from the registry)")
    p.add_argument("--revision", default="main", help="HF revision to pin")
    p.add_argument("--output", default="frontend-static/public/data")
    p.add_argument("--modality", default="", help="Only assemble this modality")
    p.add_argument("--max-battles", type=int, default=200,
                   help="Max battles per (modality, dataset, lang) pool")

    p = sub.add_parser("tally", help="Tally GitHub vote issues into leaderboards")
    p.add_argument("--data-dir", default="frontend-static/public/data")
    p.add_argument("--output", default="frontend-static/public/data")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    p.add_argument("--keep-issues-open", action="store_true",
                   help="Do not close processed issues (dry-run)")

    p = sub.add_parser(
        "verify-replay",
        help="Replay the public vote log and verify it reproduces the "
             "published leaderboards exactly (CI proof, §P5)",
    )
    p.add_argument("--data-dir", default="frontend-static/public/data")
    p.add_argument("--votes-file", default="",
                   help="Offline vote-issue JSON array (fixture/snapshot) "
                        "instead of a live GitHub fetch")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))

    p = sub.add_parser("export-index", help="Regenerate data/index.json")
    p.add_argument("--data-dir", default="frontend-static/public/data")
    p.add_argument("--output", default="frontend-static/public/data/index.json")

    p = sub.add_parser("export-bestiary", help="Flatten registry into competitors.json")
    p.add_argument("--registry", default="registry")
    p.add_argument("--data-dir", default="frontend-static/public/data",
                   help="Assembled data dir to read benchmark boards from, "
                        "for has_predictions/prediction_datasets")
    p.add_argument("--output", default="frontend-static/public/data/competitors.json")

    p = sub.add_parser(
        "export-evidence",
        help="Regenerate data/evidence.json (per-league completeness counts)",
    )
    p.add_argument("--registry", default="registry")
    p.add_argument("--data-dir", default="frontend-static/public/data")
    p.add_argument("--spec", default="docs/SPECIFICATION.md")
    p.add_argument("--output", default="frontend-static/public/data/evidence.json")

    p = sub.add_parser(
        "validate-registry",
        help="Strictly validate every registry competitor/dataset JSON file",
    )
    p.add_argument("--registry", default="registry")

    p = sub.add_parser(
        "audit-seeds",
        help="Report seed-battle Bradley-Terry weight per pair, flagging capped pairs (§4)",
    )
    p.add_argument("--data-dir", default="frontend-static/public/data")

    p = sub.add_parser(
        "prune-data",
        help="Delete battles/benchmark/leaderboard/elo-seed artifacts the "
             "registry no longer produces (standalone form of assemble's "
             "own prune step — for the sharded workflow's commit job)",
    )
    p.add_argument("--registry", default="registry")
    p.add_argument("--data-dir", default="frontend-static/public/data")

    args = parser.parse_args(argv)
    commands = {
        "assemble": cmd_assemble,
        "tally": cmd_tally,
        "verify-replay": cmd_verify_replay,
        "export-index": cmd_export_index,
        "export-bestiary": cmd_export_bestiary,
        "export-evidence": cmd_export_evidence,
        "validate-registry": cmd_validate_registry,
        "audit-seeds": cmd_audit_seeds,
        "prune-data": cmd_prune_data,
    }
    if args.command not in commands:
        parser.print_help()
        sys.exit(1)
    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
