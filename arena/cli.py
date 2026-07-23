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
    fighter-browser UI.
"""
from __future__ import annotations

import argparse
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
)
from arena.badges import emit_badges
from arena.elo import EloLedger
from arena.fraud import resolve_vote_weights
from arena.metrics import build_benchmark_board
from arena.models import (
    BattlesPool,
    EloBoard,
    EloEntry,
    EloSeed,
    VoteOutcome,
    battle_group,
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

    return EloBoard(
        modality=modality,
        lang=lang,
        generated_at=_now_iso(),
        vote_count=(seed.auto_vote_count if seed else 0) + counted,
        human_vote_count=counted,
        provisional=counted < PROVISIONAL_MIN_HUMAN_VOTES,
        entries=entries,
    )


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
    """
    if Path(source).is_dir():
        return default_revision, {}

    revision = default_revision
    try:
        from registry.loaders import list_datasets

        for dataset in list_datasets():
            if dataset.predictions_hf == source and dataset.predictions_revision:
                revision = dataset.predictions_revision
                break
    except Exception as exc:
        log.warning("Could not consult registry for %s pin: %s", source, exc)

    try:
        from arena.predictions import resolve_predictions_revision

        sha = resolve_predictions_revision(source, revision=revision)
        return sha, {"resolved_sha": sha}
    except Exception as exc:
        log.warning("Could not resolve %s@%s to a commit SHA: %s", source, revision, exc)
        return revision, {}


def cmd_assemble(args: argparse.Namespace) -> int:
    from arena.predictions import group_rows, load_predictions

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.predictions.split(",") if s.strip()]
    if not sources:
        # Registry-driven default: every eval dataset's predictions_hf repo.
        from registry.loaders import list_prediction_repos

        sources = list_prediction_repos()
        log.info("No --predictions given — using %d registry prediction "
                 "repos", len(sources))
    dataset_info = _dataset_info_lookup(sources)

    # §C reproducibility — pin every HF predictions source to an immutable
    # commit SHA (registry ``predictions_revision`` when set, else
    # ``--revision``) and record the resolved mapping for the boards.
    resolved_revisions: dict[str, str] = {}
    rows = []
    for source in sources:
        fetch_revision, meta = _predictions_revision_for(source, args.revision)
        if meta.get("resolved_sha"):
            resolved_revisions[source] = meta["resolved_sha"]
        log.info("Loading predictions from %s@%s …", source, fetch_revision)
        try:
            rows.extend(load_predictions(source, revision=fetch_revision))
        except Exception as exc:
            log.error("Skipping %s: %s", source, exc)

    if not rows:
        log.warning("No predictions loaded — nothing to assemble.")
        return 1

    grouped = group_rows(rows)
    now = _now_iso()

    # Benchmark boards stay per (modality, dataset, lang) — paradigm-pure, so a
    # template engine is never ranked against a keyword engine on metrics.
    for (modality, dataset_id, lang), samples in sorted(grouped.items()):
        if args.modality and modality != args.modality:
            continue
        by_competitor: dict[str, list] = {}
        for sample_rows in samples.values():
            for competitor_id, row in sample_rows.items():
                by_competitor.setdefault(competitor_id, []).append(row)
        board = build_benchmark_board(modality, dataset_id, lang, by_competitor, now)
        board.dataset_info = dataset_info.get(dataset_id)
        own_revisions = {
            src: sha for src, sha in resolved_revisions.items()
            if src.endswith(f"-bench-{dataset_id}")
        }
        board.predictions_revisions = own_revisions or None
        _write_json(data_dir / f"benchmark-{modality}-{dataset_id}-{lang}.json", board)

    # Battles + ELO pool by battle group: every plugin that answered the same
    # stimulus in a language competes, so the intent paradigm leagues merge into
    # one open arena (battles across all plugins, same language).
    battle_samples: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    elo_samples: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    for (modality, dataset_id, lang), samples in grouped.items():
        if args.modality and modality != args.modality:
            continue
        group = battle_group(modality)
        bs = battle_samples.setdefault((group, dataset_id, lang), {})
        es = elo_samples.setdefault((group, lang), {}).setdefault(dataset_id, {})
        for sample_id, comp_rows in samples.items():
            bs.setdefault(sample_id, {}).update(comp_rows)
            es.setdefault(sample_id, {}).update(comp_rows)

    # Sub-leagues now merged into a group leave stale per-paradigm battle/ELO
    # files behind; drop them (benchmark boards are kept).
    _clean_merged_artifacts(data_dir, {m for (m, _, _) in grouped})

    for (group, dataset_id, lang), samples in sorted(battle_samples.items()):
        battles = assemble_battles(
            group, dataset_id, lang, samples, max_battles=args.max_battles
        )
        pool = BattlesPool(
            modality=group,
            dataset_id=dataset_id,
            lang=lang,
            generated_at=now,
            dataset_info=dataset_info.get(dataset_id),
            battles=battles,
        )
        _write_json(data_dir / f"battles-{group}-{dataset_id}-{lang}.json", pool)

    ww_phrases = _wakeword_phrases()
    for (group, lang), samples_by_dataset in sorted(elo_samples.items()):
        seed = seed_elo(group, lang, samples_by_dataset, now)
        _write_json(data_dir / f"elo-seed-{group}-{lang}.json", seed)

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
        _write_json(data_dir / f"battles-{group}-freeform-{lang}.json", pool)

        # Bootstrap the ELO board when none exists yet; `tally` owns it after
        board_path = data_dir / f"leaderboard-{group}-{lang}.json"
        if not board_path.exists():
            elo_board = build_elo_board(group, lang, seed, [], {})
            _write_json(board_path, elo_board)

    return 0


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

    if counted_decisions:
        # Group votes per (modality, lang) board, carrying each vote's
        # fraud-rule weight through to the rating.
        votes_by_board: dict[tuple[str, str], list[dict]] = {}
        for d in counted_decisions:
            battle = battles_pool[d.vote["battle_id"]]
            key = (battle["modality"], battle["lang"])
            votes_by_board.setdefault(key, []).append({**d.vote, "weight": d.weight})

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
        _write_json_payload(
            out_dir / "patch-notes.json",
            build_patch_notes(patch_note_entries, _now_iso()),
        )
    else:
        # No counted votes → the boards cannot change; leave them alone so
        # the workflow's empty-diff guard skips the commit.
        log.info("No counted votes — leaderboards left untouched.")

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
    # re-fetched every time for full-history replay.
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
    return {
        "file": path.name,
        "modality": payload.get("modality"),
        "dataset_id": payload.get("dataset_id"),
        "lang": payload.get("lang"),
        "generated_at": payload.get("generated_at"),
        "count": counts[count_key],
    }


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


def cmd_export_bestiary(args: argparse.Namespace) -> int:
    registry_root = Path(args.registry).resolve()
    if str(registry_root.parent) not in sys.path:
        sys.path.insert(0, str(registry_root.parent))
    from registry.loaders import load_all_competitors

    competitors = [
        comp.model_dump(mode="json", exclude_none=True)
        for comp in load_all_competitors(registry_root=registry_root)
    ]
    competitors.sort(key=lambda c: (c["modality"], c["competitor_id"]))

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json_payload(
        out_file, {"generated_at": _now_iso(), "competitors": competitors}
    )
    log.info("Exported %d competitors", len(competitors))
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

    p = sub.add_parser("export-index", help="Regenerate data/index.json")
    p.add_argument("--data-dir", default="frontend-static/public/data")
    p.add_argument("--output", default="frontend-static/public/data/index.json")

    p = sub.add_parser("export-bestiary", help="Flatten registry into competitors.json")
    p.add_argument("--registry", default="registry")
    p.add_argument("--output", default="frontend-static/public/data/competitors.json")

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

    args = parser.parse_args(argv)
    commands = {
        "assemble": cmd_assemble,
        "tally": cmd_tally,
        "export-index": cmd_export_index,
        "export-bestiary": cmd_export_bestiary,
        "validate-registry": cmd_validate_registry,
        "audit-seeds": cmd_audit_seeds,
    }
    if args.command not in commands:
        parser.print_help()
        sys.exit(1)
    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
