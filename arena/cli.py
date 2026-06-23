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
from typing import Any, Dict, List, Optional, Tuple

from arena.assembler import assemble_battles, freeform_battles, seed_elo
from arena.elo import EloLedger
from arena.metrics import build_benchmark_board
from arena.models import (
    BattlesPool,
    EloBoard,
    EloEntry,
    EloSeed,
    VoteOutcome,
    battle_group,
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


def parse_vote_title(title: str) -> Optional[Tuple[str, str]]:
    """Parse ``vote|<battle_id>|<choice>`` — returns (battle_id, choice) or None."""
    m = VOTE_TITLE_RE.match(title.strip())
    if not m:
        return None
    return m.group("battle_id"), m.group("choice").lower()


def dedupe_votes(raw_votes: List[Dict]) -> List[Dict]:
    """Keep the first vote per (author, battle_id), ordered by issue number."""
    seen: set = set()
    out: List[Dict] = []
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


def load_battles_pools(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Map battle_id → battle dict from every ``battles-*.json`` pool."""
    battles: Dict[str, Dict[str, Any]] = {}
    for path in sorted(data_dir.glob("battles-*.json")):
        try:
            payload = json.loads(path.read_text())
            for battle in payload.get("battles", []):
                battles[battle["battle_id"]] = battle
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
    return battles


def load_elo_seeds(data_dir: Path) -> Dict[Tuple[str, str], EloSeed]:
    """Map (modality, lang) → EloSeed from every ``elo-seed-*.json``."""
    seeds: Dict[Tuple[str, str], EloSeed] = {}
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
    return ledger


def build_elo_board(
    modality: str,
    lang: str,
    seed: Optional[EloSeed],
    human_votes: List[Dict],
    battles_pool: Dict[str, Dict[str, Any]],
) -> EloBoard:
    """Replay *human_votes* (ordered) on top of *seed* and rank the result."""
    ledger = _ledger_from_seed(seed) if seed else EloLedger()
    competitor_plugin = dict(seed.competitor_plugin) if seed else {}

    counted = 0
    for vote in human_votes:
        battle = battles_pool.get(vote["battle_id"])
        if not battle:
            continue
        comp_a = battle["competitor_a"]
        comp_b = battle["competitor_b"]
        competitor_plugin.setdefault(comp_a, battle.get("plugin_a", ""))
        competitor_plugin.setdefault(comp_b, battle.get("plugin_b", ""))
        ledger.apply(comp_a, comp_b, CHOICE_TO_OUTCOME[vote["choice"]])
        counted += 1

    entries = []
    for competitor, rating in ledger.ratings.items():
        battles = ledger.battles[competitor]
        wins = ledger.wins[competitor]
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
            )
        )
    entries.sort(key=lambda e: (-e.elo, e.competitor_id))
    for i, entry in enumerate(entries, 1):
        entry.rank = i

    return EloBoard(
        modality=modality,
        lang=lang,
        generated_at=_now_iso(),
        vote_count=(seed.auto_vote_count if seed else 0) + counted,
        human_vote_count=counted,
        entries=entries,
    )


def _write_json(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n"
    )
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def _dataset_info_lookup(prediction_sources: List[str]) -> Dict[str, Dict[str, Any]]:
    """Registry metadata per dataset_id, for the benchmark board UI."""
    info: Dict[str, Dict[str, Any]] = {}
    try:
        from registry.loaders import list_datasets
    except ImportError:
        return info
    hf_repos = [s for s in prediction_sources if not Path(s).is_dir()]
    predictions_urls = [f"https://huggingface.co/datasets/{r}" for r in hf_repos]
    datasets = list_datasets()
    by_id = {d.dataset_id: d for d in datasets}
    for dataset in datasets:
        entry: Dict[str, Any] = {}
        hf_id = getattr(dataset.source, "hf_id", None)
        if hf_id:
            entry["url"] = f"https://huggingface.co/datasets/{hf_id}"
        if dataset.license:
            entry["license"] = dataset.license
        if dataset.notes:
            entry["notes"] = dataset.notes
        if predictions_urls:
            entry["predictions"] = predictions_urls
        if dataset.train_datasets:
            trains: Dict[str, Any] = {}
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


def cmd_assemble(args: argparse.Namespace) -> int:
    from arena.predictions import group_rows, load_predictions

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.predictions.split(",") if s.strip()]
    dataset_info = _dataset_info_lookup(sources)

    rows = []
    for source in sources:
        log.info("Loading predictions from %s …", source)
        try:
            rows.extend(load_predictions(source, revision=args.revision))
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
        by_competitor: Dict[str, List] = {}
        for sample_rows in samples.values():
            for competitor_id, row in sample_rows.items():
                by_competitor.setdefault(competitor_id, []).append(row)
        board = build_benchmark_board(modality, dataset_id, lang, by_competitor, now)
        board.dataset_info = dataset_info.get(dataset_id)
        _write_json(data_dir / f"benchmark-{modality}-{dataset_id}-{lang}.json", board)

    # Battles + ELO pool by battle group: every plugin that answered the same
    # stimulus in a language competes, so the intent paradigm leagues merge into
    # one open arena (battles across all plugins, same language).
    battle_samples: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    elo_samples: Dict[Tuple[str, str], Dict[str, Dict[str, Dict[str, Any]]]] = {}
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
            board = build_elo_board(group, lang, seed, [], {})
            _write_json(board_path, board)

    return 0


def _wakeword_phrases() -> Dict[str, str]:
    """Map each wake-word competitor id → its phrase (its hotword config key)."""
    phrases: Dict[str, str] = {}
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


def fetch_vote_issues(repo: str) -> List[Dict]:
    """List open ``vote``-labelled issues via the gh CLI."""
    result = subprocess.run(
        ["gh", "issue", "list",
         "--repo", repo,
         "--label", "vote",
         "--state", "open",
         "--limit", "1000",
         "--json", "number,title,author,createdAt"],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        log.warning("gh issue list failed: %s", result.stderr.decode())
        return []
    return json.loads(result.stdout.decode())


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


def cmd_tally(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    battles_pool = load_battles_pools(data_dir)
    seeds = load_elo_seeds(data_dir)
    log.info("Loaded %d battles, %d ELO seeds", len(battles_pool), len(seeds))

    issues: List[Dict] = []
    if args.repo:
        log.info("Fetching vote issues from %s …", args.repo)
        issues = fetch_vote_issues(args.repo)
        log.info("  → %d open vote issues", len(issues))

    raw_votes: List[Dict] = []
    invalid: List[Tuple[int, str]] = []
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
    log.info("  → %d valid votes (%d duplicates, %d invalid)",
             len(votes), len(duplicates), len(invalid))

    # Group votes per (modality, lang) board
    votes_by_board: Dict[Tuple[str, str], List[Dict]] = {}
    for vote in votes:
        battle = battles_pool[vote["battle_id"]]
        key = (battle["modality"], battle["lang"])
        votes_by_board.setdefault(key, []).append(vote)

    boards = set(seeds) | set(votes_by_board)
    for modality, lang in sorted(boards):
        board = build_elo_board(
            modality, lang,
            seeds.get((modality, lang)),
            votes_by_board.get((modality, lang), []),
            battles_pool,
        )
        _write_json(out_dir / f"leaderboard-{modality}-{lang}.json", board)

    # Close processed issues (votes counted, duplicates, invalid)
    if args.repo and not args.keep_issues_open:
        for vote in votes:
            close_issue(
                args.repo, vote["issue_number"],
                "Your vote has been counted — thank you! The leaderboard "
                "will reflect it once this run's commit deploys.",
                add_label="processed",
            )
        for vote in raw_votes:
            if vote["issue_number"] in duplicates:
                close_issue(
                    args.repo, vote["issue_number"],
                    "Duplicate vote on this battle — your earlier vote was "
                    "already counted.",
                )
        for number, reason in invalid:
            close_issue(args.repo, number, reason)

    return 0


# ---------------------------------------------------------------------------
# export-index / export-bestiary
# ---------------------------------------------------------------------------


def _index_entry(path: Path, count_key: str) -> Dict[str, Any]:
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
    index: Dict[str, Any] = {"generated_at": _now_iso()}
    for key in ("leaderboards", "benchmarks", "battles_pools", "freeform_pools"):
        index[key] = []
    for path in sorted(data_dir.glob("leaderboard-*.json")):
        index["leaderboards"].append(_index_entry(path, "leaderboards"))
    for path in sorted(data_dir.glob("benchmark-*.json")):
        index["benchmarks"].append(_index_entry(path, "benchmarks"))
    # blind sample battles vs free-form matchup pools (different voting UIs)
    for path in sorted(data_dir.glob("battles-*.json")):
        if "-freeform-" in path.name:
            index["freeform_pools"].append(_index_entry(path, "freeform_pools"))
        else:
            index["battles_pools"].append(_index_entry(path, "battles_pools"))
    index["has_bestiary"] = (data_dir / "competitors.json").exists()

    out_file = Path(args.output)
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
    out_file.write_text(
        json.dumps(
            {"generated_at": _now_iso(), "competitors": competitors},
            indent=2, ensure_ascii=False,
        ) + "\n"
    )
    log.info("Wrote %s (%d competitors)", out_file, len(competitors))
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
    p.add_argument("--predictions", required=True,
                   help="Comma-separated HF dataset repo ids or local predictions dirs")
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

    args = parser.parse_args(argv)
    commands = {
        "assemble": cmd_assemble,
        "tally": cmd_tally,
        "export-index": cmd_export_index,
        "export-bestiary": cmd_export_bestiary,
    }
    if args.command not in commands:
        parser.print_help()
        sys.exit(1)
    sys.exit(commands[args.command](args))


if __name__ == "__main__":
    main()
