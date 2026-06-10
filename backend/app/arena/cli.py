"""
CLI entry points for GitHub Actions workflows.

Commands
--------
assemble   — pull HF prediction datasets, build battles pool JSON
tally      — read GitHub vote issues, replay ELO, write leaderboard JSON
export-index — write data/index.json from data dir contents
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.arena.models import (
    IngestedPrediction,
    LeaderboardEntry,
    Matchup,
    PluginFamily,
    Vote,
    VoteOutcome,
    VoteSource,
)
from app.arena.elo import INITIAL_ELO, K_FACTOR, replay_from_votes, update_ratings, k_factor

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOTE_TITLE_RE = re.compile(
    r"^vote\|(?P<battle_id>[^|]+)\|(?P<choice>a|b|tie|both_wrong)$",
    re.IGNORECASE,
)

_CHOICE_TO_OUTCOME = {
    "a": VoteOutcome.CANDIDATE_A,
    "b": VoteOutcome.CANDIDATE_B,
    "tie": VoteOutcome.TIE,
    "both_wrong": VoteOutcome.BOTH_WRONG,
}

_FAMILY_MAP = {
    "stt": PluginFamily.STT,
    "tts": PluginFamily.TTS,
    "wake_word": PluginFamily.WAKE_WORD,
    "intent": PluginFamily.INTENT,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_battles_pool(data_dir: Path) -> Dict[str, Any]:
    """Load all battles from data/*.json battle-pool files.

    Returns mapping battle_id -> battle dict.
    """
    battles: Dict[str, Any] = {}
    for f in data_dir.glob("battles-*.json"):
        try:
            payload = json.loads(f.read_text())
            for b in payload.get("battles", []):
                battles[b["battle_id"]] = b
        except Exception as exc:
            log.warning("Could not read %s: %s", f, exc)
    return battles


# ---------------------------------------------------------------------------
# assemble command
# ---------------------------------------------------------------------------

def cmd_assemble(args: argparse.Namespace) -> int:
    """Pull HF prediction datasets and write battles-pool JSON files."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("huggingface `datasets` package not installed. "
                  "Add it to backend dependencies or run `pip install datasets`.")
        return 1

    from app.arena.ingestion import ingest_hf_dataset
    from app.arena.assembler import Assembler, AssemblerConfig

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    hf_datasets = [d.strip() for d in args.hf_datasets.split(",") if d.strip()]
    max_battles = int(args.max_battles)
    modality_filter = args.modality or None

    all_predictions: List[IngestedPrediction] = []
    for ds_name in hf_datasets:
        log.info("Ingesting %s …", ds_name)
        try:
            preds = ingest_hf_dataset(ds_name)
            all_predictions.extend(preds)
            log.info("  → %d predictions", len(preds))
        except Exception as exc:
            log.warning("  Skipping %s: %s", ds_name, exc)

    if not all_predictions:
        log.warning("No predictions ingested — nothing to assemble.")
        return 0

    # Group predictions by (family, lang)
    by_key: Dict[Tuple[str, str], List[IngestedPrediction]] = {}
    for p in all_predictions:
        # Derive family from source dataset name heuristic or prediction type
        fam = _derive_family(p, hf_datasets)
        lang = getattr(p, "lang", "unknown")
        key = (fam, lang)
        by_key.setdefault(key, []).append(p)

    cfg = AssemblerConfig(max_battles=max_battles)
    assembler = Assembler(cfg)

    for (fam, lang), preds in by_key.items():
        if modality_filter and fam != modality_filter:
            continue
        log.info("Assembling battles for %s/%s (%d preds)", fam, lang, len(preds))
        battles = assembler.assemble(preds)
        if not battles:
            log.info("  → no battles assembled (need ≥2 plugins on the same sample)")
            continue

        out = {
            "family": fam,
            "lang": lang,
            "generated_at": _now_iso(),
            "battles": [_battle_to_dict(b, fam, lang) for b in battles],
        }
        fname = data_dir / f"battles-{fam}-{lang}.json"
        fname.write_text(json.dumps(out, indent=2, default=str))
        log.info("  → wrote %s (%d battles)", fname.name, len(battles))

    return 0


def _derive_family(pred: IngestedPrediction, hf_datasets: List[str]) -> str:
    """Best-effort family derivation from dataset name or prediction metadata."""
    for ds in hf_datasets:
        ds_l = ds.lower()
        if "stt" in ds_l or "speech" in ds_l or "asr" in ds_l:
            return "stt"
        if "tts" in ds_l or "synth" in ds_l:
            return "tts"
        if "ww" in ds_l or "wake" in ds_l:
            return "wake_word"
        if "intent" in ds_l:
            return "intent"
    return "stt"  # default


def _battle_to_dict(b: Any, family: str, lang: str) -> Dict[str, Any]:
    """Serialize an AssembledBattle to the static JSON schema."""
    return {
        "battle_id": str(b.matchup.id),
        "family": family,
        "lang": lang,
        "sample_id": b.matchup.input_ref,
        "plugin_a": getattr(b, "plugin_a_name", str(b.matchup.plugin_a_id)),
        "plugin_b": getattr(b, "plugin_b_name", str(b.matchup.plugin_b_id)),
        "plugin_version_a": getattr(b, "plugin_version_a", ""),
        "plugin_version_b": getattr(b, "plugin_version_b", ""),
        "prediction_a": getattr(b, "prediction_a", ""),
        "prediction_b": getattr(b, "prediction_b", ""),
        "reference": getattr(b, "reference", None),
        "audio_url": getattr(b, "audio_url", None),
        "hf_dataset": getattr(b, "hf_dataset", ""),
    }


# ---------------------------------------------------------------------------
# tally command
# ---------------------------------------------------------------------------

def cmd_tally(args: argparse.Namespace) -> int:
    """Read GitHub vote issues, dedupe, replay ELO, write leaderboard JSON."""
    import subprocess

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = args.repo

    # 1. Fetch all open issues with label=vote
    log.info("Fetching vote issues from %s …", repo)
    issues = _fetch_vote_issues(repo)
    log.info("  → %d open vote issues", len(issues))

    if not issues:
        log.info("No vote issues to process.")
        return 0

    # 2. Load battles pool
    battles_pool = _load_battles_pool(data_dir)
    log.info("Loaded %d battles from pool", len(battles_pool))

    # 3. Parse + validate + dedupe votes
    valid_votes: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}  # (author, battle_id) -> first issue number

    for issue in issues:
        number = issue["number"]
        title = issue.get("title", "").strip()
        author = (issue.get("user") or {}).get("login", "anonymous")
        created_at = issue.get("created_at", "")

        m = VOTE_TITLE_RE.match(title)
        if not m:
            log.debug("  Skipping #%d — title does not match vote pattern: %r", number, title)
            _close_issue(repo, number, "This issue does not match the expected vote title format and has been closed.")
            continue

        battle_id = m.group("battle_id")
        choice = m.group("choice").lower()

        if battle_id not in battles_pool:
            log.debug("  Skipping #%d — unknown battle_id %r", number, battle_id)
            _close_issue(repo, number, f"Battle `{battle_id}` not found in the current battles pool.")
            continue

        dedup_key = f"{author}|{battle_id}"
        if dedup_key in seen:
            log.debug("  Skipping #%d — duplicate vote by %s on %s", number, author, battle_id)
            _close_issue(repo, number, "Duplicate vote on this battle — your earlier vote has been counted.")
            continue

        seen[dedup_key] = str(number)
        valid_votes.append({
            "issue_number": number,
            "battle_id": battle_id,
            "choice": choice,
            "author": author,
            "created_at": created_at,
        })

    log.info("  → %d valid votes after deduplication", len(valid_votes))

    # 4. Replay ELO deterministically (ordered by issue_number, then created_at)
    valid_votes.sort(key=lambda v: (v["issue_number"], v["created_at"]))
    ratings: Dict[str, float] = {}
    battles_count: Dict[str, int] = {}
    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}
    ties: Dict[str, int] = {}

    for vote in valid_votes:
        b = battles_pool.get(vote["battle_id"])
        if not b:
            continue
        pid_a = b["plugin_a"]
        pid_b = b["plugin_b"]
        for pid in (pid_a, pid_b):
            ratings.setdefault(pid, INITIAL_ELO)
            battles_count.setdefault(pid, 0)
            wins.setdefault(pid, 0)
            losses.setdefault(pid, 0)
            ties.setdefault(pid, 0)

        outcome = _CHOICE_TO_OUTCOME[vote["choice"]]
        r_a, r_b = ratings[pid_a], ratings[pid_b]
        new_a, new_b = update_ratings(r_a, r_b, outcome, battles_count[pid_a], battles_count[pid_b])
        ratings[pid_a] = new_a
        ratings[pid_b] = new_b
        battles_count[pid_a] += 1
        battles_count[pid_b] += 1

        if outcome == VoteOutcome.CANDIDATE_A:
            wins[pid_a] += 1
            losses[pid_b] += 1
        elif outcome == VoteOutcome.CANDIDATE_B:
            wins[pid_b] += 1
            losses[pid_a] += 1
        else:
            ties[pid_a] += 1
            ties[pid_b] += 1

    # 5. Build leaderboard JSON per (family, lang)
    by_fl: Dict[Tuple[str, str], List[str]] = {}
    for b in battles_pool.values():
        key = (b.get("family", "stt"), b.get("lang", "unknown"))
        for pid in (b["plugin_a"], b["plugin_b"]):
            if pid in ratings:
                by_fl.setdefault(key, [])
                if pid not in by_fl[key]:
                    by_fl[key].append(pid)

    for (family, lang), plugin_ids in by_fl.items():
        entries = []
        for pid in plugin_ids:
            elo = ratings.get(pid, INITIAL_ELO)
            nb = battles_count.get(pid, 0)
            w = wins.get(pid, 0)
            l_ = losses.get(pid, 0)
            t = ties.get(pid, 0)
            entries.append({
                "plugin_name": pid,
                "display_name": pid,
                "family": family,
                "lang": lang,
                "elo": round(elo, 2),
                "battles": nb,
                "wins": w,
                "losses": l_,
                "ties": t,
                "win_rate": round(w / nb, 4) if nb else 0.0,
            })
        entries.sort(key=lambda e: e["elo"], reverse=True)
        for i, e in enumerate(entries, 1):
            e["rank"] = i

        lb = {
            "family": family,
            "lang": lang,
            "generated_at": _now_iso(),
            "vote_count": len(valid_votes),
            "entries": entries,
        }
        fname = out_dir / f"leaderboard-{family}-{lang}.json"
        fname.write_text(json.dumps(lb, indent=2))
        log.info("Wrote %s (%d entries)", fname.name, len(entries))

    # 6. Thank + close processed issues
    for vote in valid_votes:
        _close_issue(
            repo,
            vote["issue_number"],
            "Your vote has been counted — thank you! "
            "The leaderboard will reflect it in the next scheduled run.",
            add_label="processed",
        )

    return 0


def _gh_api(endpoint: str, method: str = "GET", body: Optional[Dict] = None) -> Any:
    import subprocess, json as _json
    cmd = ["gh", "api", "--paginate" if method == "GET" else "--method", method, endpoint]
    if method != "GET":
        cmd = ["gh", "api", f"--method={method}", endpoint]
    if body:
        cmd += ["--input", "-"]
    result = subprocess.run(
        cmd,
        input=_json.dumps(body).encode() if body else None,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        log.debug("gh api error: %s", result.stderr.decode())
        return None
    text = result.stdout.decode().strip()
    if not text:
        return None
    try:
        return _json.loads(text)
    except Exception:
        return None


def _fetch_vote_issues(repo: str) -> List[Dict]:
    import subprocess, json as _json
    # Use gh CLI to list issues with label=vote
    result = subprocess.run(
        ["gh", "issue", "list",
         "--repo", repo,
         "--label", "vote",
         "--state", "open",
         "--limit", "1000",
         "--json", "number,title,user,createdAt,labels"],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        log.warning("gh issue list failed: %s", result.stderr.decode())
        return []
    data = json.loads(result.stdout.decode())
    # normalize key names
    out = []
    for item in data:
        out.append({
            "number": item["number"],
            "title": item["title"],
            "user": item.get("user") or {"login": item.get("author", {}).get("login", "unknown")},
            "created_at": item.get("createdAt", ""),
        })
    return out


def _close_issue(repo: str, number: int, comment: str, add_label: str = "") -> None:
    import subprocess
    try:
        if comment:
            subprocess.run(
                ["gh", "issue", "comment", str(number), "--repo", repo, "--body", comment],
                capture_output=True, timeout=30,
            )
        if add_label:
            subprocess.run(
                ["gh", "issue", "edit", str(number), "--repo", repo, "--add-label", add_label],
                capture_output=True, timeout=30,
            )
        subprocess.run(
            ["gh", "issue", "close", str(number), "--repo", repo],
            capture_output=True, timeout=30,
        )
    except Exception as exc:
        log.debug("Could not close issue #%d: %s", number, exc)


# ---------------------------------------------------------------------------
# export-index command
# ---------------------------------------------------------------------------

def cmd_export_index(args: argparse.Namespace) -> int:
    """Write data/index.json from the data directory contents."""
    data_dir = Path(args.data_dir)
    out_file = Path(args.output)

    leaderboards = []
    battles_pools = []

    for f in sorted(data_dir.glob("leaderboard-*.json")):
        try:
            payload = json.loads(f.read_text())
            leaderboards.append({
                "file": f.name,
                "family": payload.get("family"),
                "lang": payload.get("lang"),
                "generated_at": payload.get("generated_at"),
                "entry_count": len(payload.get("entries", [])),
            })
        except Exception as exc:
            log.warning("Skipping %s: %s", f, exc)

    for f in sorted(data_dir.glob("battles-*.json")):
        try:
            payload = json.loads(f.read_text())
            battles_pools.append({
                "file": f.name,
                "family": payload.get("family"),
                "lang": payload.get("lang"),
                "generated_at": payload.get("generated_at"),
                "battle_count": len(payload.get("battles", [])),
            })
        except Exception as exc:
            log.warning("Skipping %s: %s", f, exc)

    index = {
        "generated_at": _now_iso(),
        "leaderboards": leaderboards,
        "battles_pools": battles_pools,
    }
    out_file.write_text(json.dumps(index, indent=2))
    log.info("Wrote %s (%d leaderboards, %d battle pools)",
             out_file, len(leaderboards), len(battles_pools))
    return 0


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m arena.cli",
                                     description="OVOS Plugin Arena CLI")
    sub = parser.add_subparsers(dest="command")

    p_assemble = sub.add_parser("assemble", help="Assemble battles from HF datasets")
    p_assemble.add_argument("--hf-datasets", default="", help="Comma-separated HF dataset names")
    p_assemble.add_argument("--output", default="frontend-static/public/data")
    p_assemble.add_argument("--modality", default="")
    p_assemble.add_argument("--max-battles", default="200")

    p_tally = sub.add_parser("tally", help="Tally GitHub vote issues and update leaderboards")
    p_tally.add_argument("--data-dir", default="frontend-static/public/data")
    p_tally.add_argument("--output", default="frontend-static/public/data")
    p_tally.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))

    p_index = sub.add_parser("export-index", help="Regenerate data/index.json")
    p_index.add_argument("--data-dir", default="frontend-static/public/data")
    p_index.add_argument("--output", default="frontend-static/public/data/index.json")

    args = parser.parse_args(argv)
    if args.command == "assemble":
        sys.exit(cmd_assemble(args))
    elif args.command == "tally":
        sys.exit(cmd_tally(args))
    elif args.command == "export-index":
        sys.exit(cmd_export_index(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
