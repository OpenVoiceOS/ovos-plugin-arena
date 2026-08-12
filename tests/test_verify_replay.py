"""Tests for the `verify-replay` CI-gated proof (docs/operations.md
"Replay proof") — replaying the public vote log from scratch must
reproduce the published leaderboards exactly.

All tests are fully offline: votes are supplied via ``--votes-file``
(a fixture, never a live GitHub fetch), matching the "no network in
tests" rule everywhere else in this suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.cli import main

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _battle(battle_id: str, modality: str = "intent", lang: str = "en-US") -> dict:
    return {
        "battle_id": battle_id,
        "modality": modality,
        "dataset_id": "ds",
        "lang": lang,
        "sample_id": battle_id,
        "competitor_a": "alpha",
        "competitor_b": "beta",
    }


def _write_battles_pool(data_dir: Path, battles: list[dict], modality="intent", lang="en-US"):
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "modality": modality,
        "dataset_id": "ds",
        "lang": lang,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "battles": battles,
    }
    (data_dir / f"battles-{modality}-{lang}.json").write_text(json.dumps(payload))


def _issue(number: int, author: str, battle_id: str, choice: str, created_at: str) -> dict:
    return {
        "number": number,
        "title": f"vote|{battle_id}|{choice}",
        "author": {"login": author},
        "createdAt": created_at,
        "state": "OPEN",
        "labels": [{"name": "vote"}],
    }


def _votes_file(tmp_path: Path, issues: list[dict], name="votes.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(issues))
    return path


def _publish_from_tally(data_dir: Path, votes_file: Path, monkeypatch) -> None:
    """Run `tally` once (offline, `--votes-file`-fed by monkeypatching
    `fetch_vote_issues`) to produce a real, internally-consistent published
    leaderboard + vote-audit.json — the ground truth `verify-replay` is
    then checked against."""
    import arena.cli as arena_cli

    issues = json.loads(votes_file.read_text())
    monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: issues)
    monkeypatch.setattr(arena_cli, "close_issue", lambda *a, **kw: None)
    monkeypatch.setattr(arena_cli, "_now_iso", lambda: "2026-02-01T12:00:00+00:00")
    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
              "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
    assert exc.value.code == 0


def _base_setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    _write_battles_pool(data_dir, [_battle("b0"), _battle("b1"), _battle("b2")])
    (data_dir / "voter-age-cache.json").write_text(json.dumps({
        "alice": "2020-01-01T00:00:00Z",
        "bob": "2020-01-01T00:00:00Z",
    }))
    issues = [
        _issue(1, "alice", "b0", "a", "2026-02-01T00:00:00Z"),
        _issue(2, "bob", "b1", "tie", "2026-02-01T01:00:00Z"),
        _issue(3, "alice", "b2", "b", "2026-02-01T02:00:00Z"),
    ]
    votes_file = _votes_file(tmp_path, issues)
    _publish_from_tally(data_dir, votes_file, monkeypatch)
    return data_dir, votes_file


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_replay_matches_published_board_exact_match(tmp_path, monkeypatch):
    data_dir, votes_file = _base_setup(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(votes_file)])
    assert exc.value.code == 0


def test_tampered_leaderboard_rating_detected(tmp_path, monkeypatch):
    data_dir, votes_file = _base_setup(tmp_path, monkeypatch)

    board_path = data_dir / "leaderboard-intent-en-US.json"
    board = json.loads(board_path.read_text())
    board["entries"][0]["elo"] = board["entries"][0]["elo"] + 999.0
    board_path.write_text(json.dumps(board))

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(votes_file)])
    assert exc.value.code != 0


def test_vote_added_after_publication_detected(tmp_path, monkeypatch):
    """A vote cast (and present in the fetched vote log) after the
    leaderboard was published diverges the replay from the committed
    board — proving verify-replay actually re-derives standings rather
    than trivially comparing a board to itself."""
    data_dir, votes_file = _base_setup(tmp_path, monkeypatch)

    issues = json.loads(votes_file.read_text())
    issues.append(_issue(4, "bob", "b0", "b", "2026-02-01T03:00:00Z"))
    extra_votes_file = _votes_file(tmp_path, issues, name="votes-extra.json")

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(extra_votes_file)])
    assert exc.value.code != 0


def test_tampered_vote_audit_detected(tmp_path, monkeypatch):
    data_dir, votes_file = _base_setup(tmp_path, monkeypatch)

    audit_path = data_dir / "vote-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["counted"] = audit["counted"] + 1
    audit_path.write_text(json.dumps(audit))

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(votes_file)])
    assert exc.value.code != 0


def test_missing_published_board_always_fails(tmp_path, monkeypatch):
    """Strict mode, no escape hatch: a league with counted votes but no
    published leaderboard is a mismatch, full stop — published artifacts
    are derived data, never grandfathered (owner decision on PR #33)."""
    data_dir = tmp_path / "data"
    _write_battles_pool(data_dir, [_battle("tmpl-b0", modality="intent_template")],
                         modality="intent_template")
    (data_dir / "voter-age-cache.json").write_text(
        json.dumps({"alice": "2020-01-01T00:00:00Z"}))
    issues = [_issue(1, "alice", "tmpl-b0", "a", "2026-02-01T00:00:00Z")]
    votes_file = _votes_file(tmp_path, issues)

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(votes_file)])
    assert exc.value.code != 0


def test_no_network_touched(tmp_path, monkeypatch):
    """verify-replay with --votes-file must never shell out — no gh CLI,
    no subprocess at all."""
    import subprocess

    data_dir, votes_file = _base_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run",
                         lambda *a, **kw: (_ for _ in ()).throw(
                             AssertionError("verify-replay must not touch the network")))

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(votes_file)])
    assert exc.value.code == 0


def _write_elo_seed(data_dir: Path, auto_battles: int, modality="intent", lang="en-US"):
    """A minimal but internally-consistent EloSeed fixture: alpha beats beta
    every auto-battle, so growing ``auto_battles`` moves the rating."""
    seed = {
        "modality": modality,
        "lang": lang,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "auto_vote_count": auto_battles,
        "ratings": {"alpha": 1200.0, "beta": 1200.0},
        "battles": {"alpha": auto_battles, "beta": auto_battles},
        "wins": {"alpha": auto_battles, "beta": 0},
        "losses": {"alpha": 0, "beta": auto_battles},
        "ties": {"alpha": 0, "beta": 0},
        "competitor_plugin": {"alpha": "alpha-plugin", "beta": "beta-plugin"},
        "pairwise_wins": {"alpha": {"beta": float(auto_battles)}, "beta": {}},
        "pairwise_games": {"alpha": {"beta": float(auto_battles)}, "beta": {}},
    }
    (data_dir / f"elo-seed-{modality}-{lang}.json").write_text(json.dumps(seed))


def test_tally_rebuilds_board_when_seed_changes_with_zero_votes(tmp_path, monkeypatch):
    """Regression for the replay-mismatch blocker: `assemble` can regenerate
    `elo-seed-*.json` with a larger auto-battle tally (more predictions
    loaded) on a day with zero human votes. `tally` MUST still rebuild the
    published leaderboard from the fresh seed — leaving it untouched (as it
    did before this fix, gated on `if counted_decisions:`) means the
    committed board reflects a stale seed while `verify-replay` (a pure
    function of the *current* seed/battles/votes, per R19) always derives
    the fresh one, so the two permanently diverge until the next human vote
    happens to land. Before the fix this test's `verify-replay` call fails;
    after it, it passes."""
    import arena.cli as arena_cli

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    _write_elo_seed(data_dir, auto_battles=10)

    monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: [])
    monkeypatch.setattr(arena_cli, "close_issue", lambda *a, **kw: None)
    monkeypatch.setattr(arena_cli, "_now_iso", lambda: "2026-01-02T00:00:00+00:00")
    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
              "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
    assert exc.value.code == 0

    board_v1 = json.loads((data_dir / "leaderboard-intent-en-US.json").read_text())
    alpha_v1 = next(e for e in board_v1["entries"] if e["competitor_id"] == "alpha")
    assert alpha_v1["battles"] == 10

    # Simulate a same-day `assemble` re-run that loaded more predictions:
    # the seed grows, but zero human votes were cast — the exact scenario
    # that used to leave the published board stale.
    _write_elo_seed(data_dir, auto_battles=40)

    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
              "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
    assert exc.value.code == 0

    board_v2 = json.loads((data_dir / "leaderboard-intent-en-US.json").read_text())
    alpha_v2 = next(e for e in board_v2["entries"] if e["competitor_id"] == "alpha")
    assert alpha_v2["battles"] == 40, (
        "leaderboard was left stale after a zero-vote tally run despite a "
        "changed elo-seed — published board no longer matches the current "
        "seed, breaking verify-replay/R19"
    )

    # And verify-replay — the actual CI gate — must pass against this
    # freshly-rebuilt board with no votes at all.
    empty_votes_file = _votes_file(tmp_path, [], name="votes-empty.json")
    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir),
              "--votes-file", str(empty_votes_file)])
    assert exc.value.code == 0


def test_missing_votes_source_errors_cleanly(tmp_path):
    data_dir = tmp_path / "data"
    _write_battles_pool(data_dir, [_battle("b0")])
    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir), "--repo", ""])
    assert exc.value.code != 0
