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


def test_missing_votes_source_errors_cleanly(tmp_path):
    data_dir = tmp_path / "data"
    _write_battles_pool(data_dir, [_battle("b0")])
    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir), "--repo", ""])
    assert exc.value.code != 0
