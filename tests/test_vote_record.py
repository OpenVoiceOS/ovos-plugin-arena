"""Tests for the committed vote record (``votes.jsonl``) — the ingest
snapshot every replay is a pure function of (§6, §4 R4/R12/R13, §P5).

A vote is recorded once, with the battle context it was cast on. From
then on the leaderboard is rebuilt from that record: a pruned battle
pool, an edited issue title or a flaky account-age lookup can no longer
move a published rating.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import arena.cli as arena_cli
from arena.cli import main

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def permissive_registry(monkeypatch):
    """The assemble path drops rows whose competitor_id is not registered;
    this file's fighters are fixtures, not real plugins."""
    import registry.loaders as loaders_mod

    class _Stub:
        config: dict = {}
        trained_on: list = []

        def __init__(self, cid):
            self.competitor_id = cid

    monkeypatch.setattr(
        loaders_mod, "list_competitors",
        lambda modality=None: (
            [_Stub(c) for c in ("base-pt", "small-pt", "third-pt")]
            if modality == "stt" else []
        ),
    )


def _battle(battle_id: str, comp_a="alpha", comp_b="beta") -> dict:
    return {
        "battle_id": battle_id,
        "modality": "intent",
        "dataset_id": "ds",
        "lang": "en-US",
        "sample_id": battle_id,
        "competitor_a": comp_a,
        "competitor_b": comp_b,
    }


def _write_battles_pool(data_dir: Path, battles: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "battles-intent-en-US.json").write_text(json.dumps({
        "modality": "intent", "dataset_id": "ds", "lang": "en-US",
        "generated_at": "2026-01-01T00:00:00+00:00", "battles": battles,
    }))


def _write_seed(data_dir: Path, competitors: tuple[str, ...] = ("alpha", "beta")) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "elo-seed-intent-en-US.json").write_text(json.dumps({
        "modality": "intent", "lang": "en-US",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "auto_vote_count": 10,
        "ratings": {c: 1200.0 for c in competitors},
        "battles": {c: 10 for c in competitors},
        "wins": {c: 5 for c in competitors},
        "losses": {c: 5 for c in competitors},
        "ties": {c: 0 for c in competitors},
        "competitor_plugin": {c: f"{c}-plugin" for c in competitors},
        "pairwise_wins": {}, "pairwise_games": {},
    }))


def _issue(number: int, author: str, title: str, created_at: str) -> dict:
    return {
        "number": number,
        "title": title,
        "author": {"login": author},
        "createdAt": created_at,
        "state": "OPEN",
        "labels": [{"name": "vote"}],
    }


def _run_tally(data_dir: Path, issues: list[dict], monkeypatch, closed=None) -> None:
    closed = [] if closed is None else closed
    monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: issues)
    monkeypatch.setattr(arena_cli, "_now_iso", lambda: "2026-02-02T00:00:00+00:00")
    monkeypatch.setattr(
        arena_cli, "close_issue",
        lambda repo, number, comment, add_label="": closed.append(number),
    )
    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
              "--repo", "OpenVoiceOS/ovos-plugin-arena"])
    assert exc.value.code == 0


def _board(data_dir: Path, name="leaderboard-intent-en-US.json") -> dict:
    return json.loads((data_dir / name).read_text())


def _base(tmp_path: Path, monkeypatch, competitors=("alpha", "beta")) -> Path:
    data_dir = tmp_path / "data"
    _write_battles_pool(data_dir, [_battle("b0"), _battle("b1")])
    _write_seed(data_dir, competitors)
    (data_dir / "voter-age-cache.json").write_text(
        json.dumps({"alice": "2020-01-01T00:00:00Z"}))
    return data_dir


# ---------------------------------------------------------------------------
# a vote outlives the battle it was cast on
# ---------------------------------------------------------------------------


def test_vote_survives_the_battle_leaving_the_pool(tmp_path, monkeypatch):
    """The pool is pruned every assemble. A vote whose battle is gone from
    it still counts, because its competitor pair was recorded at ingest
    (§4 R4 — open votes never dangle)."""
    data_dir = _base(tmp_path, monkeypatch)
    issues = [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")]
    _run_tally(data_dir, issues, monkeypatch)
    assert _board(data_dir)["human_vote_count"] == 1

    _write_battles_pool(data_dir, [_battle("b1")])  # b0 pruned
    _run_tally(data_dir, issues, monkeypatch)

    board = _board(data_dir)
    assert board["human_vote_count"] == 1, (
        "a vote whose battle left the pool was dropped from the board"
    )
    alpha = next(e for e in board["entries"] if e["competitor_id"] == "alpha")
    assert alpha["human_votes"] == 1
    assert alpha["elo"] > 1200.0


def test_retired_competitor_is_discarded_with_a_reason(tmp_path, monkeypatch):
    """A vote whose competitor has left the seed roster cannot be replayed
    onto the board, so it lands in the audit trail rather than vanishing
    (§4 R13)."""
    data_dir = _base(tmp_path, monkeypatch)
    issues = [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")]
    _run_tally(data_dir, issues, monkeypatch)

    _write_seed(data_dir, ("alpha", "gamma"))  # beta retired
    _run_tally(data_dir, issues, monkeypatch)

    audit = json.loads((data_dir / "vote-audit.json").read_text())
    assert audit["counted"] == 0
    assert audit["discarded"] == [
        {"issue_number": 1, "author": "alice", "battle_id": "b0",
         "reason": "competitor_retired"}
    ]


def test_battle_absent_at_ingest_is_recorded_not_dropped(tmp_path, monkeypatch):
    data_dir = _base(tmp_path, monkeypatch)
    _run_tally(data_dir, [_issue(1, "alice", "vote|ghost|a", "2026-02-01T00:00:00Z")],
               monkeypatch)

    audit = json.loads((data_dir / "vote-audit.json").read_text())
    assert audit["discarded"] == [
        {"issue_number": 1, "author": "alice", "battle_id": "ghost",
         "reason": "battle_not_in_pool"}
    ]


# ---------------------------------------------------------------------------
# the record is immutable
# ---------------------------------------------------------------------------


def test_editing_the_issue_title_changes_nothing(tmp_path, monkeypatch):
    """A voter who edits the title of an already-recorded issue — flipping
    the choice, or repointing it at another battle to inherit its issue
    number and creation date — moves neither the record nor the board."""
    data_dir = _base(tmp_path, monkeypatch)
    _run_tally(data_dir, [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")],
               monkeypatch)
    record_before = (data_dir / "votes.jsonl").read_text()
    board_before = _board(data_dir)

    edited = [_issue(1, "alice", "vote|b1|b", "2026-02-01T00:00:00Z")]
    edited[0]["state"] = "CLOSED"
    _run_tally(data_dir, edited, monkeypatch)

    assert (data_dir / "votes.jsonl").read_text() == record_before, (
        "an edited issue title rewrote the vote record"
    )
    board_after = _board(data_dir)
    assert board_after["entries"] == board_before["entries"]
    assert json.loads(record_before)["choice"] == "a"


# ---------------------------------------------------------------------------
# account age is snapshotted at ingest, or the issue waits
# ---------------------------------------------------------------------------


def test_unavailable_account_age_defers_the_issue(tmp_path, monkeypatch):
    """A `gh` hiccup must not decide a vote's fate: the issue is left
    open and un-ingested, and the next run records it with the age."""
    data_dir = _base(tmp_path, monkeypatch)
    (data_dir / "voter-age-cache.json").write_text("{}")
    issues = [_issue(1, "bob", "vote|b0|a", "2026-02-01T00:00:00Z")]

    closed: list[int] = []
    monkeypatch.setattr(arena_cli, "fetch_account_created_at", lambda login: None)
    _run_tally(data_dir, issues, monkeypatch, closed=closed)

    assert closed == [], "an un-ingested issue was closed and can never be retried"
    assert not (data_dir / "votes.jsonl").exists()
    assert _board(data_dir)["human_vote_count"] == 0

    monkeypatch.setattr(arena_cli, "fetch_account_created_at",
                        lambda login: "2020-01-01T00:00:00Z")
    _run_tally(data_dir, issues, monkeypatch, closed=closed)

    assert closed == [1]
    assert _board(data_dir)["human_vote_count"] == 1


# ---------------------------------------------------------------------------
# replay is offline
# ---------------------------------------------------------------------------


def test_verify_replay_reads_the_record_without_a_network(tmp_path, monkeypatch):
    data_dir = _base(tmp_path, monkeypatch)
    _run_tally(data_dir, [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")],
               monkeypatch)

    def _boom(*args, **kwargs):
        raise AssertionError("replay must not shell out")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir)])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# assemble publishes what it can replay
# ---------------------------------------------------------------------------


def _stt_predictions(tmp_path: Path, competitors: dict[str, float]) -> Path:
    preds = tmp_path / "stt_predictions"
    preds.mkdir(parents=True, exist_ok=True)
    for competitor, wer in competitors.items():
        # Enough samples for a rankable board (arena.metrics.MIN_BOARD_SAMPLES);
        # sample_id "pt-PT/00000" is the one the recorded votes reference.
        rows = [
            {
                "competitor_id": competitor,
                "sample_id": f"pt-PT/{i:05d}",
                "dataset_id": "minds14-pt-PT",
                "lang": "pt-PT",
                "plugin_id": f"plugin-{competitor}",
                "audio_url": "https://example.com/a.wav",
                "reference_text": "ligar o alarme",
                "prediction": (
                    "ligar o alarme" if wer == 0.0 else "ligar alarme errado"
                ),
                "wer": wer,
            }
            for i in range(40)
        ]
        (preds / f"{competitor}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    return preds


def _assemble(preds: Path, out: Path) -> int:
    try:
        main(["assemble", "--predictions", str(preds), "--output", str(out)])
    except SystemExit as exc:
        return exc.code
    return 0


def test_assemble_rebuilds_a_voted_board_from_the_record(tmp_path, permissive_registry):
    """A board carrying human votes is rebuilt from (seed, record) like
    every other, so what `assemble` publishes replays exactly (§P5)."""
    out = tmp_path / "data"
    assert _assemble(_stt_predictions(tmp_path / "r1",
                                      {"base-pt": 0.5, "small-pt": 0.1}), out) == 0

    (out / "voter-age-cache.json").write_text(
        json.dumps({"alice": "2020-01-01T00:00:00Z"}))
    (out / "votes.jsonl").write_text(json.dumps({
        "issue": 1, "author": "alice", "created_at": "2026-02-01T00:00:00Z",
        "title_seen": "vote|v0|a", "battle_id": "v0", "choice": "a",
        "modality": "stt", "dataset_id": "minds14-pt-PT", "lang": "pt-PT",
        "competitor_a": "base-pt", "competitor_b": "small-pt",
        "sample_id": "pt-PT/00000",
        "account_created_at": "2020-01-01T00:00:00Z",
    }) + "\n")

    assert _assemble(_stt_predictions(tmp_path / "r2",
                                      {"base-pt": 0.5, "small-pt": 0.1,
                                       "third-pt": 0.9}), out) == 0

    board = _board(out, "leaderboard-stt-pt-PT.json")
    seed = json.loads((out / "elo-seed-stt-pt-PT.json").read_text())
    assert {e["competitor_id"] for e in board["entries"]} == set(seed["ratings"])
    assert board["human_vote_count"] == 1
    base = next(e for e in board["entries"] if e["competitor_id"] == "base-pt")
    assert base["human_votes"] == 1, (
        "assemble published a board that does not carry the recorded vote"
    )
    small = next(e for e in board["entries"] if e["competitor_id"] == "small-pt")
    assert base["bt_rating"] > small["bt_rating"], (
        "the Bradley-Terry column does not reflect the replayed vote"
    )
    for entry in board["entries"]:
        assert entry["ci_lower"] is not None and entry["ci_upper"] is not None

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(out)])
    assert exc.value.code == 0, "assemble published a board it cannot replay"


# ---------------------------------------------------------------------------
# the record is tamper-evident and never half-read
# ---------------------------------------------------------------------------


def test_a_repeated_issue_number_counts_once_and_is_audited(tmp_path, monkeypatch):
    """One issue is one vote. A second row for an issue already in the
    record is tampering, so it is discarded loudly rather than quietly
    rating twice — the board and the audit must never disagree."""
    data_dir = _base(tmp_path, monkeypatch)
    _run_tally(data_dir, [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")],
               monkeypatch)

    record = data_dir / "votes.jsonl"
    forged = json.loads(record.read_text().strip())
    forged.update(battle_id="b1", choice="b", sample_id="b1")
    record.write_text(record.read_text() + json.dumps(forged, sort_keys=True) + "\n")
    _run_tally(data_dir, [], monkeypatch)

    board = _board(data_dir)
    audit = json.loads((data_dir / "vote-audit.json").read_text())
    assert board["human_vote_count"] == 1, "a repeated issue number rated twice"
    assert audit["counted"] == board["human_vote_count"]
    assert {"issue_number": 1, "author": "alice", "battle_id": "b1",
            "reason": "duplicate_record"} in audit["discarded"]


def test_a_truncated_record_stops_the_run_loudly(tmp_path, monkeypatch, caplog):
    """A half-written line is corruption of public evidence: both commands
    name the line and exit non-zero rather than skipping a cast vote."""
    data_dir = _base(tmp_path, monkeypatch)
    _run_tally(data_dir, [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")],
               monkeypatch)

    record = data_dir / "votes.jsonl"
    record.write_text(record.read_text() + '{"issue": 2, "author": "al')

    monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: [])
    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir)])
    assert exc.value.code != 0
    # The form the workflow runs reads the record while ingesting, before
    # the replay, and must report the same way rather than traceback.
    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
              "--repo", "OpenVoiceOS/ovos-plugin-arena"])
    assert exc.value.code != 0
    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(data_dir)])
    assert exc.value.code != 0
    assert "line 2" in caplog.text


def test_a_failed_append_leaves_the_record_intact(tmp_path, monkeypatch):
    """The record is replaced atomically, so a run killed mid-write leaves
    the previous record readable instead of a line nothing can parse."""
    import os

    data_dir = _base(tmp_path, monkeypatch)
    _run_tally(data_dir, [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")],
               monkeypatch)
    record = data_dir / "votes.jsonl"
    before = record.read_text()

    def _die(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", _die)
    with pytest.raises(KeyboardInterrupt):
        arena_cli.append_vote_records(record, [{"issue": 2, "author": "bob"}])

    assert record.read_text() == before
    assert arena_cli.load_vote_records(record)[0]["issue"] == 1


# ---------------------------------------------------------------------------
# no recorded issue is left hanging
# ---------------------------------------------------------------------------


def test_a_later_run_closes_an_issue_an_earlier_run_left_open(tmp_path, monkeypatch):
    """`--keep-issues-open` records without answering the voter, and a run
    killed between recording and closing does the same. The next full run
    closes every recorded issue that is still open, not only the ones it
    recorded itself."""
    data_dir = _base(tmp_path, monkeypatch)
    issues = [_issue(1, "alice", "vote|b0|a", "2026-02-01T00:00:00Z")]

    closed: list[int] = []
    monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: issues)
    monkeypatch.setattr(
        arena_cli, "close_issue",
        lambda repo, number, comment, add_label="": closed.append(number),
    )
    with pytest.raises(SystemExit) as exc:
        main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
              "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
    assert exc.value.code == 0
    assert closed == []

    _run_tally(data_dir, issues, monkeypatch, closed=closed)
    assert closed == [1], "an issue recorded by an earlier run stayed open forever"


def test_assemble_publishes_the_audit_that_matches_its_boards(tmp_path,
                                                              permissive_registry):
    """An assemble that changes a roster can flip a recorded vote between
    counted and discarded, so it republishes the audit trail with the
    boards — otherwise the replay proof goes red on every assemble."""
    out = tmp_path / "data"
    assert _assemble(_stt_predictions(tmp_path / "r1",
                                      {"base-pt": 0.5, "small-pt": 0.1}), out) == 0
    (out / "voter-age-cache.json").write_text(
        json.dumps({"alice": "2020-01-01T00:00:00Z"}))
    (out / "votes.jsonl").write_text(json.dumps({
        "issue": 1, "author": "alice", "created_at": "2026-02-01T00:00:00Z",
        "title_seen": "vote|v0|a", "battle_id": "v0", "choice": "a",
        "modality": "stt", "dataset_id": "minds14-pt-PT", "lang": "pt-PT",
        "competitor_a": "base-pt", "competitor_b": "third-pt",
        "sample_id": "pt-PT/00000",
        "account_created_at": "2020-01-01T00:00:00Z",
    }) + "\n")

    # third-pt is not in the first seed, so the vote starts out retired and
    # becomes countable once its predictions land.
    assert _assemble(_stt_predictions(tmp_path / "r2",
                                      {"base-pt": 0.5, "small-pt": 0.1}), out) == 0
    audit = json.loads((out / "vote-audit.json").read_text())
    assert audit["discarded"][0]["reason"] == "competitor_retired"

    assert _assemble(_stt_predictions(tmp_path / "r3",
                                      {"base-pt": 0.5, "small-pt": 0.1,
                                       "third-pt": 0.9}), out) == 0
    audit = json.loads((out / "vote-audit.json").read_text())
    board = _board(out, "leaderboard-stt-pt-PT.json")
    assert audit["counted"] == board["human_vote_count"] == 1
    assert audit["discarded"] == []

    with pytest.raises(SystemExit) as exc:
        main(["verify-replay", "--data-dir", str(out)])
    assert exc.value.code == 0, "assemble published an audit it cannot replay"
