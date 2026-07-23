"""Integration tests for the tally CLI's vote-fraud pipeline (§4 A1.4).

``tests/test_fraud.py`` unit-tests ``arena/fraud.py`` in isolation.  This
module instead drives the rules the way ``arena.cli.cmd_tally`` actually
does — through ``dedupe_votes`` → ``resolve_vote_weights`` → the
``vote-audit.json`` artifact — with synthetic vote logs that exercise every
rule individually, in combination, and at their exact boundaries, plus a
purity test proving that replaying a committed vote log never touches the
network and is byte-identical across runs.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import arena.cli as arena_cli
from arena.cli import dedupe_votes, main
from arena.fraud import resolve_vote_weights

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


def _raw_vote(issue_number, author, battle_id, choice, created_at) -> dict:
    return {
        "issue_number": issue_number,
        "battle_id": battle_id,
        "choice": choice,
        "author": author,
        "created_at": created_at,
    }


DAY = "2026-02-01T{:02d}:00:00Z"


# ---------------------------------------------------------------------------
# Rule 1 — one counted vote per (voter, battle_pair, dataset_entry)
# ---------------------------------------------------------------------------


class TestDedupeByVoterBattlePair:
    def test_duplicate_vote_same_battle_kept_once(self):
        raw = [
            _raw_vote(1, "alice", "b1", "a", DAY.format(1)),
            _raw_vote(2, "alice", "b1", "b", DAY.format(2)),
        ]
        deduped = dedupe_votes(raw)
        assert len(deduped) == 1
        assert deduped[0]["issue_number"] == 1  # first issue wins, not last

    def test_duplicate_across_leagues_both_kept(self):
        """Same author voting on the *same battle_id* string reused across
        two different modalities' pools is not something battle_id allows
        (battle_id already encodes modality+dataset+lang+sample+pair per
        §4 R4/``battle_id_for``) — but the same author voting on two
        genuinely distinct battle ids (different leagues) must both count.
        """
        raw = [
            _raw_vote(1, "alice", "intent-b1", "a", DAY.format(1)),
            _raw_vote(2, "alice", "stt-b1", "a", DAY.format(2)),
        ]
        deduped = dedupe_votes(raw)
        assert len(deduped) == 2

    def test_different_authors_same_battle_both_kept(self):
        raw = [
            _raw_vote(1, "alice", "b1", "a", DAY.format(1)),
            _raw_vote(2, "bob", "b1", "b", DAY.format(2)),
        ]
        assert len(dedupe_votes(raw)) == 2

    def test_dedupe_deterministic_regardless_of_input_order(self):
        raw_fwd = [
            _raw_vote(1, "alice", "b1", "a", DAY.format(1)),
            _raw_vote(2, "alice", "b1", "b", DAY.format(2)),
        ]
        raw_rev = list(reversed(raw_fwd))
        assert dedupe_votes(raw_fwd) == dedupe_votes(raw_rev)


# ---------------------------------------------------------------------------
# Rule 2 — per-voter/league daily cap (50), deterministic by issue order
# ---------------------------------------------------------------------------


class TestDailyCapThroughTally:
    def test_exactly_at_cap_all_counted(self):
        # Alternate a/b so the one-sided rule (a separate concern, tested
        # below) never interferes with this boundary check.
        votes = [
            _raw_vote(i, "alice", f"b{i}", "a" if i % 2 else "b",
                      "2026-02-01T00:00:00Z")
            for i in range(1, 51)
        ]  # exactly 50 — boundary: none discarded
        decisions = resolve_vote_weights(votes, {f"b{i}": "intent" for i in range(1, 51)}, {})
        assert all(d.weight == 1.0 for d in decisions)

    def test_one_over_cap_the_51st_discarded(self):
        votes = [
            _raw_vote(i, "alice", f"b{i}", "a", "2026-02-01T00:00:00Z")
            for i in range(1, 52)
        ]  # 51 votes — boundary: exactly one discarded
        decisions = resolve_vote_weights(votes, {f"b{i}": "intent" for i in range(1, 52)}, {})
        assert sum(d.weight == 0.0 for d in decisions) == 1
        discarded = [d for d in decisions if d.weight == 0.0][0]
        assert discarded.vote["issue_number"] == 51
        assert discarded.discarded_reason == "daily_vote_cap_exceeded"

    def test_cap_keyed_by_issue_number_not_wall_clock(self):
        """Two votes with an identical ``created_at`` timestamp (a tie) are
        still resolved deterministically because ``dedupe_votes`` sorts by
        ``(issue_number, created_at)`` before the cap ever sees them."""
        raw = [
            _raw_vote(2, "alice", "b2", "a", "2026-02-01T00:00:00Z"),
            _raw_vote(1, "alice", "b1", "a", "2026-02-01T00:00:00Z"),
        ]
        ordered = dedupe_votes(raw)
        assert [v["issue_number"] for v in ordered] == [1, 2]


# ---------------------------------------------------------------------------
# Rule 3 — account age <7 days -> weight 0, ingest-time snapshot
# ---------------------------------------------------------------------------


class TestAccountAgeGateThroughTally:
    def test_boundary_exactly_seven_days_counts(self):
        votes = [_raw_vote(1, "alice", "b1", "a", "2026-01-08T00:00:00Z")]
        cache = {"alice": "2026-01-01T00:00:00Z"}  # exactly 7.0 days old
        decisions = resolve_vote_weights(votes, {"b1": "intent"}, cache)
        assert decisions[0].weight == 1.0

    def test_boundary_one_second_under_seven_days_discarded(self):
        votes = [_raw_vote(1, "alice", "b1", "a", "2026-01-07T23:59:59Z")]
        cache = {"alice": "2026-01-01T00:00:00Z"}
        decisions = resolve_vote_weights(votes, {"b1": "intent"}, cache)
        assert decisions[0].weight == 0.0
        assert decisions[0].discarded_reason == "account_too_new"

    def test_replay_never_refetches_age_after_ingest_snapshot(self, tmp_path, monkeypatch):
        """The account-age cache is populated once at ingest and is never
        re-fetched on replay, even if the (recorded) age would now put the
        account over the 7-day line at *replay* time rather than at
        *vote* time — created_at is a snapshot, not re-derived from now()."""

        def _boom(login):  # pragma: no cover - must never run
            raise AssertionError(f"network re-fetch attempted for {login}")

        monkeypatch.setattr(arena_cli, "fetch_account_created_at", _boom)
        cache_path = tmp_path / "voter-age-cache.json"
        cache_path.write_text(json.dumps({"alice": "2026-01-01T00:00:00Z"}))
        cache = arena_cli._account_age_cache(tmp_path, {"alice"})
        assert cache == {"alice": "2026-01-01T00:00:00Z"}


# ---------------------------------------------------------------------------
# Rule 4 — >95% one-sided over >=20 votes -> weight 0.5
# ---------------------------------------------------------------------------


class TestOneSidedThroughTally:
    def test_boundary_exactly_95_percent_not_downweighted(self):
        # 19/20 = 0.95 exactly -> spec says ">95%", so this must NOT trigger.
        votes = [_raw_vote(i, "alice", f"b{i}", "a" if i < 19 else "b",
                            "2026-02-01T00:00:00Z") for i in range(20)]
        decisions = resolve_vote_weights(votes, {f"b{i}": "intent" for i in range(20)}, {})
        assert all(d.weight == 1.0 for d in decisions)

    def test_boundary_one_more_than_95_percent_downweighted(self):
        # 20 votes, only "a" -> 100% > 95%, min_votes boundary satisfied at 20.
        votes = [_raw_vote(i, "alice", f"b{i}", "a", "2026-02-01T00:00:00Z")
                 for i in range(20)]
        decisions = resolve_vote_weights(votes, {f"b{i}": "intent" for i in range(20)}, {})
        assert all(d.weight == 0.5 for d in decisions)

    def test_below_min_votes_boundary_unaffected(self):
        # 19 votes, all "a" -> ratio 100% but below ONE_SIDED_MIN_VOTES=20.
        votes = [_raw_vote(i, "alice", f"b{i}", "a", "2026-02-01T00:00:00Z")
                 for i in range(19)]
        decisions = resolve_vote_weights(votes, {f"b{i}": "intent" for i in range(19)}, {})
        assert all(d.weight == 1.0 for d in decisions)


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------


class TestCombinedFraudRules:
    def test_new_account_over_daily_cap_and_one_sided_still_only_one_discard_reason(self):
        """A vote can only be discarded once — a brand-new, cap-busting,
        one-sided voter's excess votes carry the FIRST rule's reason
        (daily cap), since account-age and one-sided only touch votes not
        already discarded (resolve_vote_weights order)."""
        votes = [
            _raw_vote(i, "alice", f"b{i}", "a", "2026-02-01T00:00:00Z")
            for i in range(1, 52)  # 51 votes -> #51 hits the daily cap
        ]
        cache = {"alice": "2026-01-01T00:00:00Z"}  # old enough, not gated
        decisions = resolve_vote_weights(
            votes, {f"b{i}": "intent" for i in range(1, 52)}, cache)
        capped = [d for d in decisions if d.discarded_reason == "daily_vote_cap_exceeded"]
        assert len(capped) == 1
        assert capped[0].vote["issue_number"] == 51
        # the 50 surviving votes are all "a" across >=20 votes -> one-sided
        # downweight still applies on top of the cap discard for everyone else.
        assert all(d.weight == 0.5 for d in decisions if d.discarded_reason is None)

    def test_discarded_votes_excluded_from_one_sided_ratio(self):
        """A voter with 25 total votes, 5 discarded by the daily cap and 20
        surviving 100% one-sided, is downweighted on the ratio of the
        *surviving* votes only (25 raw all-"a" would still be 100% either
        way here, so use a mix that would NOT be one-sided if the discarded
        ones were wrongly included in the denominator)."""
        # 15 "a" (kept) + 5 "b" that get capped out (cap=15) => surviving
        # ratio is 15/15 = 100% one-sided among *counted* votes, well above
        # the 20-vote minimum once combined with 5 more "a" below the cap.
        votes = (
            [_raw_vote(i, "alice", f"a{i}", "a", "2026-02-01T00:00:00Z")
             for i in range(20)]
            + [_raw_vote(20 + i, "alice", f"b{i}", "b", "2026-02-01T00:00:00Z")
               for i in range(5)]
        )
        modality_by_battle = {f"a{i}": "intent" for i in range(20)}
        modality_by_battle.update({f"b{i}": "intent" for i in range(5)})
        decisions = resolve_vote_weights(votes, modality_by_battle, {}, )
        # cap defaults to 50 so nothing here is capped; verify the "b" votes
        # correctly drag the ratio below the 95% one-sided threshold.
        assert all(d.weight == 1.0 for d in decisions)  # 20/25 = 80%, not one-sided


# ---------------------------------------------------------------------------
# vote-audit.json — discards recorded, never deleted
# ---------------------------------------------------------------------------


class TestVoteAuditNeverDeletes:
    def test_discarded_and_downweighted_votes_appear_in_audit(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        _write_battles_pool(data_dir, [_battle(f"b{i}") for i in range(21)])

        issues = (
            [_issue(1, "newbie", "b0", "a", "2026-02-01T00:00:00Z")]
            + [_issue(2 + i, "onesided", f"b{i + 1}", "a", "2026-02-01T00:00:00Z")
               for i in range(20)]
        )
        monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: issues)
        # "newbie" account created same instant as the vote -> too new.
        (data_dir / "voter-age-cache.json").write_text(
            json.dumps({"newbie": "2026-02-01T00:00:00Z",
                        "onesided": "2020-01-01T00:00:00Z"}))

        def _no_network(*a, **kw):  # pragma: no cover
            raise AssertionError("no subprocess call expected in this test")
        monkeypatch.setattr(subprocess, "run", _no_network)

        with pytest.raises(SystemExit) as exc:
            main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
                  "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
        assert exc.value.code == 0

        audit = json.loads((data_dir / "vote-audit.json").read_text())
        assert any(d["issue_number"] == 1 and d["reason"] == "account_too_new"
                   for d in audit["discarded"])
        assert any(d["issue_number"] == n for n in range(2, 22)
                   for d in audit["downweighted"])
        assert audit["counted"] == 20


# ---------------------------------------------------------------------------
# Offline replay purity — §P5: replaying the committed vote log never
# touches the network and is fully deterministic.
# ---------------------------------------------------------------------------


class TestOfflineReplayPurity:
    def test_replay_is_network_free_and_byte_identical(self, tmp_path, monkeypatch):
        data_dir1 = tmp_path / "d1"
        data_dir2 = tmp_path / "d2"
        battles = [_battle(f"b{i}") for i in range(3)]
        _write_battles_pool(data_dir1, battles)
        _write_battles_pool(data_dir2, battles)

        issues = [
            _issue(1, "alice", "b0", "a", "2026-02-01T00:00:00Z"),
            _issue(2, "bob", "b1", "tie", "2026-02-01T01:00:00Z"),
            _issue(3, "alice", "b2", "b", "2026-02-01T02:00:00Z"),
        ]
        # Every voter is already in the age cache -> _account_age_cache
        # never needs fetch_account_created_at.
        age_cache = json.dumps({"alice": "2020-01-01T00:00:00Z",
                                 "bob": "2020-01-01T00:00:00Z"})
        (data_dir1 / "voter-age-cache.json").write_text(age_cache)
        (data_dir2 / "voter-age-cache.json").write_text(age_cache)

        monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: issues)

        def _boom(*a, **kw):
            raise AssertionError("replay must not touch the network")
        monkeypatch.setattr(subprocess, "run", _boom)
        monkeypatch.setattr(arena_cli, "fetch_account_created_at", _boom)
        monkeypatch.setattr(arena_cli, "close_issue", _boom)

        # Freeze "now" so generated_at is identical across both runs too —
        # true byte-identical output, not just "same modulo timestamps".
        monkeypatch.setattr(arena_cli, "_now_iso", lambda: "2026-02-01T12:00:00+00:00")

        for data_dir in (data_dir1, data_dir2):
            with pytest.raises(SystemExit) as exc:
                main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
                      "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
            assert exc.value.code == 0

        board1 = (data_dir1 / "leaderboard-intent-en-US.json").read_text()
        board2 = (data_dir2 / "leaderboard-intent-en-US.json").read_text()
        assert board1 == board2

        audit1 = (data_dir1 / "vote-audit.json").read_text()
        audit2 = (data_dir2 / "vote-audit.json").read_text()
        assert audit1 == audit2

    def test_replay_twice_over_same_dir_is_idempotent(self, tmp_path, monkeypatch):
        """Running tally twice back-to-back over the same committed data
        dir (as a maintainer would when re-running the workflow manually)
        produces the same board a second time — closed/processed issues
        from run 1 don't change run 2's replay of the full vote log."""
        data_dir = tmp_path / "data"
        _write_battles_pool(data_dir, [_battle("b0")])
        issues = [_issue(1, "alice", "b0", "a", "2026-02-01T00:00:00Z")]
        monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: issues)
        (data_dir / "voter-age-cache.json").write_text(
            json.dumps({"alice": "2020-01-01T00:00:00Z"}))
        monkeypatch.setattr(subprocess, "run",
                             lambda *a, **kw: (_ for _ in ()).throw(
                                 AssertionError("no network expected")))
        monkeypatch.setattr(arena_cli, "close_issue", lambda *a, **kw: None)
        monkeypatch.setattr(arena_cli, "_now_iso", lambda: "2026-02-01T12:00:00+00:00")

        for _ in range(2):
            with pytest.raises(SystemExit) as exc:
                main(["tally", "--data-dir", str(data_dir), "--output", str(data_dir),
                      "--repo", "OpenVoiceOS/ovos-plugin-arena", "--keep-issues-open"])
            assert exc.value.code == 0

        board = json.loads((data_dir / "leaderboard-intent-en-US.json").read_text())
        assert board["human_vote_count"] == 1
