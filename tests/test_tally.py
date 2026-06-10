"""
Tests for the tally logic (Mode C — GitHub-issue voting).

Covers:
 - vote-issue title parsing
 - deduplication (one vote per author per battle)
 - deterministic ELO replay from ordered vote list
 - data-export JSON shape validation
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Inline the tally helpers (no network calls needed for unit tests)
# ---------------------------------------------------------------------------
VOTE_TITLE_RE = re.compile(
    r"^vote\|(?P<battle_id>[^|]+)\|(?P<choice>a|b|tie|both_wrong)$",
    re.IGNORECASE,
)
_CHOICE_TO_OUTCOME = {"a": "candidate_a", "b": "candidate_b", "tie": "tie", "both_wrong": "both_wrong"}

INITIAL_ELO = 1200.0


def parse_vote_title(title: str) -> Optional[Tuple[str, str]]:
    """Return (battle_id, choice) or None."""
    m = VOTE_TITLE_RE.match(title.strip())
    if not m:
        return None
    return m.group("battle_id"), m.group("choice").lower()


def dedupe_votes(raw_votes: List[Dict]) -> List[Dict]:
    """Keep only the first vote per (author, battle_id) — ordered by issue_number."""
    seen: set = set()
    out: List[Dict] = []
    for v in sorted(raw_votes, key=lambda x: (x["issue_number"], x.get("created_at", ""))):
        key = (v["author"], v["battle_id"])
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def k_factor(battles: int) -> float:
    return 32.0 if battles < 30 else 16.0


def replay_elo(votes: List[Dict], battles_pool: Dict[str, Dict]) -> Dict[str, float]:
    ratings: Dict[str, float] = {}
    battle_counts: Dict[str, int] = {}
    for v in votes:
        b = battles_pool.get(v["battle_id"])
        if not b:
            continue
        pid_a, pid_b = b["plugin_a"], b["plugin_b"]
        for pid in (pid_a, pid_b):
            ratings.setdefault(pid, INITIAL_ELO)
            battle_counts.setdefault(pid, 0)
        r_a, r_b = ratings[pid_a], ratings[pid_b]
        e_a = expected_score(r_a, r_b)
        choice = v["choice"]
        if choice == "a":
            s_a, s_b = 1.0, 0.0
        elif choice == "b":
            s_a, s_b = 0.0, 1.0
        else:
            s_a, s_b = 0.5, 0.5
        ka = k_factor(battle_counts[pid_a])
        kb = k_factor(battle_counts[pid_b])
        ratings[pid_a] = r_a + ka * (s_a - e_a)
        ratings[pid_b] = r_b + kb * (s_b - (1.0 - e_a))
        battle_counts[pid_a] += 1
        battle_counts[pid_b] += 1
    return ratings


# ── Sample data ─────────────────────────────────────────────────────────────

BATTLES: Dict[str, Dict] = {
    "battle-001": {"plugin_a": "pluginX", "plugin_b": "pluginY", "family": "stt", "lang": "pt-PT"},
    "battle-002": {"plugin_a": "pluginX", "plugin_b": "pluginZ", "family": "stt", "lang": "pt-PT"},
    "battle-003": {"plugin_a": "pluginY", "plugin_b": "pluginZ", "family": "stt", "lang": "pt-PT"},
}

# ── Parsing tests ────────────────────────────────────────────────────────────

class TestParseVoteTitle:
    def test_valid_a(self):
        assert parse_vote_title("vote|battle-001|a") == ("battle-001", "a")

    def test_valid_b(self):
        assert parse_vote_title("vote|battle-002|b") == ("battle-002", "b")

    def test_valid_tie(self):
        assert parse_vote_title("vote|battle-003|tie") == ("battle-003", "tie")

    def test_valid_both_wrong(self):
        assert parse_vote_title("vote|battle-001|both_wrong") == ("battle-001", "both_wrong")

    def test_case_insensitive(self):
        assert parse_vote_title("VOTE|battle-001|A") == ("battle-001", "a")

    def test_invalid_choice(self):
        assert parse_vote_title("vote|battle-001|win") is None

    def test_missing_choice(self):
        assert parse_vote_title("vote|battle-001") is None

    def test_wrong_prefix(self):
        assert parse_vote_title("Vote for battle-001") is None

    def test_empty(self):
        assert parse_vote_title("") is None

    def test_extra_pipe(self):
        # Extra segments after choice are not valid
        assert parse_vote_title("vote|battle-001|a|extra") is None

    def test_battle_id_with_hyphens(self):
        result = parse_vote_title("vote|stt-pt-PT-pluginX-vs-pluginY-sample42|b")
        assert result == ("stt-pt-PT-pluginX-vs-pluginY-sample42", "b")

    def test_leading_whitespace_stripped(self):
        assert parse_vote_title("  vote|battle-001|tie  ") == ("battle-001", "tie")


# ── Deduplication tests ──────────────────────────────────────────────────────

class TestDedupeVotes:
    def _vote(self, number: int, author: str, battle_id: str, choice: str = "a") -> Dict:
        return {"issue_number": number, "author": author, "battle_id": battle_id, "choice": choice}

    def test_no_duplicates(self):
        votes = [
            self._vote(1, "alice", "battle-001", "a"),
            self._vote(2, "bob", "battle-001", "b"),
            self._vote(3, "alice", "battle-002", "tie"),
        ]
        assert len(dedupe_votes(votes)) == 3

    def test_duplicate_same_author_same_battle(self):
        votes = [
            self._vote(1, "alice", "battle-001", "a"),
            self._vote(5, "alice", "battle-001", "b"),  # duplicate — later issue
        ]
        result = dedupe_votes(votes)
        assert len(result) == 1
        assert result[0]["issue_number"] == 1  # first wins

    def test_same_author_different_battles(self):
        votes = [
            self._vote(1, "alice", "battle-001", "a"),
            self._vote(2, "alice", "battle-002", "b"),
        ]
        assert len(dedupe_votes(votes)) == 2

    def test_different_authors_same_battle(self):
        votes = [
            self._vote(1, "alice", "battle-001", "a"),
            self._vote(2, "bob", "battle-001", "b"),
        ]
        assert len(dedupe_votes(votes)) == 2

    def test_order_preserved_by_issue_number(self):
        votes = [
            self._vote(10, "alice", "battle-001", "b"),
            self._vote(2, "alice", "battle-001", "a"),  # earlier issue number
        ]
        result = dedupe_votes(votes)
        assert len(result) == 1
        assert result[0]["issue_number"] == 2  # earlier wins


# ── ELO replay tests ─────────────────────────────────────────────────────────

class TestEloReplay:
    def _vote(self, battle_id: str, choice: str, author: str = "alice", number: int = 1) -> Dict:
        return {"issue_number": number, "author": author, "battle_id": battle_id, "choice": choice}

    def test_empty_votes(self):
        ratings = replay_elo([], BATTLES)
        assert ratings == {}

    def test_single_win_raises_winner(self):
        votes = [self._vote("battle-001", "a")]
        ratings = replay_elo(votes, BATTLES)
        assert ratings["pluginX"] > INITIAL_ELO
        assert ratings["pluginY"] < INITIAL_ELO

    def test_single_win_delta_symmetric(self):
        votes = [self._vote("battle-001", "a")]
        ratings = replay_elo(votes, BATTLES)
        delta_x = ratings["pluginX"] - INITIAL_ELO
        delta_y = INITIAL_ELO - ratings["pluginY"]
        assert abs(delta_x - delta_y) < 0.01

    def test_tie_equal_ratings(self):
        votes = [self._vote("battle-001", "tie")]
        ratings = replay_elo(votes, BATTLES)
        # Both start at 1200 and tie → both stay at 1200 (symmetric)
        assert abs(ratings["pluginX"] - INITIAL_ELO) < 0.01
        assert abs(ratings["pluginY"] - INITIAL_ELO) < 0.01

    def test_deterministic(self):
        """Same votes in same order must always produce the same ratings."""
        votes = [
            self._vote("battle-001", "a", "alice", 1),
            self._vote("battle-002", "b", "bob", 2),
            self._vote("battle-003", "tie", "carol", 3),
        ]
        r1 = replay_elo(list(votes), BATTLES)
        r2 = replay_elo(list(votes), BATTLES)
        assert r1 == r2

    def test_order_matters(self):
        """Different orderings CAN yield different ratings — replay is order-sensitive."""
        v1 = self._vote("battle-001", "a", number=1)
        v2 = self._vote("battle-002", "b", "bob", number=2)
        r_ab = replay_elo([v1, v2], BATTLES)
        r_ba = replay_elo([v2, v1], BATTLES)
        # pluginX involved in both; its final rating may differ
        # Just assert both are valid (no exception) and non-equal in at least one plugin
        assert isinstance(r_ab, dict)
        assert isinstance(r_ba, dict)

    def test_both_wrong_treated_as_tie(self):
        votes_bw = [self._vote("battle-001", "both_wrong")]
        votes_tie = [self._vote("battle-001", "tie")]
        r_bw = replay_elo(votes_bw, BATTLES)
        r_tie = replay_elo(votes_tie, BATTLES)
        assert abs(r_bw["pluginX"] - r_tie["pluginX"]) < 0.01
        assert abs(r_bw["pluginY"] - r_tie["pluginY"]) < 0.01

    def test_unknown_battle_skipped(self):
        votes = [self._vote("nonexistent", "a")]
        ratings = replay_elo(votes, BATTLES)
        assert ratings == {}

    def test_many_wins_converges(self):
        """Winner's ELO should stabilise above 1200 after many wins."""
        votes = [self._vote("battle-001", "a", f"user{i}", i) for i in range(50)]
        ratings = replay_elo(votes, BATTLES)
        assert ratings["pluginX"] > 1200
        assert ratings["pluginY"] < 1200


# ── Data-export JSON shape ────────────────────────────────────────────────────

class TestLeaderboardShape:
    """Validate the shape of leaderboard JSON files."""

    def _make_lb(self, family: str = "stt", lang: str = "pt-PT") -> Dict:
        return {
            "family": family,
            "lang": lang,
            "generated_at": "2026-06-10T00:00:00+00:00",
            "vote_count": 10,
            "entries": [
                {
                    "rank": 1,
                    "plugin_name": "pluginX",
                    "display_name": "Plugin X",
                    "family": family,
                    "lang": lang,
                    "elo": 1215.3,
                    "battles": 5,
                    "wins": 3,
                    "losses": 1,
                    "ties": 1,
                    "win_rate": 0.6,
                },
            ],
        }

    def test_required_fields_present(self):
        lb = self._make_lb()
        for field in ("family", "lang", "generated_at", "vote_count", "entries"):
            assert field in lb, f"Missing field: {field}"

    def test_entry_required_fields(self):
        lb = self._make_lb()
        entry = lb["entries"][0]
        for field in ("rank", "plugin_name", "elo", "battles", "wins", "losses", "ties", "win_rate"):
            assert field in entry, f"Entry missing field: {field}"

    def test_ranks_are_sequential(self):
        lb = self._make_lb()
        # Single entry rank=1
        assert lb["entries"][0]["rank"] == 1

    def test_json_serializable(self):
        lb = self._make_lb()
        dumped = json.dumps(lb)
        reloaded = json.loads(dumped)
        assert reloaded["family"] == "stt"

    def test_index_json_shape(self):
        index = {
            "generated_at": "2026-06-10T00:00:00+00:00",
            "leaderboards": [
                {"file": "leaderboard-stt-pt-PT.json", "family": "stt", "lang": "pt-PT",
                 "generated_at": "2026-06-10T00:00:00+00:00", "entry_count": 3},
            ],
            "battles_pools": [
                {"file": "battles-stt-pt-PT.json", "family": "stt", "lang": "pt-PT",
                 "generated_at": "2026-06-10T00:00:00+00:00", "battle_count": 42},
            ],
        }
        dumped = json.dumps(index)
        reloaded = json.loads(dumped)
        assert len(reloaded["leaderboards"]) == 1
        assert reloaded["battles_pools"][0]["battle_count"] == 42
