"""Tests for arena.cli — vote parsing, tally replay, assemble pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.cli import (
    build_elo_board,
    dedupe_votes,
    load_battles_pools,
    load_elo_seeds,
    main,
    parse_vote_title,
)
from arena.elo import INITIAL_ELO
from arena.models import EloSeed, Modality


class TestParseVoteTitle:
    def test_valid_choices(self):
        for choice in ("a", "b", "tie", "both_wrong"):
            assert parse_vote_title(f"vote|abc123|{choice}") == ("abc123", choice)

    def test_case_insensitive(self):
        assert parse_vote_title("VOTE|abc|A") == ("abc", "a")

    def test_whitespace_stripped(self):
        assert parse_vote_title("  vote|abc|tie  ") == ("abc", "tie")

    def test_battle_id_with_hyphens(self):
        assert parse_vote_title("vote|intent-en-US-x-vs-y|b") == (
            "intent-en-US-x-vs-y", "b",
        )

    @pytest.mark.parametrize("title", [
        "", "vote|abc", "vote|abc|maybe", "hello|abc|a",
        "vote|a|b|c|d", "vote||a",
    ])
    def test_invalid(self, title):
        assert parse_vote_title(title) is None


class TestDedupeVotes:
    def _vote(self, n, author="alice", battle="b1", choice="a"):
        return {"issue_number": n, "author": author, "battle_id": battle,
                "choice": choice, "created_at": f"2026-01-01T00:00:{n:02d}"}

    def test_first_vote_wins(self):
        votes = dedupe_votes([
            self._vote(2, choice="b"),
            self._vote(1, choice="a"),
        ])
        assert len(votes) == 1
        assert votes[0]["choice"] == "a"  # lower issue number first

    def test_different_battles_kept(self):
        votes = dedupe_votes([self._vote(1), self._vote(2, battle="b2")])
        assert len(votes) == 2

    def test_different_authors_kept(self):
        votes = dedupe_votes([self._vote(1), self._vote(2, author="bob")])
        assert len(votes) == 2

    def test_output_ordered_by_issue_number(self):
        votes = dedupe_votes([
            self._vote(5, author="c"), self._vote(1, author="a"),
            self._vote(3, author="b"),
        ])
        assert [v["issue_number"] for v in votes] == [1, 3, 5]


def _seed(**over):
    base = dict(
        modality=Modality.INTENT, lang="en-US", generated_at="t",
        auto_vote_count=4,
        ratings={"x": 1220.0, "y": 1180.0},
        battles={"x": 4, "y": 4},
        wins={"x": 4, "y": 0},
        losses={"x": 0, "y": 4},
        ties={"x": 0, "y": 0},
        competitor_plugin={"x": "plug-x", "y": "plug-y"},
    )
    base.update(over)
    return EloSeed(**base)


BATTLE = {
    "battle_id": "bid1", "modality": "intent", "lang": "en-US",
    "dataset_id": "d", "sample_id": "s",
    "competitor_a": "x", "competitor_b": "y",
    "plugin_a": "plug-x", "plugin_b": "plug-y",
}


class TestBuildEloBoard:
    def test_seed_only(self):
        board = build_elo_board("intent", "en-US", _seed(), [], {})
        assert board.vote_count == 4
        assert board.human_vote_count == 0
        assert board.entries[0].competitor_id == "x"
        assert board.entries[0].rank == 1
        assert board.entries[0].plugin_id == "plug-x"

    def test_human_votes_on_top_of_seed(self):
        votes = [{"battle_id": "bid1", "choice": "b", "author": "alice",
                  "issue_number": 1, "created_at": ""}]
        board = build_elo_board(
            "intent", "en-US", _seed(), votes, {"bid1": BATTLE}
        )
        assert board.human_vote_count == 1
        y = next(e for e in board.entries if e.competitor_id == "y")
        assert y.human_votes == 1
        assert y.elo > 1180.0  # y won the human vote

    def test_unknown_battle_ignored(self):
        votes = [{"battle_id": "nope", "choice": "a", "author": "alice",
                  "issue_number": 1, "created_at": ""}]
        board = build_elo_board("intent", "en-US", _seed(), votes, {})
        assert board.human_vote_count == 0

    def test_no_seed_starts_at_initial(self):
        votes = [{"battle_id": "bid1", "choice": "tie", "author": "alice",
                  "issue_number": 1, "created_at": ""}]
        board = build_elo_board("intent", "en-US", None, votes, {"bid1": BATTLE})
        assert all(e.elo == pytest.approx(INITIAL_ELO) for e in board.entries)


def _write_predictions(tmp_path: Path) -> Path:
    preds = tmp_path / "predictions"
    preds.mkdir()
    for competitor, correct in (("good", True), ("bad", False)):
        rows = []
        for i in range(6):
            rows.append({
                "competitor_id": competitor,
                "sample_id": f"en-US/{i:05d}",
                "dataset_id": "intents-for-eval",
                "lang": "en-US",
                "plugin_id": f"plugin-{competitor}",
                "utterance": f"utterance number {i}",
                "reference_intent": "media:play_song",
                "prediction": "media:play_song" if correct else f"wrong{i}",
                "exact_match": correct,
            })
        (preds / f"{competitor}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    return preds


class TestAssemblePipeline:
    def test_assemble_writes_artifacts(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        rc = main_args_assemble(preds, out)
        assert rc == 0
        assert (out / "battles-intent-intents-for-eval-en-US.json").exists()
        assert (out / "benchmark-intent-intents-for-eval-en-US.json").exists()
        assert (out / "elo-seed-intent-en-US.json").exists()
        assert (out / "leaderboard-intent-en-US.json").exists()

        benchmark = json.loads((out / "benchmark-intent-intents-for-eval-en-US.json").read_text())
        assert benchmark["entries"][0]["competitor_id"] == "good"
        assert benchmark["entries"][0]["metrics"]["accuracy"] == 1.0

        seeds = load_elo_seeds(out)
        assert seeds[("intent", "en-US")].ratings["good"] > INITIAL_ELO

        # 6 blind battles (every sample disagrees) + the free-form pool
        assert (out / "battles-intent-freeform-en-US.json").exists()
        battles = load_battles_pools(out)
        assert len(battles) == 7  # 6 blind + 1 free-form pair (good vs bad)
        freeform = json.loads(
            (out / "battles-intent-freeform-en-US.json").read_text())
        assert freeform["dataset_id"] == "freeform"
        assert len(freeform["battles"]) == 1
        fb = freeform["battles"][0]
        assert {fb["competitor_a"], fb["competitor_b"]} == {"good", "bad"}
        assert fb["sample_id"] == "freeform"

    def test_assemble_then_tally_dry(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        # tally without --repo: regenerates leaderboard from seed only
        with pytest.raises(SystemExit) as exc:
            main(["tally", "--data-dir", str(out), "--output", str(out)])
        assert exc.value.code == 0
        board = json.loads((out / "leaderboard-intent-en-US.json").read_text())
        assert board["human_vote_count"] == 0
        assert board["entries"][0]["competitor_id"] == "good"

    def test_export_index(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        with pytest.raises(SystemExit) as exc:
            main(["export-index", "--data-dir", str(out),
                  "--output", str(out / "index.json")])
        assert exc.value.code == 0
        index = json.loads((out / "index.json").read_text())
        assert len(index["leaderboards"]) == 1
        assert len(index["benchmarks"]) == 1
        assert len(index["battles_pools"]) == 1
        assert index["battles_pools"][0]["count"] == 6

    def test_assemble_deterministic_battle_ids(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out1, out2 = tmp_path / "d1", tmp_path / "d2"
        assert main_args_assemble(preds, out1) == 0
        assert main_args_assemble(preds, out2) == 0
        ids1 = sorted(load_battles_pools(out1))
        ids2 = sorted(load_battles_pools(out2))
        assert ids1 == ids2


def _write_cross_league_predictions(tmp_path: Path) -> Path:
    """Template + keyword fighters answering the SAME en-US samples."""
    preds = tmp_path / "predictions"
    preds.mkdir()
    fighters = [
        ("padatious-medium", "intent_template", True),
        ("adapt-medium", "intent_keyword", False),
    ]
    for competitor, modality, correct in fighters:
        rows = []
        for i in range(5):
            rows.append({
                "competitor_id": competitor,
                "sample_id": f"en-US/{i:05d}",
                "dataset_id": "intents-for-eval",
                "lang": "en-US",
                "modality": modality,
                "plugin_id": f"plugin-{competitor}",
                "utterance": f"utterance number {i}",
                "reference_intent": "media:play_song",
                "prediction": "media:play_song" if correct else f"wrong{i}",
                "exact_match": correct,
            })
        (preds / f"{competitor}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    return preds


class TestBattleMerge:
    def test_intent_leagues_merge_for_battles_and_elo(self, tmp_path):
        preds = _write_cross_league_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0

        # battles + ELO merge into the open intent group...
        assert (out / "battles-intent-intents-for-eval-en-US.json").exists()
        assert (out / "elo-seed-intent-en-US.json").exists()
        # ...and the per-paradigm battle/ELO artifacts are NOT written
        assert not (out / "battles-intent_template-intents-for-eval-en-US.json").exists()
        assert not (out / "elo-seed-intent_template-en-US.json").exists()
        assert not (out / "elo-seed-intent_keyword-en-US.json").exists()

        # a battle pairs the template engine against the keyword engine
        pool = json.loads(
            (out / "battles-intent-intents-for-eval-en-US.json").read_text())
        pairs = {tuple(sorted((b["competitor_a"], b["competitor_b"])))
                 for b in pool["battles"]}
        assert ("adapt-medium", "padatious-medium") in pairs

        # benchmark boards stay per paradigm league
        assert (out / "benchmark-intent_template-intents-for-eval-en-US.json").exists()
        assert (out / "benchmark-intent_keyword-intents-for-eval-en-US.json").exists()

        # one merged ELO seed across both engines
        seed = json.loads((out / "elo-seed-intent-en-US.json").read_text())
        assert set(seed["ratings"]) == {"padatious-medium", "adapt-medium"}

    def test_stale_subleague_files_removed(self, tmp_path):
        preds = _write_cross_league_predictions(tmp_path)
        out = tmp_path / "data"
        out.mkdir()
        # a stale per-paradigm battle pool from a previous design
        (out / "battles-intent_template-intents-for-eval-en-US.json").write_text("{}")
        assert main_args_assemble(preds, out) == 0
        assert not (out / "battles-intent_template-intents-for-eval-en-US.json").exists()


def main_args_assemble(preds: Path, out: Path) -> int:
    try:
        main(["assemble", "--predictions", str(preds), "--output", str(out)])
    except SystemExit as exc:
        return exc.code
    return 0


class TestExportBestiary:
    def test_real_registry_export(self, tmp_path):
        registry_root = Path(__file__).parent.parent / "registry"
        out = tmp_path / "competitors.json"
        with pytest.raises(SystemExit) as exc:
            main(["export-bestiary", "--registry", str(registry_root),
                  "--output", str(out)])
        assert exc.value.code == 0
        payload = json.loads(out.read_text())
        ids = {c["competitor_id"] for c in payload["competitors"]}
        assert "padatious-medium" in ids
        entry = next(c for c in payload["competitors"]
                     if c["competitor_id"] == "padatious-medium")
        assert entry["species"] == "PadatiousPipeline"
        assert entry["types"]
