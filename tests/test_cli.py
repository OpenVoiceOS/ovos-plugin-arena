"""Tests for arena.cli — vote parsing, tally replay, assemble pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import arena.cli as arena_cli
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


class _StubCompetitor:
    def __init__(self, competitor_id):
        self.competitor_id = competitor_id


# Fictional competitor ids used by this file's fixtures (e.g. "good"/"bad"
# in `_write_predictions`) predate the board-truth registry filter added to
# ``arena.predictions.group_rows`` (rows whose competitor_id is not in the
# current registry are now excluded from boards — see
# tests/test_predictions.py::TestGroupRows for the filter's own coverage).
# Stub the registry here so this file's end-to-end fixtures keep exercising
# the assemble pipeline without being coupled to the real registry contents.
_LEGACY_TEST_COMPETITOR_IDS = {
    "intent": {"good", "bad"},
    "intent_template": {"padatious-medium"},
    "intent_keyword": {"adapt-medium"},
    "stt": {"base-pt", "small-pt", "comp-a", "comp-b", "whisper-tiny"},
    "tts": {"piper-a"},
}


@pytest.fixture(autouse=True)
def _permissive_registry(monkeypatch):
    import registry.loaders as loaders_mod

    def fake_list_competitors(modality=None):
        ids = _LEGACY_TEST_COMPETITOR_IDS.get(modality, set())
        return [_StubCompetitor(cid) for cid in ids]

    monkeypatch.setattr(loaders_mod, "list_competitors", fake_list_competitors)


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
        # x beat y every time, at the reduced auto-vote weight
        # (BT_AUTO_WEIGHT = 0.25) — mirrors what EloLedger.apply(auto=True)
        # would have accumulated for 4 auto battles.
        pairwise_wins={"x": {"y": 1.0}, "y": {"x": 0.0}},
        pairwise_games={"x": {"y": 1.0}, "y": {"x": 1.0}},
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

    def test_bt_rating_ranks_and_bounds_by_ci(self):
        votes = [{"battle_id": "bid1", "choice": "b", "author": "alice",
                  "issue_number": 1, "created_at": ""}]
        board = build_elo_board("intent", "en-US", _seed(), votes, {"bid1": BATTLE})
        x = next(e for e in board.entries if e.competitor_id == "x")
        y = next(e for e in board.entries if e.competitor_id == "y")
        assert x.bt_rating is not None and y.bt_rating is not None
        assert x.ci_lower is not None and x.ci_upper is not None
        assert x.ci_lower <= x.bt_rating <= x.ci_upper
        assert board.entries[0].rank == 1
        # ranking follows bt_rating, not the legacy sequential elo column
        assert board.entries == sorted(
            board.entries, key=lambda e: (-e.bt_rating, e.competitor_id)
        )

    def test_bt_rating_deterministic_across_rebuilds(self):
        votes = [
            {"battle_id": "bid1", "choice": "b", "author": "alice",
             "issue_number": 1, "created_at": ""},
            {"battle_id": "bid1", "choice": "a", "author": "bob",
             "issue_number": 2, "created_at": ""},
        ]
        board1 = build_elo_board("intent", "en-US", _seed(), votes, {"bid1": BATTLE})
        board2 = build_elo_board("intent", "en-US", _seed(), votes, {"bid1": BATTLE})
        ratings1 = {e.competitor_id: (e.bt_rating, e.ci_lower, e.ci_upper) for e in board1.entries}
        ratings2 = {e.competitor_id: (e.bt_rating, e.ci_lower, e.ci_upper) for e in board2.entries}
        assert ratings1 == ratings2

    def test_provisional_flag_below_threshold(self):
        board = build_elo_board("intent", "en-US", _seed(), [], {})
        assert board.provisional is True

    def test_provisional_flag_clears_with_enough_human_votes(self):
        votes = [
            {"battle_id": "bid1", "choice": "b", "author": f"voter{i}",
             "issue_number": i, "created_at": ""}
            for i in range(10)
        ]
        board = build_elo_board("intent", "en-US", _seed(), votes, {"bid1": BATTLE})
        assert board.provisional is False


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

        # §leagues — single source of truth for the frontend's league tabs
        league_ids = [entry["id"] for entry in index["leagues"]]
        assert league_ids == [
            "intent_template", "intent_keyword", "intent",
            "stt", "tts", "wake_word", "vad", "g2p",
        ]
        for entry in index["leagues"]:
            assert set(entry) == {"id", "label", "battle_group", "order", "voteless"}
        intent_entries = {e["id"]: e for e in index["leagues"]}
        # each intent league is its own battle group — no shared pool
        assert intent_entries["intent_template"]["battle_group"] == "intent_template"
        assert intent_entries["intent_keyword"]["battle_group"] == "intent_keyword"
        assert intent_entries["intent"]["battle_group"] == "intent"
        assert intent_entries["stt"]["battle_group"] == "stt"

    def test_assemble_deterministic_battle_ids(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out1, out2 = tmp_path / "d1", tmp_path / "d2"
        assert main_args_assemble(preds, out1) == 0
        assert main_args_assemble(preds, out2) == 0
        ids1 = sorted(load_battles_pools(out1))
        ids2 = sorted(load_battles_pools(out2))
        assert ids1 == ids2

    def test_assemble_then_tally_deterministic_end_to_end(self, tmp_path):
        """§P5: rerunning assemble + tally over the same corpus and vote log
        is byte-identical, including the Bradley-Terry rating and its
        bootstrap CI — not just the legacy sequential ELO."""
        preds = _write_predictions(tmp_path)
        out1, out2 = tmp_path / "d1", tmp_path / "d2"
        assert main_args_assemble(preds, out1) == 0
        assert main_args_assemble(preds, out2) == 0
        with pytest.raises(SystemExit):
            main(["tally", "--data-dir", str(out1), "--output", str(out1)])
        with pytest.raises(SystemExit):
            main(["tally", "--data-dir", str(out2), "--output", str(out2)])

        board1 = json.loads((out1 / "leaderboard-intent-en-US.json").read_text())
        board2 = json.loads((out2 / "leaderboard-intent-en-US.json").read_text())
        board1.pop("generated_at")
        board2.pop("generated_at")
        assert board1 == board2


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
    def test_intent_leagues_never_pool_battles_or_elo(self, tmp_path):
        """§R# — each intent league (keyword/template/open) is fully separate:
        its own battles pool and its own ELO seed. A template engine is
        never paired against a keyword engine."""
        preds = _write_cross_league_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0

        # no merged "intent" battle/ELO pool is produced — the fixture only
        # has template + keyword fighters, no open-league ones
        assert not (out / "battles-intent-intents-for-eval-en-US.json").exists()
        assert not (out / "elo-seed-intent-en-US.json").exists()

        # each paradigm gets its own battles pool and ELO seed
        assert (out / "battles-intent_template-intents-for-eval-en-US.json").exists()
        assert (out / "battles-intent_keyword-intents-for-eval-en-US.json").exists()
        assert (out / "elo-seed-intent_template-en-US.json").exists()
        assert (out / "elo-seed-intent_keyword-en-US.json").exists()

        # a template-league battle only ever pairs template fighters (here,
        # just the one, so no battles are assembled) — the keyword and
        # template engines are never paired against each other
        template_pool = json.loads(
            (out / "battles-intent_template-intents-for-eval-en-US.json").read_text())
        keyword_pool = json.loads(
            (out / "battles-intent_keyword-intents-for-eval-en-US.json").read_text())
        all_pairs = {
            tuple(sorted((b["competitor_a"], b["competitor_b"])))
            for pool in (template_pool, keyword_pool) for b in pool["battles"]
        }
        assert ("adapt-medium", "padatious-medium") not in all_pairs

        # benchmark boards stay per paradigm league (unchanged)
        assert (out / "benchmark-intent_template-intents-for-eval-en-US.json").exists()
        assert (out / "benchmark-intent_keyword-intents-for-eval-en-US.json").exists()

        # each ELO seed only rates its own league's fighter
        template_seed = json.loads(
            (out / "elo-seed-intent_template-en-US.json").read_text())
        keyword_seed = json.loads(
            (out / "elo-seed-intent_keyword-en-US.json").read_text())
        assert set(template_seed["ratings"]) == {"padatious-medium"}
        assert set(keyword_seed["ratings"]) == {"adapt-medium"}

    def test_no_stale_subleague_cleanup_needed(self, tmp_path):
        """§R# — battle_group is now identity for every modality, so
        ``_clean_merged_artifacts`` never removes a live league's own
        files; a pre-existing file for a still-active league survives."""
        preds = _write_cross_league_predictions(tmp_path)
        out = tmp_path / "data"
        out.mkdir()
        (out / "battles-intent_template-intents-for-eval-en-US.json").write_text("{}")
        assert main_args_assemble(preds, out) == 0
        # overwritten with real content by this run, not deleted
        pool = json.loads(
            (out / "battles-intent_template-intents-for-eval-en-US.json").read_text())
        assert pool.get("modality") == "intent_template"


def main_args_assemble(preds: Path, out: Path) -> int:
    try:
        main(["assemble", "--predictions", str(preds), "--output", str(out)])
    except SystemExit as exc:
        return exc.code
    return 0


class TestRegistryDefaultPredictions:
    def test_empty_predictions_uses_registry_repos(self, tmp_path, monkeypatch):
        """Without --predictions, sources come from list_prediction_repos()."""
        import registry.loaders as loaders

        preds = _write_predictions(tmp_path)
        monkeypatch.setattr(
            loaders, "list_prediction_repos", lambda: [str(preds)]
        )
        out = tmp_path / "data"
        with pytest.raises(SystemExit) as exc:
            main(["assemble", "--output", str(out)])
        assert exc.value.code == 0
        assert (out / "leaderboard-intent-en-US.json").exists()

    def test_explicit_predictions_override_registry(self, tmp_path, monkeypatch):
        import registry.loaders as loaders

        preds = _write_predictions(tmp_path)
        monkeypatch.setattr(
            loaders, "list_prediction_repos",
            lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
        )
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0


class TestTimestampStability:
    def test_reassemble_leaves_identical_artifacts_untouched(self, tmp_path):
        """Same predictions twice → byte-identical artifacts (stable
        generated_at), so the workflow commit guard sees an empty diff."""
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        before = {p.name: p.read_bytes() for p in out.glob("*.json")}
        assert main_args_assemble(preds, out) == 0
        after = {p.name: p.read_bytes() for p in out.glob("*.json")}
        assert before == after

    def test_changed_content_still_rewrites(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        board_path = out / "benchmark-intent-intents-for-eval-en-US.json"
        board = json.loads(board_path.read_text())
        board["entries"] = []
        board_path.write_text(json.dumps(board))
        assert main_args_assemble(preds, out) == 0
        assert json.loads(board_path.read_text())["entries"]

    def test_export_index_stable(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        index_path = out / "index.json"
        for _ in range(2):
            with pytest.raises(SystemExit) as exc:
                main(["export-index", "--data-dir", str(out),
                      "--output", str(index_path)])
            assert exc.value.code == 0
        first = index_path.read_bytes()
        with pytest.raises(SystemExit):
            main(["export-index", "--data-dir", str(out),
                  "--output", str(index_path)])
        assert index_path.read_bytes() == first

    def test_export_bestiary_stable(self, tmp_path):
        registry_root = Path(__file__).parent.parent / "registry"
        out = tmp_path / "competitors.json"
        for _ in range(2):
            with pytest.raises(SystemExit) as exc:
                main(["export-bestiary", "--registry", str(registry_root),
                      "--output", str(out)])
            assert exc.value.code == 0
        first = out.read_bytes()
        with pytest.raises(SystemExit):
            main(["export-bestiary", "--registry", str(registry_root),
                  "--output", str(out)])
        assert out.read_bytes() == first

    def test_tally_zero_votes_still_rebuilds_board_from_current_seed(self, tmp_path):
        """No new valid votes → `tally` still rebuilds every leaderboard
        from the current elo-seed/battles pool (a pure function, per R19).

        `build_elo_board` is deterministic, so a rebuild with an unchanged
        seed and zero votes reproduces byte-for-byte the same *standings*
        (mismatches would only show up via `_diff_board`, not a raw string
        compare) — but the file is NOT left untouched: hand-edits/corruption
        on disk must not survive a tally run, and a same-day `assemble`
        re-run that changes the seed (more predictions loaded) must
        propagate into the board even when zero votes were cast this run
        (see tests/test_verify_replay.py for that scenario). This replaces
        the old byte-identity assertion, which encoded the bug where a
        changed seed left a stale, verify-replay-breaking board on disk."""
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        board_path = out / "leaderboard-intent-en-US.json"
        # Corrupt the on-disk board with a sentinel: a rebuild must drop it.
        board_path.write_text(board_path.read_text() + "// sentinel\n")
        with pytest.raises(SystemExit) as exc:
            main(["tally", "--data-dir", str(out), "--output", str(out)])
        assert exc.value.code == 0
        assert not board_path.read_text().endswith("// sentinel\n")
        # The rebuilt board is still valid JSON with the same standings.
        json.loads(board_path.read_text())


def _write_stt_predictions(tmp_path: Path, competitors: dict[str, float]) -> Path:
    """*competitors*: {competitor_id: wer} — one row per competitor, same
    sample, so the assembler can discriminate on WER."""
    preds = tmp_path / "stt_predictions"
    preds.mkdir(parents=True, exist_ok=True)
    for competitor, wer in competitors.items():
        row = {
            "competitor_id": competitor,
            "sample_id": "pt-PT/00000",
            "dataset_id": "minds14-pt-PT",
            "lang": "pt-PT",
            "plugin_id": f"plugin-{competitor}",
            "audio_url": "https://example.com/a.wav",
            "reference_text": "ligar o alarme",
            "prediction": "ligar o alarme" if wer == 0.0 else "ligar alarme errado",
            "wer": wer,
        }
        (preds / f"{competitor}.jsonl").write_text(json.dumps(row) + "\n")
    return preds


class TestAssembleLeaderboardSeedGap:
    """Regression for the ghost-fighter bug: a leaderboard file that already
    exists is only ever regenerated by `tally`, and `tally` only rewrites
    boards when a vote is counted *somewhere* in that run. A fighter whose
    predictions first appear after the board already exists — and whose
    (modality, lang) never collects a human vote — could sit off the
    leaderboard forever even though it has real benchmark rows and a real
    ELO seed. `assemble` must resync every seeded fighter onto the board on
    every run, not just create the file once."""

    def test_new_fighter_added_after_board_exists_appears_on_reassemble(self, tmp_path):
        out = tmp_path / "data"

        # Round 1: only the weaker fighter has predictions — board bootstraps.
        preds1 = _write_stt_predictions(tmp_path / "r1", {"base-pt": 0.5})
        rc = main_args_assemble(preds1, out)
        assert rc == 0
        board_path = out / "leaderboard-stt-pt-PT.json"
        assert board_path.exists()
        board = json.loads(board_path.read_text())
        assert {e["competitor_id"] for e in board["entries"]} == {"base-pt"}

        # Round 2: a stronger fighter's predictions land (registry onboards
        # a new plugin). The board file already exists and no vote has ever
        # been cast, so before the fix this second run never touched it.
        preds2 = _write_stt_predictions(
            tmp_path / "r2", {"base-pt": 0.5, "small-pt": 0.1}
        )
        rc = main_args_assemble(preds2, out)
        assert rc == 0

        seed = load_elo_seeds(out)[("stt", "pt-PT")]
        assert set(seed.ratings) == {"base-pt", "small-pt"}

        board = json.loads(board_path.read_text())
        ids = {e["competitor_id"] for e in board["entries"]}
        assert ids == {"base-pt", "small-pt"}, (
            "small-pt has predictions and a seed rating but is missing from "
            "the leaderboard"
        )
        small = next(e for e in board["entries"] if e["competitor_id"] == "small-pt")
        base = next(e for e in board["entries"] if e["competitor_id"] == "base-pt")
        # small-pt's rating reflects the seed (a single-sample pair is not
        # statistically significant, so both may sit at the 1200 baseline —
        # the point under test is that it is present at all, not silently
        # dropped off the board).
        assert small["elo"] == pytest.approx(seed.ratings["small-pt"], abs=0.01)
        assert base["elo"] == pytest.approx(seed.ratings["base-pt"], abs=0.01)

    def test_missing_fighter_appended_without_disturbing_human_vote_state(self, tmp_path):
        """When the on-disk board already carries real human-vote state,
        assemble must not clobber it — it only appends the missing fighter."""
        out = tmp_path / "data"
        preds1 = _write_stt_predictions(tmp_path / "r1", {"base-pt": 0.5})
        assert main_args_assemble(preds1, out) == 0

        board_path = out / "leaderboard-stt-pt-PT.json"
        board = json.loads(board_path.read_text())
        # Simulate a prior `tally` run that replayed a real human vote.
        board["human_vote_count"] = 3
        board["entries"][0]["human_votes"] = 3
        board["entries"][0]["elo"] = 1400.0
        board["entries"][0]["bt_rating"] = 1400.0
        board_path.write_text(json.dumps(board))

        preds2 = _write_stt_predictions(
            tmp_path / "r2", {"base-pt": 0.5, "small-pt": 0.1}
        )
        assert main_args_assemble(preds2, out) == 0

        board = json.loads(board_path.read_text())
        base = next(e for e in board["entries"] if e["competitor_id"] == "base-pt")
        assert base["human_votes"] == 3
        assert base["elo"] == 1400.0  # untouched, not recomputed from seed
        ids = {e["competitor_id"] for e in board["entries"]}
        assert "small-pt" in ids


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


class TestExportEvidence:
    """§ evidence page — counts must be an exact function of the fixture
    registry + data dir, not guesses. Every number asserted here is
    independently computed from the fixture below, not read back from the
    command's own output."""

    def _write_registry(self, root: Path) -> None:
        (root / "competitors" / "stt").mkdir(parents=True)
        (root / "competitors" / "tts").mkdir(parents=True)
        (root / "datasets" / "stt").mkdir(parents=True)
        (root / "datasets" / "tts").mkdir(parents=True)

        for cid in ("whisper-tiny", "onnx-asr-a", "onnx-asr-b"):
            (root / "competitors" / "stt" / f"{cid}.json").write_text(json.dumps({
                "competitor_id": cid, "modality": "stt",
                "plugin": f"ovos-stt-plugin-{cid}", "species": "Whisper",
            }))
        (root / "competitors" / "tts" / "piper-a.json").write_text(json.dumps({
            "competitor_id": "piper-a", "modality": "tts",
            "plugin": "ovos-tts-plugin-piper-a", "species": "Piper",
        }))

        # Two STT eval datasets: one with a published benchmark board, one
        # without — the exact "gap" shape the evidence page must surface.
        (root / "datasets" / "stt" / "fleurs-en.json").write_text(json.dumps({
            "dataset_id": "fleurs-en", "modality": "stt",
            "source": {"type": "huggingface", "hf_id": "google/fleurs",
                       "revision": "main", "split": "test"},
            "lang": "en", "role": "eval",
            "predictions_hf": "OpenVoiceOS/ovos-stt-bench-fleurs-en",
        }))
        (root / "datasets" / "stt" / "fleurs-de.json").write_text(json.dumps({
            "dataset_id": "fleurs-de", "modality": "stt",
            "source": {"type": "huggingface", "hf_id": "google/fleurs",
                       "revision": "main", "split": "test"},
            "lang": "de", "role": "eval",
            "predictions_hf": "OpenVoiceOS/ovos-stt-bench-fleurs-de",
        }))
        # One TTS eval dataset, no benchmark board published for it yet.
        (root / "datasets" / "tts" / "intents-prompts.json").write_text(json.dumps({
            "dataset_id": "intents-prompts", "modality": "tts",
            "source": {"type": "huggingface", "hf_id": "OpenVoiceOS/x",
                       "revision": "main", "split": "test"},
            "lang": "en", "role": "eval",
            "predictions_hf": "OpenVoiceOS/ovos-tts-bench-intents-prompts",
        }))

    def test_counts_match_fixture(self, tmp_path):
        registry_root = tmp_path / "registry"
        self._write_registry(registry_root)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "benchmark-stt-fleurs-en.json").write_text(json.dumps({
            "modality": "stt", "dataset_id": "fleurs-en", "lang": "en",
            "entries": [{"competitor_id": "whisper-tiny"}],
        }))
        (data_dir / "leaderboard-stt-en.json").write_text(json.dumps({
            "modality": "stt", "lang": "en", "entries": [],
        }))

        # Built at runtime (not a literal "R1" etc. substring in this file)
        # so tests/test_spec_coverage.py's raw-text R-number scan across
        # tests/*.py doesn't mistake this fixture spec for a real citation.
        r = "R"
        spec_path = tmp_path / "SPECIFICATION.md"
        spec_path.write_text(
            f"- **{r}1 — One.** Text.\n- **{r}2 — Two.** Text.\n"
            f"- **{r}2a — Two-a.** Text citing {r}1 again.\n"
        )

        out = tmp_path / "evidence.json"
        with pytest.raises(SystemExit) as exc:
            main(["export-evidence", "--registry", str(registry_root),
                  "--data-dir", str(data_dir), "--spec", str(spec_path),
                  "--output", str(out)])
        assert exc.value.code == 0

        payload = json.loads(out.read_text())
        assert payload["totals"]["fighters"] == 4
        assert payload["totals"]["datasets"] == 3
        assert payload["totals"]["spec_requirements"] == 3
        # § fighter-coverage rollup — only "whisper-tiny" has any rows
        # anywhere (one benchmark-board entry); the other 3 fighters
        # (onnx-asr-a, onnx-asr-b, piper-a) are registered-but-untested
        # ghosts, matching this fixture's data files exactly.
        assert payload["totals"]["fighter_coverage"] == {
            "registered": 4, "on_boards": 1, "ghosts": 3,
        }
        assert "notes" in payload and "fighter_coverage" in payload["notes"]

        by_id = {row["id"]: row for row in payload["leagues"]}
        stt = by_id["stt"]
        assert stt["fighters"] == 3
        assert stt["datasets"] == 2
        assert stt["datasets_with_predictions"] == 1
        assert stt["benchmark_boards"] == 1
        assert stt["elo_leaderboards"] == 1
        links = {link["dataset_id"]: link["has_predictions"]
                 for link in stt["predictions_links"]}
        assert links == {"fleurs-en": True, "fleurs-de": False}
        assert stt["fighter_coverage"] == {
            "registered": 3, "on_boards": 1, "ghosts": 2,
        }

        tts = by_id["tts"]
        assert tts["fighters"] == 1
        assert tts["datasets"] == 1
        assert tts["datasets_with_predictions"] == 0
        assert tts["benchmark_boards"] == 0
        assert tts["elo_leaderboards"] == 0
        assert tts["fighter_coverage"] == {
            "registered": 1, "on_boards": 0, "ghosts": 1,
        }

        vad = by_id["vad"]
        assert vad["fighters"] == 0
        assert vad["datasets"] == 0
        assert vad["fighter_coverage"] == {
            "registered": 0, "on_boards": 0, "ghosts": 0,
        }

    def test_fighter_coverage_counts_battles_pool_too(self, tmp_path):
        """A fighter with zero benchmark-board rows but a battles-pool
        appearance (competitor_a/competitor_b) must still count as
        on_boards, not a ghost — battles are a separate coverage source."""
        registry_root = tmp_path / "registry"
        self._write_registry(registry_root)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # onnx-asr-a has no benchmark-board row, only a battles-pool row.
        (data_dir / "battles-stt-fleurs-en.json").write_text(json.dumps({
            "battles": [{
                "battle_id": "b1", "modality": "stt", "lang": "en",
                "competitor_a": "onnx-asr-a", "competitor_b": "onnx-asr-b",
            }],
        }))

        spec_path = tmp_path / "SPECIFICATION.md"
        r = "R"
        spec_path.write_text(f"- **{r}1 — One.** Text.\n")

        out = tmp_path / "evidence.json"
        with pytest.raises(SystemExit) as exc:
            main(["export-evidence", "--registry", str(registry_root),
                  "--data-dir", str(data_dir), "--spec", str(spec_path),
                  "--output", str(out)])
        assert exc.value.code == 0

        payload = json.loads(out.read_text())
        by_id = {row["id"]: row for row in payload["leagues"]}
        stt = by_id["stt"]
        # onnx-asr-a and onnx-asr-b both counted via the battles pool;
        # whisper-tiny is the ghost this time (no benchmark board, no battle).
        assert stt["fighter_coverage"] == {
            "registered": 3, "on_boards": 2, "ghosts": 1,
        }

    def test_stable_regeneration(self, tmp_path):
        registry_root = tmp_path / "registry"
        self._write_registry(registry_root)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        spec_path = tmp_path / "SPECIFICATION.md"
        spec_path.write_text(f"- **{'R'}1 — One.** Text.\n")
        out = tmp_path / "evidence.json"

        for _ in range(2):
            with pytest.raises(SystemExit) as exc:
                main(["export-evidence", "--registry", str(registry_root),
                      "--data-dir", str(data_dir), "--spec", str(spec_path),
                      "--output", str(out)])
            assert exc.value.code == 0
        first = out.read_bytes()
        with pytest.raises(SystemExit):
            main(["export-evidence", "--registry", str(registry_root),
                  "--data-dir", str(data_dir), "--spec", str(spec_path),
                  "--output", str(out)])
        assert out.read_bytes() == first


class TestAuditSeeds:
    def test_reports_capped_pair(self, tmp_path, capsys):
        preds = tmp_path / "predictions"
        preds.mkdir()
        for competitor, correct in (("good", True), ("bad", False)):
            rows = [
                {
                    "competitor_id": competitor, "sample_id": f"en-US/{i:05d}",
                    "dataset_id": "intents-for-eval", "lang": "en-US",
                    "plugin_id": f"plugin-{competitor}", "utterance": f"utterance {i}",
                    "reference_intent": "media:play_song",
                    "prediction": "media:play_song" if correct else f"wrong{i}",
                    "exact_match": correct,
                }
                for i in range(220)
            ]
            (preds / f"{competitor}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0

        with pytest.raises(SystemExit) as exc:
            main(["audit-seeds", "--data-dir", str(out)])
        assert exc.value.code == 0
        printed = capsys.readouterr().out
        assert "CAPPED" in printed
        assert "1/1 pair(s) at the auto-vote weight cap." in printed

    def test_no_seeds_found_exits_cleanly(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit) as exc:
            main(["audit-seeds", "--data-dir", str(empty)])
        assert exc.value.code == 0


class TestTallyFraudIntegration:
    """cmd_tally wired to arena.fraud — network calls monkeypatched."""

    def _setup_board(self, tmp_path):
        preds = _write_predictions(tmp_path)
        out = tmp_path / "data"
        assert main_args_assemble(preds, out) == 0
        battles = load_battles_pools(out)
        battle_id = next(iter(battles))
        return out, battle_id

    def _fake_issue(self, number, author, battle_id, choice="a",
                     created_at="2026-06-01T00:00:00Z", state="OPEN"):
        return {
            "number": number,
            "title": f"vote|{battle_id}|{choice}",
            "author": {"login": author},
            "createdAt": created_at,
            "state": state,
            "labels": [],
        }

    def test_new_account_vote_excluded_from_board(self, tmp_path, monkeypatch):
        out, battle_id = self._setup_board(tmp_path)
        issue = self._fake_issue(1, "newbie", battle_id,
                                  created_at="2026-06-05T00:00:00Z")
        monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: [issue])
        monkeypatch.setattr(
            arena_cli, "fetch_account_created_at",
            lambda login: "2026-06-01T00:00:00Z",  # 4 days old, gate is 7
        )
        closed: list[int] = []
        monkeypatch.setattr(
            arena_cli, "close_issue",
            lambda repo, number, comment, add_label="": closed.append(number),
        )

        with pytest.raises(SystemExit) as exc:
            main(["tally", "--data-dir", str(out), "--output", str(out),
                  "--repo", "fake/repo"])
        assert exc.value.code == 0

        board = json.loads((out / "leaderboard-intent-en-US.json").read_text())
        assert board["human_vote_count"] == 0  # excluded, not counted

        audit = json.loads((out / "vote-audit.json").read_text())
        assert audit["counted"] == 0
        assert audit["discarded"][0]["reason"] == "account_too_new"
        assert closed == [1]  # still commented/closed despite not counting

    def test_established_account_vote_counted(self, tmp_path, monkeypatch):
        out, battle_id = self._setup_board(tmp_path)
        issue = self._fake_issue(2, "veteran", battle_id,
                                  created_at="2026-06-10T00:00:00Z")
        monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: [issue])
        monkeypatch.setattr(
            arena_cli, "fetch_account_created_at",
            lambda login: "2020-01-01T00:00:00Z",
        )
        monkeypatch.setattr(arena_cli, "close_issue", lambda *a, **k: None)

        with pytest.raises(SystemExit) as exc:
            main(["tally", "--data-dir", str(out), "--output", str(out),
                  "--repo", "fake/repo"])
        assert exc.value.code == 0

        board = json.loads((out / "leaderboard-intent-en-US.json").read_text())
        assert board["human_vote_count"] == 1
        audit = json.loads((out / "vote-audit.json").read_text())
        assert audit["counted"] == 1
        assert audit["discarded"] == []

    def test_age_cache_persisted_and_reused(self, tmp_path, monkeypatch):
        out, battle_id = self._setup_board(tmp_path)
        issue = self._fake_issue(3, "cached-user", battle_id,
                                  created_at="2026-06-10T00:00:00Z")
        monkeypatch.setattr(arena_cli, "fetch_vote_issues", lambda repo: [issue])
        monkeypatch.setattr(arena_cli, "close_issue", lambda *a, **k: None)

        fetch_calls = []

        def _fetch(login):
            fetch_calls.append(login)
            return "2020-01-01T00:00:00Z"

        monkeypatch.setattr(arena_cli, "fetch_account_created_at", _fetch)

        with pytest.raises(SystemExit):
            main(["tally", "--data-dir", str(out), "--output", str(out),
                  "--repo", "fake/repo"])
        assert fetch_calls == ["cached-user"]
        assert (out / "voter-age-cache.json").exists()

        # second run: same author, network must not be hit again
        monkeypatch.setattr(
            arena_cli, "fetch_account_created_at",
            lambda login: (_ for _ in ()).throw(AssertionError("should be cached")),
        )
        with pytest.raises(SystemExit):
            main(["tally", "--data-dir", str(out), "--output", str(out),
                  "--repo", "fake/repo"])

    def test_only_open_issues_get_closed(self, tmp_path, monkeypatch):
        out, battle_id = self._setup_board(tmp_path)
        open_issue = self._fake_issue(4, "alice", battle_id, state="OPEN")
        closed_issue = self._fake_issue(5, "bob", battle_id, choice="b", state="CLOSED")
        monkeypatch.setattr(
            arena_cli, "fetch_vote_issues", lambda repo: [open_issue, closed_issue]
        )
        monkeypatch.setattr(
            arena_cli, "fetch_account_created_at", lambda login: "2020-01-01T00:00:00Z",
        )
        closed_calls: list[int] = []
        monkeypatch.setattr(
            arena_cli, "close_issue",
            lambda repo, number, comment, add_label="": closed_calls.append(number),
        )

        with pytest.raises(SystemExit):
            main(["tally", "--data-dir", str(out), "--output", str(out),
                  "--repo", "fake/repo"])

        # both votes count toward the board (full history replay)...
        board = json.loads((out / "leaderboard-intent-en-US.json").read_text())
        assert board["human_vote_count"] == 2
        # ...but only the still-open issue gets a close/comment action
        assert closed_calls == [4]
