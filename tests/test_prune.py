"""Prune-guard tests: an artifact this run did not touch survives when the
registry still produces its key; only genuinely dead keys are deleted."""

import json
from pathlib import Path

import pytest

from arena import cli


def _touch(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    path.write_text(json.dumps({"stub": True}))
    return path


@pytest.fixture()
def registry_stub(monkeypatch):
    monkeypatch.setattr(
        cli, "_registry_dataset_langs",
        lambda: {"minds14-en-US": {"en-US"}, "intents-for-eval": {"en-US", "fr-FR"}},
    )
    monkeypatch.setattr(
        cli, "_registry_battle_groups",
        lambda: {"stt", "intent_template"},
    )
    monkeypatch.setattr(
        cli, "_registry_dataset_modalities",
        lambda: {"minds14-en-US": "stt", "intents-for-eval": "intent_template"},
    )
    monkeypatch.setattr(cli, "_registry_trained_on_by_modality", lambda: {})


def test_untouched_live_leaderboard_survives_transient_miss(tmp_path, registry_stub):
    kept = _touch(tmp_path, "leaderboard-stt-en-US.json")
    seed = _touch(tmp_path, "elo-seed-intent_template-fr-FR.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert pruned == []
    assert kept.exists() and seed.exists()


def test_dead_short_lang_leaderboard_is_pruned(tmp_path, registry_stub):
    dead = _touch(tmp_path, "leaderboard-stt-en.json")
    dead_seed = _touch(tmp_path, "elo-seed-stt-en.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert sorted(pruned) == ["elo-seed-stt-en.json", "leaderboard-stt-en.json"]
    assert not dead.exists() and not dead_seed.exists()


def test_dead_group_leaderboard_is_pruned(tmp_path, registry_stub):
    dead = _touch(tmp_path, "leaderboard-wake_word-en-US.json")
    cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert not dead.exists()


def test_untouched_live_benchmark_survives(tmp_path, registry_stub):
    kept = _touch(tmp_path, "benchmark-stt-minds14-en-US-en-US.json")
    dead = _touch(tmp_path, "benchmark-stt-minds14-en-US-en.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert kept.exists()
    assert not dead.exists()
    assert pruned == ["benchmark-stt-minds14-en-US-en.json"]


def test_untouched_live_freeform_battles_survives_transient_miss(tmp_path, registry_stub):
    # Group-scoped freeform pool (dataset_id "freeform", not a real dataset
    # id) — must not be treated as a dead dataset-scoped pool just because
    # "freeform" never appears as a dataset_id in the registry.
    kept = _touch(tmp_path, "battles-stt-freeform-en-US.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert pruned == []
    assert kept.exists()


def test_dead_lang_freeform_battles_is_pruned(tmp_path, registry_stub):
    dead = _touch(tmp_path, "battles-stt-freeform-en.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert pruned == ["battles-stt-freeform-en.json"]
    assert not dead.exists()


def test_dead_group_freeform_battles_is_pruned(tmp_path, registry_stub):
    dead = _touch(tmp_path, "battles-vad-freeform-en-US.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert pruned == ["battles-vad-freeform-en-US.json"]
    assert not dead.exists()


def test_paradigm_league_benchmark_survives_dataset_dir_move(tmp_path, monkeypatch):
    # A paradigm-league dataset (e.g. an intent_keyword/intent_template
    # corpus under registry/datasets/intent_keyword|intent_template/) must
    # still read as live purely off its (dataset_id, lang) pair — the
    # prune's liveness check goes through ``_registry_dataset_langs``,
    # which is registry-derived, not path-derived, so relocating the
    # dataset file between paradigm subdirectories can never make an
    # untouched paradigm-league artifact look stale.
    monkeypatch.setattr(
        cli, "_registry_dataset_langs",
        lambda: {"jurebes": {"en-US"}},
    )
    monkeypatch.setattr(
        cli, "_registry_battle_groups",
        lambda: {"intent_template"},
    )
    kept = _touch(tmp_path, "benchmark-intent_template-jurebes-en-US.json")
    pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
    assert pruned == []
    assert kept.exists()


# ---------------------------------------------------------------------------
# Standalone ``prune-data`` CLI entry point (§assemble scalability — the
# sharded workflow's commit job runs this after merging every matrix leg's
# artifacts, since download-artifact --merge-multiple only adds/overwrites
# and never deletes what a leg's own in-run prune removed).
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, data_dir, registry="registry"):
        self.data_dir = str(data_dir)
        self.registry = registry


def test_prune_data_deletes_stale_artifact(tmp_path, registry_stub):
    dead = _touch(tmp_path, "leaderboard-stt-en.json")
    rc = cli.cmd_prune_data(_Args(tmp_path))
    assert rc == 0
    assert not dead.exists()


def test_prune_data_keeps_live_paradigm_league_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "_registry_dataset_langs",
        lambda: {"jurebes": {"en-US"}},
    )
    monkeypatch.setattr(
        cli, "_registry_battle_groups",
        lambda: {"intent_template"},
    )
    kept = _touch(tmp_path, "benchmark-intent_template-jurebes-en-US.json")
    rc = cli.cmd_prune_data(_Args(tmp_path))
    assert rc == 0
    assert kept.exists()


def test_prune_data_missing_data_dir_is_a_noop(tmp_path):
    rc = cli.cmd_prune_data(_Args(tmp_path / "does-not-exist"))
    assert rc == 0


class TestTrainedOnPruning:
    """A board/battle-pool entry for a (fighter, dataset) pair the registry
    now excludes via ``trained_on`` must not survive a prune just because
    the file's ``(dataset_id, lang)`` key is otherwise still live —
    reproduces the review finding: a stale board with the excluded pair as
    its sole (or one of several) entrants must have that entry stripped."""

    def _stub(self, monkeypatch, trained_on_by_modality):
        monkeypatch.setattr(
            cli, "_registry_dataset_langs",
            lambda: {"community-athena": {"en-US"}},
        )
        monkeypatch.setattr(cli, "_registry_battle_groups", lambda: {"wake_word"})
        monkeypatch.setattr(
            cli, "_registry_dataset_modalities",
            lambda: {"community-athena": "wake_word"},
        )
        monkeypatch.setattr(
            cli, "_registry_trained_on_by_modality", lambda: trained_on_by_modality,
        )

    def test_stale_benchmark_board_sole_entrant_is_deleted(self, tmp_path, monkeypatch):
        self._stub(monkeypatch, {"wake_word": {"precise-onnx-athena": {"community-athena"}}})
        path = tmp_path / "benchmark-wake_word-community-athena-en-US.json"
        path.write_text(json.dumps({
            "entries": [{"competitor_id": "precise-onnx-athena", "score": 0.99}],
        }))
        pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
        assert pruned == ["benchmark-wake_word-community-athena-en-US.json"]
        assert not path.exists()

    def test_stale_benchmark_board_keeps_other_entrants(self, tmp_path, monkeypatch):
        self._stub(monkeypatch, {"wake_word": {"precise-onnx-athena": {"community-athena"}}})
        path = tmp_path / "benchmark-wake_word-community-athena-en-US.json"
        path.write_text(json.dumps({
            "entries": [
                {"competitor_id": "precise-onnx-athena", "score": 0.99},
                {"competitor_id": "vosk-ww-athena", "score": 0.5},
            ],
        }))
        pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
        assert pruned == []
        remaining = json.loads(path.read_text())
        assert [e["competitor_id"] for e in remaining["entries"]] == ["vosk-ww-athena"]

    def test_stale_battles_pool_drops_battles_naming_the_excluded_fighter(
        self, tmp_path, monkeypatch,
    ):
        self._stub(monkeypatch, {"wake_word": {"precise-onnx-athena": {"community-athena"}}})
        path = tmp_path / "battles-wake_word-community-athena-en-US.json"
        path.write_text(json.dumps({
            "battles": [
                {"competitor_a": "precise-onnx-athena", "competitor_b": "vosk-ww-athena"},
                {"competitor_a": "vosk-ww-athena", "competitor_b": "openwakeword-alexa"},
            ],
        }))
        cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
        remaining = json.loads(path.read_text())
        assert len(remaining["battles"]) == 1
        assert remaining["battles"][0]["competitor_a"] == "vosk-ww-athena"

    def test_untainted_pair_survives_untouched(self, tmp_path, monkeypatch):
        self._stub(monkeypatch, {})
        path = tmp_path / "benchmark-wake_word-community-athena-en-US.json"
        original = {"entries": [{"competitor_id": "precise-onnx-athena", "score": 0.99}]}
        path.write_text(json.dumps(original))
        pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
        assert pruned == []
        assert json.loads(path.read_text()) == original

    def test_written_this_run_is_left_alone_by_trained_on_pruning(self, tmp_path, monkeypatch):
        # A file the assemble loop itself just (re)wrote this run is never
        # touched by the prune pass at all, trained_on or not.
        self._stub(monkeypatch, {"wake_word": {"precise-onnx-athena": {"community-athena"}}})
        name = "benchmark-wake_word-community-athena-en-US.json"
        path = tmp_path / name
        original = {"entries": [{"competitor_id": "precise-onnx-athena", "score": 0.99}]}
        path.write_text(json.dumps(original))
        pruned = cli._prune_stale_artifacts(tmp_path, written_files={name}, modality_scope=None)
        assert pruned == []
        assert json.loads(path.read_text()) == original

    def test_leaderboard_entry_dropped_when_every_live_dataset_is_excluded(
        self, tmp_path, monkeypatch,
    ):
        # A group-scoped board only drops a competitor entry when EVERY
        # live dataset feeding it is one the competitor trained on — here
        # community-athena is the only live wake_word dataset in scope.
        self._stub(monkeypatch, {"wake_word": {"precise-onnx-athena": {"community-athena"}}})
        path = tmp_path / "leaderboard-wake_word-en-US.json"
        path.write_text(json.dumps({
            "entries": [
                {"competitor_id": "precise-onnx-athena", "elo": 1675.32},
                {"competitor_id": "vosk-ww-athena", "elo": 1500.0},
            ],
        }))
        pruned = cli._prune_stale_artifacts(tmp_path, written_files=set(), modality_scope=None)
        assert pruned == []
        remaining = json.loads(path.read_text())
        assert [e["competitor_id"] for e in remaining["entries"]] == ["vosk-ww-athena"]
