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
