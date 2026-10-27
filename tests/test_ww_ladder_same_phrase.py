"""A wake-word fighter is scored, ranked, battled and seeded only on its
own phrase, and only a ranked fighter battles or seeds a rating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import arena.cli as arena_cli
from arena.cli import _registry_dataset_revisions, main
from arena.models import PredictionRow
from arena.predictions import group_rows
from registry.schemas import CompetitorDef, DatasetDef
from runner.queue_tools import is_compatible

DATASET = "synthetic-wakewords-hey_mycroft"

#: competitor id -> the phrase its hotword config is keyed by
PHRASES = {
    "mycroft-a": "hey_mycroft",
    "mycroft-b": "hey_mycroft",
    "mycroft-thin": "hey_mycroft",
    "jarvis-x": "hey_jarvis",
}


class _StubCompetitor:
    def __init__(self, competitor_id: str, phrase: str | None) -> None:
        self.competitor_id = competitor_id
        self.label_set = None
        self.config = {"hotwords": {phrase: {"module": "m"}}} if phrase else {}
        self.trained_on = []


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    import registry.loaders as loaders_mod

    def fake_list_competitors(modality=None):
        if modality == "stt":
            return [_StubCompetitor(cid, None) for cid in ("stt-a", "stt-b")]
        if modality != "wake_word":
            return []
        return [_StubCompetitor(cid, phrase) for cid, phrase in PHRASES.items()]

    monkeypatch.setattr(loaders_mod, "list_competitors", fake_list_competitors)
    monkeypatch.setattr(arena_cli, "_load_sample_set", lambda *args: None)


def _rows(competitor: str, sample_ids: range, wrong_every: int) -> list[dict]:
    """Even sample ids are positives, odd are negatives. The fighter
    answers wrongly on every ``wrong_every``-th sample, so two fighters
    with different ``wrong_every`` disagree on some samples."""
    revision = _registry_dataset_revisions()[DATASET]
    rows = []
    for i in sample_ids:
        label = "negative" if i % 2 else "positive"
        flipped = "positive" if label == "negative" else "negative"
        rows.append({
            "competitor_id": competitor,
            "sample_id": f"en-US/{i:05d}",
            "dataset_id": DATASET,
            "dataset_revision": revision,
            "lang": "en-US",
            "plugin_id": f"plugin-{competitor}",
            "label": label,
            "prediction": flipped if i % wrong_every == 0 else label,
        })
    return rows


def _assemble(tmp_path: Path, rows_by_competitor: dict[str, list[dict]]) -> Path:
    preds = tmp_path / "predictions"
    preds.mkdir()
    for competitor, rows in rows_by_competitor.items():
        (preds / f"{competitor}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "data"
    with pytest.raises(SystemExit) as exit_info:
        main(["assemble", "--predictions", str(preds), "--output", str(out)])
    assert exit_info.value.code == 0
    return out


def _board(out: Path) -> dict[str, dict]:
    payload = json.loads((out / f"benchmark-wake_word-{DATASET}-en-US.json").read_text())
    return {e["competitor_id"]: e for e in payload["entries"]}


def _battle_fighters(out: Path) -> set[str]:
    path = out / f"battles-wake_word-{DATASET}-en-US.json"
    if not path.exists():
        return set()
    battles = json.loads(path.read_text())["battles"]
    return {b[side] for b in battles for side in ("competitor_a", "competitor_b")}


def _seeded_fighters(out: Path) -> set[str]:
    path = out / "elo-seed-wake_word-en-US.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text())["competitor_plugin"])


def test_other_phrase_fighter_is_not_scored_battled_or_seeded(tmp_path):
    out = _assemble(tmp_path, {
        "mycroft-a": _rows("mycroft-a", range(40), wrong_every=7),
        "mycroft-b": _rows("mycroft-b", range(40), wrong_every=3),
        "jarvis-x": _rows("jarvis-x", range(40), wrong_every=2),
    })
    board = _board(out)
    assert {cid for cid, e in board.items() if not e.get("unranked")} == {
        "mycroft-a", "mycroft-b"}
    assert "jarvis-x" not in board
    assert _battle_fighters(out) == {"mycroft-a", "mycroft-b"}
    assert _seeded_fighters(out) == {"mycroft-a", "mycroft-b"}
    summary = json.loads((out / "assemble-summary.json").read_text())
    assert summary["cross_phrase_rows_excluded"] == {f"jarvis-x/{DATASET}": 40}


def _stream_row(competitor: str, dataset_id: str) -> PredictionRow:
    return PredictionRow(
        competitor_id=competitor, sample_id="s0", dataset_id=dataset_id,
        lang="en-US", plugin_id="p", modality="ww_stream", prediction="WW_STREAM",
        extras={"events": [], "truth_onsets": [], "duration_s": 60.0},
    )


def test_stream_set_keeps_only_its_phrase_and_negatives_keep_every_fighter():
    phrase_set = "ww_stream_synthetic-wakewords-hey_mycroft"
    negatives = "ww-negatives-en-US"
    dropped: dict[str, int] = {}
    grouped = group_rows(
        [_stream_row(cid, ds) for ds in (phrase_set, negatives)
         for cid in ("mycroft-a", "jarvis-x")],
        cross_phrase_dropped=dropped,
    )
    assert set(grouped[("ww_stream", phrase_set, "en-US")]["s0"]) == {"mycroft-a"}
    assert set(grouped[("ww_stream", negatives, "en-US")]["s0"]) == {
        "mycroft-a", "jarvis-x"}
    assert dropped == {f"jarvis-x/{phrase_set}": 1}


def test_fighter_the_board_leaves_unranked_never_battles_or_seeds(tmp_path):
    out = _assemble(tmp_path, {
        "mycroft-a": _rows("mycroft-a", range(40), wrong_every=7),
        "mycroft-b": _rows("mycroft-b", range(40), wrong_every=3),
        "mycroft-thin": _rows("mycroft-thin", range(10), wrong_every=2),
    })
    board = _board(out)
    assert board["mycroft-thin"]["unranked"] is True
    assert _battle_fighters(out) == {"mycroft-a", "mycroft-b"}
    assert _seeded_fighters(out) == {"mycroft-a", "mycroft-b"}


def test_board_that_ranks_nobody_writes_no_battles_or_seed(tmp_path):
    # 20 rows each, under the board's 30-row floor, but 30 distinct
    # sample ids between them.
    out = _assemble(tmp_path, {
        "mycroft-a": _rows("mycroft-a", range(0, 20), wrong_every=7),
        "mycroft-b": _rows("mycroft-b", range(10, 30), wrong_every=3),
    })
    board = _board(out)
    assert [e.get("unranked") for e in board.values()] == [True, True]
    assert _battle_fighters(out) == set()
    assert _seeded_fighters(out) == set()


def _ww_competitor(phrase: str) -> CompetitorDef:
    return CompetitorDef(
        competitor_id="c", modality="wake_word", plugin="p", langs=["en-US"],
        config={"hotwords": {phrase: {"module": "m"}}},
    )


def _ww_dataset(wakeword: str | None) -> DatasetDef:
    return DatasetDef(
        dataset_id="d", modality="wake_word", lang="en-US", wakeword=wakeword,
        source={"type": "huggingface", "hf_id": "x"},
    )


@pytest.mark.parametrize(("phrase", "wakeword", "expected"), [
    ("hey_mycroft", "hey_mycroft", True),
    ("hey_jarvis", "hey_mycroft", False),
    ("smart_mirror", "smart mirror", True),
    ("hey_jarvis", None, True),
])
def test_autorun_schedules_a_wake_word_fighter_only_on_its_phrase(phrase, wakeword, expected):
    assert is_compatible(_ww_competitor(phrase), _ww_dataset(wakeword)) is expected


def test_merged_summary_sums_cross_phrase_rows_per_pair():
    merged = arena_cli.merge_assemble_summaries([
        {"cross_phrase_rows_excluded": {"jarvis-x/d1": 3, "jarvis-x/d2": 1}},
        {"cross_phrase_rows_excluded": {"jarvis-x/d1": 4}},
    ])
    assert merged["cross_phrase_rows_excluded"] == {"jarvis-x/d1": 7, "jarvis-x/d2": 1}


def _stt_rows(competitor: str, sample_ids: range, wer: float) -> list[dict]:
    return [{
        "competitor_id": competitor,
        "sample_id": f"pt-PT/{i:05d}",
        "dataset_id": "minds14-pt-PT",
        "lang": "pt-PT",
        "plugin_id": f"plugin-{competitor}",
        "audio_url": "https://example.com/a.wav",
        "reference_text": "ligar o alarme",
        "prediction": "ligar o alarme" if wer == 0.0 else "ligar alarme errado",
        "wer": wer,
    } for i in sample_ids]


def test_other_leagues_keep_battles_and_seed_for_unranked_fighters(tmp_path):
    # 20 rows each, under the board's 30-row floor, over 30 distinct sample
    # ids, so the board ranks nobody. Outside the wake-word ladder the
    # sample-count gate decides, and the 10 shared samples still battle.
    out = _assemble(tmp_path, {
        "stt-a": _stt_rows("stt-a", range(0, 20), wer=0.0),
        "stt-b": _stt_rows("stt-b", range(10, 30), wer=0.5),
    })
    board = json.loads((out / "benchmark-stt-minds14-pt-PT-pt-PT.json").read_text())
    assert [e.get("unranked") for e in board["entries"]] == [True, True]
    battles = json.loads((out / "battles-stt-minds14-pt-PT-pt-PT.json").read_text())
    assert {b[side] for b in battles["battles"]
            for side in ("competitor_a", "competitor_b")} == {"stt-a", "stt-b"}
    seed = json.loads((out / "elo-seed-stt-pt-PT.json").read_text())
    assert set(seed["competitor_plugin"]) == {"stt-a", "stt-b"}
