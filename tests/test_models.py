"""Unit tests for arena.models — artifact contracts and battle ids."""
from __future__ import annotations

import json

from arena.models import (
    Battle,
    BattlesPool,
    BenchmarkBoard,
    BenchmarkEntry,
    EloBoard,
    EloEntry,
    EloSeed,
    Modality,
    PredictionRow,
    VoteOutcome,
    battle_id_for,
)


class TestBattleId:
    def test_deterministic(self):
        a = battle_id_for("intent", "ds", "en-US", "s1", "x", "y")
        b = battle_id_for("intent", "ds", "en-US", "s1", "x", "y")
        assert a == b

    def test_competitor_order_invariant(self):
        a = battle_id_for("intent", "ds", "en-US", "s1", "x", "y")
        b = battle_id_for("intent", "ds", "en-US", "s1", "y", "x")
        assert a == b

    def test_distinct_per_sample(self):
        a = battle_id_for("intent", "ds", "en-US", "s1", "x", "y")
        b = battle_id_for("intent", "ds", "en-US", "s2", "x", "y")
        assert a != b

    def test_distinct_per_lang(self):
        a = battle_id_for("intent", "ds", "en-US", "s1", "x", "y")
        b = battle_id_for("intent", "ds", "pt-PT", "s1", "x", "y")
        assert a != b

    def test_shape(self):
        bid = battle_id_for("intent", "ds", "en-US", "s1", "x", "y")
        assert len(bid) == 16
        int(bid, 16)  # hex


class TestPredictionRow:
    def test_minimal_intent_row(self):
        row = PredictionRow(
            competitor_id="padatious-medium",
            sample_id="en-US/00001",
            dataset_id="intents-for-eval",
            lang="en-US",
            plugin_id="ovos-padatious-pipeline-plugin",
            utterance="play a song",
            reference_intent="media:play_song",
            prediction="media:play_song",
            exact_match=True,
        )
        assert row.exact_match is True
        assert row.extras == {}

    def test_ood_row_none_fields(self):
        row = PredictionRow(
            competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
            plugin_id="p", reference_intent=None, prediction=None,
            exact_match=True,
        )
        assert row.reference_intent is None
        assert row.prediction is None


class TestArtifactRoundtrip:
    def test_battles_pool_json_roundtrip(self):
        pool = BattlesPool(
            modality=Modality.INTENT,
            dataset_id="intents-for-eval",
            lang="en-US",
            generated_at="2026-01-01T00:00:00+00:00",
            battles=[Battle(
                battle_id="ab12", modality=Modality.INTENT,
                dataset_id="intents-for-eval", lang="en-US", sample_id="s1",
                input_text="play a song",
                prediction_a={"intent": "media:play_song"},
                prediction_b=None,
                competitor_a="x", competitor_b="y",
            )],
        )
        payload = json.loads(json.dumps(pool.model_dump(mode="json")))
        again = BattlesPool(**payload)
        assert again == pool

    def test_elo_seed_roundtrip(self):
        seed = EloSeed(
            modality=Modality.INTENT, lang="en-US",
            generated_at="2026-01-01T00:00:00+00:00",
            auto_vote_count=2,
            ratings={"x": 1210.0, "y": 1190.0},
            battles={"x": 2, "y": 2},
            wins={"x": 2, "y": 0},
            losses={"x": 0, "y": 2},
            ties={"x": 0, "y": 0},
            competitor_plugin={"x": "plug-x", "y": "plug-y"},
        )
        again = EloSeed(**json.loads(json.dumps(seed.model_dump(mode="json"))))
        assert again == seed

    def test_boards_serialise(self):
        bench = BenchmarkBoard(
            modality=Modality.INTENT, dataset_id="d", lang="en-US",
            generated_at="t", primary_metric="accuracy",
            entries=[BenchmarkEntry(rank=1, competitor_id="x",
                                    metrics={"accuracy": 0.9})],
        )
        elo = EloBoard(
            modality=Modality.INTENT, lang="en-US", generated_at="t",
            entries=[EloEntry(rank=1, competitor_id="x", elo=1234.5)],
        )
        json.dumps(bench.model_dump(mode="json"))
        json.dumps(elo.model_dump(mode="json"))

    def test_vote_outcomes_complete(self):
        assert {o.value for o in VoteOutcome} == {
            "candidate_a", "candidate_b", "tie", "both_wrong",
        }
