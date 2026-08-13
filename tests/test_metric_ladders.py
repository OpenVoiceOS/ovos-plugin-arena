"""Tests for per-metric ladders (§ per-metric ladders campaign).

Covers: secondary metric seeding determinism, auto_only flagging, aggregate
metrics excluded from the ladderable registry, and the metric_ladders field
surviving a full assemble/build_elo_board replay.
"""
from __future__ import annotations

from arena.assembler import seed_elo, seed_secondary_metrics
from arena.cli import build_elo_board
from arena.metrics import (
    ROW_METRIC_ACCESSORS,
    ladder_metrics_for,
    row_metric_value,
    secondary_ladder_metrics_for,
)
from arena.models import PredictionRow


def _intent_row(competitor, prediction, sample_id, reference="media:play_song", **over):
    base = dict(
        competitor_id=competitor,
        sample_id=sample_id,
        dataset_id="intents-for-eval",
        lang="en-US",
        plugin_id=f"plugin-{competitor}",
        utterance="play a song",
        reference_intent=reference,
        prediction=prediction,
    )
    base.update(over)
    return PredictionRow(**base)


def _tts_row(competitor, sample_id, utmos, **over):
    base = dict(
        competitor_id=competitor,
        sample_id=sample_id,
        dataset_id="tts-eval",
        lang="en-US",
        plugin_id=f"plugin-{competitor}",
        input_text="hello world",
        prediction=f"https://example.com/{competitor}/{sample_id}.wav",
        extras={"utmos": utmos},
    )
    base.update(over)
    return PredictionRow(**base)


def _stt_row(competitor, sample_id, reference_text, prediction, **over):
    base = dict(
        competitor_id=competitor,
        sample_id=sample_id,
        dataset_id="stt-eval",
        lang="en-US",
        plugin_id=f"plugin-{competitor}",
        reference_text=reference_text,
        prediction=prediction,
    )
    base.update(over)
    return PredictionRow(**base)


def _samples_by_dataset(dataset_id, *rows_per_sample):
    samples = {}
    for i, rows in enumerate(rows_per_sample):
        sample_id = f"en-US/{i:05d}"
        samples[sample_id] = {
            r.competitor_id: r.model_copy(update={"sample_id": sample_id})
            for r in rows
        }
    return {dataset_id: samples}


class TestLadderRegistry:
    def test_aggregate_only_metrics_excluded(self):
        # ECE and macro_f1 are dataset-aggregate-only — no per-row value —
        # and must never appear as ladderable intent metrics.
        for modality in ("intent", "intent_template", "intent_keyword"):
            keys = set(ROW_METRIC_ACCESSORS.get(modality, {}))
            assert "ece" not in keys
            assert "macro_f1" not in keys
            assert "ood_fpr" not in keys

    def test_tts_secondary_metrics_are_row_level(self):
        secondary = secondary_ladder_metrics_for("tts")
        assert "utmos" not in secondary  # primary, not secondary
        assert "sigmos.noise" in secondary
        assert "dnsmos.bak" in secondary

    def test_stt_primary_first_in_ladder_metrics(self):
        assert ladder_metrics_for("stt")[0] == "wer_mean"

    def test_wake_word_has_no_ladderable_metrics(self):
        # No secondary row-level metric exists for wake_word today.
        assert secondary_ladder_metrics_for("wake_word") == []


class TestRowMetricValue:
    def test_intent_accuracy_row_value(self):
        correct = _intent_row("x", "media:play_song", "s1")
        wrong = _intent_row("x", "media:stop", "s1")
        assert row_metric_value(correct, "intent", "accuracy") == 1.0
        assert row_metric_value(wrong, "intent", "accuracy") == 0.0

    def test_slot_exact_match_none_when_no_gold_slots(self):
        row = _intent_row("x", "media:play_song", "s1")
        assert row_metric_value(row, "intent", "slot_exact_match") is None

    def test_unknown_metric_returns_none(self):
        row = _intent_row("x", "media:play_song", "s1")
        assert row_metric_value(row, "intent", "not_a_real_metric") is None


class TestSeedSecondaryMetricsDeterminism:
    def test_tts_quality_dims_deterministic(self):
        samples_by_dataset = _samples_by_dataset(
            "tts-eval",
            [_tts_row("a", "s0", 4.5), _tts_row("b", "s0", 3.0)],
            [_tts_row("a", "s1", 4.2), _tts_row("b", "s1", 3.5)],
        )
        first = seed_secondary_metrics("tts", samples_by_dataset)
        second = seed_secondary_metrics("tts", samples_by_dataset)
        assert set(first) == set(second) == set(secondary_ladder_metrics_for("tts"))
        for key in first:
            assert first[key].model_dump() == second[key].model_dump()

    def test_utmos_not_in_secondary_seeds(self):
        samples_by_dataset = _samples_by_dataset(
            "tts-eval", [_tts_row("a", "s0", 4.5), _tts_row("b", "s0", 3.0)]
        )
        secondary = seed_secondary_metrics("tts", samples_by_dataset)
        assert "utmos" not in secondary

    def test_higher_utmos_wins_pairwise(self):
        samples_by_dataset = _samples_by_dataset(
            "tts-eval",
            [
                _tts_row("a", "s0", 4.8, extras={"utmos": 4.8, "sigmos.noise": 4.8}),
                _tts_row("b", "s0", 3.0, extras={"utmos": 3.0, "sigmos.noise": 1.0}),
            ],
        )
        secondary = seed_secondary_metrics("tts", samples_by_dataset)
        noise = secondary["sigmos.noise"]
        assert noise.wins.get("a", 0) == 1
        assert noise.losses.get("a", 0) == 0
        assert noise.wins.get("b", 0) == 0


class TestMetricLaddersOnBoard:
    def test_board_has_ladder_per_metric_and_auto_only_flag(self):
        samples_by_dataset = _samples_by_dataset(
            "tts-eval",
            [_tts_row("a", "s0", 4.8), _tts_row("b", "s0", 3.0)],
            [_tts_row("a", "s1", 4.2), _tts_row("b", "s1", 3.9)],
        )
        seed = seed_elo("tts", "en-US", samples_by_dataset, "2026-08-13T00:00:00Z")
        seed.secondary_metrics = seed_secondary_metrics("tts", samples_by_dataset)
        board = build_elo_board("tts", "en-US", seed, [], {})

        assert "utmos" in board.metric_ladders
        assert board.metric_ladders["utmos"].auto_only is False
        assert "sigmos.noise" in board.metric_ladders
        assert board.metric_ladders["sigmos.noise"].auto_only is True
        ranked_ids = [e.competitor_id for e in board.metric_ladders["sigmos.noise"].entries]
        assert set(ranked_ids) == {"a", "b"}

    def test_intent_board_slot_ladder_auto_only(self):
        samples_by_dataset = _samples_by_dataset(
            "intents-for-eval",
            [
                _intent_row(
                    "a", "media:play_song", "s0",
                    reference_slots={"song": "x"}, predicted_slots={"song": "x"},
                ),
                _intent_row(
                    "b", "media:play_song", "s0",
                    reference_slots={"song": "x"}, predicted_slots={"song": "y"},
                ),
            ],
        )
        seed = seed_elo("intent", "en-US", samples_by_dataset, "2026-08-13T00:00:00Z")
        seed.secondary_metrics = seed_secondary_metrics("intent", samples_by_dataset)
        board = build_elo_board("intent", "en-US", seed, [], {})

        assert board.metric_ladders["accuracy"].auto_only is False
        assert board.metric_ladders["slot_exact_match"].auto_only is True

    def test_replay_reproduces_metric_ladders_byte_identical(self):
        samples_by_dataset = _samples_by_dataset(
            "stt-eval",
            [_stt_row("a", "s0", "play a song", "play a song"),
             _stt_row("b", "s0", "play a song", "play the song")],
        )
        seed = seed_elo("stt", "en-US", samples_by_dataset, "2026-08-13T00:00:00Z")
        seed.secondary_metrics = seed_secondary_metrics("stt", samples_by_dataset)
        board_1 = build_elo_board("stt", "en-US", seed, [], {})
        board_2 = build_elo_board("stt", "en-US", seed, [], {})
        d1 = board_1.model_dump(mode="json", exclude={"generated_at"})
        d2 = board_2.model_dump(mode="json", exclude={"generated_at"})
        assert d1["metric_ladders"] == d2["metric_ladders"]
