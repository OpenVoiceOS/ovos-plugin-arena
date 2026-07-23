"""Tests for the streaming wake-word league (§A3.2 / R14).

Synthetic-first (per the roadmap): no real audio corpus or plugin is
required. ``TestDetectStream`` drives the runner's frame-loop against an
in-memory dummy detector over a synthesized tone-burst clip; everything else
builds ``PredictionRow``/``CompetitorDef`` objects directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from arena.metrics import (
    EVENT_TOLERANCE_S,
    TARGET_FA_PER_HOUR,
    score_ww_stream,
)
from arena.models import PredictionRow
from registry.schemas import CompetitorDef, DatasetDef
from runner.ww_bench import FRAME_SAMPLES, SAMPLE_RATE, WWStack, _detect_stream


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p", prediction="WW_STREAM",
    )
    base.update(over)
    return PredictionRow(**base)


def _stream_row(events, truth_onsets, duration_s):
    return _row(extras={
        "events": events, "truth_onsets": truth_onsets, "duration_s": duration_s,
    })


# ---------------------------------------------------------------------------
# registry: capabilities field
# ---------------------------------------------------------------------------


class TestCapabilitiesSchema:
    def _valid(self, **kw):
        defaults = dict(
            competitor_id="c", modality="wake_word",
            plugin="ovos-ww-plugin-openwakeword", config={}, langs=["en"],
        )
        defaults.update(kw)
        return CompetitorDef(**defaults)

    def test_default_is_clip_only(self):
        c = self._valid()
        assert c.capabilities == ["clip"]

    def test_stream_capability_accepted(self):
        c = self._valid(capabilities=["clip", "stream"])
        assert "stream" in c.capabilities

    def test_unknown_capability_rejected(self):
        with pytest.raises(Exception):
            self._valid(capabilities=["clip", "teleport"])

    def test_all_wake_word_fighters_parse(self):
        from registry.loaders import list_competitors

        fighters = list_competitors("wake_word")
        assert fighters, "expected wake_word registry fixtures to be present"
        for fighter in fighters:
            assert fighter.capabilities  # non-empty, defaults to ["clip"]
            assert set(fighter.capabilities) <= {"clip", "stream"}

    def test_stream_fighters_are_the_expected_families(self):
        from registry.loaders import list_competitors

        stream_ids = {
            c.competitor_id for c in list_competitors("wake_word")
            if "stream" in c.capabilities
        }
        assert stream_ids, "expected at least one stream-capable fighter"
        for cid in stream_ids:
            assert cid.startswith(("openwakeword-", "microwakeword-", "precise-onnx-"))
        # a representative clip-only fighter stays clip-only
        clip_only = {
            c.competitor_id for c in list_competitors("wake_word")
            if c.capabilities == ["clip"]
        }
        assert any(cid.startswith("vosk-ww-") for cid in clip_only)


# ---------------------------------------------------------------------------
# registry: ww_stream dataset entry stays inert
# ---------------------------------------------------------------------------


class TestStreamDatasetEntry:
    def test_loads_and_pins_sample_rate(self):
        from registry.loaders import load_dataset

        d = load_dataset("ww_stream", "ww_stream_hey_mycroft")
        assert isinstance(d, DatasetDef)
        assert d.sample_rate_hz == 16000
        assert d.event_tolerance_s == EVENT_TOLERANCE_S
        assert d.role == "eval"

    def test_predictions_repo_404_is_skipped_not_fatal(self, monkeypatch):
        """assemble's fetch loop must tolerate an unpublished predictions_hf
        repo — this dataset is legitimately unbuilt (§A3.2 is scaffolding
        only), so it must never crash a full assemble run. Simulates the
        404 without touching the network: any exception from fetching must
        propagate as a normal exception, which ``arena.cli.cmd_assemble``
        already catches per-source (log + continue)."""
        import huggingface_hub

        from arena.predictions import load_predictions

        def _boom(*a, **kw):
            raise huggingface_hub.utils.RepositoryNotFoundError("404")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
        with pytest.raises(Exception):
            load_predictions("TigreGotico/ww-stream-bench-hey_mycroft-does-not-exist")


# ---------------------------------------------------------------------------
# arena.metrics.score_ww_stream
# ---------------------------------------------------------------------------


class TestScoreWwStream:
    def test_no_stream_rows_returns_empty(self):
        rows = [_row(prediction="detected", label="positive")]
        assert score_ww_stream(rows) == {}

    def test_perfect_detector_zero_frr_and_fa(self):
        rows = [_stream_row(events=[[10.0, 1.0]], truth_onsets=[10.0],
                            duration_s=3600.0)]
        m = score_ww_stream(rows)
        assert m["frr"] == 0.0
        assert m["fa_per_hour"] == 0.0
        assert m["n_onsets"] == 1.0
        assert m["negative_hours"] == pytest.approx(1.0)

    def test_missed_onset_counts_as_false_reject(self):
        rows = [_stream_row(events=[], truth_onsets=[5.0], duration_s=3600.0)]
        m = score_ww_stream(rows)
        assert m["frr"] == 1.0
        assert m["fa_per_hour"] == 0.0

    def test_unmatched_event_counts_as_false_accept(self):
        rows = [_stream_row(events=[[100.0, 1.0]], truth_onsets=[],
                            duration_s=3600.0)]
        m = score_ww_stream(rows)
        assert m["frr"] == 0.0
        assert m["fa_per_hour"] == pytest.approx(1.0)

    def test_boundary_at_exact_tolerance_is_a_true_positive(self):
        onset = 20.0
        rows = [_stream_row(
            events=[[onset + EVENT_TOLERANCE_S, 1.0]],
            truth_onsets=[onset], duration_s=3600.0,
        )]
        m = score_ww_stream(rows)
        assert m["frr"] == 0.0
        assert m["fa_per_hour"] == 0.0
        assert m["latency_s_median"] == pytest.approx(EVENT_TOLERANCE_S)

    def test_just_past_tolerance_is_a_miss_and_a_false_accept(self):
        onset = 20.0
        rows = [_stream_row(
            events=[[onset + EVENT_TOLERANCE_S + 0.01, 1.0]],
            truth_onsets=[onset], duration_s=3600.0,
        )]
        m = score_ww_stream(rows)
        assert m["frr"] == 1.0  # onset unmatched
        assert m["fa_per_hour"] == pytest.approx(1.0)  # event unmatched

    def test_fa_per_hour_scales_with_duration(self):
        # 4 false accepts over 2 hours of audio -> 2/hour
        rows = [_stream_row(
            events=[[100.0, 1.0], [200.0, 1.0], [300.0, 1.0], [400.0, 1.0]],
            truth_onsets=[], duration_s=7200.0,
        )]
        m = score_ww_stream(rows)
        assert m["fa_per_hour"] == pytest.approx(2.0)

    def test_aggregates_across_multiple_rows(self):
        rows = [
            _stream_row(events=[[10.0, 1.0]], truth_onsets=[10.0], duration_s=1800.0),
            _stream_row(events=[], truth_onsets=[50.0], duration_s=1800.0),
        ]
        m = score_ww_stream(rows)
        assert m["n_onsets"] == 2.0
        assert m["frr"] == pytest.approx(0.5)
        assert m["negative_hours"] == pytest.approx(1.0)

    def test_det_points_flattened_as_float_metrics(self):
        rows = [_stream_row(events=[[10.0, 0.6]], truth_onsets=[10.0],
                            duration_s=3600.0)]
        m = score_ww_stream(rows)
        assert "det_frr@0.5" in m and isinstance(m["det_frr@0.5"], float)
        assert "det_fa_per_hour@0.7" in m
        # score 0.6 misses the 0.7 threshold bucket -> onset unmatched there
        assert m["det_frr@0.7"] == 1.0
        assert m["det_frr@0.5"] == 0.0

    def test_primary_metric_respects_fa_budget(self):
        # threshold 0.5 keeps every low-score nuisance firing -> way over
        # budget; only threshold >=0.9 drops them, at the cost of also
        # losing the real (lower-score) detections -> worse FRR there.
        events = [[float(i), 0.55] for i in range(0, 3600, 60)]  # 1/min noise
        events.append([10.0, 0.95])
        rows = [_stream_row(events=events, truth_onsets=[10.0], duration_s=3600.0)]
        m = score_ww_stream(rows)
        assert m["fa_per_hour"] > TARGET_FA_PER_HOUR  # at 0.5 this is over budget
        assert m["error_at_2fa_per_hour"] <= 1.0
        # the chosen operating point must actually respect the budget
        # (or be the most conservative one scanned, if none do)
        assert m["det_fa_per_hour@0.9"] <= TARGET_FA_PER_HOUR


# ---------------------------------------------------------------------------
# runner.ww_bench: fighter eligibility (exclusion, not zero-scoring)
# ---------------------------------------------------------------------------


class TestStreamEligibility:
    def _fighter(self, cid, capabilities):
        return CompetitorDef(
            competitor_id=cid, modality="wake_word",
            plugin="ovos-ww-plugin-x", config={}, langs=["en"],
            capabilities=capabilities,
        )

    def test_filter_keeps_only_stream_capable(self):
        from runner.ww_bench import WakeWordStreamBench

        fighters = [
            self._fighter("clip-only", ["clip"]),
            self._fighter("stream-capable", ["clip", "stream"]),
        ]
        kept = WakeWordStreamBench().filter_competitors(fighters)
        assert [c.competitor_id for c in kept] == ["stream-capable"]

    def test_clip_only_excluded_entirely_not_present_at_all(self):
        """A clip-only fighter must be absent from the stream board's input
        set, not present with a zero/undefined score — it structurally
        cannot compete on continuous audio, so scoring it would be a
        fabricated number, not a real result."""
        from runner.ww_bench import WakeWordStreamBench

        fighters = [self._fighter("clip-only", ["clip"])]
        kept = WakeWordStreamBench().filter_competitors(fighters)
        assert kept == []

    def test_competitor_modality_pulls_from_wake_word_pool(self):
        from runner.ww_bench import WakeWordStreamBench

        adapter = WakeWordStreamBench()
        assert adapter.modality == "ww_stream"
        assert adapter.competitor_modality == "wake_word"


# ---------------------------------------------------------------------------
# runner.ww_bench._detect_stream: synthetic tone-burst detector
# ---------------------------------------------------------------------------


class _DummyStreamEngine:
    """Fires once per contiguous loud stretch, then stays in cooldown until a
    quiet frame is seen — mirrors a real hotword engine's own refractory/
    debounce logic (the engine owns re-arm behaviour, not the runner;
    ``runner.ww_bench._detect_stream`` never force-resets mid-clip)."""

    def __init__(self, loud_threshold: float = 0.2):
        self.loud_threshold = loud_threshold
        self._cooldown = False
        self._latched = False

    def update(self, chunk: bytes) -> None:
        arr = np.frombuffer(chunk, dtype="<i2").astype("float32") / 32767.0
        loud = float(np.abs(arr).mean()) > self.loud_threshold
        if not loud:
            self._cooldown = False
        elif not self._cooldown:
            self._latched = True
            self._cooldown = True

    def found_wake_word(self) -> bool:
        if self._latched:
            self._latched = False
            return True
        return False

    def reset(self) -> None:
        self._cooldown = False
        self._latched = False


def _tone_burst(start_s: float, dur_s: float, total_s: float) -> np.ndarray:
    """A synthetic clip: silence, except a loud tone burst at *start_s*."""
    n_total = int(total_s * SAMPLE_RATE)
    arr = np.zeros(n_total, dtype="float32")
    start = int(start_s * SAMPLE_RATE)
    end = start + int(dur_s * SAMPLE_RATE)
    t = np.arange(end - start) / SAMPLE_RATE
    arr[start:end] = 0.8 * np.sin(2 * np.pi * 440.0 * t)
    return arr


class TestDetectStream:
    def test_single_onset_detected_near_true_time(self):
        clip = _tone_burst(start_s=5.0, dur_s=0.5, total_s=10.0)
        stack = WWStack(ww=_DummyStreamEngine())
        events = _detect_stream(stack, clip)
        assert len(events) == 1
        t, score = events[0]
        assert 5.0 - 0.1 <= t <= 5.0 + (FRAME_SAMPLES / SAMPLE_RATE) + 0.1
        assert score == 1.0

    def test_multiple_onsets_each_produce_one_event(self):
        clip = np.concatenate([
            _tone_burst(start_s=2.0, dur_s=0.3, total_s=5.0),
            _tone_burst(start_s=1.0, dur_s=0.3, total_s=5.0),
        ])
        stack = WWStack(ww=_DummyStreamEngine())
        events = _detect_stream(stack, clip)
        assert len(events) == 2
        # second onset lands ~6s into the concatenated clip (5s + 1s)
        assert events[1][0] > events[0][0]

    def test_silence_produces_no_events(self):
        clip = np.zeros(int(3.0 * SAMPLE_RATE), dtype="float32")
        stack = WWStack(ww=_DummyStreamEngine())
        assert _detect_stream(stack, clip) == []

    def test_engine_confidence_attribute_is_recorded(self):
        class _ScoredEngine(_DummyStreamEngine):
            confidence = 0.87

        clip = _tone_burst(start_s=1.0, dur_s=0.3, total_s=3.0)
        stack = WWStack(ww=_ScoredEngine())
        events = _detect_stream(stack, clip)
        assert len(events) == 1
        assert events[0][1] == pytest.approx(0.87)
