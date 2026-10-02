"""Tests for the streaming wake-word league (§A3.2 / R15).

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
            plugin="ovos-ww-plugin-openwakeword", config={}, langs=["en-US"],
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
            plugin="ovos-ww-plugin-x", config={}, langs=["en-US"],
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


# ---------------------------------------------------------------------------
# negatives_dataset_ids: parquet-pooled negatives (PR #125 follow-up)
# ---------------------------------------------------------------------------


class TestPooledDatasetNegatives:
    """stream_ww pools extra negatives from other registry datasets whose
    audio ships in parquet row groups (e.g. ml_spoken_words negatives),
    resolved via the registry loader + stream_audio_dataset — the path
    negatives_sources/negatives_hf cannot use (they only list audiofolder
    repo files via huggingface_hub.list_repo_files).
    """

    def _fighter_def(self, **over):
        from registry.schemas import DatasetDef

        base = dict(
            dataset_id="fighter", modality="wake_word",
            source={"type": "huggingface", "hf_id": "org/fighter-repo",
                    "revision": "main"},
            wakeword="hey_test", lang="en-US",
        )
        base.update(over)
        return DatasetDef(**base)

    def _neg_registry_def(self):
        from registry.schemas import DatasetDef

        return DatasetDef(
            dataset_id="mlsw-negatives-en-US", modality="wake_word",
            source={"type": "huggingface", "hf_id": "MLCommons/ml_spoken_words",
                    "revision": "refs/convert/parquet", "subset": "en_wav",
                    "split": "partial-test"},
            reference_fields={"audio": "audio"},
            lang="en-US", role="unrestricted",
        )

    def test_negatives_dataset_ids_adds_parquet_pooled_negatives(self, monkeypatch):
        import numpy as np

        from runner import audio_io

        # no network: same-corpus negatives come back empty so the only
        # negatives observed are the parquet-pooled ones under test.
        monkeypatch.setattr(audio_io, "_repo_audio", lambda hf, rev: [])

        def fake_load_dataset(modality, dataset_id):
            assert modality == "wake_word"
            assert dataset_id == "mlsw-negatives-en-US"
            return self._neg_registry_def()

        monkeypatch.setattr("registry.loaders.load_dataset", fake_load_dataset)

        def fake_stream_audio_dataset(source, audio_key, extra_keys, revision,
                                      max_samples=0, id_key=None, seed=None):
            assert source.hf_id == "MLCommons/ml_spoken_words"
            for i in range(3):
                yield f"clip{i}", {
                    "array": np.zeros(16000, dtype=np.float32), "sr": 16000,
                }

        monkeypatch.setattr(audio_io, "stream_audio_dataset",
                            fake_stream_audio_dataset)
        monkeypatch.setattr(audio_io, "_emit_ww", lambda pos, neg, cap: iter(()))

        dataset_def = self._fighter_def(negatives_dataset_ids=["mlsw-negatives-en-US"])
        results = list(audio_io.stream_ww(dataset_def, "main"))

        assert results, "expected pooled parquet negatives to be yielded"
        assert all(sample["label"] == "negative" for _, sample in results)
        assert all(sid.startswith("mlsw-negatives-en-US/") for sid, _ in results)


# ---------------------------------------------------------------------------
# operating points at several FA budgets, thresholds from observed scores
# ---------------------------------------------------------------------------


class TestOperatingPoints:
    def test_primary_metric_is_error_at_one_fa_per_hour(self):
        from arena.metrics import PRIMARY_METRIC

        assert PRIMARY_METRIC["ww_stream"] == "error_at_1fa_per_hour"
        assert TARGET_FA_PER_HOUR == 1.0

    def test_saturated_scores_are_separated_by_observed_thresholds(self):
        # every nuisance event scores 0.9995, the real onset only 0.9985:
        # no threshold under 0.9995 respects the budget, and a 0.1..0.9 grid
        # cannot see that, so it would report the 0.9 point (FRR 0).
        nuisance = [[float(100 * i), 0.9995] for i in range(1, 4)]
        rows = [_stream_row(
            events=[[10.0, 0.9985], *nuisance], truth_onsets=[10.0],
            duration_s=3600.0,
        )]
        m = score_ww_stream(rows)
        assert m["error_at_2fa_per_hour"] == 1.0
        assert m["error_at_1fa_per_hour"] == 1.0
        assert m["recall_at_1fa_per_hour"] == 0.0

    def test_threshold_can_sit_between_close_scores(self):
        # onset 0.998 beats the single nuisance at 0.9975; a grid stops at 0.9
        rows = [_stream_row(
            events=[[10.0, 0.998], [900.0, 0.9975], [1800.0, 0.9975]],
            truth_onsets=[10.0], duration_s=3600.0,
        )]
        m = score_ww_stream(rows)
        assert m["error_at_1fa_per_hour"] == 0.0
        assert m["recall_at_1fa_per_hour"] == 1.0

    def test_budgets_are_ordered_and_recall_complements_error(self):
        # two onsets at 0.97 / 0.92; three nuisance at 0.96, 0.95, 0.9 in 1 hour
        rows = [_stream_row(
            events=[[10.0, 0.97], [500.0, 0.92], [1000.0, 0.96],
                    [2000.0, 0.95], [3000.0, 0.9]],
            truth_onsets=[10.0, 500.0], duration_s=3600.0,
        )]
        m = score_ww_stream(rows)
        assert m["error_at_0.5fa_per_hour"] == 0.5
        assert m["error_at_1fa_per_hour"] == 0.5
        assert m["error_at_2fa_per_hour"] == 0.0
        for budget in ("0.5", "1", "2"):
            assert m[f"recall_at_{budget}fa_per_hour"] == pytest.approx(
                1.0 - m[f"error_at_{budget}fa_per_hour"]
            )
        errs = [m[f"error_at_{b}fa_per_hour"] for b in ("0.5", "1", "2")]
        assert errs == sorted(errs, reverse=True)

    def test_scoring_is_deterministic(self):
        events = [[float(i), 0.5 + (i % 7) / 20.0] for i in range(0, 3600, 37)]
        rows = [_stream_row(events=events, truth_onsets=[74.0, 111.0],
                            duration_s=3600.0)]
        assert score_ww_stream(rows) == score_ww_stream(rows)

    def test_no_events_means_every_onset_missed_at_every_budget(self):
        rows = [_stream_row(events=[], truth_onsets=[5.0], duration_s=3600.0)]
        m = score_ww_stream(rows)
        assert m["error_at_1fa_per_hour"] == 1.0
        assert m["recall_at_2fa_per_hour"] == 0.0


class TestWwStreamCi:
    def test_ci_is_deterministic_and_ordered(self):
        from arena.metrics import primary_metric_ci

        rows = [
            _row(sample_id=f"s{i}", extras={
                "events": [[10.0, 0.9]] if i % 2 else [],
                "truth_onsets": [10.0], "duration_s": 600.0,
            })
            for i in range(20)
        ]
        ci = primary_metric_ci("ww_stream", rows)
        assert ci is not None and ci[0] <= ci[1]
        assert 0.0 <= ci[0] and ci[1] <= 1.0
        assert primary_metric_ci("ww_stream", rows) == ci

    def test_no_onsets_has_no_ci(self):
        from arena.metrics import primary_metric_ci

        rows = [_stream_row(events=[], truth_onsets=[], duration_s=600.0)]
        assert primary_metric_ci("ww_stream", rows) is None


# ---------------------------------------------------------------------------
# Elo seed from per-onset auto-battles
# ---------------------------------------------------------------------------


def _seed_samples(per_competitor_events, n_clips=40, nuisance=None):
    """{sample_id: {competitor: row}} — n_clips one-hour recordings with an
    onset at 100 s; *per_competitor_events(comp, i)* gives the events."""
    samples: dict = {}
    for i in range(n_clips):
        sid = f"clip{i:03d}"
        samples[sid] = {
            comp: _row(
                competitor_id=comp, sample_id=sid, plugin_id=f"plugin-{comp}",
                extras={
                    "events": events, "truth_onsets": [100.0],
                    "duration_s": 3600.0,
                },
            )
            for comp, events in per_competitor_events(i).items()
        }
    return samples


class TestWwStreamEloSeed:
    def _seed(self, per_competitor_events, **kw):
        from arena.assembler import seed_elo

        samples = _seed_samples(per_competitor_events, **kw)
        return seed_elo("ww_stream", "en-US", {"ds": samples}, "2026-01-01T00:00:00Z")

    def test_detected_beats_missed_per_onset(self):
        seed = self._seed(lambda i: {"good": [[100.2, 0.99]], "deaf": []})
        assert seed.modality == "ww_stream"
        assert seed.auto_vote_count == 40
        assert seed.ratings["good"] > seed.ratings["deaf"]
        assert seed.wins["good"] == 40 and seed.losses["deaf"] == 40
        assert seed.competitor_plugin == {"good": "plugin-good", "deaf": "plugin-deaf"}
        assert seed.pairwise_games["good"]["deaf"] > 0

    def test_both_or_neither_detected_is_no_battle(self):
        seed = self._seed(lambda i: {"a": [[100.2, 0.99]], "b": [[100.4, 0.97]]})
        assert seed.auto_vote_count == 0
        assert set(seed.ratings) == {"a", "b"}

    def test_each_competitor_is_judged_at_its_one_fa_per_hour_point(self):
        # "noisy" hears every onset (0.9) but fires 3 false accepts per clip
        # at 0.95 — 3 FA/h, so its 1 FA/h point sits above 0.95 and it
        # misses the onsets. Counting any event ever fired would tie it.
        noisy_events = [[100.1, 0.9], [500.0, 0.95], [900.0, 0.95], [1300.0, 0.95]]
        seed = self._seed(
            lambda i: {"quiet": [[100.2, 0.99]], "noisy": noisy_events}
        )
        assert seed.wins["quiet"] == 40
        assert seed.ratings["quiet"] > seed.ratings["noisy"]

    def test_overlapping_cis_gate_out_the_pair(self):
        # each detects a different half of the recordings: FRR 0.5 both, CIs
        # overlap, so the per-onset disagreements seed nothing.
        seed = self._seed(lambda i: {
            "a": [[100.2, 0.99]] if i % 2 else [],
            "b": [] if i % 2 else [[100.2, 0.99]],
        })
        assert seed.auto_vote_count == 0

    def test_seed_is_deterministic(self):
        def build():
            return self._seed(lambda i: {"good": [[100.2, 0.99]], "deaf": []})

        assert build().model_dump() == build().model_dump()

    def test_league_is_listed_for_the_frontend(self):
        from arena.models import leagues

        entry = next(x for x in leagues({"ww_stream"}) if x["id"] == "ww_stream")
        assert entry["label"] and entry["voteless"] is False


# ---------------------------------------------------------------------------
# registry: test_set_kind
# ---------------------------------------------------------------------------


class TestTestSetKind:
    def _def(self, **over):
        base = dict(
            dataset_id="ww_stream_x", modality="ww_stream",
            source={"type": "huggingface", "hf_id": "org/x", "revision": "main"},
            lang="en-US",
        )
        base.update(over)
        return DatasetDef(**base)

    def test_accepts_real_and_synthetic(self):
        assert self._def(test_set_kind="real").test_set_kind == "real"
        assert self._def(test_set_kind="synthetic").test_set_kind == "synthetic"
        assert self._def().test_set_kind is None

    def test_rejects_other_values(self):
        with pytest.raises(Exception):
            self._def(test_set_kind="mixed")

    def test_rejected_outside_ww_stream(self):
        with pytest.raises(Exception):
            self._def(modality="wake_word", test_set_kind="synthetic")

    def test_surfaced_in_board_dataset_info(self, monkeypatch):
        from arena.cli import _dataset_info_lookup

        monkeypatch.setattr(
            "registry.loaders.list_datasets",
            lambda *a, **k: [self._def(test_set_kind="synthetic")],
        )
        info = _dataset_info_lookup([])
        assert info["ww_stream_x"]["test_set_kind"] == "synthetic"


class TestOnsetEndWindow:
    def _clip_row(self, event_t, ends=True):
        extras = {"events": [[event_t, 1.0]], "truth_onsets": [10.0],
                  "duration_s": 3600.0}
        if ends:
            extras["truth_ends"] = [12.0]  # a 2 s phrase starting at 10 s
        return _row(extras=extras)

    def test_detection_after_a_long_phrase_matches(self):
        m = score_ww_stream([self._clip_row(12.5)])  # 2.5 s after the start
        assert m["frr"] == 0.0 and m["fa_per_hour"] == 0.0
        assert m["latency_s_median"] == pytest.approx(2.5)

    def test_detection_up_to_end_plus_tolerance_matches(self):
        assert score_ww_stream([self._clip_row(12.0 + EVENT_TOLERANCE_S)])["frr"] == 0.0
        late = score_ww_stream([self._clip_row(12.0 + EVENT_TOLERANCE_S + 0.01)])
        assert late["frr"] == 1.0 and late["fa_per_hour"] == pytest.approx(1.0)

    def test_detection_before_the_start_does_not_match(self):
        m = score_ww_stream([self._clip_row(9.5)])
        assert m["frr"] == 1.0
        assert m["fa_per_hour"] == pytest.approx(1.0)

    def test_manifest_without_ends_scores_as_before(self):
        assert score_ww_stream([self._clip_row(12.5, ends=False)])["frr"] == 1.0
        assert score_ww_stream([self._clip_row(9.5, ends=False)])["frr"] == 0.0

    def test_runner_carries_ends_into_the_row(self):
        from runner.ww_bench import WakeWordStreamBench

        stack = WWStack(ww=_DummyStreamEngine())
        sample = {"array": np.zeros(SAMPLE_RATE, dtype="float32"),
                  "truth_onsets": [0.1], "truth_ends": [0.6]}
        out = WakeWordStreamBench().predict(stack, sample, None)
        assert out["extras"]["truth_ends"] == [0.6]
        sample["truth_ends"] = None
        out = WakeWordStreamBench().predict(stack, sample, None)
        assert out["extras"]["truth_ends"] == []
