"""Tests for M2 performance boards: RTF, peak memory, model size, Pareto.

See docs/methodology.md (perf-metrics section) for the median-RTF and
tier-separation rationale these tests pin down.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arena.metrics import (
    hw_tier_of,
    pareto_frontier,
    perf_metrics_by_tier,
    primary_perf_tier,
    row_rtf,
)
from arena.model_size import likely_hf_repo_id, model_repo_size_mb
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestRowRtf:
    def test_computes_elapsed_over_audio(self):
        row = _row(elapsed_ms=2000.0, audio_secs=4.0)
        assert row_rtf(row) == 0.5

    def test_none_without_elapsed_ms(self):
        assert row_rtf(_row(audio_secs=4.0)) is None

    def test_none_without_audio_secs(self):
        assert row_rtf(_row(elapsed_ms=2000.0)) is None

    def test_none_for_zero_or_negative_audio_secs(self):
        assert row_rtf(_row(elapsed_ms=2000.0, audio_secs=0.0)) is None
        assert row_rtf(_row(elapsed_ms=2000.0, audio_secs=-1.0)) is None


class TestHwTierOf:
    def test_extracts_host_class(self):
        row = _row(hw={"host_class": "cpu-x86", "hostname": "x"})
        assert hw_tier_of(row) == "cpu-x86"

    def test_none_without_hw(self):
        assert hw_tier_of(_row()) is None

    def test_none_without_host_class_key(self):
        assert hw_tier_of(_row(hw={"hostname": "x"})) is None


class TestPerfMetricsByTier:
    def test_median_rtf_robust_to_cold_start_outlier(self):
        # One huge cold-start outlier among steady-state calls: the MEDIAN
        # should stay close to steady state, not get dragged by the mean.
        hw = {"host_class": "cpu-x86"}
        rows = [
            _row(elapsed_ms=30000.0, audio_secs=1.0, hw=hw),  # cold start: rtf=30
            _row(elapsed_ms=500.0, audio_secs=1.0, hw=hw),    # rtf=0.5
            _row(elapsed_ms=500.0, audio_secs=1.0, hw=hw),    # rtf=0.5
            _row(elapsed_ms=500.0, audio_secs=1.0, hw=hw),    # rtf=0.5
            _row(elapsed_ms=500.0, audio_secs=1.0, hw=hw),    # rtf=0.5
        ]
        perf = perf_metrics_by_tier(rows)
        assert perf["cpu-x86"]["rtf"] == 0.5  # median, not mean (~6.4)
        assert perf["cpu-x86"]["rtf_n"] == 5.0

    def test_peak_rss_aggregated_as_max(self):
        hw = {"host_class": "cpu-x86"}
        rows = [
            _row(peak_rss_mb=100.0, hw=hw),
            _row(peak_rss_mb=250.0, hw=hw),
            _row(peak_rss_mb=180.0, hw=hw),
        ]
        perf = perf_metrics_by_tier(rows)
        assert perf["cpu-x86"]["peak_rss_mb"] == 250.0
        assert perf["cpu-x86"]["peak_rss_mb_n"] == 3.0

    def test_tiers_never_blended(self):
        cpu_hw = {"host_class": "cpu-x86"}
        gpu_hw = {"host_class": "gpu"}
        rows = [
            _row(elapsed_ms=2000.0, audio_secs=1.0, peak_rss_mb=500.0, hw=cpu_hw),
            _row(elapsed_ms=200.0, audio_secs=1.0, peak_rss_mb=1200.0, hw=gpu_hw),
        ]
        perf = perf_metrics_by_tier(rows)
        assert set(perf.keys()) == {"cpu-x86", "gpu"}
        assert perf["cpu-x86"]["rtf"] == 2.0
        assert perf["gpu"]["rtf"] == 0.2
        assert perf["cpu-x86"]["peak_rss_mb"] == 500.0
        assert perf["gpu"]["peak_rss_mb"] == 1200.0

    def test_rows_without_hw_are_excluded_not_zeroed(self):
        rows = [_row(elapsed_ms=100.0, audio_secs=1.0)]  # no hw — pre-#90 row
        assert perf_metrics_by_tier(rows) == {}

    def test_empty_rows_gives_empty_dict(self):
        assert perf_metrics_by_tier([]) == {}


class TestPrimaryPerfTier:
    def test_picks_tier_with_most_samples(self):
        perf = {
            "cpu-x86": {"rtf": 1.0, "rtf_n": 2.0},
            "gpu": {"rtf": 0.2, "rtf_n": 10.0},
        }
        assert primary_perf_tier(perf) == "gpu"

    def test_none_when_no_perf(self):
        assert primary_perf_tier({}) is None


class TestModelRepoSizeMb:
    def test_none_for_falsy_repo_id(self):
        assert model_repo_size_mb(None) is None
        assert model_repo_size_mb("") is None

    def test_sums_sibling_sizes_via_hf_api(self):
        sib1 = MagicMock(size=1024 * 1024, lfs=None)  # 1 MB
        sib2 = MagicMock(size=2 * 1024 * 1024, lfs=None)  # 2 MB
        info = MagicMock(siblings=[sib1, sib2])
        fake_api = MagicMock()
        fake_api.model_info.return_value = info
        cache: dict = {}
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            size = model_repo_size_mb("org/repo", cache=cache)
        assert size == 3.0
        fake_api.model_info.assert_called_once()

    def test_caches_across_calls(self):
        sib1 = MagicMock(size=1024 * 1024, lfs=None)
        info = MagicMock(siblings=[sib1])
        fake_api = MagicMock()
        fake_api.model_info.return_value = info
        cache: dict = {}
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            model_repo_size_mb("org/repo", cache=cache)
            model_repo_size_mb("org/repo", cache=cache)
        assert fake_api.model_info.call_count == 1

    def test_none_on_lookup_failure(self):
        fake_api = MagicMock()
        fake_api.model_info.side_effect = RuntimeError("network down")
        cache: dict = {}
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            assert model_repo_size_mb("org/repo", cache=cache) is None


class TestLikelyHfRepoId:
    def test_accepts_owner_slash_name(self):
        assert likely_hf_repo_id("OpenVoiceOS/ARK-ASR-0.6B-onnx") is True
        assert likely_hf_repo_id("openai/whisper-tiny") is True
        assert likely_hf_repo_id("k2-fsa/OmniVoice") is True

    def test_rejects_free_text_model_values(self):
        # Real values seen in the registry's pre-existing "model" field
        # that are NOT HF repo ids — must never be sent to HF as a lookup.
        assert likely_hf_repo_id("sabela") is False  # cotovia voice id
        assert likely_hf_repo_id("tts_models/en/ljspeech/vits") is False  # coqui path — has slash but 3 segments? actually 3 slashes
        assert likely_hf_repo_id("gTTS en (tld=us)") is False  # spaces/parens
        assert likely_hf_repo_id("YatharthS/LuxTTS (zipvoice)") is False  # trailing free text

    def test_rejects_none_and_empty(self):
        assert likely_hf_repo_id(None) is False
        assert likely_hf_repo_id("") is False

    def test_rejects_no_slash(self):
        assert likely_hf_repo_id("f1") is False


class TestAttachModelSizesFallsBackToModelField:
    def test_falls_back_to_model_field_when_no_model_hf_repo(self):
        from arena.cli import _attach_model_sizes
        from arena.models import BenchmarkBoard, BenchmarkEntry

        fake_comp = MagicMock(competitor_id="c", model_hf_repo=None,
                               model="OpenVoiceOS/ARK-ASR-0.6B-onnx")
        board = BenchmarkBoard(
            modality="stt", dataset_id="d", lang="en-US", generated_at="t",
            primary_metric="wer_mean",
            entries=[BenchmarkEntry(competitor_id="c", plugin_id="p", samples=1,
                                     metrics={"wer_mean": 0.1})],
        )
        with patch("registry.loaders.list_competitors", return_value=[fake_comp]), \
             patch("arena.model_size.model_repo_size_mb", return_value=42.0) as fake_size:
            _attach_model_sizes(board, "stt")
        fake_size.assert_called_once_with("OpenVoiceOS/ARK-ASR-0.6B-onnx")
        assert board.entries[0].model_mb == 42.0

    def test_model_hf_repo_takes_priority_over_model_field(self):
        from arena.cli import _attach_model_sizes
        from arena.models import BenchmarkBoard, BenchmarkEntry

        fake_comp = MagicMock(competitor_id="c", model_hf_repo="explicit/override",
                               model="OpenVoiceOS/ARK-ASR-0.6B-onnx")
        board = BenchmarkBoard(
            modality="stt", dataset_id="d", lang="en-US", generated_at="t",
            primary_metric="wer_mean",
            entries=[BenchmarkEntry(competitor_id="c", plugin_id="p", samples=1,
                                     metrics={"wer_mean": 0.1})],
        )
        with patch("registry.loaders.list_competitors", return_value=[fake_comp]), \
             patch("arena.model_size.model_repo_size_mb", return_value=7.0) as fake_size:
            _attach_model_sizes(board, "stt")
        fake_size.assert_called_once_with("explicit/override")

    def test_non_repo_shaped_model_field_never_looked_up(self):
        from arena.cli import _attach_model_sizes
        from arena.models import BenchmarkBoard, BenchmarkEntry

        fake_comp = MagicMock(competitor_id="c", model_hf_repo=None,
                               model="gTTS en (tld=us)")
        board = BenchmarkBoard(
            modality="tts", dataset_id="d", lang="en-US", generated_at="t",
            primary_metric="utmos",
            entries=[BenchmarkEntry(competitor_id="c", plugin_id="p", samples=1,
                                     metrics={"utmos": 4.0})],
        )
        with patch("registry.loaders.list_competitors", return_value=[fake_comp]), \
             patch("arena.model_size.model_repo_size_mb") as fake_size:
            _attach_model_sizes(board, "tts")
        fake_size.assert_not_called()
        assert board.entries[0].model_mb is None


class TestParetoFrontier:
    def test_dominated_fighter_is_excluded(self):
        # b is worse quality AND slower than a — strictly dominated.
        points = [("a", 0.9, 0.5), ("b", 0.8, 0.6)]
        frontier, dominated = pareto_frontier(points)
        assert frontier == ["a"]
        assert dominated == 1

    def test_tradeoff_fighters_both_stay_on_frontier(self):
        # a: best quality, slower. c: fastest, lower quality. Neither
        # dominates the other — both are genuine trade-offs.
        points = [("a", 0.95, 1.0), ("c", 0.70, 0.1)]
        frontier, dominated = pareto_frontier(points)
        assert set(frontier) == {"a", "c"}
        assert dominated == 0

    def test_hand_fixture_mixed_dominated_and_frontier(self):
        # a: quality .95, rtf 1.0   -> frontier (best quality)
        # b: quality .90, rtf 1.2   -> dominated by a (a is better AND faster)
        # c: quality .85, rtf 0.3   -> frontier (much faster, decent quality)
        # d: quality .80, rtf 0.5   -> dominated by c (c is better AND faster)
        points = [
            ("a", 0.95, 1.0),
            ("b", 0.90, 1.2),
            ("c", 0.85, 0.3),
            ("d", 0.80, 0.5),
        ]
        frontier, dominated = pareto_frontier(points)
        assert set(frontier) == {"a", "c"}
        assert dominated == 2

    def test_ties_do_not_dominate_each_other(self):
        points = [("a", 0.9, 0.5), ("b", 0.9, 0.5)]
        frontier, dominated = pareto_frontier(points)
        assert set(frontier) == {"a", "b"}
        assert dominated == 0

    def test_lower_is_better_quality_supported(self):
        # e.g. WER as the "quality" axis — lower WER is better.
        points = [("a", 0.05, 0.5), ("b", 0.10, 0.6)]
        frontier, dominated = pareto_frontier(points, quality_higher_is_better=False)
        assert frontier == ["a"]
        assert dominated == 1

    def test_empty_points(self):
        assert pareto_frontier([]) == ([], 0)


class TestBuildBenchmarkBoardAttachesPerf:
    def test_entry_perf_populated_from_rows(self):
        from arena.metrics import build_benchmark_board

        hw = {"host_class": "cpu-x86"}
        rows = [
            _row(competitor_id="a", reference_intent="x", prediction="x",
                 elapsed_ms=500.0, audio_secs=1.0, peak_rss_mb=300.0, hw=hw),
            _row(competitor_id="a", reference_intent="y", prediction="y",
                 elapsed_ms=500.0, audio_secs=1.0, peak_rss_mb=320.0, hw=hw),
        ]
        by_competitor = {"a": rows}
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        entry = board.entries[0]
        assert entry.perf is not None
        assert entry.perf["cpu-x86"]["rtf"] == 0.5
        assert entry.perf["cpu-x86"]["peak_rss_mb"] == 320.0

    def test_entry_perf_none_without_hw(self):
        from arena.metrics import build_benchmark_board

        rows = [_row(competitor_id="a", reference_intent="x", prediction="x")]
        board = build_benchmark_board("intent", "d", "en-US", {"a": rows}, "t")
        assert board.entries[0].perf is None
