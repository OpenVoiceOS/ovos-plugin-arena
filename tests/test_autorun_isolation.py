"""Regression tests for the child-process isolation around
``AutoRunner._run_pair_batch``.

Each ``(fighter, dataset, lang)`` batch now runs ``mb.run_competitor_lang``
in a ``multiprocessing`` (spawn) child instead of in-process, so a wedged or
leaking model load in one pair can never hang or bloat the long-lived
autorun daemon. These adapters stand in for real STT/TTS plugins — no audio
stack, no onnxruntime/torch — but they must be defined at module scope
(not as local closures) so ``spawn`` can pickle a reference to them and
re-import them in the child process.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from registry.loaders import load_competitor
from runner.autorun import AutoRunConfig, AutoRunner
from runner.media_bench import MediaBenchAdapter


def _competitor():
    return load_competitor("stt", "whispercpp-base")


def _eval_def(dataset_id="minds14-en"):
    return SimpleNamespace(dataset_id=dataset_id,
                            source=SimpleNamespace(hf_id="PolyAI/minds14",
                                                    revision="main"))


class NormalAdapter(MediaBenchAdapter):
    """Well-behaved stand-in: loads instantly, predicts instantly."""

    modality = "stt"

    def iter_samples(self, dataset_def, lang, revision, max_samples):
        n = min(3, max_samples) if max_samples else 3
        for i in range(n):
            yield f"{lang}/{i:05d}", {"i": i}

    def load_engine(self, competitor, lang):
        return "ENGINE"

    def predict(self, engine, sample, ctx):
        return {"reference_text": "ref", "prediction": f"hyp{sample['i']}",
                "latency_ms": 1.0}


class WedgedLoadAdapter(MediaBenchAdapter):
    """Simulates a model load that never returns (the on-call defect:
    ``adapter.load_engine`` hanging with no timeout anywhere above it)."""

    modality = "stt"

    def iter_samples(self, dataset_def, lang, revision, max_samples):
        yield f"{lang}/00000", {"i": 0}

    def load_engine(self, competitor, lang):
        time.sleep(3600)  # "forever" relative to test timeouts
        return "ENGINE"

    def predict(self, engine, sample, ctx):
        return {"reference_text": "ref", "prediction": "hyp", "latency_ms": 1.0}


class CrashingLoadAdapter(MediaBenchAdapter):
    """Simulates a fighter that hard-fails to load."""

    modality = "stt"

    def iter_samples(self, dataset_def, lang, revision, max_samples):
        yield f"{lang}/00000", {"i": 0}

    def load_engine(self, competitor, lang):
        raise RuntimeError("plugin failed to import: boom")

    def predict(self, engine, sample, ctx):
        raise AssertionError("predict should never be reached")


@pytest.mark.timeout(60)
class TestChildProcessIsolation:
    def test_normal_batch_matches_in_process_rows(self, tmp_path):
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False,
                                seed_from_hf=False)
        runner = AutoRunner(config)
        competitor = _competitor()
        dataset = _eval_def()

        import runner.autorun as autorun_module
        autorun_module._ADAPTER_FACTORIES["stt"] = NormalAdapter

        result = runner._run_pair_batch("stt", competitor, dataset, "en-US", 10)

        assert result.written == 3
        assert result.errored == 0
        out_path = (tmp_path / dataset.dataset_id / "stt" / "predictions"
                    / "en-US" / f"{competitor.competitor_id}.jsonl")
        assert out_path.exists()
        assert len(out_path.read_text().splitlines()) == 3

    def test_wedged_load_times_out_and_quarantines(self, tmp_path):
        import runner.autorun as autorun_module
        autorun_module._ADAPTER_FACTORIES["stt"] = WedgedLoadAdapter

        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False,
                                seed_from_hf=False, load_timeout=2.0)
        runner = AutoRunner(config)
        competitor = _competitor()
        dataset = _eval_def()

        result = runner.process_pair("stt", competitor, dataset, "en-US")

        assert result is None
        assert runner.scheduler.is_quarantined(competitor.competitor_id, now=0.0)
        entry = runner.scheduler.quarantined[competitor.competitor_id]
        assert "load_or_batch_timeout" in entry.reason

    def test_child_raise_is_captured_and_quarantines(self, tmp_path):
        import runner.autorun as autorun_module
        autorun_module._ADAPTER_FACTORIES["stt"] = CrashingLoadAdapter

        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False,
                                seed_from_hf=False, load_timeout=30.0)
        runner = AutoRunner(config)
        competitor = _competitor()
        dataset = _eval_def()

        result = runner.process_pair("stt", competitor, dataset, "en-US")

        assert result is None
        assert runner.scheduler.is_quarantined(competitor.competitor_id, now=0.0)
        entry = runner.scheduler.quarantined[competitor.competitor_id]
        assert "plugin failed to import: boom" in entry.reason
