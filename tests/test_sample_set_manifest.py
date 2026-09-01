"""Unit tests for the sample-set manifest gap: comparability of predictions
already swept over different sample populations for a ``sample_policy``-
capped dataset (runner/publish_sample_set.py, arena.metrics.build_benchmark_
board's sample_set_ids, arena.cli._load_sample_set).
"""
from __future__ import annotations

from types import SimpleNamespace

from arena.metrics import build_benchmark_board
from arena.models import PredictionRow
from registry.schemas import SamplePolicy


def _row(**over):
    base = dict(competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
                plugin_id="p")
    base.update(over)
    return PredictionRow(**base)


class _FakeParquetCorpus:
    """A fake in-memory parquet corpus shared by publish_sample_set and
    stream_audio_dataset tests, so both exercise the identical row walk."""

    def __init__(self, monkeypatch, n=50):
        self.n = n
        import numpy as np

        from runner import audio_io

        class FakeMeta:
            def __init__(self, num_rows):
                self.num_rows = num_rows

        class FakeParquetFile:
            def __init__(self, path):
                self.metadata = FakeMeta(n)

        class FakeTable:
            def to_batches(self, max_chunksize=64):
                for start in range(0, n, max_chunksize):
                    end = min(start + max_chunksize, n)
                    yield SimpleNamespace(to_pydict=lambda start=start, end=end: {
                        "audio": [
                            {"array": [0.0], "sampling_rate": 16000}
                            for _ in range(start, end)
                        ],
                        "id": list(range(start, end)),
                    })

        monkeypatch.setattr(audio_io, "_parquet_files",
                            lambda hf_id, subset, split, revision: ["a.parquet"])

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **k: "/fake/a.parquet")

        import pyarrow.parquet as pq
        monkeypatch.setattr(pq, "ParquetFile", FakeParquetFile)
        monkeypatch.setattr(pq, "read_table", lambda path: FakeTable())

    def source(self):
        return SimpleNamespace(hf_id="org/corpus", subset=None, split="test")

    def dataset_def(self, max_samples=8, seed=5, lang="en-US"):
        return SimpleNamespace(
            dataset_id="fake-corpus",
            lang=lang,
            modality=SimpleNamespace(value="stt"),
            source=self.source(),
            reference_fields={"audio": "audio"},
            sample_policy=SamplePolicy(max_samples=max_samples, seed=seed),
        )


class TestComputeSampleSetDeterminism:
    def test_same_policy_same_manifest_across_two_calls(self, monkeypatch):
        from runner.publish_sample_set import compute_sample_set

        corpus = _FakeParquetCorpus(monkeypatch, n=50)
        dataset_def = corpus.dataset_def(max_samples=10, seed=42)
        a = compute_sample_set(dataset_def, "main")
        b = compute_sample_set(dataset_def, "main")
        assert a["sample_ids"] == b["sample_ids"]
        assert len(a["sample_ids"]) == 10
        assert a["total_rows"] == 50
        assert a["seed"] == 42
        assert a["max_samples"] == 10

    def test_manifest_ids_match_what_stream_audio_dataset_yields(self, monkeypatch):
        """The manifest's ids must be byte-identical to a real sweep's ids
        for the same policy — both derive from the same _sample_id/
        _iter_parquet_rows row-walk."""
        from runner.audio_io import stream_audio_dataset
        from runner.publish_sample_set import compute_sample_set

        corpus = _FakeParquetCorpus(monkeypatch, n=50)
        dataset_def = corpus.dataset_def(max_samples=10, seed=7)
        manifest = compute_sample_set(dataset_def, "main")

        # id_key=None (default): real sweeps (stt_bench.py, ww_bench.py) call
        # stream_audio_dataset without an explicit id_key, so this must match
        # compute_sample_set's derivation, which also uses the fallback path.
        streamed_ids = {
            sid for sid, _ in stream_audio_dataset(
                corpus.source(), audio_key="audio", extra_keys={},
                revision="main", max_samples=10, seed=7)
        }
        assert set(manifest["sample_ids"]) == streamed_ids

    def test_no_policy_raises(self, monkeypatch):
        import pytest

        from runner.publish_sample_set import compute_sample_set

        corpus = _FakeParquetCorpus(monkeypatch, n=50)
        dataset_def = corpus.dataset_def()
        dataset_def.sample_policy = None
        with pytest.raises(ValueError):
            compute_sample_set(dataset_def, "main")


class TestFallbackIdStability:
    """The fallback sample_id (no 'path' on the audio cell) must be derived
    from the row's ABSOLUTE position in the corpus, not its position in the
    filtered/yielded output — otherwise the same physical row gets a
    different id whenever the cap changes, and a manifest built at one cap
    can never match a sweep run at another."""

    def test_same_row_same_fallback_id_across_different_caps(self, monkeypatch):
        from runner.audio_io import stream_audio_dataset

        corpus = _FakeParquetCorpus(monkeypatch, n=50)
        src = corpus.source()
        # id_key=None forces the fallback id path (no "path" field on our
        # fake audio cell, so _sample_id falls through to sample_{pos}).
        wide = dict(stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"},
            revision="main", max_samples=50, seed=None))
        narrow = dict(stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"},
            revision="main", max_samples=5, seed=None))

        # every row present in the narrow (smaller-cap) run must carry the
        # SAME sample_id it has in the wide (larger-cap) run.
        wide_by_row_id = {v["id"]: k for k, v in wide.items()}
        for sid, sample in narrow.items():
            assert wide_by_row_id[sample["id"]] == sid


class TestBenchmarkBoardSampleSetFiltering:
    def test_full_coverage_fighter_is_comparable_and_ranked(self):
        manifest_ids = {"s0", "s1", "s2", "s3"}
        rows = [_row(competitor_id="full", sample_id=sid, wer=0.1)
                for sid in manifest_ids]
        board = build_benchmark_board(
            "stt", "d", "en-US", {"full": rows}, "t", sample_set_ids=manifest_ids)
        entry = board.entries[0]
        assert entry.sample_set == "manifest"
        assert entry.sample_set_coverage == 1.0
        assert entry.unranked is False
        assert entry.rank == 1

    def test_partial_coverage_fighter_is_unranked(self):
        manifest_ids = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"}
        # only 3 of 10 manifest rows present -> coverage 0.3
        rows = [_row(competitor_id="partial", sample_id=sid, wer=0.1)
                for sid in ("s0", "s1", "s2")]
        board = build_benchmark_board(
            "stt", "d", "en-US", {"partial": rows}, "t", sample_set_ids=manifest_ids)
        entry = board.entries[0]
        assert entry.sample_set == "manifest"
        assert abs(entry.sample_set_coverage - 0.3) < 1e-9
        assert entry.unranked is True
        assert entry.unranked_reason is not None
        assert "sample_set_partial" in entry.unranked_reason
        assert entry.rank == 0

    def test_rows_outside_manifest_are_dropped_before_scoring(self):
        manifest_ids = {"s0", "s1"}
        rows = [
            _row(competitor_id="c", sample_id="s0", wer=0.0),
            _row(competitor_id="c", sample_id="s1", wer=0.0),
            _row(competitor_id="c", sample_id="not-in-manifest", wer=1.0),
        ]
        board = build_benchmark_board(
            "stt", "d", "en-US", {"c": rows}, "t", sample_set_ids=manifest_ids)
        entry = board.entries[0]
        assert entry.samples == 2
        assert entry.metrics["wer_mean"] == 0.0  # the outlier row was excluded

    def test_no_sample_set_ids_is_unmanaged(self):
        rows = [_row(competitor_id="c", sample_id="s0", wer=0.1)]
        board = build_benchmark_board("stt", "d", "en-US", {"c": rows}, "t")
        entry = board.entries[0]
        assert entry.sample_set == "unmanaged"
        assert entry.sample_set_coverage is None
        assert entry.unranked is False

    def test_two_fighters_different_populations_become_comparable(self):
        """The exact regression the reviewer flagged: a full-corpus-swept
        fighter and a fighter swept on the seeded subset must be scored
        over the SAME rows once a manifest is applied."""
        manifest_ids = {"s0", "s1"}
        full_corpus_fighter = [
            _row(competitor_id="full", sample_id=sid, wer=0.2)
            for sid in ("s0", "s1", "s2", "s3", "s4")  # swept before the cap existed
        ]
        subset_fighter = [
            _row(competitor_id="subset", sample_id=sid, wer=0.2)
            for sid in ("s0", "s1")
        ]
        board = build_benchmark_board(
            "stt", "d", "en-US",
            {"full": full_corpus_fighter, "subset": subset_fighter}, "t",
            sample_set_ids=manifest_ids,
        )
        by_id = {e.competitor_id: e for e in board.entries}
        assert by_id["full"].samples == 2
        assert by_id["subset"].samples == 2
        assert by_id["full"].sample_set_coverage == 1.0
        assert by_id["subset"].sample_set_coverage == 1.0


class TestLoadSampleSetFallback:
    def test_absent_manifest_logs_warning_and_returns_none(self, monkeypatch, caplog):
        from arena import cli

        cli._SAMPLE_SET_CACHE.clear()

        def fake_load_dataset(modality, dataset_id):
            return SimpleNamespace(
                sample_policy=SamplePolicy(max_samples=100, seed=1),
                predictions_hf="OpenVoiceOS/ovos-stt-bench-fake",
            )

        monkeypatch.setattr("registry.loaders.load_dataset", fake_load_dataset)

        def fake_hf_hub_download(*a, **k):
            raise FileNotFoundError("404")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

        import logging
        with caplog.at_level(logging.WARNING):
            result = cli._load_sample_set("stt", "fake", "en-US")
        assert result is None
        assert any("sample_sets manifest is" in r.message for r in caplog.records)

    def test_no_policy_returns_none_without_warning(self, monkeypatch, caplog):
        from arena import cli

        cli._SAMPLE_SET_CACHE.clear()

        def fake_load_dataset(modality, dataset_id):
            return SimpleNamespace(sample_policy=None, predictions_hf=None)

        monkeypatch.setattr("registry.loaders.load_dataset", fake_load_dataset)

        import logging
        with caplog.at_level(logging.WARNING):
            result = cli._load_sample_set("stt", "small-set", "en-US")
        assert result is None
        assert not any("sample_sets manifest" in r.message for r in caplog.records)

    def test_present_manifest_is_parsed(self, monkeypatch, tmp_path):
        import json as jsonmod

        from arena import cli

        cli._SAMPLE_SET_CACHE.clear()

        def fake_load_dataset(modality, dataset_id):
            return SimpleNamespace(
                sample_policy=SamplePolicy(max_samples=2, seed=1),
                predictions_hf="OpenVoiceOS/ovos-stt-bench-fake",
            )

        monkeypatch.setattr("registry.loaders.load_dataset", fake_load_dataset)

        manifest_path = tmp_path / "en_US.json"
        manifest_path.write_text(jsonmod.dumps({"sample_ids": ["a", "b"]}))
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            lambda *a, **k: str(manifest_path))

        result = cli._load_sample_set("stt", "fake", "en-US")
        assert result == {"a", "b"}
