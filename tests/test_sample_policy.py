"""Unit tests for the registry-owned deterministic sampling policy.

``sample_policy`` on ``DatasetDef`` (registry/schemas.py) replaces the
operator-typed ``--max-samples`` CLI flag as the source of truth for how many
rows a sweep draws per (dataset, lang): a registry cap is comparable across
runs and fighters, and the selected subset must be the SAME every time.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from registry.schemas import DatasetDef, SamplePolicy
from runner.audio_io import resolve_sample_cap, select_sample_positions


class TestSamplePolicySchema:
    def _def(self, **over):
        base = dict(
            dataset_id="x", modality="stt",
            source={"type": "path", "path": "/x.jsonl"},
            lang="en-US",
        )
        base.update(over)
        return DatasetDef(**base)

    def test_defaults_to_unset(self):
        d = self._def()
        assert d.sample_policy is None

    def test_accepts_max_samples_and_seed(self):
        d = self._def(sample_policy={"max_samples": 2000, "seed": 7})
        assert d.sample_policy.max_samples == 2000
        assert d.sample_policy.seed == 7

    def test_seed_has_a_fixed_default(self):
        d = self._def(sample_policy={"max_samples": 500})
        assert isinstance(d.sample_policy.seed, int)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            SamplePolicy(max_samples=10, bogus_field=True)


class TestSelectSamplePositions:
    def test_no_cap_returns_none(self):
        assert select_sample_positions(1000, None, seed=1) is None
        assert select_sample_positions(1000, 0, seed=1) is None

    def test_cap_at_or_above_total_returns_none(self):
        assert select_sample_positions(10, 10, seed=1) is None
        assert select_sample_positions(10, 50, seed=1) is None

    def test_selection_size_matches_cap(self):
        sel = select_sample_positions(1000, 200, seed=1)
        assert sel is not None
        assert len(sel) == 200
        assert all(0 <= p < 1000 for p in sel)

    def test_deterministic_same_inputs_same_subset(self):
        a = select_sample_positions(1000, 200, seed=42)
        b = select_sample_positions(1000, 200, seed=42)
        assert a == b

    def test_different_seed_generally_different_subset(self):
        a = select_sample_positions(1000, 200, seed=42)
        b = select_sample_positions(1000, 200, seed=43)
        assert a != b

    def test_not_a_trivial_head_slice(self):
        # a real shuffle, not just range(max_samples) — otherwise seeding buys
        # nothing over the pre-policy plain head-of-stream truncation.
        sel = select_sample_positions(1000, 200, seed=42)
        assert sel != set(range(200))


class TestResolveSampleCap:
    def _def(self, sample_policy=None):
        return SimpleNamespace(sample_policy=sample_policy)

    def test_no_policy_passes_cli_cap_through_no_seed(self):
        eff, seed = resolve_sample_cap(self._def(None), cli_max_samples=50)
        assert eff == 50
        assert seed is None

    def test_no_policy_no_cli_cap_stays_unbounded(self):
        eff, seed = resolve_sample_cap(self._def(None), cli_max_samples=0)
        assert eff == 0
        assert seed is None

    def test_policy_honoured_when_cli_cap_is_zero(self):
        policy = SamplePolicy(max_samples=2000, seed=99)
        eff, seed = resolve_sample_cap(self._def(policy), cli_max_samples=0)
        assert eff == 2000
        assert seed == 99

    def test_cli_cap_smaller_than_policy_still_wins(self):
        # smoke runs: an operator-typed small cap must not be overridden by
        # a large registry policy cap.
        policy = SamplePolicy(max_samples=2000, seed=99)
        eff, seed = resolve_sample_cap(self._def(policy), cli_max_samples=10)
        assert eff == 10
        assert seed == 99  # selection is still the deterministic policy subset

    def test_cli_cap_larger_than_policy_policy_still_wins(self):
        policy = SamplePolicy(max_samples=2000, seed=99)
        eff, seed = resolve_sample_cap(self._def(policy), cli_max_samples=9000)
        assert eff == 2000
        assert seed == 99


class TestStreamAudioDatasetHonoursPolicy:
    """End-to-end determinism through ``stream_audio_dataset`` itself,
    against a fake parquet corpus (no network)."""

    def _fake_corpus(self, monkeypatch, n=50):
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

    def _source(self):
        return SimpleNamespace(hf_id="org/corpus", subset=None, split="test")

    def test_same_seed_same_subset_across_two_runs(self, monkeypatch):
        from runner.audio_io import stream_audio_dataset

        self._fake_corpus(monkeypatch, n=50)
        src = self._source()
        a = [sid for sid, _ in stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"}, id_key="id",
            revision="main", max_samples=10, seed=1337)]
        b = [sid for sid, _ in stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"}, id_key="id",
            revision="main", max_samples=10, seed=1337)]
        assert a == b
        assert len(a) == 10

    def test_same_subset_for_two_different_fighters(self, monkeypatch):
        # "fighters" here just means two independent call sites drawing from
        # the same dataset def with the same registry seed.
        from runner.audio_io import stream_audio_dataset

        self._fake_corpus(monkeypatch, n=50)
        src = self._source()
        fighter_a = {sid for sid, _ in stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"}, id_key="id",
            revision="main", max_samples=10, seed=7)}
        fighter_b = {sid for sid, _ in stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"}, id_key="id",
            revision="main", max_samples=10, seed=7)}
        assert fighter_a == fighter_b

    def test_no_seed_falls_back_to_plain_head_truncation(self, monkeypatch):
        from runner.audio_io import stream_audio_dataset

        self._fake_corpus(monkeypatch, n=50)
        src = self._source()
        rows = list(stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"}, id_key="id",
            revision="main", max_samples=5, seed=None))
        ids = [sample["id"] for _, sample in rows]
        assert ids == [0, 1, 2, 3, 4]

    def test_cli_zero_defers_to_policy_cap(self, monkeypatch):
        # simulates STTBench.iter_samples: resolve_sample_cap(dataset_def, 0)
        # then stream_audio_dataset(..., max_samples=eff, seed=seed)
        from registry.schemas import SamplePolicy
        from runner.audio_io import resolve_sample_cap, stream_audio_dataset

        self._fake_corpus(monkeypatch, n=50)
        src = self._source()
        dataset_def = SimpleNamespace(
            sample_policy=SamplePolicy(max_samples=8, seed=5))
        eff, seed = resolve_sample_cap(dataset_def, cli_max_samples=0)
        rows = list(stream_audio_dataset(
            src, audio_key="audio", extra_keys={"id": "id"}, id_key="id",
            revision="main", max_samples=eff, seed=seed))
        assert len(rows) == 8
