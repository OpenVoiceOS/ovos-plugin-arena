"""Unit tests for resumable sample-set publication (runner/publish_sample_set.py):
--skip-existing (present/absent/mismatched policy) and the bounded
per-dataset timeout that keeps one wedged corpus from taking the whole
run down (production incident: a hung per-shard HTTP request against
dominguesm/mTEDx-ptbr, then facebook/voxpopuli en, each silenced the log
for 25+ minutes and a restart redid every previously-published manifest).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import concurrent.futures
import pytest

from registry.schemas import SamplePolicy


def _dataset_def(dataset_id="ds", lang="en-US", max_samples=10, seed=5):
    return SimpleNamespace(
        dataset_id=dataset_id,
        lang=lang,
        modality=SimpleNamespace(value="stt"),
        reference_fields={"audio": "audio"},
        source=SimpleNamespace(hf_id="org/corpus", subset=None, split="test",
                               revision="main"),
        sample_policy=SamplePolicy(max_samples=max_samples, seed=seed),
        predictions_hf=None,
    )


class TestSampleSetIsCurrent:
    def test_matching_policy_is_current(self):
        from runner.publish_sample_set import sample_set_is_current

        existing = {"seed": 5, "max_samples": 10, "total_rows": 50,
                    "sample_ids": ["a", "b"]}
        assert sample_set_is_current(existing, _dataset_def(max_samples=10, seed=5))

    def test_different_seed_is_not_current(self):
        from runner.publish_sample_set import sample_set_is_current

        existing = {"seed": 999, "max_samples": 10}
        assert not sample_set_is_current(existing, _dataset_def(max_samples=10, seed=5))

    def test_different_max_samples_is_not_current(self):
        from runner.publish_sample_set import sample_set_is_current

        existing = {"seed": 5, "max_samples": 2}
        assert not sample_set_is_current(existing, _dataset_def(max_samples=10, seed=5))

    def test_missing_fields_falls_back_to_existence_only(self):
        from runner.publish_sample_set import sample_set_is_current

        existing = {"sample_ids": ["a"]}  # legacy/foreign manifest shape
        assert sample_set_is_current(existing, _dataset_def())

    def test_no_policy_never_current(self):
        from runner.publish_sample_set import sample_set_is_current

        dataset_def = _dataset_def()
        dataset_def.sample_policy = None
        assert not sample_set_is_current({"seed": 5, "max_samples": 10}, dataset_def)


class TestExistingSampleSet:
    def test_absent_manifest_returns_none(self, monkeypatch):
        from runner.publish_sample_set import existing_sample_set

        def boom(*a, **k):
            raise FileNotFoundError("404")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
        assert existing_sample_set(_dataset_def(), "OpenVoiceOS") is None

    def test_present_manifest_is_parsed(self, monkeypatch, tmp_path):
        import json

        from runner.publish_sample_set import existing_sample_set

        manifest_path = tmp_path / "en_US.json"
        manifest_path.write_text(json.dumps({"seed": 5, "max_samples": 10}))
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            lambda *a, **k: str(manifest_path))
        result = existing_sample_set(_dataset_def(), "OpenVoiceOS")
        assert result == {"seed": 5, "max_samples": 10}


class TestMainSkipExisting:
    """End-to-end: main() actually skips a dataset whose manifest already
    matches the current policy, and does NOT skip one that's absent or
    whose policy has drifted."""

    def _patch_common(self, monkeypatch, dataset_def):
        monkeypatch.setattr("registry.loaders.list_datasets",
                            lambda modality=None: [dataset_def])

    def test_skips_when_manifest_matches_policy(self, monkeypatch):
        from runner import publish_sample_set as pss

        dataset_def = _dataset_def()
        self._patch_common(monkeypatch, dataset_def)
        monkeypatch.setattr(pss, "existing_sample_set",
                            lambda dd, owner: {"seed": 5, "max_samples": 10})

        called = []
        monkeypatch.setattr(pss, "run_with_timeout",
                            lambda fn, timeout_secs: called.append(1))

        rc = pss.main(["--skip-existing", "--dry-run"])
        assert rc == 0
        assert called == [], "a matching manifest must not trigger recomputation"

    def test_recomputes_when_manifest_absent(self, monkeypatch):
        from runner import publish_sample_set as pss

        dataset_def = _dataset_def()
        self._patch_common(monkeypatch, dataset_def)
        monkeypatch.setattr(pss, "existing_sample_set", lambda dd, owner: None)

        called = []
        monkeypatch.setattr(pss, "run_with_timeout",
                            lambda fn, timeout_secs: called.append(1))

        rc = pss.main(["--skip-existing", "--dry-run"])
        assert rc == 0
        assert called == [1]

    def test_recomputes_when_policy_mismatched(self, monkeypatch):
        from runner import publish_sample_set as pss

        dataset_def = _dataset_def(max_samples=10, seed=5)
        self._patch_common(monkeypatch, dataset_def)
        monkeypatch.setattr(pss, "existing_sample_set",
                            lambda dd, owner: {"seed": 999, "max_samples": 10})

        called = []
        monkeypatch.setattr(pss, "run_with_timeout",
                            lambda fn, timeout_secs: called.append(1))

        rc = pss.main(["--skip-existing", "--dry-run"])
        assert rc == 0
        assert called == [1]

    def test_without_skip_existing_flag_always_recomputes(self, monkeypatch):
        from runner import publish_sample_set as pss

        dataset_def = _dataset_def()
        self._patch_common(monkeypatch, dataset_def)
        existing_calls = []
        monkeypatch.setattr(pss, "existing_sample_set",
                            lambda dd, owner: existing_calls.append(1))

        called = []
        monkeypatch.setattr(pss, "run_with_timeout",
                            lambda fn, timeout_secs: called.append(1))

        rc = pss.main(["--dry-run"])
        assert rc == 0
        assert called == [1]
        assert existing_calls == [], "skip-existing check must be opt-in"


class TestRunWithTimeout:
    def test_fast_call_returns_normally(self):
        from runner.publish_sample_set import run_with_timeout

        assert run_with_timeout(lambda: 42, timeout_secs=5) == 42

    def test_slow_call_raises_timeout_error(self):
        from runner.publish_sample_set import run_with_timeout

        def _hang():
            time.sleep(5)
            return "too late"

        with pytest.raises(concurrent.futures.TimeoutError):
            run_with_timeout(_hang, timeout_secs=0.05)

    def test_exception_inside_fn_propagates(self):
        from runner.publish_sample_set import run_with_timeout

        def _boom():
            raise ValueError("bad corpus")

        with pytest.raises(ValueError):
            run_with_timeout(_boom, timeout_secs=5)


class TestMainTimeoutPath:
    """A dataset whose work hangs past --timeout-secs is logged and
    skipped; the run continues to the next dataset and the failure is
    listed at the end instead of the whole process wedging."""

    def test_hung_dataset_is_skipped_and_reported(self, monkeypatch, caplog):
        import logging

        from runner import publish_sample_set as pss

        hung = _dataset_def(dataset_id="hung-corpus")
        fine = _dataset_def(dataset_id="fine-corpus")
        monkeypatch.setattr("registry.loaders.list_datasets",
                            lambda modality=None: [hung, fine])

        calls = []

        def fake_run_with_timeout(fn, timeout_secs):
            # simulate: the FIRST dataset's work (hung-corpus, visited
            # first — main() walks targets in list order) would block
            # forever; the second (fine-corpus) returns immediately.
            # run_with_timeout itself is tested standalone above; this
            # isolates main()'s handling of a timeout it raises.
            calls.append(fn)
            if len(calls) == 1:
                raise concurrent.futures.TimeoutError()
            return {"ok": True}

        monkeypatch.setattr(pss, "run_with_timeout", fake_run_with_timeout)
        monkeypatch.setattr(pss, "_publish_one",
                            lambda dd, owner, dry_run, request_timeout: dd.dataset_id)
        # main() forces os._exit(1) when a timeout occurred, to avoid
        # blocking on a leaked hung worker thread at interpreter exit — stub
        # that out so the test process itself doesn't get force-exited.
        exit_codes = []
        monkeypatch.setattr("os._exit", lambda code: exit_codes.append(code))

        with caplog.at_level(logging.ERROR):
            pss.main(["--dry-run", "--timeout-secs", "1"])

        assert exit_codes == [1]
        messages = "\n".join(r.message for r in caplog.records)
        assert "hung-corpus" in messages
        assert "timed out" in messages.lower()

    def test_no_timeouts_returns_zero_without_force_exit(self, monkeypatch):
        from runner import publish_sample_set as pss

        dataset_def = _dataset_def()
        monkeypatch.setattr("registry.loaders.list_datasets",
                            lambda modality=None: [dataset_def])
        monkeypatch.setattr(pss, "run_with_timeout",
                            lambda fn, timeout_secs: {"ok": True})

        exit_codes = []
        monkeypatch.setattr("os._exit", lambda code: exit_codes.append(code))

        rc = pss.main(["--dry-run"])
        assert rc == 0
        assert exit_codes == []
