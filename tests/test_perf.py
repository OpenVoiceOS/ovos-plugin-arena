"""Unit tests for runner.perf — the per-row performance capture helper."""
from __future__ import annotations

import time

from runner.perf import hw_fingerprint, measure_call


class TestMeasureCall:
    def test_elapsed_positive_on_dummy_call(self):
        result, elapsed_ms, _peak_rss_mb = measure_call(lambda: time.sleep(0.01))
        assert result is None
        assert elapsed_ms > 0

    def test_returns_call_result(self):
        result, _elapsed_ms, _peak_rss_mb = measure_call(lambda: 42)
        assert result == 42

    def test_propagates_exceptions(self):
        def _boom():
            raise ValueError("boom")

        try:
            measure_call(_boom)
        except ValueError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("measure_call swallowed the exception")

    def test_peak_rss_is_none_or_positive(self):
        # psutil may or may not be installed in the test environment; either
        # way the contract holds: None, or a positive MB figure.
        _result, _elapsed_ms, peak_rss_mb = measure_call(lambda: sum(range(1000)))
        assert peak_rss_mb is None or peak_rss_mb > 0


class TestHwFingerprint:
    def test_shape(self):
        fp = hw_fingerprint()
        assert set(fp) == {"host_class", "cpu_model", "threads", "accelerator", "hostname"}
        assert fp["host_class"] in ("cpu-x86", "arm64", "gpu")
        assert isinstance(fp["cpu_model"], str) and fp["cpu_model"]
        assert isinstance(fp["threads"], int) and fp["threads"] >= 0
        assert fp["accelerator"] is None or isinstance(fp["accelerator"], str)
        assert isinstance(fp["hostname"], str)

    def test_cached_per_process(self):
        # Same dict object every call — captured once, not per-row.
        assert hw_fingerprint() is hw_fingerprint()
