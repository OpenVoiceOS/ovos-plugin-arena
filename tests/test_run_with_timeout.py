"""Regression test for run_with_timeout's executor lifecycle (runner/publish_sample_set.py).

The original implementation spun up a brand-new single-worker
ThreadPoolExecutor on EVERY call — success or timeout — and abandoned it
via ``executor.shutdown(wait=False)``. That per-call churn is what the fix
removes: a single persistent executor is now reused across successful
calls, only replaced when a call actually times out.

The per-timeout abandoned worker itself is UNCHANGED by this rework — a
function that blocks forever still leaves a live thread behind whether or
not the executor around it is freshly created or reused, since Python
cannot safely kill a thread stuck in a C-level read. That worker is
self-healed by the process-wide ``socket.setdefaulttimeout()`` set in
``runner/autorun.py``'s ``main()``: it bounds the blocked read, which then
raises and lets the thread exit on its own.
"""
import concurrent.futures

import pytest

from runner import publish_sample_set as pss


def _shutdown_and_clear():
    if pss._executor is not None:
        pss._executor.shutdown(wait=False)
    pss._executor = None
    pss._abandoned_executors = 0


@pytest.fixture(autouse=True)
def _reset_executor():
    _shutdown_and_clear()
    yield
    _shutdown_and_clear()


def test_timeout_abandons_and_replaces_the_executor():
    first = pss.run_with_timeout(lambda: 1, timeout_secs=5)
    assert first == 1
    executor_before = pss._executor
    assert executor_before is not None

    with pytest.raises(concurrent.futures.TimeoutError):
        pss.run_with_timeout(lambda: __import__("time").sleep(5), timeout_secs=0.05)

    assert pss._executor is not executor_before
    assert pss._executor is None
    assert pss._abandoned_executors == 1


def test_successful_calls_reuse_the_same_executor_object():
    pss.run_with_timeout(lambda: 1, timeout_secs=5)
    executor_after_first = pss._executor
    assert executor_after_first is not None

    pss.run_with_timeout(lambda: 2, timeout_secs=5)
    assert pss._executor is executor_after_first
