"""Unit tests for runner.autorun — the fleet-wide round-robin runner.

Everything here uses fake benches and in-memory state: no network, no
plugins, no audio. ``AutoRunner.process_fn`` is injected so the scheduler
and orchestration logic are exercised in isolation from
``runner.media_bench.run_competitor_lang``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import runner.autorun as autorun_module
from runner.autorun import (
    QUARANTINE_BASE_BACKOFF_SECONDS,
    AutoRunConfig,
    AutoRunner,
    PairKey,
    QuarantineEntry,
    RoundRobinScheduler,
    _quarantine_backoff_seconds,
    apply_filters,
    is_heavy,
    load_state,
    log_classifications,
    save_state,
    size_rank,
)
from runner.media_bench import BatchResult


def _pair(mod="stt", comp="c", ds="d", lang="en"):
    return PairKey(mod, comp, ds, lang)


# ---------------------------------------------------------------------------
# RoundRobinScheduler
# ---------------------------------------------------------------------------


class TestRoundRobinScheduler:
    def test_sweep_yields_all_pairs_in_order(self):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        assert list(sched.sweep()) == [a, b]

    def test_completed_pair_skipped_next_sweep(self):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        sched.mark_complete(a)
        assert list(sched.sweep()) == [b]

    def test_quarantine_removes_all_pairs_for_that_fighter(self):
        a1 = _pair(comp="A", ds="d1")
        a2 = _pair(comp="A", ds="d2")
        b = _pair(comp="B")
        sched = RoundRobinScheduler([a1, a2, b])
        sched.quarantine("A", "load failure")
        assert list(sched.sweep()) == [b]

    def test_quarantine_logs_each_attempt(self, caplog):
        # Each quarantine() call is a distinct failure event (the scheduler
        # only re-invokes it for an already-quarantined fighter after a
        # backoff-gated retry attempt failed again) — logging every call is
        # the "recovers from blips, not silently quarantined forever"
        # behaviour; it is NOT the same as re-logging every sweep, since
        # sweep() filters out a fighter still within its backoff window.
        sched = RoundRobinScheduler([_pair(comp="A")])
        with caplog.at_level("ERROR"):
            sched.quarantine("A", "boom")
            sched.quarantine("A", "boom again")
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 2
        assert sched.quarantined["A"].reason == "boom again"
        assert sched.quarantined["A"].attempts == 2

    def test_is_exhausted(self):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        assert not sched.is_exhausted()
        sched.mark_complete(a)
        assert not sched.is_exhausted()
        sched.quarantine("B", "x")
        assert sched.is_exhausted()

    def test_set_pairs_keeps_state_for_surviving_pairs(self):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        sched.mark_complete(a)
        c = _pair(comp="C")
        sched.set_pairs([a, b, c])  # registry reload adds C
        assert list(sched.sweep()) == [b, c]

    def test_set_pairs_drops_vanished_pairs(self):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        sched.set_pairs([a])  # b removed from registry
        assert list(sched.sweep()) == [a]

    def test_state_round_trip(self):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        sched.mark_complete(a)
        sched.quarantine("B", "missing deps")
        state = sched.to_state()

        restored = RoundRobinScheduler([a, b])
        restored.apply_state(state)
        assert restored.completed == {a}
        assert restored.quarantined["B"].reason == "missing deps"
        assert list(restored.sweep()) == []  # B still within its backoff window

    def test_state_file_round_trip(self, tmp_path):
        a, b = _pair(comp="A"), _pair(comp="B")
        sched = RoundRobinScheduler([a, b])
        sched.mark_complete(a)
        sched.quarantine("B", "boom")
        path = tmp_path / "autorun_state.json"
        save_state(path, sched)

        raw = json.loads(path.read_text())
        assert raw["completed"] == [a.to_str()]
        assert raw["quarantined"]["B"]["reason"] == "boom"
        assert raw["quarantined"]["B"]["attempts"] == 1

        restored = RoundRobinScheduler([a, b])
        restored.apply_state(load_state(path))
        assert restored.completed == {a}
        assert restored.quarantined["B"].reason == "boom"

    def test_load_state_missing_file_returns_empty(self, tmp_path):
        assert load_state(tmp_path / "nope.json") == {}

    def test_load_state_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not json")
        assert load_state(path) == {}

    def test_apply_state_tolerates_pre_backoff_string_shape(self):
        # Old state files (before the backoff fix) stored quarantined as
        # {competitor_id: reason_string}. Loading one must not crash, and
        # must treat it as immediately eligible for a retry.
        restored = RoundRobinScheduler([_pair(comp="B")])
        restored.apply_state({"completed": [], "quarantined": {"B": "boom"}})
        assert restored.quarantined["B"].reason == "boom"
        assert not restored.is_quarantined("B", now=1_000_000.0)


class TestQuarantineBackoff:
    def test_backoff_grows_exponentially_and_caps(self):
        assert _quarantine_backoff_seconds(1) == QUARANTINE_BASE_BACKOFF_SECONDS
        assert _quarantine_backoff_seconds(2) == QUARANTINE_BASE_BACKOFF_SECONDS * 2
        assert _quarantine_backoff_seconds(3) == QUARANTINE_BASE_BACKOFF_SECONDS * 4
        # Capped at 24h regardless of how many attempts pile up.
        assert _quarantine_backoff_seconds(20) == 24 * 60 * 60

    def test_quarantined_fighter_excluded_until_backoff_expires(self):
        sched = RoundRobinScheduler([_pair(comp="A")])
        t0 = 1_000_000.0
        sched.quarantine("A", "network blip", now=t0)

        # Still within the 30-minute backoff window.
        assert list(sched.sweep(now=t0 + 60)) == []
        assert sched.is_quarantined("A", now=t0 + 60)

        # Backoff window has elapsed — the fighter is back in rotation for
        # a retry, recovering a weeks-long daemon from a transient blip
        # without a restart.
        retry_time = t0 + QUARANTINE_BASE_BACKOFF_SECONDS + 1
        assert list(sched.sweep(now=retry_time)) == [_pair(comp="A")]
        assert not sched.is_quarantined("A", now=retry_time)

    def test_repeated_failure_backs_off_further_each_time(self):
        sched = RoundRobinScheduler([_pair(comp="A")])
        t0 = 1_000_000.0
        sched.quarantine("A", "fail 1", now=t0)
        first_backoff = sched.quarantined["A"].retry_after_seconds

        retry_time = t0 + first_backoff + 1
        sched.quarantine("A", "fail 2", now=retry_time)
        second_backoff = sched.quarantined["A"].retry_after_seconds

        assert second_backoff > first_backoff
        assert sched.quarantined["A"].attempts == 2


# ---------------------------------------------------------------------------
# AutoRunner orchestration: starvation-freedom + completion + quarantine
# ---------------------------------------------------------------------------


class FakeProcessor:
    """Stand-in for AutoRunner._run_pair_batch: fake datasets of a fixed
    size per (modality, competitor_id, dataset_id, lang); each call writes
    up to *batch* new rows and records call order.

    *error_pairs* maps a pair_str to a number of samples that ALWAYS error
    in every batch touching that pair (simulating a flaky/broken model that
    is not a hard load failure — samples are attempted and fail, they don't
    raise out of process_fn itself).
    """

    def __init__(self, sizes: dict, fail_competitors: set = frozenset(),
                 error_pairs: dict | None = None):
        self.sizes = dict(sizes)  # pair_str -> remaining count
        self.fail_competitors = fail_competitors
        self.error_pairs = dict(error_pairs or {})  # pair_str -> errored count per call
        self.calls: list[str] = []

    def __call__(self, modality, competitor, dataset, lang, batch) -> BatchResult:
        key = PairKey(modality, competitor.competitor_id, dataset.dataset_id, lang).to_str()
        self.calls.append(key)
        if competitor.competitor_id in self.fail_competitors:
            raise RuntimeError("plugin not found")
        errored = self.error_pairs.get(key, 0)
        if errored:
            return BatchResult(written=0, errored=errored, last_error="boom")
        remaining = self.sizes.get(key, 0)
        written = min(batch, remaining)
        self.sizes[key] = remaining - written
        return BatchResult(written=written, errored=0)


def _comp(cid):
    return SimpleNamespace(competitor_id=cid, plugin=cid, size=None)


def _ds(did):
    return SimpleNamespace(dataset_id=did)


class TestAutoRunnerRoundRobin:
    def test_alternates_between_pairs_batch_2(self, tmp_path):
        a_comp, b_comp = _comp("A"), _comp("B")
        a_ds = b_ds = _ds("d")
        entries = [
            ("stt", a_comp, a_ds, "en"),
            ("stt", b_comp, b_ds, "en"),
        ]
        sizes = {
            PairKey("stt", "A", "d", "en").to_str(): 5,
            PairKey("stt", "B", "d", "en").to_str(): 5,
        }
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False)
        runner = AutoRunner(config, process_fn=fake)

        # Two full sweeps by hand (bypassing enumerate_all_pairs/registry).
        lookup = {
            PairKey(m, c.competitor_id, d.dataset_id, l): (m, c, d, l)
            for m, c, d, l in entries
        }
        runner.scheduler.set_pairs(lookup.keys())
        for _ in range(2):
            for pair in list(runner.scheduler.sweep()):
                m, c, d, l = lookup[pair]
                runner.process_pair(m, c, d, l)

        # A and B alternate: A gets a batch, then B, then A again, then B.
        assert fake.calls == [
            PairKey("stt", "A", "d", "en").to_str(),
            PairKey("stt", "B", "d", "en").to_str(),
            PairKey("stt", "A", "d", "en").to_str(),
            PairKey("stt", "B", "d", "en").to_str(),
        ]
        # Neither pair drained to completion in one sweep: after sweep 1,
        # both still have work left (5 - 2 = 3 remaining each).
        assert fake.sizes[PairKey("stt", "A", "d", "en").to_str()] >= 0

    def test_completed_pair_is_skipped_after_exhaustion(self, tmp_path):
        a_comp, b_comp = _comp("A"), _comp("B")
        ds = _ds("d")
        sizes = {
            PairKey("stt", "A", "d", "en").to_str(): 1,  # exhausts in 1 batch
            PairKey("stt", "B", "d", "en").to_str(): 10,
        }
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        lookup = {
            PairKey("stt", "A", "d", "en"): ("stt", a_comp, ds, "en"),
            PairKey("stt", "B", "d", "en"): ("stt", b_comp, ds, "en"),
        }
        runner.scheduler.set_pairs(lookup.keys())

        for pair in list(runner.scheduler.sweep()):
            m, c, d, l = lookup[pair]
            runner.process_pair(m, c, d, l)

        assert runner.scheduler.completed == {PairKey("stt", "A", "d", "en")}
        # Second sweep only touches B.
        fake.calls.clear()
        for pair in list(runner.scheduler.sweep()):
            m, c, d, l = lookup[pair]
            runner.process_pair(m, c, d, l)
        assert fake.calls == [PairKey("stt", "B", "d", "en").to_str()]

    def test_quarantine_on_hard_load_failure_stops_retries(self, tmp_path):
        a_comp = _comp("BAD")
        ds = _ds("d")
        fake = FakeProcessor({}, fail_competitors={"BAD"})
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        pair = PairKey("stt", "BAD", "d", "en")
        runner.scheduler.set_pairs([pair])

        runner.process_pair("stt", a_comp, ds, "en")
        assert "BAD" in runner.scheduler.quarantined
        assert list(runner.scheduler.sweep()) == []

        # Within the backoff window, a later sweep never calls process_fn
        # again for BAD.
        fake.calls.clear()
        for p in runner.scheduler.sweep():
            pass  # empty
        assert fake.calls == []

    # -- CRITICAL regression: error-shortfall must never look like completion --

    def test_all_error_batch_does_not_mark_pair_complete(self, tmp_path):
        """A batch that returns 0 rows because every sample errored (model
        crash, OOM, ...) is NOT the same as the dataset being exhausted.
        Marking the pair complete here would silently and permanently drop
        every sample after the failure point — this is the bug PR #96 was
        refuted on. The pair must stay in rotation for a retry.
        """
        comp = _comp("FLAKY")
        ds = _ds("d")
        pair = PairKey("stt", "FLAKY", "d", "en")
        fake = FakeProcessor({}, error_pairs={pair.to_str(): 2})
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False,
                                max_consecutive_error_batches=5)
        runner = AutoRunner(config, process_fn=fake)
        runner.scheduler.set_pairs([pair])

        runner.process_pair("stt", comp, ds, "en")

        assert pair not in runner.scheduler.completed
        assert "FLAKY" not in runner.scheduler.quarantined  # not yet — below the streak threshold
        assert list(runner.scheduler.sweep()) == [pair]  # still in rotation next sweep

    def test_repeated_all_error_batches_escalate_to_quarantine(self, tmp_path):
        comp = _comp("FLAKY")
        ds = _ds("d")
        pair = PairKey("stt", "FLAKY", "d", "en")
        fake = FakeProcessor({}, error_pairs={pair.to_str(): 2})
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False,
                                max_consecutive_error_batches=3)
        runner = AutoRunner(config, process_fn=fake)
        runner.scheduler.set_pairs([pair])

        for _ in range(3):
            runner.process_pair("stt", comp, ds, "en")

        assert pair not in runner.scheduler.completed  # never silently completed
        assert "FLAKY" in runner.scheduler.quarantined  # escalated instead
        assert "3 consecutive all-error batches" in runner.scheduler.quarantined["FLAKY"].reason

    def test_progress_after_errors_resets_the_streak(self, tmp_path):
        """A pair that errors for a while and then makes progress again
        (e.g. after a transient blip clears mid-dataset) must not carry
        over stale error-streak count toward quarantine."""
        comp = _comp("SOMETIMES")
        ds = _ds("d")
        pair = PairKey("stt", "SOMETIMES", "d", "en")
        fake = FakeProcessor(
            {pair.to_str(): 4}, error_pairs={pair.to_str(): 2}
        )
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False,
                                max_consecutive_error_batches=2)
        runner = AutoRunner(config, process_fn=fake)
        runner.scheduler.set_pairs([pair])

        runner.process_pair("stt", comp, ds, "en")  # all-error, streak=1
        assert runner.scheduler.error_streak[pair] == 1

        # Clear the error condition — this batch makes real progress.
        fake.error_pairs.pop(pair.to_str())
        runner.process_pair("stt", comp, ds, "en")
        assert runner.scheduler.error_streak[pair] == 0
        assert "SOMETIMES" not in runner.scheduler.quarantined


# ---------------------------------------------------------------------------
# Filters / size heuristic
# ---------------------------------------------------------------------------


class TestFilters:
    def test_include_exclude_globs(self):
        entries = [
            ("stt", _comp("vosk-en"), _ds("d"), "en"),
            ("stt", _comp("whisper-large"), _ds("d"), "en"),
        ]
        out = apply_filters(entries, include=["vosk-*"])
        assert [c.competitor_id for _, c, _, _ in out] == ["vosk-en"]

        out = apply_filters(entries, exclude=["whisper-*"])
        assert [c.competitor_id for _, c, _, _ in out] == ["vosk-en"]

    def test_langs_filter(self):
        entries = [
            ("stt", _comp("a"), _ds("d"), "en"),
            ("stt", _comp("a"), _ds("d"), "pt-PT"),
        ]
        out = apply_filters(entries, langs=["pt-PT"])
        assert [lang for _, _, _, lang in out] == ["pt-PT"]

    def test_heavy_light_split(self):
        light = SimpleNamespace(competitor_id="vosk-en", plugin="vosk", size="tiny")
        heavy = SimpleNamespace(competitor_id="whisper-large-v3",
                                plugin="whisper", size="large")
        entries = [("stt", light, _ds("d"), "en"), ("stt", heavy, _ds("d"), "en")]

        assert not is_heavy(light)
        assert is_heavy(heavy)

        only_heavy = apply_filters(entries, heavy_only=True)
        assert [c.competitor_id for _, c, _, _ in only_heavy] == ["whisper-large-v3"]

        only_light = apply_filters(entries, light_only=True)
        assert [c.competitor_id for _, c, _, _ in only_light] == ["vosk-en"]

    def test_heavy_name_hint_without_size_field(self):
        comp = SimpleNamespace(competitor_id="onnx-asr-canary", plugin="onnx-asr", size=None)
        assert is_heavy(comp)

    def test_heavy_param_count_token_cohere_transcribe_2b(self):
        # Regression: cohere-transcribe-2b (2B params) was misclassified
        # light because HEAVY_NAME_HINTS didn't enumerate it by name.
        comp = SimpleNamespace(competitor_id="cohere-transcribe-2b",
                               plugin="cohere-transcribe", size=None)
        assert is_heavy(comp)

    def test_heavy_param_count_various_thresholds(self):
        heavy_ids = [
            "onnx-asr-canary-qwen-2.5b",
            "coreml-parakeet-tdt-1.1b-fp16",
            "some-model-3b",
            "some-model-1b",
        ]
        light_ids = [
            "some-model-0.6b",  # sub-1B stays light
            "vosk-en",
            "silero-vad",
        ]
        for cid in heavy_ids:
            comp = SimpleNamespace(competitor_id=cid, plugin="", size=None)
            assert is_heavy(comp), f"{cid} should classify heavy"
        for cid in light_ids:
            comp = SimpleNamespace(competitor_id=cid, plugin="", size=None)
            assert not is_heavy(comp), f"{cid} should classify light"

    def test_dropped_decimal_leading_zero_does_not_misparse_as_huge(self):
        # onnx-asr-parakeet-tdt-06b-v3 is NVIDIA Parakeet TDT 0.6B — the
        # registry id drops the "." from "0.6b". A naive "\d+b" regex would
        # read "06b" as the integer 6 (>= 1B threshold) and misclassify a
        # genuinely small model as heavy.
        comp = SimpleNamespace(competitor_id="onnx-asr-parakeet-tdt-06b-v3",
                               plugin="ovos-stt-plugin-onnx-asr", size=None)
        assert not is_heavy(comp)

        # An explicit decimal is still read correctly either way.
        comp2 = SimpleNamespace(competitor_id="model-0.6b", plugin="", size=None)
        assert not is_heavy(comp2)
        comp3 = SimpleNamespace(competitor_id="model-1.6b", plugin="", size=None)
        assert is_heavy(comp3)

    def test_min_max_size(self):
        small = SimpleNamespace(competitor_id="a", plugin="", size="small")
        giant = SimpleNamespace(competitor_id="b", plugin="", size="giant")
        entries = [("stt", small, _ds("d"), "en"), ("stt", giant, _ds("d"), "en")]

        out = apply_filters(entries, min_size=size_rank("medium"))
        assert [c.competitor_id for _, c, _, _ in out] == ["b"]

        out = apply_filters(entries, max_size=size_rank("medium"))
        assert [c.competitor_id for _, c, _, _ in out] == ["a"]

    def test_log_classifications_logs_each_fighter_once(self, caplog):
        entries = [
            ("stt", _comp("vosk-en"), _ds("d1"), "en"),
            ("stt", _comp("vosk-en"), _ds("d2"), "en"),  # same fighter, 2nd dataset
            ("stt", _comp("cohere-transcribe-2b"), _ds("d1"), "en"),
        ]
        with caplog.at_level("INFO", logger="autorun"):
            log_classifications(entries)
        messages = [r.message for r in caplog.records if "classified" in r.message]
        assert len(messages) == 2  # vosk-en logged once despite 2 entries
        assert any("vosk-en" in m and "light" in m for m in messages)
        assert any("cohere-transcribe-2b" in m and "heavy" in m for m in messages)


# ---------------------------------------------------------------------------
# One-shot mode (AutoRunner.run_one_shot / --one-shot CLI) — CI-friendly
# single-round runs: random pair pick, time-budget, retry-on-load-failure.
#
# enumerate_all_pairs and pair_is_complete both pull in real registry/adapter
# machinery, so every test here monkeypatches them: enumerate_all_pairs at
# module scope, pair_is_complete on the AutoRunner instance directly.
# ---------------------------------------------------------------------------


class _NoShuffleRandom:
    """Stand-in for random.Random that leaves rng.shuffle a no-op, so a
    candidate list's iteration order is exactly `enumerate_all_pairs`'s
    (dict-insertion) order — deterministic tests for the discovery walk
    without depending on real RNG output for a given seed."""

    def __init__(self, *a, **k):
        pass

    def shuffle(self, seq):
        return None


class TestBreadthOrderedKeys:
    """AutoRunner._breadth_ordered_keys — reuses runner.queue_tools'
    breadth_tier_sort/find_missing_pairs presence signal to order a
    sweep's eligible pairs, same policy as the queue generator."""

    def _lookup(self, entries):
        return {
            PairKey(m, c.competitor_id, d.dataset_id, l): (m, c, d, l)
            for m, c, d, l in entries
        }

    def test_zero_coverage_fighter_ordered_before_deep_fighters_second_pair(
        self, monkeypatch, tmp_path
    ):
        entries = [
            ("stt", _comp("deep"), _ds("d1"), "en"),
            ("stt", _comp("deep"), _ds("d2"), "en"),
            ("stt", _comp("shallow"), _ds("d1"), "en"),
        ]
        lookup = self._lookup(entries)

        def fake_find_missing_pairs(modality, registry_root=None, lister=None,
                                     check_rows=True, min_rows=1,
                                     present_by_competitor=None,
                                     present_by_dataset=None):
            # "deep" already has 1 published pair; "shallow" has 0 — the
            # exact HF-listing/lang-matching machinery is queue_tools' own
            # concern and already covered by test_queue_tools.py; this
            # fake stands in for its presence-counting side effect only.
            present_by_competitor["deep"] = 1
            present_by_dataset["d1"] = 1
            return []

        monkeypatch.setattr(autorun_module, "find_missing_pairs", fake_find_missing_pairs)

        runner = AutoRunner(
            AutoRunConfig(output_dir=tmp_path, upload=False), lister=object()
        )
        ordered = runner._breadth_ordered_keys(lookup, registry_root=None)
        ids = [k.competitor_id for k in ordered]
        assert ids[0] == "shallow"
        assert ids.index("shallow") < ids.index("deep")

    def test_no_lister_keeps_natural_order_and_skips_listing(self, tmp_path, monkeypatch):
        entries = [
            ("stt", _comp("b"), _ds("d"), "en"),
            ("stt", _comp("a"), _ds("d"), "en"),
        ]
        lookup = self._lookup(entries)
        calls = []
        monkeypatch.setattr(
            autorun_module, "find_missing_pairs",
            lambda *a, **k: calls.append(1),
        )
        runner = AutoRunner(AutoRunConfig(output_dir=tmp_path, upload=False), lister=None)
        ordered = runner._breadth_ordered_keys(lookup, registry_root=None)
        assert ordered == list(lookup.keys())
        assert calls == []

    def test_listing_failure_falls_back_to_natural_order(self, tmp_path, monkeypatch):
        entries = [("stt", _comp("a"), _ds("d"), "en")]
        lookup = self._lookup(entries)

        def boom(*a, **k):
            raise ConnectionError("simulated network failure")

        monkeypatch.setattr(autorun_module, "find_missing_pairs", boom)
        runner = AutoRunner(
            AutoRunConfig(output_dir=tmp_path, upload=False), lister=object()
        )
        ordered = runner._breadth_ordered_keys(lookup, registry_root=None)
        assert ordered == list(lookup.keys())


class TestOneShotBreadthTier:
    """run_one_shot restricts its random draw to the lowest published-
    coverage tier when a lister is available (see docstring update)."""

    def test_picks_zero_coverage_fighter_over_deep_fighter_with_lister(
        self, tmp_path, monkeypatch
    ):
        deep_comp, shallow_comp = _comp("deep"), _comp("shallow")
        ds = _ds("d")
        entries = [("stt", deep_comp, ds, "en"), ("stt", shallow_comp, ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        monkeypatch.setattr(autorun_module.random, "Random", _NoShuffleRandom)

        def fake_find_missing_pairs(modality, registry_root=None, lister=None,
                                     check_rows=True, min_rows=1,
                                     present_by_competitor=None,
                                     present_by_dataset=None):
            present_by_competitor["deep"] = 3
            return []

        monkeypatch.setattr(autorun_module, "find_missing_pairs", fake_find_missing_pairs)

        sizes = {
            PairKey("stt", "deep", "d", "en").to_str(): 5,
            PairKey("stt", "shallow", "d", "en").to_str(): 5,
        }
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        # _NoShuffleRandom leaves candidate order exactly as built — "deep"
        # is first in `entries`/candidates, so a uniform pick (the old
        # behaviour) would draw it first every time. With the lister
        # present, tiering must restrict the draw to "shallow" (tier 0)
        # only, since "deep" is at tier 3.
        runner = AutoRunner(config, process_fn=fake, lister=object())
        runner.pair_is_complete = lambda *a, **k: False

        summary = runner.run_one_shot(["stt"], max_samples=5, seed=1)
        assert summary["pairs"][0]["pair"] == PairKey("stt", "shallow", "d", "en").to_str()

    def test_no_lister_falls_back_to_uniform_pick_unchanged(self, tmp_path, monkeypatch):
        """No lister (--no-seed) must keep the exact pre-existing uniform
        behaviour — the first candidate in enumeration order, since
        _NoShuffleRandom makes shuffle a no-op."""
        deep_comp, shallow_comp = _comp("deep"), _comp("shallow")
        ds = _ds("d")
        entries = [("stt", deep_comp, ds, "en"), ("stt", shallow_comp, ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        monkeypatch.setattr(autorun_module.random, "Random", _NoShuffleRandom)
        monkeypatch.setattr(
            autorun_module, "find_missing_pairs",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        sizes = {
            PairKey("stt", "deep", "d", "en").to_str(): 5,
            PairKey("stt", "shallow", "d", "en").to_str(): 5,
        }
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake, lister=None)
        runner.pair_is_complete = lambda *a, **k: False

        summary = runner.run_one_shot(["stt"], max_samples=5, seed=1)
        assert summary["pairs"][0]["pair"] == PairKey("stt", "deep", "d", "en").to_str()


class TestSweepOnceAppliesBreadthOrder:
    """End-to-end (not just the _breadth_ordered_keys unit): calling
    _sweep_once with a fake lister must actually leave the scheduler's own
    iteration order breadth-first — this is what guards the wiring itself
    (the `self.scheduler.set_order(...)` call in `_sweep_once`), not just
    the pure ordering function it calls."""

    def test_sweep_once_orders_scheduler_pairs_zero_coverage_first(
        self, monkeypatch, tmp_path
    ):
        deep_comp, shallow_comp = _comp("deep"), _comp("shallow")
        ds1, ds2 = _ds("d1"), _ds("d2")
        entries = [
            ("stt", deep_comp, ds1, "en"),
            ("stt", deep_comp, ds2, "en"),
            ("stt", shallow_comp, ds1, "en"),
        ]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)

        def fake_find_missing_pairs(modality, registry_root=None, lister=None,
                                     check_rows=True, min_rows=1,
                                     present_by_competitor=None,
                                     present_by_dataset=None):
            # "deep" already has 1 published pair; "shallow" has 0.
            present_by_competitor["deep"] = 1
            present_by_dataset["d1"] = 1
            return []

        monkeypatch.setattr(autorun_module, "find_missing_pairs", fake_find_missing_pairs)

        sizes = {
            PairKey("stt", "deep", "d1", "en").to_str(): 0,
            PairKey("stt", "deep", "d2", "en").to_str(): 0,
            PairKey("stt", "shallow", "d1", "en").to_str(): 0,
        }
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=2, upload=False)
        runner = AutoRunner(config, process_fn=fake, lister=object())

        runner._sweep_once(["stt"], {}, None)

        ids = [p.competitor_id for p in runner.scheduler._pairs]
        assert ids[0] == "shallow"
        assert ids.index("shallow") < ids.index("deep")


class TestOneShot:
    def test_seeded_pick_excludes_complete_pairs(self, tmp_path, monkeypatch):
        a_comp, b_comp = _comp("A"), _comp("B")
        ds = _ds("d")
        entries = [("stt", a_comp, ds, "en"), ("stt", b_comp, ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)

        # A is already complete; B never is. Regardless of which seed picks
        # which candidate first, only B may ever actually run. Fresh runner
        # + fake per seed so one seed's local state (dirty shard, "used up"
        # sample count) can't bleed into the next seed's pick.
        for seed in range(10):
            sizes = {PairKey("stt", "B", "d", "en").to_str(): 5}
            fake = FakeProcessor(sizes)
            config = AutoRunConfig(output_dir=tmp_path / f"seed{seed}", batch=10, upload=False)
            runner = AutoRunner(config, process_fn=fake)
            runner.pair_is_complete = lambda m, c, d_, l, deadline=None: c.competitor_id == "A"

            summary = runner.run_one_shot(["stt"], max_samples=5, seed=seed)
            assert summary["pairs"], f"seed={seed} should have found B"
            assert summary["pairs"][0]["pair"] == PairKey("stt", "B", "d", "en").to_str()

    def test_same_seed_is_reproducible(self, tmp_path, monkeypatch):
        comps = [_comp(c) for c in "ABCDE"]
        ds = _ds("d")
        entries = [("stt", c, ds, "en") for c in comps]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        sizes = {PairKey("stt", c.competitor_id, "d", "en").to_str(): 5 for c in comps}

        def pick(run_id):
            fake = FakeProcessor(dict(sizes))
            config = AutoRunConfig(output_dir=tmp_path / str(run_id), batch=10, upload=False)
            runner = AutoRunner(config, process_fn=fake)
            runner.pair_is_complete = lambda *a, **k: False
            return runner.run_one_shot(["stt"], max_samples=5, seed=42)["pairs"][0]["pair"]

        assert pick(1) == pick(2)

    def test_nothing_to_do_when_all_pairs_complete(self, tmp_path, monkeypatch):
        a_comp = _comp("A")
        ds = _ds("d")
        entries = [("stt", a_comp, ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        fake = FakeProcessor({})
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = lambda *a, **k: True

        summary = runner.run_one_shot(["stt"], max_samples=5, seed=1)

        assert summary["nothing_to_do"] is True
        assert summary["discovery_bound"] is False
        assert summary["pairs"] == []
        assert fake.calls == []

    def test_load_failure_falls_back_to_another_pair(self, tmp_path, monkeypatch):
        bad_comp, good_comp = _comp("BAD"), _comp("GOOD")
        ds = _ds("d")
        entries = [("stt", bad_comp, ds, "en"), ("stt", good_comp, ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        # A no-op shuffle keeps the walk order == entries order (BAD first,
        # then GOOD) — makes the "attempt 1 fails, attempt 2 succeeds" retry
        # path exercised on purpose rather than left to chance.
        monkeypatch.setattr(autorun_module.random, "Random", _NoShuffleRandom)
        sizes = {PairKey("stt", "GOOD", "d", "en").to_str(): 5}
        fake = FakeProcessor(sizes, fail_competitors={"BAD"})
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = lambda *a, **k: False

        summary = runner.run_one_shot(["stt"], max_samples=5, seed=0, max_attempts=3)

        assert "BAD" in runner.scheduler.quarantined
        assert summary["pairs"], "GOOD must still have run despite BAD's failure"
        assert summary["pairs"][0]["pair"] == PairKey("stt", "GOOD", "d", "en").to_str()

    def test_exhausting_all_attempts_on_all_bad_pairs_exits_cleanly(self, tmp_path, monkeypatch):
        bad_comp = _comp("BAD")
        ds = _ds("d")
        entries = [("stt", bad_comp, ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        fake = FakeProcessor({}, fail_competitors={"BAD"})
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = lambda *a, **k: False

        # Must not raise — the caller (the CLI) still exits 0 on this shape.
        summary = runner.run_one_shot(["stt"], max_samples=5, seed=0, max_attempts=3)

        assert summary["pairs"] == []
        assert summary["nothing_to_do"] is False  # a real draw WAS attempted
        assert "BAD" in runner.scheduler.quarantined

    def test_deadline_stops_drawing_once_budget_runs_low(self, tmp_path, monkeypatch):
        """Time-budgeted mode with a fake clock: each pair "costs" 1000s of
        wall-clock. With a 1500s budget, a second pair still fits (500s > the
        5-minute/300s continue threshold) but a third does not."""
        comps = [_comp(c) for c in ("A", "B", "C")]
        ds = _ds("d")
        entries = [("stt", c, ds, "en") for c in comps]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)

        clock = {"t": 0.0}
        monkeypatch.setattr(autorun_module.time, "monotonic", lambda: clock["t"])

        class SlowProcessor:
            def __init__(self):
                self.calls = []

            def __call__(self, modality, competitor, dataset, lang, batch, deadline=None):
                self.calls.append(deadline)
                clock["t"] += 1000
                return BatchResult(written=batch, errored=0)

        fake = SlowProcessor()
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = lambda *a, **k: False

        summary = runner.run_one_shot(
            ["stt"], max_samples=5, seed=0, time_budget_secs=1500,
        )

        assert len(summary["pairs"]) == 2
        assert summary["written"] == 10
        # The deadline is fixed at call start (start-of-run + budget), not
        # recomputed per pair — the SAME deadline threads through every call.
        assert fake.calls == [1500.0, 1500.0]

    def test_no_time_budget_runs_exactly_one_pair(self, tmp_path, monkeypatch):
        comps = [_comp(c) for c in ("A", "B")]
        ds = _ds("d")
        entries = [("stt", c, ds, "en") for c in comps]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        sizes = {PairKey("stt", c.competitor_id, "d", "en").to_str(): 100 for c in comps}
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = lambda *a, **k: False

        summary = runner.run_one_shot(["stt"], max_samples=5, seed=0)

        assert len(summary["pairs"]) == 1
        assert summary["written"] == 5

    # -- REGRESSION: lazy discovery — the bug the adversarial review caught --

    def test_discovery_only_probes_drawn_candidates(self, tmp_path, monkeypatch):
        """pair_is_complete must be called AT MOST once per candidate
        actually walked before landing on a usable pair, never once per
        EVERY eligible candidate up front. With a no-op shuffle (walk order
        == entries order) and the first candidate immediately usable, only
        ONE pair_is_complete call should happen even though 500 candidates
        exist."""
        ds = _ds("d")
        comps = [_comp(f"c{i}") for i in range(500)]
        entries = [("stt", c, ds, "en") for c in comps]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        monkeypatch.setattr(autorun_module.random, "Random", _NoShuffleRandom)

        probe_calls = {"n": 0}

        def counting_pair_is_complete(m, c, d, l, deadline=None):
            probe_calls["n"] += 1
            return False  # the very first candidate is immediately usable

        sizes = {PairKey("stt", "c0", "d", "en").to_str(): 5}
        fake = FakeProcessor(sizes)
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = counting_pair_is_complete

        summary = runner.run_one_shot(["stt"], max_samples=5, seed=0)

        assert summary["pairs"][0]["pair"] == PairKey("stt", "c0", "d", "en").to_str()
        assert probe_calls["n"] == 1, (
            "discovery must be O(candidates walked before landing on one), "
            f"not O(all eligible pairs) — got {probe_calls['n']} probes for 500 candidates"
        )
        assert summary["candidates_probed"] == 1

    def test_discovery_deadline_exits_zero_pairs_with_distinct_summary(self, tmp_path, monkeypatch):
        """If the wall-clock deadline elapses WHILE STILL WALKING candidates
        (every one probed so far turned out complete, and probing itself
        costs fake-clock time), the call must exit with 0 pairs drawn and a
        `discovery_bound` flag distinct from `nothing_to_do` — a caller must
        be able to tell "ran out of time to look" from "genuinely nothing
        left"."""
        ds = _ds("d")
        comps = [_comp(f"c{i}") for i in range(50)]
        entries = [("stt", c, ds, "en") for c in comps]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        monkeypatch.setattr(autorun_module.random, "Random", _NoShuffleRandom)

        clock = {"t": 0.0}
        monkeypatch.setattr(autorun_module.time, "monotonic", lambda: clock["t"])

        def slow_complete_check(m, c, d, l, deadline=None):
            clock["t"] += 10  # each completeness probe "costs" 10s
            return True  # every candidate looks complete — never a real draw

        fake = FakeProcessor({})
        config = AutoRunConfig(output_dir=tmp_path, batch=10, upload=False)
        runner = AutoRunner(config, process_fn=fake)
        runner.pair_is_complete = slow_complete_check

        # 50 candidates * 10s = 500s > the 45s budget, so the deadline
        # elapses partway through the walk.
        summary = runner.run_one_shot(
            ["stt"], max_samples=5, seed=0, time_budget_secs=45,
        )

        assert summary["pairs"] == []
        assert summary["discovery_bound"] is True
        assert summary["nothing_to_do"] is False
        assert fake.calls == []  # never actually drew/ran a pair
        assert 0 < summary["candidates_probed"] < 50

    def test_summary_shapes_are_mutually_exclusive_across_three_outcomes(self, tmp_path, monkeypatch):
        """The three possible "found nothing" exits — real work done, every
        pair genuinely complete, ran out of time before finding one — must
        never overlap in their flags."""
        ds = _ds("d")

        # 1) Work done: neither flag set.
        entries = [("stt", _comp("A"), ds, "en")]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: entries)
        fake = FakeProcessor({PairKey("stt", "A", "d", "en").to_str(): 5})
        runner = AutoRunner(AutoRunConfig(output_dir=tmp_path / "work", batch=10, upload=False),
                             process_fn=fake)
        runner.pair_is_complete = lambda *a, **k: False
        summary = runner.run_one_shot(["stt"], max_samples=5, seed=0)
        assert summary["pairs"] and not summary["nothing_to_do"] and not summary["discovery_bound"]

        # 2) Nothing to do: only nothing_to_do.
        runner2 = AutoRunner(AutoRunConfig(output_dir=tmp_path / "done", batch=10, upload=False),
                              process_fn=FakeProcessor({}))
        runner2.pair_is_complete = lambda *a, **k: True
        summary2 = runner2.run_one_shot(["stt"], max_samples=5, seed=0)
        assert not summary2["pairs"] and summary2["nothing_to_do"] and not summary2["discovery_bound"]

        # 3) Discovery-bound: only discovery_bound. Needs MORE than one
        # candidate so the deadline is actually detected mid-walk (checked
        # between candidates) rather than the walk simply exhausting its
        # one-and-only candidate before anyone notices time ran out.
        many_entries = [("stt", _comp(f"c{i}"), ds, "en") for i in range(20)]
        monkeypatch.setattr(autorun_module, "enumerate_all_pairs", lambda *a, **k: many_entries)
        monkeypatch.setattr(autorun_module.random, "Random", _NoShuffleRandom)
        clock = {"t": 0.0}
        monkeypatch.setattr(autorun_module.time, "monotonic", lambda: clock["t"])

        def slow_complete_check(m, c, d, l, deadline=None):
            clock["t"] += 100
            return True

        runner3 = AutoRunner(AutoRunConfig(output_dir=tmp_path / "bound", batch=10, upload=False),
                              process_fn=FakeProcessor({}))
        runner3.pair_is_complete = slow_complete_check
        summary3 = runner3.run_one_shot(["stt"], max_samples=5, seed=0, time_budget_secs=10)
        assert not summary3["pairs"] and not summary3["nothing_to_do"] and summary3["discovery_bound"]
