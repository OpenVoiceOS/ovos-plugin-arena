"""A wake-word dataset's sample_policy must reach the sampler.

Wake-word corpora take an early branch that used to pass the operator's
``--max-samples`` straight through, so a registry ``sample_policy`` was
accepted by the schema and then silently ignored: two fighters swept at
different times could draw different depths with nothing pinning the draw.
"""
from dataclasses import dataclass

from runner import ww_bench


@dataclass
class _Policy:
    max_samples: int
    seed: int = 1337


@dataclass
class _Source:
    hf_id: str = "OpenVoiceOS/synthetic-wakewords"
    file_pattern: str | None = None
    subset: str | None = None


@dataclass
class _DatasetDef:
    dataset_id: str = "synthetic-wakewords-wake_up"
    wakeword: str = "wake_up"
    sample_policy: _Policy | None = None
    reference_fields: dict | None = None
    source: _Source = None


def _capture(monkeypatch):
    seen = {}

    def fake_stream_ww(dataset_def, revision, max_per_class=0):
        seen["max_per_class"] = max_per_class
        return iter(())

    monkeypatch.setattr(ww_bench, "stream_ww", fake_stream_ww)
    return seen


def _drain(dataset_def):
    list(ww_bench.WakeWordBench().iter_samples(
        dataset_def, "en-US", "main", 0))


def test_a_registry_sample_policy_reaches_the_wake_word_sampler(monkeypatch):
    seen = _capture(monkeypatch)
    _drain(_DatasetDef(sample_policy=_Policy(max_samples=5),
                       source=_Source()))
    assert seen["max_per_class"] == 5


def test_an_operator_cap_below_the_policy_still_wins(monkeypatch):
    seen = _capture(monkeypatch)
    d = _DatasetDef(sample_policy=_Policy(max_samples=50), source=_Source())
    list(ww_bench.WakeWordBench().iter_samples(d, "en-US", "main", 7))
    assert seen["max_per_class"] == 7


def test_no_policy_leaves_the_operator_cap_alone(monkeypatch):
    seen = _capture(monkeypatch)
    d = _DatasetDef(sample_policy=None, source=_Source())
    list(ww_bench.WakeWordBench().iter_samples(d, "en-US", "main", 9))
    assert seen["max_per_class"] == 9
