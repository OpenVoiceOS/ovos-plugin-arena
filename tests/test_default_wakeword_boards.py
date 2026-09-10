"""The shipped default wake phrases must each have an eval corpus.

A default phrase with no registry entry cannot be swept, so it gets no board,
so a newly trained model for it has nothing to rank against and no baseline to
beat. That failure is invisible from the boards themselves: an unregistered
phrase produces no rows, and no rows looks the same as a phrase nobody has
gotten to yet.

The set is pinned rather than derived. Deriving it from ovos-config would make
the arena depend on the core stack to answer a question about its own
registry, and a set that grows silently is the thing this guards against: a
third default phrase added upstream should fail here until someone gives it a
corpus.
"""
from __future__ import annotations

import pytest

from registry.loaders import list_datasets

#: The wake words a stock install listens for, per the shipped mycroft.conf.
DEFAULT_WAKE_PHRASES = {"hey_mycroft", "wake_up"}


@pytest.fixture(scope="module")
def wake_word_datasets():
    return list(list_datasets("wake_word"))


def test_every_default_phrase_has_an_eval_corpus(wake_word_datasets):
    covered = {
        d.wakeword for d in wake_word_datasets
        if getattr(d, "wakeword", None) and d.role == "eval"
    }
    missing = DEFAULT_WAKE_PHRASES - covered
    assert not missing, (
        f"default wake phrase(s) with no eval corpus: {sorted(missing)} — "
        f"a phrase with no corpus gets no board, so a model trained for it "
        f"has nothing to rank against"
    )


def test_each_default_phrase_names_a_distinct_predictions_repo(wake_word_datasets):
    """Two phrases sharing a repo would overwrite each other's rows."""
    repos = {}
    for d in wake_word_datasets:
        if getattr(d, "wakeword", None) in DEFAULT_WAKE_PHRASES and d.role == "eval":
            repos.setdefault(d.predictions_hf, []).append(d.dataset_id)
    clashes = {k: v for k, v in repos.items() if len(v) > 1}
    assert not clashes, f"default phrases sharing a predictions repo: {clashes}"
