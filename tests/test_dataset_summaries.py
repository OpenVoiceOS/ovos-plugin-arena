"""Every number a dataset summary states must be traceable to the registry.

The summaries are published prose: they head each board on the public
leaderboard, so a figure invented for one corpus and pasted across a family
would read as measured fact. These tests pin every digit in every summary to
something the registry itself records — the sampling cap, the language list,
or the corpus's own ``notes`` — so a future summary cannot carry a number
nobody can source.
"""
from __future__ import annotations

import json
import re

import pytest

from registry.loaders import list_datasets

EVAL_DATASETS = [d for d in list_datasets() if d.role == "eval"]
NUMBER = re.compile(r"\b\d[\d,]*\b")
# "Runs on this corpus are capped at 2,000 clips, ..."
CAP_CLAIM = re.compile(r"capped at ([\d,]+) (?:clips|rows|prompts)")
LANG_CLAIM = re.compile(r"\((\d+) languages\)|across (\d+) languages")


def _ints(text: str) -> set[int]:
    return {int(m.replace(",", "")) for m in NUMBER.findall(text or "")}


class TestSummariesExist:
    def test_every_eval_dataset_has_one(self):
        missing = [d.dataset_id for d in EVAL_DATASETS if not (d.summary or "").strip()]
        assert not missing, f"eval corpora with no summary: {missing}"

    def test_the_registry_actually_loaded(self):
        # Guards the filter above: an empty list would make every
        # parametrised test below vacuous.
        assert len(EVAL_DATASETS) > 100


@pytest.mark.parametrize("dataset", EVAL_DATASETS, ids=lambda d: d.dataset_id)
class TestNumbersAreSourced:
    def test_cap_claim_matches_the_sample_policy(self, dataset):
        """A stated sampling cap is the one the runner will actually apply
        (registry ``sample_policy.max_samples``, honoured by
        ``runner.audio_io.resolve_sample_cap``), never a round number."""
        claim = CAP_CLAIM.search(dataset.summary or "")
        policy_cap = dataset.sample_policy.max_samples if dataset.sample_policy else None
        if claim is None:
            assert policy_cap is None, (
                f"{dataset.dataset_id} is capped at {policy_cap} but its "
                "summary does not say so"
            )
        else:
            assert int(claim.group(1).replace(",", "")) == policy_cap

    def test_language_count_matches_the_langs_list(self, dataset):
        for match in LANG_CLAIM.finditer(dataset.summary or ""):
            claimed = int(match.group(1) or match.group(2))
            assert claimed == len(dataset.langs or []), (
                f"{dataset.dataset_id} claims {claimed} languages, "
                f"registry lists {len(dataset.langs or [])}"
            )

    def test_no_number_is_unsourced(self, dataset):
        """Any remaining figure must appear in the corpus's own ``notes``,
        its language list, or its sampling cap — the three places the
        registry records a real count."""
        summary = dataset.summary or ""
        sourced = _ints(dataset.notes or "")
        sourced.add(len(dataset.langs or []))
        if dataset.sample_policy and dataset.sample_policy.max_samples:
            sourced.add(dataset.sample_policy.max_samples)
        unsourced = sorted(_ints(summary) - sourced)
        assert not unsourced, (
            f"{dataset.dataset_id}: summary states {unsourced}, which appears "
            "neither in its notes, its langs list, nor its sample_policy"
        )


class TestNoUnmeasuredRanking:
    """Difficulty rankings between boards are not measured anywhere, so the
    summaries describe a corpus's properties instead of ranking it."""

    BANNED = re.compile(
        r"\b(hardest|easiest|harder than|easier than|most comparable|"
        r"the easy end|best board|worst board)\b", re.I)

    @pytest.mark.parametrize("dataset", EVAL_DATASETS, ids=lambda d: d.dataset_id)
    def test_summary_makes_no_ranking_claim(self, dataset):
        found = self.BANNED.findall(dataset.summary or "")
        assert not found, f"{dataset.dataset_id}: unmeasured ranking claim {found}"


class TestSummaryShape:
    @pytest.mark.parametrize("dataset", EVAL_DATASETS, ids=lambda d: d.dataset_id)
    def test_reads_as_plain_sentences(self, dataset):
        summary = dataset.summary
        assert summary.endswith("."), dataset.dataset_id
        sentences = [s for s in summary.split(". ") if s.strip()]
        assert 2 <= len(sentences) <= 6, (
            f"{dataset.dataset_id}: {len(sentences)} sentences, "
            "expected two to four plain ones"
        )
        assert json.dumps(summary)  # no control characters
