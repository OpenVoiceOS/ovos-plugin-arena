"""Unit tests for arena.rover — word-level ROVER consensus."""
from __future__ import annotations

import pytest

from arena.rover import rover_consensus, rover_consensus_and_agreement


class TestRoverConsensus:
    def test_single_hypothesis_is_identity(self):
        assert rover_consensus(["turn on the lights"]) == "turn on the lights"

    def test_identical_hypotheses_passthrough(self):
        hyps = ["turn on the lights"] * 3
        assert rover_consensus(hyps) == "turn on the lights"

    def test_two_vs_one_disagreement_majority_wins(self):
        hyps = ["turn on the lights", "turn on the lights", "turn on the lites"]
        assert rover_consensus(hyps) == "turn on the lights"

    def test_insertion_is_captured_by_majority(self):
        # Two of three hypotheses agree the word "big" was said.
        hyps = ["the big cat sat", "the big cat sat", "the cat sat"]
        assert rover_consensus(hyps) == "the big cat sat"

    def test_deletion_is_captured_by_majority(self):
        # Two of three hypotheses agree "on" was dropped.
        hyps = ["the cat sat mat", "the cat sat mat", "the cat sat on mat"]
        assert rover_consensus(hyps) == "the cat sat mat"

    def test_tie_broken_by_primary_judge(self):
        # Two hypotheses, one word apiece disagreeing — a tie must resolve
        # to whatever the primary (index 0) judge said.
        hyps = ["hello world", "hallo world"]
        assert rover_consensus(hyps) == "hello world"

    def test_empty_hypotheses_returns_empty_string(self):
        assert rover_consensus([]) == ""

    def test_none_entries_are_skipped(self):
        assert rover_consensus([None, "hi there", None]) == "hi there"


class TestRoverAgreement:
    def test_unanimous_panel_has_full_agreement(self):
        _, agreement = rover_consensus_and_agreement(["hi there"] * 3)
        assert agreement == 1.0

    def test_single_hypothesis_has_full_agreement(self):
        _, agreement = rover_consensus_and_agreement(["hi there"])
        assert agreement == 1.0

    def test_empty_hypotheses_has_full_agreement(self):
        _, agreement = rover_consensus_and_agreement([])
        assert agreement == 1.0

    def test_two_of_three_slots_disagreement_is_exact(self):
        # 2-of-3 judges agree "hello"; all 3 agree "world" — mean per-slot
        # vote share is (2/3 + 3/3) / 2 = 5/6.
        consensus, agreement = rover_consensus_and_agreement(
            ["hello world", "hello world", "hallo world"])
        assert consensus == "hello world"
        assert agreement == pytest.approx(5 / 6)
