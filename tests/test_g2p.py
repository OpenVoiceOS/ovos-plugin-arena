"""Tests for the g2p league (§A6): PER scoring, provenance validation, and
the runner adapter's engine-facing contract (fake plugin, no real OVOS
install required).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from arena.metrics import normalize_ipa, row_per, row_per_components, score_g2p
from arena.models import PredictionRow
from registry.schemas import CIRCULAR_PROVENANCE_TIERS, DatasetDef, HuggingFaceSource


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p", modality="g2p",
    )
    base.update(over)
    return PredictionRow(**base)


# ---------------------------------------------------------------------------
# normalize_ipa / PER
# ---------------------------------------------------------------------------


class TestNormalizeIpa:
    def test_strips_stress_marks(self):
        assert normalize_ipa("hɐˈloʊ") == normalize_ipa("hɐloʊ")

    def test_strips_punctuation_and_tie_bars(self):
        assert normalize_ipa("t͡ʃiz.") == normalize_ipa("tʃiz")

    def test_folds_ascii_g_confusable(self):
        assert normalize_ipa("gow") == normalize_ipa("ɡow")


class TestScoreG2P:
    def test_identical_strings_score_zero_per(self):
        rows = [_row(reference_text="hɛˈloʊ", prediction="hɛˈloʊ")]
        metrics = score_g2p(rows)
        assert metrics["per"] == 0.0

    def test_stress_only_difference_scores_zero(self):
        """A hypothesis differing from gold ONLY in stress marking must
        score 0 PER — the fairness rule this whole port exists for."""
        rows = [_row(reference_text="hɛˈloʊ", prediction="hɛloʊ")]
        metrics = score_g2p(rows)
        assert metrics["per"] == 0.0

    def test_known_edit_distance(self):
        # "kat" vs "kot": one substitution over 3 reference phonemes.
        errors, ref_len = row_per_components(
            _row(reference_text="kat", prediction="kot")
        )
        assert (errors, ref_len) == (1.0, 3.0)
        assert row_per(_row(reference_text="kat", prediction="kot")) == pytest.approx(1 / 3, abs=1e-4)

    def test_aggregate_is_error_ratio_not_row_average(self):
        # Same convention as STT's wer_mean: sum(errors)/sum(ref_len), not
        # the mean of per-row ratios — a 1-word gold shouldn't outweigh a
        # 10-word gold in the pooled metric.
        rows = [
            _row(sample_id="s1", reference_text="a", prediction="b"),  # 1/1
            _row(sample_id="s2", reference_text="aaaaaaaaaa", prediction="aaaaaaaaaa"),  # 0/10
        ]
        metrics = score_g2p(rows)
        assert metrics["per"] == pytest.approx(1 / 11, abs=1e-4)

    def test_missing_prediction_excluded_not_crashed(self):
        rows = [_row(reference_text="kat", prediction=None)]
        metrics = score_g2p(rows)
        assert metrics["n_scored"] == 0.0
        assert "per" not in metrics


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------


def _dataset(**over):
    base = dict(
        dataset_id="d", modality="g2p",
        source=HuggingFaceSource(hf_id="org/dataset"),
        lang="en-US", role="eval",
    )
    base.update(over)
    return DatasetDef(**base)


class TestG2PProvenanceValidation:
    def test_eval_dataset_requires_provenance_tier(self):
        with pytest.raises(ValidationError):
            _dataset(provenance_tier=None)

    def test_circular_tiers_rejected(self):
        for tier in CIRCULAR_PROVENANCE_TIERS:
            with pytest.raises(ValidationError):
                _dataset(provenance_tier=tier)

    def test_tool_derived_tier_requires_explicit_flag(self):
        with pytest.raises(ValidationError):
            _dataset(provenance_tier="machine-generated", tool_derived=False)
        # Explicit ack passes.
        d = _dataset(provenance_tier="machine-generated", tool_derived=True)
        assert d.tool_derived is True

    def test_lexicon_derived_gold_registers_cleanly(self):
        d = _dataset(provenance_tier="lexicon-derived")
        assert d.provenance_tier == "lexicon-derived"
        assert d.tool_derived is False

    def test_unknown_tier_rejected(self):
        with pytest.raises(ValidationError):
            _dataset(provenance_tier="definitely-not-a-real-tier")

    def test_train_role_does_not_require_tier(self):
        # The gate only applies to eval (gold) datasets.
        d = _dataset(role="train", provenance_tier=None)
        assert d.provenance_tier is None

    def test_non_g2p_modality_unaffected(self):
        d = DatasetDef(
            dataset_id="d", modality="stt",
            source=HuggingFaceSource(hf_id="org/dataset"),
            lang="en-US", role="eval",
        )
        assert d.provenance_tier is None


# ---------------------------------------------------------------------------
# Adapter contract (fake plugin, no OVOS install required)
# ---------------------------------------------------------------------------


class _FakeG2PPlugin:
    """Mimics ovos_plugin_manager.templates.g2p.Grapheme2PhonemePlugin's
    public surface just enough to exercise runner.g2p_bench.predict()."""

    def __init__(self, config=None):
        self.config = config or {}

    def get_ipa(self, word, lang, ignore_oov=False):
        table = {"hello": ["h", "ɛ", "l", "oʊ"], "cat": ["k", "æ", "t"]}
        phones = table.get(word.lower())
        if phones is None:
            if ignore_oov:
                return None
            raise KeyError(word)
        return phones

    def utterance2ipa(self, utterance, lang, ignore_oov=False):
        out = []
        for w in utterance.split():
            out += self.get_ipa(w, lang, ignore_oov) + ["."]
        return out[:-1] if out else []


class TestG2PBenchAdapter:
    def test_predict_single_word(self):
        from runner.g2p_bench import predict

        engine = _FakeG2PPlugin()
        result = predict(engine, {"grapheme": "cat", "reference_ipa": "kæt"}, "en")
        assert result["prediction"] == "kæt"
        assert result["reference_text"] == "kæt"
        assert result["extras"]["oov"] is False

    def test_predict_oov_word_flagged_not_crashed(self):
        from runner.g2p_bench import predict

        engine = _FakeG2PPlugin()
        result = predict(
            engine, {"grapheme": "xyzzy", "reference_ipa": "??"}, "en"
        )
        assert result["prediction"] == ""
        assert result["extras"]["oov"] is True

    def test_predict_multiword_uses_utterance2ipa(self):
        from runner.g2p_bench import predict

        engine = _FakeG2PPlugin()
        result = predict(
            engine, {"grapheme": "hello cat", "reference_ipa": "hɛloʊ.kæt"}, "en"
        )
        assert result["prediction"] == "hɛloʊ.kæt"

    def test_competitor_lang_matching_primary_subtag(self):
        from registry.schemas import CompetitorDef
        from runner.g2p_bench import competitor_langs

        comp = CompetitorDef(
            competitor_id="c", modality="g2p", plugin="fake-g2p",
            langs=["en-GB"],
        )
        assert competitor_langs(comp, "en-US") == ["en-US"]
        assert competitor_langs(comp, "pt-PT") == []

    def test_arpa_to_ipa_strips_stress_digits(self):
        from runner.g2p_bench import _arpa_to_ipa

        # ARPABET stress digits (0/1/2) must not survive into the IPA
        # string used as gold — they are not IPA symbols.
        result = _arpa_to_ipa("HH AH0 L OW1")
        assert result and all(c not in "012" for c in result)
