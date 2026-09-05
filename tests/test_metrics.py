"""Unit tests for arena.metrics — benchmark scoring."""
from __future__ import annotations

import pytest

from arena.metrics import (
    TTS_QUALITY_DIMENSION_KEYS,
    build_benchmark_board,
    domain_of,
    expected_calibration_error,
    intelligibility_scores,
    row_intelligibility_agreement,
    row_intelligibility_cer,
    row_intelligibility_wer,
    row_is_correct,
    row_quality_dimension,
    row_utmos,
    row_wer,
    score_intent,
    score_stt,
    score_tts,
    score_wake_word,
    tts_seed_score,
    ww_row_correct,
)
from arena.models import PredictionRow


def _row(**over):
    base = dict(
        competitor_id="c", sample_id="s", dataset_id="d", lang="en-US",
        plugin_id="p",
    )
    base.update(over)
    return PredictionRow(**base)


class TestRowIsCorrect:
    def test_exact_match_field_wins(self):
        assert row_is_correct(_row(reference_intent="a", prediction="b",
                                   exact_match=True))

    def test_string_comparison_fallback(self):
        assert row_is_correct(_row(reference_intent="a", prediction="a"))
        assert not row_is_correct(_row(reference_intent="a", prediction="b"))

    def test_ood_correct_rejection(self):
        assert row_is_correct(_row(reference_intent=None, prediction=None))
        assert not row_is_correct(_row(reference_intent=None, prediction="a"))


class TestDomainOf:
    """domain_of — the 'text before the first :' rule domain-granularity
    scoring is built on (meteocat and future domain-only corpora)."""

    def test_strips_intent_suffix(self):
        assert domain_of("weather:current_conditions") == "weather"

    def test_bare_domain_passes_through(self):
        assert domain_of("weather") == "weather"

    def test_only_first_colon_splits(self):
        assert domain_of("weather:forecast:tomorrow") == "weather"

    def test_none_stays_none(self):
        assert domain_of(None) is None


class TestGeneralizationAccuracy:
    """The ranked intent metric must ignore the buckets that leak training
    data (``template``/``near_ood``), so a memorizer cannot outrank an engine
    that actually handles unseen phrasings."""

    def test_contaminated_buckets_do_not_enter_the_metric(self):
        rows = [
            _row(reference_intent="a", prediction="a", bucket="template"),
            _row(reference_intent="a", prediction="a", bucket="near_ood"),
            _row(reference_intent="a", prediction="b", bucket="paraphrase"),
            _row(reference_intent="a", prediction="a", bucket="typos"),
        ]
        metrics = score_intent(rows)
        assert metrics["accuracy"] == 0.75
        assert metrics["generalization_accuracy"] == 0.5
        assert metrics["acc_template"] == 1.0
        assert metrics["acc_in_distribution"] == 1.0
        assert "acc_near_ood" not in metrics

    def test_all_contaminated_fighter_is_unranked_but_not_called_a_failure(self):
        # Every row lands in a bucket the ranked metric excludes: there is no
        # generalization_accuracy to rank on, but the run itself succeeded.
        by_competitor = {
            "memorizer": [
                _row(competitor_id="memorizer", sample_id=f"t{i}",
                     bucket="template", reference_intent="a", prediction="a")
                for i in range(4)
            ],
            "generalizer": [
                _row(competitor_id="generalizer", sample_id=f"p{i}",
                     bucket="paraphrase", reference_intent="a", prediction="b")
                for i in range(4)
            ],
        }
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        memorizer = next(e for e in board.entries
                         if e.competitor_id == "memorizer")
        assert memorizer.metrics["accuracy"] == 1.0
        assert "generalization_accuracy" not in memorizer.metrics
        assert memorizer.unranked is True
        assert memorizer.rank == 0
        assert "run failed" not in memorizer.unranked_reason
        assert "generalization_accuracy" in memorizer.unranked_reason

    def test_zero_row_fighter_still_reads_as_a_failed_run(self):
        by_competitor = {"crashed": [], "ok": [
            _row(competitor_id="ok", bucket="paraphrase",
                 reference_intent="a", prediction="a"),
        ]}
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        crashed = next(e for e in board.entries
                       if e.competitor_id == "crashed")
        assert crashed.metrics["n_scored"] == 0.0
        assert crashed.unranked_reason == "run failed — no scored samples"

    def test_crashed_stt_run_reads_as_a_failed_run_not_off_metric(self):
        # score_stt returns {} (not an n_scored=0.0 placeholder) when a row
        # never produced a transcript or reference to compute WER from.
        by_competitor = {"crashed": [
            _row(competitor_id="crashed", dataset_id="d", lang="en-US"),
        ]}
        board = build_benchmark_board("stt", "d", "en-US", by_competitor, "t")
        crashed = next(e for e in board.entries
                       if e.competitor_id == "crashed")
        assert crashed.metrics == {}
        assert crashed.unranked_reason == "run failed — no scored samples"

    def test_memorizer_does_not_outrank_generalizer(self):
        def rows(competitor, memorized, generalized):
            out = []
            for i in range(10):
                for bucket, hit in (("template", memorized),
                                    ("near_ood", memorized),
                                    ("paraphrase", generalized),
                                    ("typos", generalized)):
                    out.append(_row(
                        competitor_id=competitor, sample_id=f"{bucket}-{i}",
                        bucket=bucket, reference_intent="a",
                        prediction="a" if hit else "b",
                    ))
            return out

        by_competitor = {
            "memorizer": rows("memorizer", memorized=True, generalized=False),
            "generalizer": rows("generalizer", memorized=False, generalized=True),
        }
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        assert board.primary_metric == "generalization_accuracy"
        ranked = [e.competitor_id for e in board.entries]
        assert ranked == ["generalizer", "memorizer"]
        scores = {e.competitor_id: e.metrics for e in board.entries}
        assert scores["memorizer"]["accuracy"] == scores["generalizer"]["accuracy"]


class TestScoreIntent:
    def test_accuracy_counts_ood_rejections(self):
        rows = [
            _row(reference_intent="a", prediction="a", bucket="template"),
            _row(reference_intent="a", prediction="b", bucket="template"),
            _row(reference_intent=None, prediction=None, bucket="far_ood"),
            _row(reference_intent=None, prediction="a", bucket="far_ood"),
        ]
        metrics = score_intent(rows)
        assert metrics["accuracy"] == 0.5
        assert metrics["ood_fpr"] == 0.5
        assert metrics["acc_template"] == 0.5
        assert metrics["acc_far_ood"] == 0.5

    def test_perfect_run(self):
        rows = [
            _row(reference_intent="a", prediction="a"),
            _row(reference_intent="b", prediction="b"),
            _row(reference_intent=None, prediction=None),
        ]
        metrics = score_intent(rows)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["ood_fpr"] == 0.0

    def test_ood_false_positive_hurts_macro_f1(self):
        clean = score_intent([
            _row(reference_intent="a", prediction="a"),
            _row(reference_intent=None, prediction=None),
        ])
        noisy = score_intent([
            _row(reference_intent="a", prediction="a"),
            _row(reference_intent=None, prediction="a"),
        ])
        assert noisy["macro_f1"] < clean["macro_f1"]

    def test_slot_exact_match(self):
        rows = [
            _row(reference_intent="a", prediction="a",
                 reference_slots={"song": "Africa"},
                 predicted_slots={"song": "africa"}),  # case-insensitive
            _row(reference_intent="a", prediction="a",
                 reference_slots={"song": "x"},
                 predicted_slots={}),
        ]
        assert score_intent(rows)["slot_exact_match"] == 0.5

    def test_latency_median(self):
        rows = [
            _row(reference_intent="a", prediction="a", latency_ms=10.0),
            _row(reference_intent="a", prediction="a", latency_ms=20.0),
            _row(reference_intent="a", prediction="a", latency_ms=30.0),
        ]
        assert score_intent(rows)["latency_ms_median"] == 20.0

    def test_empty(self):
        assert score_intent([])["accuracy"] == 0.0

    def test_ece_column_present_when_confidence_available(self):
        rows = [
            _row(reference_intent="a", prediction="a", confidence=0.9),
            _row(reference_intent="a", prediction="b", confidence=0.9),
        ]
        assert "ece" in score_intent(rows)

    def test_ece_absent_without_confidence(self):
        rows = [_row(reference_intent="a", prediction="a")]
        assert "ece" not in score_intent(rows)


class TestExpectedCalibrationError:
    """§M4 — ECE from existing prediction rows, no re-benching."""

    def test_missing_confidence_is_none_not_zero(self):
        rows = [_row(reference_intent="a", prediction="a")]
        assert expected_calibration_error(rows) is None

    def test_empty_rows_is_none(self):
        assert expected_calibration_error([]) is None

    def test_perfectly_calibrated_is_zero(self):
        # Every row's confidence matches the bin's actual accuracy exactly:
        # 10 rows at confidence 0.75, 7.5 correct — use a bin count where the
        # split is exact (8 correct out of 10 -> acc 0.8, conf must be 0.8).
        rows = (
            [_row(reference_intent="a", prediction="a", confidence=0.8)] * 8
            + [_row(reference_intent="a", prediction="b", confidence=0.8)] * 2
        )
        assert expected_calibration_error(rows) == 0.0

    def test_always_confident_always_wrong_is_high(self):
        rows = [
            _row(reference_intent="a", prediction="b", confidence=1.0)
            for _ in range(10)
        ]
        # 100% confident, 0% accurate in that bin -> ECE == 1.0 (max).
        assert expected_calibration_error(rows) == 1.0

    def test_confidence_exactly_one_is_included_not_dropped(self):
        # confidence == 1.0 is the top edge of [0, 1] and must land in the
        # last of the 10 equal-width bins ([0.9, 1.0]), not be excluded or
        # overflow into a phantom 11th bin.
        rows = [_row(reference_intent="a", prediction="a", confidence=1.0)]
        assert expected_calibration_error(rows) == 0.0

    def test_single_bin_all_rows_same_confidence(self):
        rows = [
            _row(reference_intent="a", prediction="a", confidence=0.5),
            _row(reference_intent="a", prediction="b", confidence=0.5),
        ]
        # acc 0.5, mean conf 0.5 -> perfectly calibrated even though every
        # row falls in a single bin.
        assert expected_calibration_error(rows) == 0.0

    def test_ood_rows_use_row_is_correct_for_bin_accuracy(self):
        # An OOD row (reference_intent=None) is "correct" iff prediction is
        # also None — confidence-bearing OOD rejections must count toward
        # bin accuracy via the same row_is_correct() rule as everywhere else.
        rows = [
            _row(reference_intent=None, prediction=None, confidence=0.9),
            _row(reference_intent=None, prediction="a", confidence=0.9),
        ]
        # acc 0.5, mean conf 0.9 -> |0.5 - 0.9| = 0.4
        assert expected_calibration_error(rows) == 0.4

    def test_mixed_confidence_and_no_confidence_scores_only_the_former(self):
        with_conf = [
            _row(reference_intent="a", prediction="a", confidence=0.8)
            for _ in range(8)
        ] + [
            _row(reference_intent="a", prediction="b", confidence=0.8)
            for _ in range(2)
        ]
        without_conf = [_row(reference_intent="a", prediction="a")]
        assert expected_calibration_error(with_conf) == 0.0
        assert expected_calibration_error(with_conf + without_conf) == 0.0


class TestScoreStt:
    def test_wer_present(self):
        rows = [_row(wer=0.1), _row(wer=0.3)]
        metrics = score_stt(rows)
        assert metrics["wer_mean"] == pytest.approx(0.2)
        assert metrics["wer_median"] == pytest.approx(0.2)

    def test_wer_computed_from_reference(self):
        rows = [_row(reference_text="hello world", prediction="hello world"),
                _row(reference_text="hello world", prediction="hello there")]
        assert score_stt(rows)["wer_mean"] == pytest.approx(0.25)

    def test_row_wer_priority(self):
        assert row_wer(_row(wer=0.42)) == 0.42
        assert row_wer(_row(reference_text="a b", prediction="a c")) == 0.5
        assert row_wer(_row()) is None


class TestScoreWakeWord:
    def _clip(self, label, prediction, **over):
        return _row(label=label, prediction=prediction, **over)

    def test_row_correct(self):
        assert ww_row_correct(self._clip("positive", "detected")) is True
        assert ww_row_correct(self._clip("positive", "not_detected")) is False
        assert ww_row_correct(self._clip("negative", "not_detected")) is True
        assert ww_row_correct(self._clip("negative", "detected")) is False
        assert ww_row_correct(self._clip(None, "detected")) is None

    def test_label_aliases(self):
        # numeric / boolean-ish labels normalise to presence
        assert ww_row_correct(self._clip("1", "detected")) is True
        assert ww_row_correct(self._clip("0", "not_detected")) is True
        assert ww_row_correct(self._clip("adversarial", "not_detected")) is True

    def test_rates(self):
        rows = [
            self._clip("positive", "detected"),       # TP
            self._clip("positive", "not_detected"),   # FR
            self._clip("negative", "not_detected"),   # TN
            self._clip("negative", "detected"),        # FA
        ]
        m = score_wake_word(rows)
        assert m["error_rate"] == 0.5
        assert m["accuracy"] == 0.5
        assert m["false_accept_rate"] == 0.5
        assert m["false_reject_rate"] == 0.5

    def test_perfect(self):
        rows = [self._clip("positive", "detected"),
                self._clip("negative", "not_detected")]
        m = score_wake_word(rows)
        assert m["error_rate"] == 0.0
        assert m["false_accept_rate"] == 0.0
        assert m["false_reject_rate"] == 0.0

    def test_unscorable_rows_ignored(self):
        assert score_wake_word([self._clip(None, None)]) == {}

    def test_latency(self):
        rows = [self._clip("positive", "detected", latency_ms=5.0),
                self._clip("negative", "not_detected", latency_ms=15.0)]
        assert score_wake_word(rows)["latency_ms_median"] == 10.0


class TestFaPerHour:
    """§M5 — fa_per_hour on the isolated-clip wake_word board."""

    def _clip(self, label, prediction, audio_secs=None, **over):
        return _row(label=label, prediction=prediction, audio_secs=audio_secs,
                    **over)

    def test_denominator_math_by_hand(self):
        # 1800 s of covered negative audio (0.5h) = 3 * 600s clips; one FA.
        rows = [
            self._clip("negative", "not_detected", audio_secs=600.0),
            self._clip("negative", "detected", audio_secs=600.0),
            self._clip("negative", "not_detected", audio_secs=600.0),
            self._clip("positive", "detected", audio_secs=1.0),
        ]
        m = score_wake_word(rows)
        assert m["fa_per_hour_hours"] == 0.5
        # 1 false accept / 0.5h = 2.0 FA/h
        assert m["fa_per_hour"] == 2.0

    def test_below_threshold_omits_metric(self):
        # 300s = 0.0833h of covered negative audio, well under MIN_FA_HOURS
        # (0.25h) — fa_per_hour must be omitted (None), not printed as noise,
        # even though the coverage figure itself is still reported.
        rows = [self._clip("negative", "detected", audio_secs=300.0)]
        m = score_wake_word(rows)
        assert "fa_per_hour" not in m
        assert m["fa_per_hour_hours"] == pytest.approx(300.0 / 3600.0, abs=1e-4)

    def test_at_threshold_included(self):
        # exactly MIN_FA_HOURS (0.25h = 900s) — boundary is inclusive.
        rows = [self._clip("negative", "detected", audio_secs=900.0)]
        m = score_wake_word(rows)
        assert m["fa_per_hour_hours"] == 0.25
        assert m["fa_per_hour"] == 4.0

    def test_rows_without_audio_secs_excluded_from_both(self):
        # a row with no audio_secs (pre-#90) must not contribute to the
        # numerator (its false accept) OR the denominator (its duration).
        rows = [
            self._clip("negative", "detected", audio_secs=None),  # excluded
            self._clip("negative", "not_detected", audio_secs=1800.0),  # 0.5h
        ]
        m = score_wake_word(rows)
        assert m["fa_per_hour_hours"] == 0.5
        assert m["fa_per_hour"] == 0.0  # the covered clip had no FA

    def test_no_negatives_with_audio_secs_omits_both(self):
        rows = [self._clip("negative", "detected", audio_secs=None),
                self._clip("positive", "detected", audio_secs=5.0)]
        m = score_wake_word(rows)
        assert "fa_per_hour" not in m
        assert "fa_per_hour_hours" not in m


class TestRowUtmos:
    def test_present(self):
        assert row_utmos(_row(extras={"utmos": 3.5})) == 3.5

    def test_missing(self):
        assert row_utmos(_row()) is None

    def test_non_numeric_ignored(self):
        assert row_utmos(_row(extras={"utmos": "not-a-number"})) is None

    def test_nan_guard(self):
        assert row_utmos(_row(extras={"utmos": float("nan")})) is None


class TestScoreTts:
    def test_mean(self):
        rows = [_row(extras={"utmos": 3.0}), _row(extras={"utmos": 4.0})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(3.5)
        assert metrics["n_scored"] == 2.0

    def test_missing_rows_excluded_not_fatal(self):
        rows = [_row(extras={"utmos": 4.0}), _row(extras={}),
                _row(extras={"utmos": None})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(4.0)
        assert metrics["n_scored"] == 1.0

    def test_all_rows_missing_utmos(self):
        rows = [_row(extras={}), _row(extras={})]
        metrics = score_tts(rows)
        assert "utmos" not in metrics
        assert metrics["n_scored"] == 0.0

    def test_single_row(self):
        metrics = score_tts([_row(extras={"utmos": 4.2})])
        assert metrics["utmos"] == pytest.approx(4.2)

    def test_empty(self):
        assert score_tts([]) == {"n_scored": 0.0}

    def test_latency(self):
        rows = [_row(extras={"utmos": 4.0}, latency_ms=5.0),
                _row(extras={"utmos": 3.0}, latency_ms=15.0)]
        assert score_tts(rows)["latency_ms_median"] == 10.0

    def test_nan_score_excluded(self):
        rows = [_row(extras={"utmos": 4.0}),
                _row(extras={"utmos": float("nan")})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(4.0)
        assert metrics["n_scored"] == 1.0

    def test_intelligibility_wer_mean_and_ci(self):
        rows = [_row(extras={"utmos": 4.0, "intelligibility_wer": 0.0}),
                _row(extras={"utmos": 3.0, "intelligibility_wer": 0.5})]
        metrics = score_tts(rows)
        # UTMOS stays primary, intelligibility_wer rides along as secondary
        assert metrics["utmos"] == pytest.approx(3.5)
        assert metrics["intelligibility_wer"] == pytest.approx(0.25)
        assert metrics["intelligibility_n_scored"] == 2.0
        assert "intelligibility_wer_ci_lower" in metrics
        assert "intelligibility_wer_ci_upper" in metrics
        assert (metrics["intelligibility_wer_ci_lower"]
                <= metrics["intelligibility_wer"]
                <= metrics["intelligibility_wer_ci_upper"])

    def test_intelligibility_wer_missing_rows_excluded_not_fatal(self):
        rows = [_row(extras={"intelligibility_wer": 0.2}), _row(extras={})]
        metrics = score_tts(rows)
        assert metrics["intelligibility_wer"] == pytest.approx(0.2)
        assert metrics["intelligibility_n_scored"] == 1.0

    def test_no_intelligibility_data_omits_secondary_metric(self):
        rows = [_row(extras={"utmos": 4.0})]
        metrics = score_tts(rows)
        assert "intelligibility_wer" not in metrics

    def test_sigmos_dnsmos_nisqa_dimensions_aggregated_as_mean(self):
        rows = [
            _row(extras={"utmos": 4.0, "sigmos.noise": 4.0, "sigmos.col": 3.0,
                          "sigmos.disc": 4.5, "dnsmos.bak": 3.5, "dnsmos.ovrl": 3.0,
                          "nisqa.mos": 4.2, "nisqa.noi": 4.0}),
            _row(extras={"utmos": 3.0, "sigmos.noise": 2.0, "sigmos.col": 5.0,
                          "sigmos.disc": 3.5, "dnsmos.bak": 4.5, "dnsmos.ovrl": 4.0,
                          "nisqa.mos": 3.8, "nisqa.noi": 2.0}),
        ]
        metrics = score_tts(rows)
        assert metrics["sigmos.noise"] == pytest.approx(3.0)
        assert metrics["sigmos.col"] == pytest.approx(4.0)
        assert metrics["sigmos.disc"] == pytest.approx(4.0)
        assert metrics["dnsmos.bak"] == pytest.approx(4.0)
        assert metrics["dnsmos.ovrl"] == pytest.approx(3.5)
        assert metrics["nisqa.mos"] == pytest.approx(4.0)
        assert metrics["nisqa.noi"] == pytest.approx(3.0)

    def test_quality_dimensions_tolerate_old_rows_missing_the_key(self):
        # rows benched before this metric existed have no sigmos.*/dnsmos.*/
        # nisqa.* extras at all — aggregation must not crash or fabricate a
        # score.
        rows = [_row(extras={"utmos": 4.0}), _row(extras={"utmos": 3.5})]
        metrics = score_tts(rows)
        assert metrics["utmos"] == pytest.approx(3.75)
        for key in TTS_QUALITY_DIMENSION_KEYS:
            assert key not in metrics

    def test_quality_dimension_partial_coverage_excludes_missing_rows(self):
        rows = [_row(extras={"sigmos.noise": 4.0}), _row(extras={})]
        metrics = score_tts(rows)
        assert metrics["sigmos.noise"] == pytest.approx(4.0)


class TestRowQualityDimension:
    def test_present(self):
        assert row_quality_dimension(_row(extras={"sigmos.noise": 4.2}), "sigmos.noise") == 4.2

    def test_missing(self):
        assert row_quality_dimension(_row(), "sigmos.noise") is None

    def test_non_numeric_ignored(self):
        assert row_quality_dimension(
            _row(extras={"dnsmos.bak": "nope"}), "dnsmos.bak") is None

    def test_nan_guard(self):
        assert row_quality_dimension(
            _row(extras={"sigmos.col": float("nan")}), "sigmos.col") is None


class TestRowIntelligibilityWer:
    def test_present(self):
        assert row_intelligibility_wer(_row(extras={"intelligibility_wer": 0.3})) == 0.3

    def test_missing(self):
        assert row_intelligibility_wer(_row()) is None

    def test_non_numeric_ignored(self):
        assert row_intelligibility_wer(
            _row(extras={"intelligibility_wer": "nope"})) is None

    def test_nan_guard(self):
        assert row_intelligibility_wer(
            _row(extras={"intelligibility_wer": float("nan")})) is None


class TestRowIntelligibilityCer:
    def test_present(self):
        assert row_intelligibility_cer(_row(extras={"intelligibility_cer": 0.2})) == 0.2

    def test_missing(self):
        assert row_intelligibility_cer(_row()) is None

    def test_non_numeric_ignored(self):
        assert row_intelligibility_cer(
            _row(extras={"intelligibility_cer": "nope"})) is None

    def test_nan_guard(self):
        assert row_intelligibility_cer(
            _row(extras={"intelligibility_cer": float("nan")})) is None


class TestRowIntelligibilityAgreement:
    def test_present(self):
        assert row_intelligibility_agreement(
            _row(extras={"intelligibility_agreement": 0.6})) == 0.6

    def test_missing_defaults_to_one(self):
        # Legacy rows scored before panels existed default to full
        # agreement — nothing to disagree with.
        assert row_intelligibility_agreement(_row()) == 1.0

    def test_non_numeric_defaults_to_one(self):
        assert row_intelligibility_agreement(
            _row(extras={"intelligibility_agreement": "nope"})) == 1.0

    def test_nan_guard_defaults_to_one(self):
        assert row_intelligibility_agreement(
            _row(extras={"intelligibility_agreement": float("nan")})) == 1.0


class TestTtsSeedScore:
    def test_formula(self):
        row = _row(extras={"utmos": 4.0, "intelligibility_cer": 0.2,
                            "intelligibility_agreement": 0.5})
        # (4/5) * (1-0.2) * 0.5 = 0.32
        assert tts_seed_score(row) == pytest.approx(0.32)

    def test_defaults_agreement_to_one_when_absent(self):
        row = _row(extras={"utmos": 4.0, "intelligibility_cer": 0.2})
        # (4/5) * (1-0.2) * 1.0 = 0.64
        assert tts_seed_score(row) == pytest.approx(0.64)

    def test_missing_utmos_is_none(self):
        row = _row(extras={"intelligibility_cer": 0.2})
        assert tts_seed_score(row) is None

    def test_missing_cer_is_none(self):
        row = _row(extras={"utmos": 4.0})
        assert tts_seed_score(row) is None

    def test_caps_utmos_and_cer_before_combining(self):
        row = _row(extras={"utmos": 6.0, "intelligibility_cer": 1.5,
                            "intelligibility_agreement": 1.0})
        # utmos capped to 5 -> 5/5=1.0; cer capped to 1 -> (1-1)=0.0
        assert tts_seed_score(row) == pytest.approx(0.0)


class TestIntelligibilityScores:
    def test_perfect_transcript_zero_wer_cer(self):
        wer, cer = intelligibility_scores("hello there", "hello there")
        assert wer == 0.0
        assert cer == 0.0

    def test_mismatched_transcript_nonzero(self):
        wer, cer = intelligibility_scores("hello there", "goodbye world")
        assert wer > 0.0
        assert cer > 0.0

    def test_reuses_canonical_normalizer_punct_and_case_insensitive(self):
        wer, _cer = intelligibility_scores("Hello, there!", "hello there")
        assert wer == 0.0


class TestBenchmarkBoard:
    def test_intent_ranked_by_generalization_accuracy_desc(self):
        by_competitor = {
            "weak": [_row(competitor_id="weak", reference_intent="a",
                          prediction="b")],
            "strong": [_row(competitor_id="strong", reference_intent="a",
                            prediction="a")],
        }
        board = build_benchmark_board("intent", "d", "en-US", by_competitor, "t")
        assert board.primary_metric == "generalization_accuracy"
        assert [e.competitor_id for e in board.entries] == ["strong", "weak"]
        assert [e.rank for e in board.entries] == [1, 2]

    def test_stt_ranked_by_wer_asc(self):
        by_competitor = {
            "bad": [_row(competitor_id="bad", wer=0.9)],
            "good": [_row(competitor_id="good", wer=0.1)],
        }
        board = build_benchmark_board("stt", "d", "pt-PT", by_competitor, "t")
        assert [e.competitor_id for e in board.entries] == ["good", "bad"]

    def test_wake_word_ranked_by_error_rate_asc(self):
        by_competitor = {
            "noisy": [_row(competitor_id="noisy", label="negative",
                           prediction="detected")],
            "clean": [_row(competitor_id="clean", label="negative",
                           prediction="not_detected")],
        }
        board = build_benchmark_board("wake_word", "d", "en", by_competitor, "t")
        assert board.primary_metric == "error_rate"
        assert [e.competitor_id for e in board.entries] == ["clean", "noisy"]

    def test_no_competitors_yields_empty_board(self):
        board = build_benchmark_board("tts", "d", "en-US", {}, "t")
        assert board.entries == []

    def test_tts_ranked_by_utmos_desc(self):
        by_competitor = {
            "bad": [_row(competitor_id="bad", extras={"utmos": 2.0})],
            "good": [_row(competitor_id="good", extras={"utmos": 4.0})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        assert board.primary_metric == "utmos"
        assert [e.competitor_id for e in board.entries] == ["good", "bad"]

    def test_solo_zero_scored_entry_is_unranked_not_rank_one(self):
        # A fighter whose entire TTS run failed (no row got a usable utmos)
        # has n_scored == 0 and no "utmos" key at all. As the board's only
        # entry it must not land at rank 1 — it has no signal to rank on.
        by_competitor = {
            "phoonnx-dii-es-es": [
                _row(competitor_id="phoonnx-dii-es-es", extras={}),
            ],
        }
        board = build_benchmark_board("tts", "d", "es-ES", by_competitor, "t")
        entry = board.entries[0]
        assert entry.metrics.get("n_scored") == 0.0
        assert "utmos" not in entry.metrics
        assert entry.unranked is True
        assert entry.rank == 0
        assert entry.unranked_reason

    def test_zero_scored_entry_ranked_below_scored_peers(self):
        by_competitor = {
            "failed": [_row(competitor_id="failed", extras={})],
            "good": [_row(competitor_id="good", extras={"utmos": 4.0})],
        }
        board = build_benchmark_board("tts", "d", "en-US", by_competitor, "t")
        good = next(e for e in board.entries if e.competitor_id == "good")
        failed = next(e for e in board.entries if e.competitor_id == "failed")
        assert good.rank == 1
        assert good.unranked is False
        assert failed.unranked is True
        assert failed.rank == 0


class TestTtsSeedScoreWithoutAnAsrJudge:
    def test_judged_language_seed_is_unchanged(self):
        row = _row(modality="tts", lang="pt-PT", extras={
            "utmos": 3.5843, "intelligibility_wer": 0.4173,
            "intelligibility_cer": 0.2011, "intelligibility_agreement": 0.87,
        })
        # (3.5843 / 5) * (1 - 0.2011) * 0.87, computed independently.
        assert tts_seed_score(row) == pytest.approx(
            (3.5843 / 5.0) * (1.0 - 0.2011) * 0.87)

    def test_unjudgeable_language_seeds_on_utmos_alone(self):
        row = _row(modality="tts", lang="an-ES", extras={
            "utmos": 3.3061, "intelligibility": "not_available",
            "intelligibility_wer": None, "intelligibility_cer": None,
            "intelligibility_judge": "none",
        })
        assert tts_seed_score(row) == pytest.approx(3.3061 / 5.0)

    def test_missing_cer_without_the_marker_is_still_not_computable(self):
        row = _row(modality="tts", lang="pt-PT", extras={"utmos": 3.5})
        assert tts_seed_score(row) is None
