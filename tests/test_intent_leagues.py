"""Intent leagues are keyed by training regime.

A fighter's league follows from what it needs before it can answer: nothing
(zero-shot), a training pass at boot (online), or a pretrained artefact
(offline). Keyword supervision keeps its own league. These tests pin the
derivation, the registry's refusal to file a fighter anywhere else, the
label-set eligibility rule for pretrained fighters, and the data-driven
hiding of a league nobody has swept yet.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from arena.models import leagues
from registry.loaders import load_dataset, validate_registry
from registry.schemas import (
    CompetitorDef,
    Modality,
    TrainingRegime,
    expected_league,
    pipeline_regime,
)
from runner.intent_bench import check_league, trained_on


def _fighter(pipeline, regime, modality, **kw):
    intents = {"pipeline": pipeline}
    intents.update(kw.pop("stage_config", {}))
    return CompetitorDef(
        competitor_id=kw.pop("competitor_id", "fixture"),
        modality=modality,
        config={"intents": intents},
        langs=["en-US"],
        training_regime=regime,
        **kw,
    )


class TestRegimeDerivation:
    def test_fusion_takes_its_heaviest_stage(self):
        stages = ["ovos-m2v-pipeline-high", "ovos-padatious-pipeline-plugin-medium"]
        assert pipeline_regime(
            ["ovos-m2v-pipeline", "ovos-padatious-pipeline-plugin"]
        ) is TrainingRegime.OFFLINE
        assert expected_league(
            ["ovos-m2v-pipeline", "ovos-padatious-pipeline-plugin"]
        ) is Modality.INTENT_OFFLINE
        assert stages  # the same pair as the shipped m2v-first fighter

    def test_zero_shot_stage_does_not_lift_an_online_fusion(self):
        plugins = ["ovos-m2v-prototype-pipeline", "ovos-nebulento-pipeline-plugin"]
        assert pipeline_regime(plugins) is TrainingRegime.ONLINE

    def test_m2v_prototype_mode_is_zero_shot(self):
        cfg = {"ovos-m2v-pipeline": {"mode": "prototype"}}
        assert pipeline_regime(["ovos-m2v-pipeline"], cfg) is TrainingRegime.ZERO_SHOT
        assert pipeline_regime(["ovos-m2v-pipeline"]) is TrainingRegime.OFFLINE

    def test_all_keyword_fighter_is_a_keyword_fighter(self):
        assert expected_league(
            ["ovos-adapt-pipeline-plugin", "ovos-palavreado-pipeline-plugin"]
        ) is Modality.INTENT_KEYWORD

    def test_bench_rejects_a_fighter_filed_in_the_wrong_league(self):
        comp = _fighter(
            ["ovos-padatious-pipeline-plugin-medium"],
            TrainingRegime.ONLINE, Modality.INTENT_ONLINE,
        )
        check_league(comp)
        comp.modality = Modality.INTENT_ZERO_SHOT
        with pytest.raises(ValueError, match="intent_online"):
            check_league(comp)


class TestSchemaRefusals:
    def test_regime_must_match_the_stages(self):
        with pytest.raises(ValidationError, match="stages need 'offline'"):
            _fighter(["ovos-m2v-pipeline-medium"],
                     TrainingRegime.ONLINE, Modality.INTENT_ONLINE)

    def test_league_must_match_the_regime(self):
        with pytest.raises(ValidationError, match="belongs in the 'intent_online'"):
            _fighter(["ovos-padatious-pipeline-plugin-medium"],
                     TrainingRegime.ONLINE, Modality.INTENT_OFFLINE,
                     label_set=[])

    def test_intent_fighter_must_declare_a_regime(self):
        with pytest.raises(ValidationError, match="must declare training_regime"):
            CompetitorDef(
                competitor_id="no-regime", modality=Modality.INTENT_ONLINE,
                config={"intents": {"pipeline": ["ovos-padatious-pipeline-plugin-medium"]}},
                langs=["en-US"],
            )

    def test_offline_fighter_must_declare_a_label_set(self):
        with pytest.raises(ValidationError, match="must declare label_set"):
            _fighter(["ovos-m2v-pipeline-medium"],
                     TrainingRegime.OFFLINE, Modality.INTENT_OFFLINE)

    def test_label_set_is_offline_only(self):
        with pytest.raises(ValidationError, match="only meaningful for"):
            _fighter(["ovos-padatious-pipeline-plugin-medium"],
                     TrainingRegime.ONLINE, Modality.INTENT_ONLINE,
                     label_set=["ovos-intents-v5"])

    def test_unknown_label_set_dataset_is_a_registry_error(self, tmp_path):
        comp_dir = tmp_path / "competitors" / "intent_offline"
        comp_dir.mkdir(parents=True)
        (comp_dir / "ghost.json").write_text(json.dumps({
            "competitor_id": "ghost",
            "modality": "intent_offline",
            "config": {"intents": {
                "pipeline": ["ovos-m2v-pipeline-medium"],
                "ovos-m2v-pipeline": {"model": "x", "mode": "classifier"},
            }},
            "langs": ["en-US"],
            "training_regime": "offline",
            "label_set": ["no-such-corpus"],
        }))
        errors = validate_registry(registry_root=tmp_path)
        assert any("unknown dataset_id 'no-such-corpus'" in e for e in errors)


class TestLabelSetEligibility:
    def _offline(self, label_set):
        return _fighter(
            ["ovos-m2v-pipeline-medium"], TrainingRegime.OFFLINE,
            Modality.INTENT_OFFLINE, label_set=label_set,
        )

    def test_offline_fighter_runs_only_on_its_own_label_set(self):
        comp = self._offline(["ovos-intents-v5"])
        assert trained_on(comp, "ovos-intents-v5")
        assert not trained_on(comp, "intents-for-eval")

    def test_a_fighter_matching_no_corpus_is_unranked_everywhere(self):
        comp = self._offline([])
        assert not trained_on(comp, "ovos-intents-v5")

    def test_online_fighters_are_never_restricted(self):
        comp = _fighter(["ovos-padatious-pipeline-plugin-medium"],
                        TrainingRegime.ONLINE, Modality.INTENT_ONLINE)
        assert trained_on(comp, "any-corpus-at-all")


class TestLeagueVisibility:
    def test_a_league_with_no_ranked_board_is_not_a_tab(self):
        shown = {entry["id"] for entry in leagues({"intent_online", "stt"})}
        assert shown == {"intent_online", "stt"}
        assert Modality.INTENT_KEYWORD.value not in shown

    def test_a_league_with_a_ranked_board_is_a_tab(self):
        shown = {entry["id"] for entry in leagues({"intent_keyword"})}
        assert shown == {"intent_keyword"}

    def test_unfiltered_leagues_cover_every_intent_league(self):
        ids = {entry["id"] for entry in leagues()}
        assert {"intent_zero_shot", "intent_online",
                "intent_offline", "intent_keyword"} <= ids


class TestV5Corpus:
    def test_eval_corpus_is_pinned_over_forty_locales(self):
        ds = load_dataset("intent", "ovos-intents-v5")
        assert len(ds.langs) == 40
        assert ds.source.revision == "70798090ab515a6d8c78d7c260decf9a6b3e72b4"
        assert ds.reference_fields["bucket"] == "split"
        assert ds.train_datasets == {"template": "ovos-intents-v5-templates-train"}

    def test_iso_639_3_locales_are_kept_bare(self):
        ds = load_dataset("intent", "ovos-intents-v5")
        assert {"arb", "kab"} <= set(ds.langs)
        assert all("-" in tag or len(tag) == 3 for tag in ds.langs)

    def test_template_corpus_matches_the_eval_locales(self):
        train = load_dataset("intent_template", "ovos-intents-v5-templates-train")
        assert train.paradigm == "template"
        assert train.role == "train"
        assert train.langs == load_dataset("intent", "ovos-intents-v5").langs


class TestBucketRoles:
    """The ranked metric must mean the same thing on every corpus, whatever
    each one calls its buckets."""

    def _row(self, bucket, correct, dataset_id="ovos-intents-v5"):
        from arena.models import PredictionRow

        return PredictionRow(
            competitor_id="fixture", sample_id=f"en-US/{bucket}-{correct}",
            dataset_id=dataset_id, lang="en-US", plugin_id="p",
            modality="intent_online", utterance="turn on the lights",
            reference_intent="lights:on",
            prediction="lights:on" if correct else "weather:now",
            exact_match=correct, bucket=bucket,
        )

    def test_v5_id_test_rows_do_not_count_as_generalization(self):
        from arena.metrics import score_intent

        rows = [self._row("id_test", True) for _ in range(8)]
        rows += [self._row("ood", False) for _ in range(4)]
        metrics = score_intent(rows)
        assert metrics["generalization_accuracy"] == 0.0
        assert metrics["accuracy"] == round(8 / 12, 4)
        assert metrics["acc_id_test"] == 1.0

    def test_declared_generalization_bucket_is_ranked(self):
        from arena.metrics import score_intent

        rows = [self._row("id_test", False) for _ in range(4)]
        rows += [self._row("ood", True) for _ in range(4)]
        assert score_intent(rows)["generalization_accuracy"] == 1.0

    def test_a_corpus_that_declares_nothing_keeps_the_default_buckets(self):
        from arena.metrics import score_intent

        rows = [self._row("template", True, dataset_id="snips") for _ in range(4)]
        rows += [self._row("paraphrase", False, dataset_id="snips") for _ in range(4)]
        assert score_intent(rows)["generalization_accuracy"] == 0.0

    def test_both_intent_corpora_declare_their_buckets(self):
        v5 = load_dataset("intent", "ovos-intents-v5")
        ife = load_dataset("intent", "intents-for-eval")
        assert v5.bucket_roles == {"id_test": "in_distribution",
                                   "ood": "generalization"}
        assert ife.bucket_roles["template"] == "in_distribution"
        assert ife.bucket_roles["far_ood"] == "generalization"


class TestUnexpandedUtteranceGuard:
    """An eval row stands for something a person said. Template markup in
    that column means the corpus was never expanded, and scoring it measures
    nothing."""

    def _dataset(self):
        return load_dataset("intent", "ovos-intents-v5")

    def test_markup_row_is_refused(self):
        from runner.intent_bench import (
            UnexpandedUtterances,
            reject_unexpanded_utterances,
        )

        rows = [
            {"utterance": "add milk to my list", "expected_intent": "x"},
            {"utterance": "(create|add) list [items]", "expected_intent": "x"},
        ]
        with pytest.raises(UnexpandedUtterances, match="row 1"):
            reject_unexpanded_utterances(self._dataset(), "en-US", rows)

    def test_slot_placeholder_is_refused(self):
        from runner.intent_bench import (
            UnexpandedUtterances,
            reject_unexpanded_utterances,
        )

        rows = [{"utterance": "set an alarm for {time}", "expected_intent": "x"}]
        with pytest.raises(UnexpandedUtterances):
            reject_unexpanded_utterances(self._dataset(), "en-US", rows)

    def test_plain_utterances_pass(self):
        from runner.intent_bench import reject_unexpanded_utterances

        rows = [
            {"utterance": "remove the dentist appointment", "expected_intent": "x"},
            {"utterance": "what's the weather like (today)", "expected_intent": "x"},
        ]
        reject_unexpanded_utterances(self._dataset(), "en-US", rows)


class TestModelPinIsHonoured:
    """A declared model revision has to reach the download, not just the row:
    a row stamping a sha the run never fetched is false provenance."""

    def _fighter(self, revision="deadbeef"):
        return _fighter(
            ["ovos-m2v-pipeline-medium"], TrainingRegime.OFFLINE,
            Modality.INTENT_OFFLINE, label_set=["ovos-intents-v5"],
            model="OpenVoiceOS/some-head", model_revision=revision,
            stage_config={"ovos-m2v-pipeline": {
                "model": "OpenVoiceOS/some-head", "mode": "classifier"}},
        )

    def test_the_declared_revision_is_downloaded_and_becomes_the_model_path(
        self, monkeypatch, tmp_path
    ):
        from runner import intent_bench

        snapshot = tmp_path / "snapshots" / "cafebabe0000"
        snapshot.mkdir(parents=True)
        seen = {}

        def fake_snapshot_download(repo_id, revision, **kw):
            seen["repo_id"] = repo_id
            seen["revision"] = revision
            return str(snapshot)

        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", fake_snapshot_download)
        competitor = self._fighter()
        intents = {k: dict(v) if isinstance(v, dict) else v
                   for k, v in competitor.config["intents"].items()}
        resolved = intent_bench.resolve_model_pin(competitor, intents)

        assert seen == {"repo_id": "OpenVoiceOS/some-head",
                        "revision": "deadbeef"}
        assert intents["ovos-m2v-pipeline"]["model"] == str(snapshot)
        # the sha the snapshot resolved to, never the declared one
        assert resolved == "cafebabe0000"

    def test_an_unresolvable_pin_fails_loudly(self, monkeypatch):
        from runner.intent_bench import UnresolvableModelPin, resolve_model_pin

        def boom(repo_id, revision, **kw):
            raise OSError("404")

        monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
        competitor = self._fighter()
        with pytest.raises(UnresolvableModelPin, match="cannot resolve"):
            resolve_model_pin(competitor, dict(competitor.config["intents"]))

    def test_a_fighter_without_a_pin_is_left_alone(self, monkeypatch):
        from runner.intent_bench import resolve_model_pin

        def boom(*a, **kw):
            raise AssertionError("must not download")

        monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
        competitor = _fighter(
            ["ovos-padatious-pipeline-plugin-medium"],
            TrainingRegime.ONLINE, Modality.INTENT_ONLINE,
        )
        assert resolve_model_pin(competitor, {}) is None


class TestMeasuredLabelOverlap:
    """`label_set` is a claim; only the loaded model can settle it."""

    class _Pipeline:
        def __init__(self, classes):
            model = type("M", (), {"classes_": list(classes)})()
            self.plugins = {"stage": type("P", (), {"model": model})()}

    def test_overlap_counts_the_labels_the_model_can_emit(self):
        from runner.intent_bench import model_label_overlap

        pipeline = self._Pipeline(["skill:a", "skill:b", "skill:z"])
        assert model_label_overlap(pipeline, {"skill:a", "skill:b"}) == 2

    def test_zero_overlap_is_reported_as_zero_not_none(self):
        from runner.intent_bench import model_label_overlap

        pipeline = self._Pipeline(["other:a"])
        assert model_label_overlap(pipeline, {"skill:a"}) == 0

    def test_an_engine_with_no_class_list_is_not_checked(self):
        from runner.intent_bench import model_label_overlap

        pipeline = type("P", (), {"plugins": {"stage": object()}})()
        assert model_label_overlap(pipeline, {"skill:a"}) is None


class TestMinimumBoardSamples:
    """A rank computed from a handful of utterances reads as a result and is
    not one."""

    def _rows(self, n, competitor="fixture"):
        from arena.models import PredictionRow

        return [
            PredictionRow(
                competitor_id=competitor, sample_id=f"en-US/{i:05d}",
                dataset_id="ovos-intents-v5", lang="en-US", plugin_id="p",
                modality="intent_online", utterance="turn on the lights",
                reference_intent="lights:on", prediction="lights:on",
                exact_match=True, bucket="ood",
            )
            for i in range(n)
        ]

    def test_a_thin_board_is_unranked_with_the_reason(self):
        from arena.metrics import MIN_BOARD_SAMPLES, build_benchmark_board

        board = build_benchmark_board(
            "intent_online", "ovos-intents-v5", "en-US",
            {"fixture": self._rows(5)}, "t",
        )
        entry = board.entries[0]
        assert entry.unranked
        assert entry.rank == 0
        assert entry.unranked_reason.startswith("too_few_samples")
        assert str(MIN_BOARD_SAMPLES) in entry.unranked_reason

    def _mixed_rows(self, in_distribution, generalization,
                    competitor="fixture"):
        """Rows split between an in-distribution bucket and a ranked one."""
        from arena.models import PredictionRow

        buckets = ["in_distribution"] * in_distribution + ["ood"] * generalization
        return [
            PredictionRow(
                competitor_id=competitor, sample_id=f"en-US/{i:05d}",
                dataset_id="ovos-intents-v5", lang="en-US", plugin_id="p",
                modality="intent_online", utterance="turn on the lights",
                reference_intent="lights:on", prediction="lights:on",
                exact_match=True, bucket=bucket,
            )
            for i, bucket in enumerate(buckets)
        ]

    def test_the_floor_counts_the_rows_the_rank_is_computed_from(self):
        """A mostly in-distribution run clears a row-count floor while the
        ranked metric rests on a handful of rows."""
        from arena.metrics import MIN_BOARD_SAMPLES, build_benchmark_board

        rows = self._mixed_rows(in_distribution=100, generalization=3)
        board = build_benchmark_board(
            "intent_online", "ovos-intents-v5", "en-US",
            {"fixture": rows}, "t",
        )
        entry = board.entries[0]
        assert entry.samples > MIN_BOARD_SAMPLES
        assert entry.unranked
        assert entry.rank == 0
        assert entry.unranked_reason.startswith("too_few_samples")
        assert entry.metrics["generalization_n"] == 3

    def test_enough_ranked_rows_ranks_whatever_the_bucket_mix(self):
        from arena.metrics import MIN_BOARD_SAMPLES, build_benchmark_board

        rows = self._mixed_rows(
            in_distribution=100, generalization=MIN_BOARD_SAMPLES)
        board = build_benchmark_board(
            "intent_online", "ovos-intents-v5", "en-US",
            {"fixture": rows}, "t",
        )
        entry = board.entries[0]
        assert entry.metrics["generalization_n"] == MIN_BOARD_SAMPLES
        assert not entry.unranked
        assert entry.rank == 1

    def test_a_full_board_ranks_normally(self):
        from arena.metrics import MIN_BOARD_SAMPLES, build_benchmark_board

        board = build_benchmark_board(
            "intent_online", "ovos-intents-v5", "en-US",
            {"fixture": self._rows(MIN_BOARD_SAMPLES)}, "t",
        )
        entry = board.entries[0]
        assert not entry.unranked
        assert entry.rank == 1

    def test_a_thin_lang_seeds_no_battles(self, tmp_path):
        from arena.cli import main

        preds = tmp_path / "predictions" / "en-US"
        preds.mkdir(parents=True)
        for competitor in ("padatious-medium", "adapt-medium"):
            rows = [
                {
                    "competitor_id": competitor,
                    "sample_id": f"en-US/{i:05d}",
                    "dataset_id": "ovos-intents-v5", "lang": "en-US",
                    "modality": "intent_online", "plugin_id": "p",
                    "utterance": f"utterance {i}",
                    "reference_intent": "lights:on",
                    "prediction": "lights:on" if competitor.startswith("pad")
                    else "weather:now",
                    "exact_match": competitor.startswith("pad"),
                }
                for i in range(5)
            ]
            (preds / f"{competitor}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n")
        out = tmp_path / "data"
        with pytest.raises(SystemExit) as exc:
            main(["assemble", "--predictions", str(preds.parent),
                  "--output", str(out), "--modality", "intent_online"])
        assert exc.value.code == 0
        assert not list(out.glob("battles-intent-*.json"))
        assert not list(out.glob("elo-seed-intent-*.json"))
