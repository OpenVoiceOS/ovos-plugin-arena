"""Unit tests for runner.intent_bench — pure helpers, no engines needed."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import runner.intent_bench as intent_bench_mod
from registry.loaders import load_competitor
from registry.loaders import load_dataset as load_dataset_def
from registry.schemas import DatasetDef
from runner.intent_bench import (
    fetch_hf_classification_rows,
    fetch_rows,
    load_stt_engine,
    make_row,
    needed_paradigms,
    normalize_hierarchical_label,
    results_repo_for,
    run_benchmark,
    split_name,
)


class TestNormalizeHierarchicalLabel:
    """MTOP's own intent column is prefixed with a generic parser tag
    ('IN:SEND_MESSAGE') rather than the real domain; the arena scores
    against a 'domain:intent' string (see intents-for-eval)."""

    def test_strips_generic_prefix_and_lowercases(self):
        assert normalize_hierarchical_label(
            "messaging", "IN:SEND_MESSAGE"
        ) == "messaging:send_message"

    def test_already_bare_intent_still_lowercased(self):
        assert normalize_hierarchical_label("weather", "GET_WEATHER") == (
            "weather:get_weather"
        )

    def test_domain_casing_and_whitespace_normalized(self):
        assert normalize_hierarchical_label(
            " Messaging ", "IN:GET_MESSAGE"
        ) == "messaging:get_message"


class TestNaming:
    def test_repo_per_benchmark_modality(self):
        assert results_repo_for("intent_template", "intents-for-eval") == (
            "OpenVoiceOS/ovos-intent-template-bench-intents-for-eval"
        )
        assert results_repo_for("intent", "massive-templates") == (
            "OpenVoiceOS/ovos-intent-bench-massive-templates"
        )

    def test_split_name_word_chars_only(self):
        assert split_name("pt-PT") == "pt_PT"
        assert split_name("en-US") == "en_US"


class TestEligibility:
    def test_single_engine_paradigm(self):
        comp = load_competitor("intent_keyword", "adapt-medium")
        assert needed_paradigms(comp) == {"keyword"}

    def test_fusion_needs_both(self):
        comp = load_competitor("intent_online", "frankenparse")
        assert needed_paradigms(comp) == {"template", "keyword"}

    def test_template_pure_fusion(self):
        comp = load_competitor("intent_online", "nebulatious")
        assert needed_paradigms(comp) == {"template"}


class TestMakeRow:
    def _row(self, prediction, reference="media:play_song"):
        comp = load_competitor("intent_online", "padacioso-medium")
        return make_row(
            comp, "intents-for-eval", "en-US", 7,
            {"utterance": "play a song", "expected_intent": reference,
             "expected_slots": {"song": "a song"}, "split": "template"},
            prediction, {}, None, 1.5,
            "ovos-padacioso-pipeline-plugin-medium", "rev123",
        )

    def test_row_contract(self):
        row = self._row("media:play_song")
        assert row["sample_id"] == "en-US/00007"
        assert row["modality"] == "intent_online"
        assert row["dataset_revision"] == "rev123"
        assert row["stage"] == "ovos-padacioso-pipeline-plugin-medium"
        assert row["exact_match"] is True

    def test_wrong_prediction(self):
        assert self._row("media:stop")["exact_match"] is False

    def test_ood_correct_rejection(self):
        row = self._row(None, reference=None)
        assert row["exact_match"] is True
        assert row["reference_intent"] is None


class TestMakeRowDomainGranularity:
    """meteocat and other domain-only corpora set granularity='domain':
    a prediction is correct as long as its domain (text before the first
    ':') matches the bare domain reference — the full intent name is
    unconstrained since these corpora carry no per-intent label."""

    def _row(self, prediction, reference="weather", granularity="domain"):
        comp = load_competitor("intent_online", "padacioso-medium")
        return make_row(
            comp, "meteocat", "ca-ES", 3,
            {"utterance": "quin temps fara demà", "expected_intent": reference,
             "split": "test"},
            prediction, {}, None, 1.5,
            "ovos-padacioso-pipeline-plugin-medium", "rev123",
            granularity=granularity,
        )

    def test_domain_prediction_scores_correct(self):
        # a domain/hierarchical fighter's fired intent still carries the
        # sub-intent after the domain — e.g. "weather:current" — but
        # meteocat only asserts the domain part.
        row = self._row("weather:current")
        assert row["exact_match"] is True

    def test_wrong_domain_scores_incorrect(self):
        row = self._row("music:play_song")
        assert row["exact_match"] is False

    def test_bare_domain_prediction_still_matches(self):
        row = self._row("weather")
        assert row["exact_match"] is True

    def test_ood_rejection_unaffected_by_granularity(self):
        row = self._row(None, reference=None)
        assert row["exact_match"] is True

    def test_intent_granularity_is_the_default_and_unchanged(self):
        # regression guard: omitting granularity keeps exact full-string
        # comparison, so a same-domain-different-intent prediction that
        # would score 'correct' under domain granularity still scores
        # 'incorrect' under the (default) intent granularity.
        comp = load_competitor("intent_online", "padacioso-medium")
        row = make_row(
            comp, "intents-for-eval", "en-US", 7,
            {"utterance": "play a song", "expected_intent": "media:play_song",
             "split": "template"},
            "media:stop", {}, None, 1.5,
            "ovos-padacioso-pipeline-plugin-medium", "rev123",
        )
        assert row["exact_match"] is False


class _FakeClassLabel:
    """Mimics ``datasets.ClassLabel`` well enough for int2str()."""

    def __init__(self, names):
        self.names = names

    def int2str(self, value):
        return self.names[value]


class _FakeHFDataset:
    """Minimal stand-in for a ``datasets.Dataset`` — iterable + .features."""

    def __init__(self, rows, features):
        self._rows = rows
        self.features = features

    def __iter__(self):
        return iter(self._rows)

    def filter(self, predicate):
        return _FakeHFDataset(
            [r for r in self._rows if predicate(r)], self.features
        )


class TestFetchHFClassificationRows:
    """runner.intent_bench.fetch_hf_classification_rows — the path absorbed
    text-classification datasets (SNIPS/BANKING77/CLINC150/MASSIVE) go
    through instead of the file_pattern JSONL path."""

    def test_eval_decodes_int_labels_via_features(self):
        ds_def = load_dataset_def("intent", "banking77")
        rows = [
            {"text": "why was my card declined", "label": 0},
            {"text": "how do I top up", "label": 1},
        ]
        feat = _FakeClassLabel(["card_declined", "top_up"])
        fake_ds = _FakeHFDataset(rows, {"label": feat})
        # banking77's reference_fields point at 'label_text' (already a
        # string) in the real mirror; point the fake at the int column
        # 'label' instead, purely to exercise ClassLabel decoding.
        ds_def.reference_fields = {"utterance": "text", "intent": "label"}
        with patch("datasets.load_dataset", return_value=fake_ds) as mocked:
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        mocked.assert_called_once_with(
            "mteb/banking77", name=None, split="test", revision="rev123",
        )
        assert out == [
            {"utterance": "why was my card declined",
             "expected_intent": "card_declined", "split": "test"},
            {"utterance": "how do I top up",
             "expected_intent": "top_up", "split": "test"},
        ]

    def test_eval_accepts_plain_string_labels(self):
        ds_def = load_dataset_def("intent", "snips")
        rows = [{"text": "play some jazz", "category": "PlayMusic"}]
        fake_ds = _FakeHFDataset(rows, {"category": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"utterance": "play some jazz", "expected_intent": "PlayMusic",
             "split": "test"},
        ]

    def test_train_role_emits_intent_id_and_template(self):
        ds_def = load_dataset_def("intent_template", "snips-train")
        rows = [{"text": "play some jazz", "category": "PlayMusic"}]
        fake_ds = _FakeHFDataset(rows, {"category": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [{"intent_id": "PlayMusic", "template": "play some jazz"}]

    def test_oos_label_routed_to_far_ood_on_eval(self):
        ds_def = load_dataset_def("intent", "clinc150")
        rows = [
            {"text": "in scope question", "intent": 0},
            {"text": "totally unrelated nonsense", "intent": 1},
        ]
        feat = _FakeClassLabel(["book_flight", "oos"])
        fake_ds = _FakeHFDataset(rows, {"intent": feat})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"utterance": "in scope question",
             "expected_intent": "book_flight", "split": "test"},
            {"utterance": "totally unrelated nonsense",
             "expected_intent": None, "split": "far_ood"},
        ]

    def test_oos_label_dropped_on_train(self):
        ds_def = load_dataset_def("intent_template", "clinc150-train")
        rows = [
            {"text": "in scope question", "intent": 0},
            {"text": "totally unrelated nonsense", "intent": 1},
        ]
        feat = _FakeClassLabel(["book_flight", "oos"])
        fake_ds = _FakeHFDataset(rows, {"intent": feat})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"intent_id": "book_flight", "template": "in scope question"},
        ]

    def test_fetch_rows_dispatches_to_classification_path(self):
        """fetch_rows() routes file_pattern-less sources to the classification
        loader instead of hf_hub_download (the JSONL path)."""
        ds_def = load_dataset_def("intent", "snips")
        with patch(
            "runner.intent_bench.fetch_hf_classification_rows",
            return_value=[{"utterance": "x", "expected_intent": "y",
                            "split": "test"}],
        ) as mocked:
            out = fetch_rows(ds_def, "en-US", "rev123")
        mocked.assert_called_once_with(ds_def, "en-US", "rev123")
        assert out == [{"utterance": "x", "expected_intent": "y", "split": "test"}]

    def test_missing_reference_fields_raise(self):
        ds_def = load_dataset_def("intent", "snips")
        ds_def.reference_fields = {}
        with patch("datasets.load_dataset"):
            try:
                fetch_hf_classification_rows(ds_def, "en-US", "rev123")
            except ValueError as exc:
                assert "reference_fields" in str(exc)
            else:
                raise AssertionError("expected ValueError")

    def test_domain_label_supplies_constant_reference_when_no_intent_column(self):
        """meteocat has no per-intent label column at all — every row's
        reference comes from domain_label instead of reference_fields['intent']."""
        ds_def = load_dataset_def("intent", "meteocat")
        assert ds_def.reference_fields.get("intent") is None
        rows = [
            {"instruction": "Quin temps farà demà a Girona?"},
            {"instruction": "Plourà aquesta tarda a Vic?"},
        ]
        fake_ds = _FakeHFDataset(rows, {})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "ca-ES", "rev123")
        assert out == [
            {"utterance": "Quin temps farà demà a Girona?",
             "expected_intent": "weather", "split": "test"},
            {"utterance": "Plourà aquesta tarda a Vic?",
             "expected_intent": "weather", "split": "test"},
        ]

    def test_missing_intent_and_domain_label_still_raises(self):
        """Without an intent column AND without domain_label set, the
        source genuinely has no usable reference — still an error."""
        ds_def = load_dataset_def("intent", "meteocat").model_copy(deep=True)
        ds_def.domain_label = None
        with patch("datasets.load_dataset"):
            try:
                fetch_hf_classification_rows(ds_def, "ca-ES", "rev123")
            except ValueError as exc:
                assert "reference_fields" in str(exc)
            else:
                raise AssertionError("expected ValueError")

    def test_id_field_dedupes_duplicated_rows(self):
        """source.id_field is general hygiene, not a MASSIVE special case —
        no currently-registered dataset needs it (their sources ship each
        row once), but a source that turns it on must have duplicates
        collapsed to one row per id, keeping first occurrence. Built off
        a synthetic source rather than a registered one, since none of
        the absorbed datasets (SNIPS/BANKING77/CLINC150) require dedup."""
        ds_def = load_dataset_def("intent", "banking77").model_copy(deep=True)
        ds_def.source.id_field = "id"
        base_rows = [
            {"id": "1", "text": "why was my card declined", "label_text": "card_declined"},
            {"id": "2", "text": "how do I top up", "label_text": "top_up"},
        ]
        duplicated = base_rows * 3  # e.g. a mirror shipping every row 3x
        fake_ds = _FakeHFDataset(duplicated, {"label_text": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert out == [
            {"utterance": "why was my card declined",
             "expected_intent": "card_declined", "split": "test"},
            {"utterance": "how do I top up",
             "expected_intent": "top_up", "split": "test"},
        ]

    def test_id_field_dedup_is_noop_on_clean_sources(self):
        """Sources without id_field set (SNIPS/BANKING77/CLINC150 as
        registered) are unaffected — dedup only fires when id_field is set."""
        ds_def = load_dataset_def("intent", "banking77")
        assert ds_def.source.id_field is None
        rows = [
            {"text": "why was my card declined", "label_text": "card_declined"},
            {"text": "how do I top up", "label_text": "top_up"},
        ]
        fake_ds = _FakeHFDataset(rows, {"label_text": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert len(out) == 2

    def test_lang_field_filters_and_domain_field_normalizes(self):
        """MTOP ships every language mixed into one split with a 'lang'
        column instead of a per-language config; source.lang_field/
        lang_value filters to the registered language, and
        reference_fields['domain'] composes 'domain:intent' labels."""
        ds_def = load_dataset_def("intent", "mtop-de-DE")
        assert ds_def.source.lang_field == "lang"
        assert ds_def.source.lang_value == "de_XX"
        rows = [
            {"question": "Antworte im Thread", "intent": "IN:SEND_MESSAGE",
             "domain": "messaging", "lang": "de_XX"},
            {"question": "play a song", "intent": "IN:PLAY_MUSIC",
             "domain": "music", "lang": "en_XX"},
        ]
        fake_ds = _FakeHFDataset(rows, {})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "de-DE", "rev123")
        assert out == [
            {"utterance": "Antworte im Thread",
             "expected_intent": "messaging:send_message", "split": "test"},
        ]

    def test_missing_id_field_treated_as_a_dedup_key(self):
        """Rows genuinely missing the id column collapse to a single
        None-keyed row rather than raising — a defensive edge case, not
        the expected shape for any registered dataset."""
        ds_def = load_dataset_def("intent", "banking77").model_copy(deep=True)
        ds_def.source.id_field = "id"
        rows = [
            {"text": "a", "label_text": "x"},
            {"text": "b", "label_text": "y"},
        ]
        fake_ds = _FakeHFDataset(rows, {"label_text": None})
        with patch("datasets.load_dataset", return_value=fake_ds):
            out = fetch_hf_classification_rows(ds_def, "en-US", "rev123")
        assert len(out) == 1


def test_train_fetch_uses_train_repo_revision():
    """Train corpora pin their own repo's revision, not the eval repo's."""
    from pathlib import Path
    from types import SimpleNamespace

    from runner import intent_bench

    eval_def = SimpleNamespace(
        source=SimpleNamespace(hf_id="org/eval-repo", revision="main",
                               file_pattern=None, subset=None, split="test"),
        train_datasets={"template": "train-ds"},
        input="text",
    )
    train_def = SimpleNamespace(
        source=SimpleNamespace(hf_id="org/train-repo", revision="main",
                               file_pattern="{lang}/train.jsonl",
                               subset=None, split="train"),
    )
    seen = {}

    def fake_resolve(hf_id, revision):
        return {"org/eval-repo": "EVALSHA", "org/train-repo": "TRAINSHA"}[hf_id]

    def fake_fetch(dataset_def, lang, revision):
        seen[dataset_def.source.hf_id] = revision
        if dataset_def is eval_def:
            return [{"utterance": "quin temps fa", "expected_intent": "weather"}]
        return []

    competitor = SimpleNamespace(competitor_id="x")
    with patch.object(intent_bench, "resolve_revision", side_effect=fake_resolve), \
         patch.object(intent_bench, "fetch_rows", side_effect=fake_fetch), \
         patch.object(intent_bench, "needed_paradigms", return_value={"template"}), \
         patch.object(intent_bench, "done_samples", return_value=set()):
        try:
            intent_bench.run_competitor_lang(
                competitor, "meteocat", "ca-ES", eval_def,
                {"template": train_def}, "EVALSHA",
                Path("/nonexistent/out.jsonl"))
        except Exception:
            # pipeline construction fails on the stub competitor —
            # irrelevant: the train fetch has happened by then.
            pass
    assert seen.get("org/train-repo") == "TRAINSHA", \
        "train corpus must pin its own repo's sha"


def test_run_competitor_lang_survives_one_crashing_utterance(tmp_path):
    """One raising ``pipeline.predict`` call must not abort the cell — the
    remaining rows are still written and the failure is counted, mirroring
    ``runner.media_bench.run_competitor_lang``'s per-sample try/except."""
    from types import SimpleNamespace

    from runner import intent_bench

    eval_def = SimpleNamespace(
        source=SimpleNamespace(hf_id="org/eval-repo", revision="main",
                               file_pattern=None, subset=None, split="test"),
        train_datasets={}, input="text", reference_granularity="flat",
    )
    test_rows = [
        {"utterance": "quin temps fa", "expected_intent": "weather"},
        {"utterance": "atura la musica", "expected_intent": "stop"},
        {"utterance": "posa una alarma", "expected_intent": "alarm"},
    ]

    class ExplodingPipeline:
        stage_names = ["stub"]

        def __init__(self, *a, **kw):
            pass

        def train(self, *a, **kw):
            pass

        def predict(self, utterance):
            if utterance == "atura la musica":
                raise RuntimeError("boom")
            return utterance, {}, 1.0, 1.0, "stub"

    competitor = SimpleNamespace(
        competitor_id="x", config={"intents": {}}, model_revision=None,
        training_regime=None,
        pipeline_plugins=[], modality=SimpleNamespace(value="intent_online"),
        plugin="stub-plugin", pipeline="stub-pipeline",
    )
    out_path = tmp_path / "out.jsonl"

    with patch.object(intent_bench, "resolve_revision", return_value="EVALSHA"), \
         patch.object(intent_bench, "fetch_rows", return_value=test_rows), \
         patch.object(intent_bench, "needed_paradigms", return_value=set()), \
         patch.object(intent_bench, "done_samples", return_value=set()), \
         patch.object(intent_bench, "IntentPipeline", ExplodingPipeline):
        written = intent_bench.run_competitor_lang(
            competitor, "meteocat", "ca-ES", eval_def, {}, "EVALSHA", out_path)

    assert written == 2, "the two non-crashing rows must still be written"
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    utterances = [json.loads(line)["utterance"] for line in lines]
    assert "atura la musica" not in utterances


def _guard_fixture(tmp_path, pipeline_cls):
    """Shared eval/competitor fixtures for the training-guard tests, with
    ``IntentPipeline`` swapped for ``pipeline_cls``."""
    from types import SimpleNamespace

    from runner import intent_bench

    eval_def = SimpleNamespace(
        source=SimpleNamespace(hf_id="org/eval-repo", revision="main",
                               file_pattern=None, subset=None, split="test"),
        train_datasets={}, input="text", reference_granularity="flat",
    )
    test_rows = [{"utterance": "quin temps fa", "expected_intent": "weather"}]
    competitor = SimpleNamespace(
        competitor_id="x", config={"intents": {}}, model_revision=None,
        training_regime=None,
        pipeline_plugins=[], modality=SimpleNamespace(value="intent_online"),
        plugin="stub-plugin", pipeline="stub-pipeline",
    )
    out_path = tmp_path / "out.jsonl"
    patches = [
        patch.object(intent_bench, "resolve_revision", return_value="EVALSHA"),
        patch.object(intent_bench, "fetch_rows", return_value=test_rows),
        patch.object(intent_bench, "needed_paradigms", return_value=set()),
        patch.object(intent_bench, "done_samples", return_value=set()),
        patch.object(intent_bench, "IntentPipeline", pipeline_cls),
    ]
    return intent_bench, competitor, eval_def, out_path, patches


class TestTrainingGuard:
    """Wall-clock timeout and RSS ceiling per benchmark cell (#trainguard):
    a single runaway fighter must not OOM-kill or hang an entire sweep."""

    def test_timeout_kills_a_slow_train_and_skips_the_cell(self, tmp_path, caplog):
        import time

        class SlowPipeline:
            stage_names = ["stub"]

            def __init__(self, *a, **kw):
                pass

            def train(self, *a, **kw):
                time.sleep(5)

            def predict(self, utterance):
                return utterance, {}, 1.0, 1.0, "stub"

        intent_bench, competitor, eval_def, out_path, patches = _guard_fixture(
            tmp_path, SlowPipeline)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with caplog.at_level("ERROR"):
                written = intent_bench.run_competitor_lang(
                    competitor, "meteocat", "ca-ES", eval_def, {}, "EVALSHA",
                    out_path, train_timeout_secs=1, train_max_rss_mb=0,
                )
        assert written == 0
        assert not out_path.exists() or out_path.read_text() == ""
        assert any(
            "trained=False" in r.message and "reason=train_timeout" in r.message
            for r in caplog.records
        )

    def test_rss_ceiling_kills_a_memory_hungry_train_and_skips_the_cell(
        self, tmp_path, caplog
    ):
        class GreedyPipeline:
            stage_names = ["stub"]

            def __init__(self, *a, **kw):
                pass

            def train(self, *a, **kw):
                # Allocate well past a tiny ceiling and touch every page so
                # it actually lands in RSS, then hold it for the watchdog.
                import time
                self._hog = bytearray(200 * 1024 * 1024)
                for i in range(0, len(self._hog), 4096):
                    self._hog[i] = 1
                time.sleep(5)

            def predict(self, utterance):
                return utterance, {}, 1.0, 1.0, "stub"

        intent_bench, competitor, eval_def, out_path, patches = _guard_fixture(
            tmp_path, GreedyPipeline)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with caplog.at_level("ERROR"):
                written = intent_bench.run_competitor_lang(
                    competitor, "meteocat", "ca-ES", eval_def, {}, "EVALSHA",
                    out_path, train_timeout_secs=0, train_max_rss_mb=50,
                )
        assert written == 0
        assert not out_path.exists() or out_path.read_text() == ""
        assert any(
            "trained=False" in r.message and "reason=train_memory" in r.message
            for r in caplog.records
        )

    def test_guard_off_trains_normally(self, tmp_path):
        class QuickPipeline:
            stage_names = ["stub"]

            def __init__(self, *a, **kw):
                pass

            def train(self, *a, **kw):
                pass

            def predict(self, utterance):
                return utterance, {}, 1.0, 1.0, "stub"

        intent_bench, competitor, eval_def, out_path, patches = _guard_fixture(
            tmp_path, QuickPipeline)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            written = intent_bench.run_competitor_lang(
                competitor, "meteocat", "ca-ES", eval_def, {}, "EVALSHA",
                out_path, train_timeout_secs=0, train_max_rss_mb=0,
            )
        assert written == 1
        assert len(out_path.read_text().splitlines()) == 1


class TestLoadSttEngine:
    """load_stt_engine (§ audio-input intent) — a missing STT plugin entry
    point must raise a clear RuntimeError, not fall through to a
    ``clazz(...)`` call on ``None``."""

    def _audio_dataset_def(self):
        return DatasetDef(
            dataset_id="speech-massive-en-US", modality="intent",
            source={"type": "path", "path": "/x.jsonl"},
            lang="en-US", input="audio",
            stt_plugin="ovos-stt-plugin-does-not-exist",
            stt_config={"lang": "en-US"},
            reference_fields={"audio": "audio", "intent": "intent_str"},
        )

    def test_missing_plugin_raises_runtime_error(self, monkeypatch):
        import ovos_plugin_manager.stt as opm_stt

        monkeypatch.setattr(opm_stt, "load_stt_plugin", lambda module: None)
        with pytest.raises(RuntimeError, match="ovos-stt-plugin-does-not-exist"):
            load_stt_engine(self._audio_dataset_def(), "en-US")

    def test_installed_plugin_still_instantiates(self, monkeypatch):
        import ovos_plugin_manager.stt as opm_stt

        calls = []

        class FakeEngine:
            def __init__(self, config):
                calls.append(config)

        monkeypatch.setattr(opm_stt, "load_stt_plugin", lambda module: FakeEngine)
        engine = load_stt_engine(self._audio_dataset_def(), "en-US")
        assert isinstance(engine, FakeEngine)
        assert calls[0]["module"] == "ovos-stt-plugin-does-not-exist"


class TestRunBenchmarkSttPreflight:
    """run_benchmark refuses an audio-input dataset outright when its
    pinned STT plugin isn't installed, instead of letting every fighter's
    cell independently crash inside transcribe_dataset/load_stt_engine."""

    def _audio_dataset_def(self, dataset_id):
        return DatasetDef(
            dataset_id=dataset_id, modality="intent",
            source={"type": "huggingface", "hf_id": "org/corpus"},
            lang="en-US", langs=["en-US"], input="audio",
            stt_plugin="ovos-stt-plugin-does-not-exist",
            stt_config={"lang": "en-US"},
            reference_fields={"audio": "audio", "intent": "intent_str"},
        )

    def test_dataset_skipped_once_when_stt_plugin_missing(
            self, monkeypatch, caplog, tmp_path):
        dataset_id = "speech-massive-en-US"
        eval_def = self._audio_dataset_def(dataset_id)
        monkeypatch.setattr(intent_bench_mod, "load_dataset",
                            lambda modality, did: eval_def)

        def _boom(*a, **kw):
            raise AssertionError(
                "eligible_competitors must not run for a dataset whose STT "
                "plugin preflight failed")

        monkeypatch.setattr(intent_bench_mod, "eligible_competitors", _boom)
        monkeypatch.setattr("runner.media_bench.plugin_is_installed",
                            lambda modality, plugin: False)

        with caplog.at_level("ERROR"):
            rc = run_benchmark(
                dataset_id, "test dataset",
                argv=["--output-dir", str(tmp_path)],
            )
        assert rc == 0
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert dataset_id in errors[0].message
        assert "ovos-stt-plugin-does-not-exist" in errors[0].message

    def test_dataset_runs_when_stt_plugin_installed(self, monkeypatch, tmp_path):
        dataset_id = "speech-massive-en-US"
        eval_def = self._audio_dataset_def(dataset_id)
        monkeypatch.setattr(intent_bench_mod, "load_dataset",
                            lambda modality, did: eval_def)
        monkeypatch.setattr(intent_bench_mod, "eligible_competitors",
                            lambda paradigms, dataset_id: [])
        monkeypatch.setattr("runner.media_bench.plugin_is_installed",
                            lambda modality, plugin: True)
        monkeypatch.setattr(intent_bench_mod, "resolve_revision",
                            lambda hf_id, revision, timeout=None: "deadbeef")

        rc = run_benchmark(
            dataset_id, "test dataset",
            argv=["--output-dir", str(tmp_path)],
        )
        assert rc == 0


class TestRunBenchmarkTrainedOnSkip:
    """A fighter never runs against an intent dataset it lists in
    ``trained_on`` (owner ruling: never scored on a corpus containing its
    own training recordings) — enforced in ``run_benchmark``'s
    per-competitor loop, the same way ``runner.media_bench`` does it."""

    def _competitor(self, competitor_id, trained_on=None):
        from types import SimpleNamespace
        return SimpleNamespace(
            competitor_id=competitor_id, modality=SimpleNamespace(value="intent"),
            langs=[], trained_on=trained_on or [], pipeline_plugins=[], plugin=None,
        )

    def test_trained_on_pair_is_skipped(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        import runner.intent_bench as intent_bench

        trained_comp = self._competitor("trained-fighter", trained_on=["ds-a"])
        clean_comp = self._competitor("clean-fighter")

        monkeypatch.setattr(
            intent_bench, "load_dataset",
            lambda modality, dataset_id: SimpleNamespace(
                dataset_id=dataset_id, input=None,
                source=SimpleNamespace(hf_id="org/ds", revision="main"),
                langs=["en-US"], lang="en-US", train_datasets={},
            ),
        )
        monkeypatch.setattr(intent_bench, "resolve_revision", lambda hf_id, rev: "abc123")
        monkeypatch.setattr(intent_bench, "eligible_competitors",
                             lambda paradigms, dataset_id="": [trained_comp, clean_comp])

        run_calls: list[str] = []

        def fake_run_competitor_lang(competitor, dataset_id, lang, *a, **kw):
            run_calls.append(competitor.competitor_id)
            return 0

        monkeypatch.setattr(intent_bench, "run_competitor_lang", fake_run_competitor_lang)

        rc = intent_bench.run_benchmark(
            "ds-a", "test", argv=["--output-dir", str(tmp_path)],
        )
        assert rc == 0
        assert run_calls == ["clean-fighter"]


class TestModelLabelOverlap:
    """model_label_overlap measures overlap against the matcher's effective
    class list — classes_ narrowed by the plugin's own registered-intents
    set, when the plugin exposes one — not raw classes_ alone."""

    def _pipeline(self, plugins):
        from types import SimpleNamespace
        return SimpleNamespace(plugins=plugins)

    def test_overlap_narrowed_by_registered_intents(self):
        """A plugin whose matcher only fires on a subset of its own
        classes_ (self.intents narrower than the model's class list) must
        be measured on that narrower, matcher-effective set."""
        from types import SimpleNamespace

        from runner.intent_bench import model_label_overlap

        model = SimpleNamespace(classes_=["a", "b", "c", "d"])
        plugin = SimpleNamespace(model=model, intents={"a", "b"})
        pipeline = self._pipeline({"stub": plugin})

        overlap = model_label_overlap(pipeline, {"a", "b", "c", "d"})

        assert overlap == 2, (
            "matcher only ever returns 'a'/'b' — 'c'/'d' are unreachable "
            "even though they're in classes_"
        )

    def test_falls_back_to_raw_classes_without_a_registered_set(self):
        """A plugin that exposes no ``intents`` attribute at all (or an
        empty one) is measured on raw ``classes_``, same as before."""
        from types import SimpleNamespace

        from runner.intent_bench import model_label_overlap

        model = SimpleNamespace(classes_=["a", "b", "c"])
        plugin = SimpleNamespace(model=model, intents=set())
        pipeline = self._pipeline({"stub": plugin})

        overlap = model_label_overlap(pipeline, {"a", "b", "c", "d"})

        assert overlap == 3

    def test_no_class_list_returns_none(self):
        from types import SimpleNamespace

        from runner.intent_bench import model_label_overlap

        plugin = SimpleNamespace(model=None, intents=set())
        pipeline = self._pipeline({"stub": plugin})

        assert model_label_overlap(pipeline, {"a"}) is None


class TestNullProbeAbort:
    """A cell whose leading predictions are all null is refused outright —
    a matcher that never sees the labels it claims to know (the plugin's
    registered-intents set stayed empty, e.g.) predicts None for every
    utterance, and that must not publish a board of zeros."""

    def _eval_def(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            source=SimpleNamespace(hf_id="org/eval-repo", revision="main",
                                   file_pattern=None, subset=None, split="test"),
            train_datasets={}, input="text", reference_granularity="flat",
        )

    def _competitor(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            competitor_id="x", config={"intents": {}}, model_revision=None,
            training_regime=None,
            pipeline_plugins=[], modality=SimpleNamespace(value="intent_offline"),
            plugin="stub-plugin", pipeline="stub-pipeline",
        )

    def test_all_null_leading_predictions_abort_the_cell(self, tmp_path):
        from runner import intent_bench

        class NullPipeline:
            stage_names = ["stub"]

            def __init__(self, *a, **kw):
                pass

            def train(self, *a, **kw):
                pass

            def predict(self, utterance):
                return None, {}, None, 1.0, None

        test_rows = [
            {"utterance": f"utterance {i}", "expected_intent": "weather"}
            for i in range(intent_bench.NULL_PROBE_ROWS + 5)
        ]
        out_path = tmp_path / "out.jsonl"

        with patch.object(intent_bench, "resolve_revision", return_value="EVALSHA"), \
             patch.object(intent_bench, "fetch_rows", return_value=test_rows), \
             patch.object(intent_bench, "needed_paradigms", return_value=set()), \
             patch.object(intent_bench, "done_samples", return_value=set()), \
             patch.object(intent_bench, "IntentPipeline", NullPipeline):
            written = intent_bench.run_competitor_lang(
                self._competitor(), "corpus", "en-US", self._eval_def(), {},
                "EVALSHA", out_path,
            )

        assert written == 0, "an all-null cell must write no rows"
        assert not out_path.exists() or out_path.read_text() == ""

    def test_a_few_real_predictions_clear_the_probe(self, tmp_path):
        """Not every leading prediction has to be null — one real match
        among the probe rows is enough to trust the rest of the cell."""
        from runner import intent_bench

        class MostlyNullPipeline:
            stage_names = ["stub"]

            def __init__(self, *a, **kw):
                pass

            def train(self, *a, **kw):
                pass

            def predict(self, utterance):
                if utterance == "utterance 0":
                    return "weather", {}, 1.0, 1.0, "stub"
                return None, {}, None, 1.0, None

        test_rows = [
            {"utterance": f"utterance {i}", "expected_intent": "weather"}
            for i in range(intent_bench.NULL_PROBE_ROWS + 5)
        ]
        out_path = tmp_path / "out.jsonl"

        with patch.object(intent_bench, "resolve_revision", return_value="EVALSHA"), \
             patch.object(intent_bench, "fetch_rows", return_value=test_rows), \
             patch.object(intent_bench, "needed_paradigms", return_value=set()), \
             patch.object(intent_bench, "done_samples", return_value=set()), \
             patch.object(intent_bench, "IntentPipeline", MostlyNullPipeline):
            written = intent_bench.run_competitor_lang(
                self._competitor(), "corpus", "en-US", self._eval_def(), {},
                "EVALSHA", out_path,
            )

        assert written == len(test_rows), "one real match clears the probe"
