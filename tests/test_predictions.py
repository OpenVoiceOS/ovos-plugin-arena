"""Unit tests for arena.predictions — JSONL loading and grouping."""
from __future__ import annotations

import json

import pytest

import arena.predictions as predictions_mod
from arena.predictions import (
    group_rows,
    infer_modality,
    load_predictions,
    load_predictions_dir,
    parse_row,
    read_jsonl,
)


def _intent_row(**over):
    row = {
        "competitor_id": "padatious-medium",
        "sample_id": "en-US/00001",
        "dataset_id": "intents-for-eval",
        "lang": "en-US",
        "plugin_id": "ovos-padatious-pipeline-plugin",
        "utterance": "play a song",
        "reference_intent": "media:play_song",
        "prediction": "media:play_song",
        "exact_match": True,
    }
    row.update(over)
    return row


class TestInferModality:
    def test_intent(self):
        assert infer_modality(_intent_row()) == "intent"

    def test_stt(self):
        assert infer_modality({"reference_text": "ola", "prediction": "ola"}) == "stt"

    def test_unknown(self):
        assert infer_modality({"prediction": "?"}) == "unknown"

    def test_explicit_modality_wins(self):
        assert infer_modality(_intent_row(modality="intent_template")) == (
            "intent_template"
        )
        assert infer_modality(
            {"label": "positive", "prediction": "detected", "modality": "vad"}
        ) == "vad"

    def test_wake_word(self):
        assert infer_modality(
            {"label": "positive", "prediction": "detected"}
        ) == "wake_word"
        assert infer_modality(
            {"label": "negative", "prediction": "not_detected"}
        ) == "wake_word"

    def test_vad_from_label_vocabulary(self):
        # runner.vad_bench rows: label speech/non_speech, prediction speech/silence
        assert infer_modality(
            {"label": "speech", "prediction": "silence"}
        ) == "vad"
        assert infer_modality(
            {"label": "non_speech", "prediction": "speech"}
        ) == "vad"

    def test_vad_from_prediction_only(self):
        # even a mislabelled row is caught by the decision vocabulary
        assert infer_modality(
            {"label": "positive", "prediction": "silence"}
        ) == "vad"


class TestParseRow:
    def test_known_fields_mapped(self):
        row = parse_row(_intent_row(), "fallback-id")
        assert row.competitor_id == "padatious-medium"
        assert row.reference_intent == "media:play_song"

    def test_competitor_falls_back_to_filename(self):
        raw = _intent_row()
        del raw["competitor_id"]
        row = parse_row(raw, "from-filename")
        assert row.competitor_id == "from-filename"

    def test_unknown_keys_preserved_in_extras(self):
        row = parse_row(_intent_row(some_new_column="abc123"), "c")
        assert row.extras["some_new_column"] == "abc123"


class TestReadJsonl:
    def test_reads_rows_and_skips_malformed(self, tmp_path):
        path = tmp_path / "competitor-x.jsonl"
        path.write_text(
            json.dumps(_intent_row()) + "\n"
            + "NOT JSON\n"
            + json.dumps(_intent_row(sample_id="en-US/00002")) + "\n"
        )
        rows = read_jsonl(path)
        assert len(rows) == 2
        assert rows[0].competitor_id == "padatious-medium"

    def test_dir_loader(self, tmp_path):
        for name in ("a.jsonl", "b.jsonl"):
            (tmp_path / name).write_text(json.dumps(_intent_row()) + "\n")
        assert len(load_predictions_dir(tmp_path)) == 2

    def test_load_predictions_local_path(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(json.dumps(_intent_row()) + "\n")
        assert len(load_predictions(str(tmp_path))) == 1

    def test_concrete_lang_excludes_other_lang_dirs(self, tmp_path):
        # Regression for the merged-lang-pools blocker: a prediction repo
        # for a concrete-lang dataset (e.g. minds14 de-DE) commonly has an
        # orphaned shard from an earlier wrong-lang run (e.g. an
        # English-forced decode published under predictions/en/) alongside
        # its own predictions/de-DE/ dir. Loading with lang="de-DE" must
        # only pick up the de-DE shard, never the en orphan.
        de_dir = tmp_path / "de-DE"
        de_dir.mkdir()
        (de_dir / "canary.jsonl").write_text(
            json.dumps(_intent_row(lang="de-DE", sample_id="de-DE/00001")) + "\n"
        )
        en_dir = tmp_path / "en"
        en_dir.mkdir()
        (en_dir / "canary.jsonl").write_text(
            json.dumps(_intent_row(lang="en", sample_id="en/00001")) + "\n"
        )

        all_rows = load_predictions_dir(tmp_path)
        assert len(all_rows) == 2  # unfiltered: both shards merge (the bug)

        de_only = load_predictions_dir(tmp_path, lang="de-DE")
        assert len(de_only) == 1
        assert de_only[0].lang == "de-DE"

        de_only_via_load_predictions = load_predictions(str(tmp_path), lang="de-DE")
        assert len(de_only_via_load_predictions) == 1

    def test_concrete_lang_still_loads_flat_legacy_root_files(self, tmp_path):
        # The flat legacy predictions/<competitor>.jsonl form predates the
        # per-lang layout and is always accepted regardless of lang filter.
        (tmp_path / "legacy.jsonl").write_text(json.dumps(_intent_row()) + "\n")
        en_dir = tmp_path / "en"
        en_dir.mkdir()
        (en_dir / "canary.jsonl").write_text(
            json.dumps(_intent_row(lang="en", sample_id="en/00001")) + "\n"
        )
        rows = load_predictions_dir(tmp_path, lang="de-DE")
        assert len(rows) == 1
        assert rows[0].lang == "en-US"  # from the flat legacy.jsonl row, not en/

    def test_multi_lang_dataset_keeps_loading_every_lang_dir(self, tmp_path):
        # A multi/unknown-lang dataset (lang=None) must NOT be filtered —
        # every lang dir stays in scope.
        for tag in ("de-DE", "en", "fr"):
            d = tmp_path / tag
            d.mkdir()
            (d / "canary.jsonl").write_text(
                json.dumps(_intent_row(lang=tag, sample_id=f"{tag}/00001")) + "\n"
            )
        rows = load_predictions_dir(tmp_path, lang=None)
        assert len(rows) == 3


class _FakeCompetitor:
    def __init__(self, competitor_id):
        self.competitor_id = competitor_id


def _stub_registry(monkeypatch, registered_by_modality):
    """Make ``registry.loaders.list_competitors`` return only the given ids.

    ``group_rows`` imports ``list_competitors`` locally (inside the
    function) from ``registry.loaders``, so patching the attribute on that
    module is sufficient — no need to touch ``arena.predictions``.
    """
    import registry.loaders as loaders_mod

    def fake_list_competitors(modality=None):
        return [
            _FakeCompetitor(cid)
            for cid in registered_by_modality.get(modality, [])
        ]

    monkeypatch.setattr(loaders_mod, "list_competitors", fake_list_competitors)


class TestGroupRows:
    def test_groups_by_modality_dataset_lang(self, monkeypatch):
        _stub_registry(monkeypatch, {"intent": ["padatious-medium", "other"]})
        rows = [
            parse_row(_intent_row(), "a"),
            parse_row(_intent_row(lang="pt-PT", sample_id="pt-PT/00001"), "a"),
            parse_row(_intent_row(competitor_id="other"), "other"),
        ]
        grouped = group_rows(rows)
        assert ("intent", "intents-for-eval", "en-US") in grouped
        assert ("intent", "intents-for-eval", "pt-PT") in grouped
        en = grouped[("intent", "intents-for-eval", "en-US")]
        assert set(en["en-US/00001"]) == {"padatious-medium", "other"}

    def test_duplicate_rows_keep_last(self, monkeypatch):
        _stub_registry(monkeypatch, {"intent": ["padatious-medium"]})
        rows = [
            parse_row(_intent_row(prediction="first"), "a"),
            parse_row(_intent_row(prediction="second"), "a"),
        ]
        grouped = group_rows(rows)
        sample = grouped[("intent", "intents-for-eval", "en-US")]["en-US/00001"]
        assert sample["padatious-medium"].prediction == "second"

    def test_unknown_modality_dropped(self, monkeypatch):
        _stub_registry(monkeypatch, {})
        rows = [parse_row({"competitor_id": "c", "sample_id": "s",
                           "dataset_id": "d", "lang": "x", "plugin_id": "p",
                           "prediction": "?"}, "c")]
        assert group_rows(rows) == {}

    def test_unregistered_competitor_excluded_from_boards(self, monkeypatch):
        # §board-truth — a fighter removed from the registry (e.g.
        # jurebes-medium, replaced by the per-baseline fan-out) must not
        # keep appearing on published boards just because its orphaned HF
        # prediction shards are still fetched and loaded.
        _stub_registry(monkeypatch, {"intent": ["padatious-medium"]})
        rows = [
            parse_row(_intent_row(competitor_id="padatious-medium"), "padatious-medium"),
            parse_row(_intent_row(competitor_id="padatious-medium"), "padatious-medium"),
            parse_row(_intent_row(competitor_id="jurebes-medium"), "jurebes-medium"),
        ]
        unregistered: dict[str, int] = {}
        grouped = group_rows(rows, unregistered=unregistered)

        en = grouped[("intent", "intents-for-eval", "en-US")]["en-US/00001"]
        assert set(en) == {"padatious-medium"}
        assert "jurebes-medium" not in en
        assert unregistered == {"jurebes-medium": 1}


def _legacy_stt_row(**over):
    row = {
        "dataset_entry_id": "pt-PT/00007",
        "plugin_name": "ovos-stt-plugin-fasterwhisper",
        "model_id": "ovos-stt-plugin-fasterwhisper/small",
        "prediction_transcript": "ola mundo",
        "transcript": "olá mundo",
        "prediction_confidence": 0.91,
        "prediction_type": "STT",
        "dataset_id": "minds14-pt-PT",
        "lang": "pt-PT",
    }
    row.update(over)
    return row


class TestLegacySttSchemaConvergence:
    """§4 A2 — legacy STTRow-shaped rows convert to canonical PredictionRow
    at load time, and are re-keyed via registry alias when possible."""

    def setup_method(self):
        predictions_mod._alias_cache.clear()

    def test_legacy_shape_detected_and_converted(self, monkeypatch):
        monkeypatch.setattr(
            predictions_mod, "_resolve_competitor_id", lambda modality, plugin_id: None
        )
        row = parse_row(_legacy_stt_row(), "fallback-competitor")
        assert row.sample_id == "pt-PT/00007"
        assert row.reference_text == "olá mundo"
        assert row.prediction == "ola mundo"
        assert row.confidence == 0.91
        assert row.modality == "stt"
        assert row.schema_version == 1
        assert row.extras["model_id"] == "ovos-stt-plugin-fasterwhisper/small"
        assert row.competitor_id == "fallback-competitor"  # alias resolution failed

    def test_legacy_shape_rekeyed_via_alias(self, monkeypatch):
        monkeypatch.setattr(
            predictions_mod, "_resolve_competitor_id",
            lambda modality, plugin_id: "fasterwhisper-small-pt" if plugin_id.endswith(
                "fasterwhisper") else None,
        )
        row = parse_row(_legacy_stt_row(), "fallback-competitor")
        assert row.competitor_id == "fasterwhisper-small-pt"

    def test_canonical_row_missing_competitor_id_rekeyed_via_plugin_id(self, monkeypatch):
        # Rows written by the current runner (no registry dependency) carry
        # plugin_id but not competitor_id.
        monkeypatch.setattr(
            predictions_mod, "_resolve_competitor_id",
            lambda modality, plugin_id: "resolved-id",
        )
        raw = {
            "sample_id": "s1", "dataset_id": "d", "lang": "en-US",
            "plugin_id": "ovos-stt-plugin-x", "modality": "stt",
            "prediction": "hi", "reference_text": "hi",
        }
        row = parse_row(raw, "filename-fallback")
        assert row.competitor_id == "resolved-id"

    def test_canonical_row_with_explicit_competitor_id_not_rekeyed(self, monkeypatch):
        monkeypatch.setattr(
            predictions_mod, "_resolve_competitor_id",
            lambda modality, plugin_id: (_ for _ in ()).throw(
                AssertionError("should not be called")),
        )
        row = parse_row(_intent_row(), "fallback")
        assert row.competitor_id == "padatious-medium"

    def test_alias_resolution_is_memoized(self, monkeypatch):
        calls = []

        def _fake_get_by_alias(modality, plugin_id):
            calls.append((modality, plugin_id))
            return None

        import registry.loaders
        monkeypatch.setattr(registry.loaders, "get_competitor_by_alias", _fake_get_by_alias)

        parse_row(_legacy_stt_row(), "fallback")
        parse_row(_legacy_stt_row(), "fallback")
        assert len(calls) == 1  # second call hits the cache


class TestPerfFieldsBackwardCompat:
    """Performance-metrics campaign M1: rows WITHOUT elapsed_ms/peak_rss_mb/
    audio_secs/hw are the vast majority of already-published data and MUST
    stay valid — no KeyError, no validation error. This is a genuine
    regression guard: making these fields required on ``PredictionRow``
    (dropping their ``= None`` default) makes ``test_old_row_without_perf_fields``
    fail with a pydantic ``ValidationError`` — verified by hand while writing
    this test, the two-line diff being ``elapsed_ms: float | None = None`` ->
    ``elapsed_ms: float``, etc.
    """

    def test_old_row_without_perf_fields_parses_fine(self):
        raw = _intent_row()  # no elapsed_ms / peak_rss_mb / audio_secs / hw
        row = parse_row(raw, "fallback")
        assert row.elapsed_ms is None
        assert row.peak_rss_mb is None
        assert row.audio_secs is None
        assert row.hw is None

    def test_new_row_with_perf_fields_round_trips(self):
        raw = _intent_row(
            elapsed_ms=123.4,
            peak_rss_mb=512.0,
            audio_secs=None,  # intent rows have no audio
            hw={"host_class": "cpu-x86", "cpu_model": "x", "threads": 4,
                "accelerator": None, "hostname": "box1"},
        )
        row = parse_row(raw, "fallback")
        assert row.elapsed_ms == 123.4
        assert row.peak_rss_mb == 512.0
        assert row.hw["host_class"] == "cpu-x86"

    def test_group_rows_mixes_old_and_new_rows_without_error(self, monkeypatch):
        _stub_registry(monkeypatch, {"intent": ["padatious-medium"]})
        old = _intent_row(sample_id="en-US/00001")
        new = _intent_row(sample_id="en-US/00002", elapsed_ms=10.0,
                          hw={"host_class": "cpu-x86"})
        rows = [parse_row(old, "padatious-medium"),
                parse_row(new, "padatious-medium")]
        grouped = group_rows(rows)
        key = ("intent", "intents-for-eval", "en-US")
        assert len(grouped[key]["en-US/00001"]) == 1
        assert len(grouped[key]["en-US/00002"]) == 1


class TestHfFetchRetry:
    """The daily assemble walks ~120 prediction repos unauthenticated and
    routinely draws a 429 from the Hub; a single-shot download turned that
    transient into a dropped dataset."""

    def test_retries_a_rate_limited_download(self, monkeypatch):
        import sys
        import types

        attempts = []

        def flaky_snapshot_download(**kwargs):
            attempts.append(kwargs["repo_id"])
            if len(attempts) < 3:
                raise RuntimeError("429 Client Error: Too Many Requests")
            return "/tmp/snapshot"

        monkeypatch.setitem(
            sys.modules, "huggingface_hub",
            types.SimpleNamespace(snapshot_download=flaky_snapshot_download),
        )
        monkeypatch.setattr(predictions_mod, "HF_FETCH_BACKOFF_SECONDS",
                            (0.0, 0.0, None))

        path = predictions_mod.fetch_hf_predictions("Org/some-bench", "main")

        assert len(attempts) == 3
        assert path.name == "predictions"

    def test_gives_up_after_the_last_attempt(self, monkeypatch):
        import sys
        import types

        attempts = []

        def always_429(**kwargs):
            attempts.append(kwargs["repo_id"])
            raise RuntimeError("429 Client Error: Too Many Requests")

        monkeypatch.setitem(
            sys.modules, "huggingface_hub",
            types.SimpleNamespace(snapshot_download=always_429),
        )
        monkeypatch.setattr(predictions_mod, "HF_FETCH_BACKOFF_SECONDS",
                            (0.0, 0.0, None))

        with pytest.raises(RuntimeError, match="429"):
            predictions_mod.fetch_hf_predictions("Org/some-bench", "main")
        assert len(attempts) == 3

    def test_default_backoff_actually_pauses_between_attempts(self):
        """The retry loop is worthless against a real rate limiter if the
        production constant has no positive pauses — this pins the contract
        without sleeping, so a defaults-mutation regression fails fast."""
        backoff = predictions_mod.HF_FETCH_BACKOFF_SECONDS
        pauses = [p for p in backoff[:-1] if p is not None]
        assert backoff[-1] is None
        assert len(pauses) >= 2
        assert pauses[0] >= 1.0
        assert pauses == sorted(pauses)

    def test_default_backoff_is_used_to_sleep_between_retries(self, monkeypatch):
        import sys
        import types

        attempts = []

        def flaky_snapshot_download(**kwargs):
            attempts.append(kwargs["repo_id"])
            if len(attempts) < 3:
                raise RuntimeError("429 Client Error: Too Many Requests")
            return "/tmp/snapshot"

        monkeypatch.setitem(
            sys.modules, "huggingface_hub",
            types.SimpleNamespace(snapshot_download=flaky_snapshot_download),
        )

        sleeps = []
        monkeypatch.setattr(predictions_mod.time, "sleep", sleeps.append)

        path = predictions_mod.fetch_hf_predictions("Org/some-bench", "main")

        assert len(attempts) == 3
        assert path.name == "predictions"
        expected = [p for p in predictions_mod.HF_FETCH_BACKOFF_SECONDS if p is not None][:2]
        assert sleeps == expected
