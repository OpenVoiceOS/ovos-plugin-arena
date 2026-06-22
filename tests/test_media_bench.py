"""Unit tests for runner.media_bench — the shared audio-benchmark engine.

A stub adapter stands in for the real STT/WW/TTS plugins so the row contract,
resume and publishing logic are exercised without any audio stack or plugin
installed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from registry.loaders import load_competitor
from runner import media_bench as mb


class StubAdapter(mb.MediaBenchAdapter):
    modality = "stt"
    card_tags = ("automatic-speech-recognition",)
    card_task = "Per-clip transcripts"

    def __init__(self, n=3, fail_on=None):
        self.n = n
        self.fail_on = fail_on
        self.loaded = 0

    def iter_samples(self, dataset_def, lang, revision, max_samples):
        n = min(self.n, max_samples) if max_samples else self.n
        for i in range(n):
            yield f"{lang}/{i:05d}", {"i": i}

    def load_engine(self, competitor, lang):
        self.loaded += 1
        return "ENGINE"

    def predict(self, engine, sample, ctx):
        if sample["i"] == self.fail_on:
            raise RuntimeError("boom")
        return {"reference_text": "ref", "prediction": f"hyp{sample['i']}",
                "latency_ms": 1.0}


def _competitor():
    return load_competitor("stt", "whispercpp-base")


def _eval_def():
    return SimpleNamespace(
        source=SimpleNamespace(hf_id="PolyAI/minds14", revision="main"))


# ---------------------------------------------------------------------------


class TestMakeRow:
    def test_base_and_merged_fields(self):
        row = mb.make_row(_competitor(), "minds14-pt-PT", "pt-PT", "pt-PT/0",
                          "abc123", {"prediction": "hi", "wer": 0.0})
        assert row["competitor_id"] == "whispercpp-base"
        assert row["modality"] == "stt"
        assert row["dataset_revision"] == "abc123"
        assert row["plugin_id"] == "ovos-stt-plugin-whispercpp"
        assert row["prediction"] == "hi" and row["wer"] == 0.0
        assert row["runner_version"].startswith("ovos-plugin-arena==")
        assert row["created_at"]

    def test_parses_into_prediction_row(self):
        from arena.models import PredictionRow

        row = mb.make_row(_competitor(), "d", "pt-PT", "s0", "r",
                          {"reference_text": "x", "prediction": "x"})
        PredictionRow(**row)  # must not raise


class TestCompetitorsFor:
    def test_filters_by_modality(self):
        ids = {c.competitor_id for c in mb.competitors_for("wake_word")}
        assert "openwakeword-hey-mycroft" in ids
        assert all(c.modality.value == "wake_word"
                   for c in mb.competitors_for("wake_word"))

    def test_wanted_subset(self):
        comps = mb.competitors_for("stt", {"whispercpp-base"})
        assert [c.competitor_id for c in comps] == ["whispercpp-base"]


class TestDatasetCard:
    def test_card_lists_langs_and_tags(self):
        card = mb.dataset_card(StubAdapter(), "minds14-pt-PT", _eval_def(),
                               ["pt-PT", "en-US"])
        assert "config_name: default" in card
        assert "split: pt_PT" in card and "split: en_US" in card
        assert "automatic-speech-recognition" in card
        assert "Per-clip transcripts" in card
        assert "ovos-stt" not in card.split("\n")[0]  # not a malformed header


class TestRunCompetitorLang:
    def _run(self, adapter, tmp_path, **kw):
        out = tmp_path / "out.jsonl"
        written = mb.run_competitor_lang(
            adapter, _competitor(), "minds14-pt-PT", "pt-PT", _eval_def(),
            "rev", out, tmp_path / "audio", "owner/repo", **kw)
        return out, written

    def test_writes_all_rows(self, tmp_path):
        out, written = self._run(StubAdapter(n=3), tmp_path)
        assert written == 3
        assert len(out.read_text().splitlines()) == 3

    def test_resume_skips_done(self, tmp_path):
        out, _ = self._run(StubAdapter(n=3), tmp_path)
        # second pass: everything already done, engine never loaded
        adapter = StubAdapter(n=3)
        out2, written = self._run(adapter, tmp_path)
        assert written == 0
        assert adapter.loaded == 0  # lazy: no model load when nothing to do
        assert len(out.read_text().splitlines()) == 3

    def test_failed_sample_skipped(self, tmp_path):
        out, written = self._run(StubAdapter(n=3, fail_on=1), tmp_path)
        assert written == 2  # the failing sample is dropped, others kept

    def test_max_samples(self, tmp_path):
        out, written = self._run(StubAdapter(n=10), tmp_path, max_samples=4)
        assert written == 4


class TestLangMatch:
    def test_primary_subtag_matches_both_ways(self):
        assert mb._lang_matches("en", "en-US")
        assert mb._lang_matches("en-US", "en")
        assert mb._lang_matches("pt-BR", "pt-PT")  # same primary, different region
        assert mb._lang_matches("EN-us", "en-US")
        assert not mb._lang_matches("en", "pt-PT")

    def test_competitor_langs_primary_subtag(self):
        # a plugin advertising "en" must run on an "en-US" corpus
        comp = SimpleNamespace(langs=["en"])
        adapter = StubAdapter()
        assert adapter.competitor_langs(comp, ["en-US", "pt-PT"]) == ["en-US"]

    def test_competitor_langs_empty_means_all(self):
        comp = SimpleNamespace(langs=[])
        assert StubAdapter().competitor_langs(comp, ["en-US", "pt-PT"]) == \
            ["en-US", "pt-PT"]


class TestLoadPluginClass:
    def test_resolves_underscore_entrypoint(self):
        # plugins like ovos-tts-plugin-espeakng register with underscores
        seen = []

        def loader(name):
            seen.append(name)
            return "CLS" if name == "ovos_tts_plugin_x" else None

        assert mb.load_plugin_class(loader, "ovos-tts-plugin-x") == "CLS"

    def test_resolves_dashed_entrypoint(self):
        loader = lambda n: "CLS" if n == "ovos-stt-plugin-x" else None  # noqa: E731
        assert mb.load_plugin_class(loader, "ovos-stt-plugin-x") == "CLS"

    def test_raises_when_unresolvable(self):
        with pytest.raises(RuntimeError):
            mb.load_plugin_class(lambda n: None, "missing-plugin")


class TestPredictContext:
    def test_hf_audio_url(self):
        ctx = mb.PredictContext(_competitor(), "en-US", "d", "tts",
                                Path("/tmp"), "OpenVoiceOS/ovos-tts-bench-d")
        url = ctx.hf_audio_url("en-US/edge/abc.wav")
        assert url == ("https://huggingface.co/datasets/"
                       "OpenVoiceOS/ovos-tts-bench-d/resolve/main/"
                       "audio/en-US/edge/abc.wav")
