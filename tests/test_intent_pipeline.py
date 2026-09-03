"""Unit tests for runner.intent_pipeline — pure logic, no plugin loading."""
from __future__ import annotations

import pytest

from runner.intent_pipeline import (
    ENGINE_REGISTRY,
    IntentPipeline,
    expand_template,
)


class TestEngineRegistry:
    def test_expected_engines_present(self):
        assert set(ENGINE_REGISTRY) == {
            "ovos-padatious-pipeline-plugin",
            "ovos-padacioso-pipeline-plugin",
            "ovos-nebulento-pipeline-plugin",
            "ovos-adapt-pipeline-plugin",
            "ovos-palavreado-pipeline-plugin",
            "ovos-jurebes-pipeline-plugin",
            "ovos-linha-fina-pipeline-plugin",
            "ovos-markov-pipeline-plugin",
            "ovos-m2v-pipeline",
            "ovos-hierarchical-knn-pipeline",
            "ovos-nebulento-hierarchical-pipeline-plugin",
            "ovos-linha-fina-domain-pipeline-plugin",
            "ovos-linha-fina-hierarchical-pipeline-plugin",
            "ovos-markov-domain-pipeline-plugin",
            "ovos-adapt-domain-pipeline-plugin",
            "ovos-adapt-hierarchical-pipeline-plugin",
            "ovos-palavreado-hierarchical-pipeline",
        }

    def test_paradigms(self):
        assert ENGINE_REGISTRY["ovos-adapt-pipeline-plugin"].paradigm == "keyword"
        assert ENGINE_REGISTRY["ovos-padatious-pipeline-plugin"].paradigm == "template"

    def test_unknown_plugin_raises(self):
        with pytest.raises(KeyError):
            IntentPipeline({"pipeline": ["not-a-plugin-high"]})

    def test_bad_tier_raises(self):
        with pytest.raises(ValueError):
            IntentPipeline({"pipeline": ["ovos-adapt-pipeline-plugin-extreme"]})

    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError):
            IntentPipeline({"pipeline": []})


class TestExpandTemplate:
    def test_no_slots_passthrough(self):
        assert expand_template("hello world", []) == ["hello world"]

    def test_slot_expansion(self):
        out = expand_template(
            "play {song}",
            [{"name": "song", "examples": ["africa", "hey jude"]}],
        )
        assert out == ["play africa", "play hey jude"]

    def test_multi_slot_cycling(self):
        out = expand_template(
            "play {song} by {artist}",
            [{"name": "song", "examples": ["a", "b"]},
             {"name": "artist", "examples": ["x"]}],
        )
        assert out == ["play a by x", "play b by x"]

    def test_unused_slot_ignored(self):
        out = expand_template(
            "pause", [{"name": "song", "examples": ["a"]}]
        )
        assert out == ["pause"]

    def test_cap(self):
        examples = [str(i) for i in range(20)]
        out = expand_template(
            "n {x}", [{"name": "x", "examples": examples}], cap=6
        )
        assert len(out) == 6


class TestExtractSlots:
    def test_adapt_style(self):
        slots = IntentPipeline._extract_slots({
            "intent_type": "media:play_song",
            "media:play_song__PlayKw": "play",
            "media:play_song__song": "bohemian rhapsody",
            "confidence": 0.95,
            "target": None,
            "__tags__": [],
            "utterance": "play bohemian rhapsody",
        })
        assert slots == {"song": "bohemian rhapsody"}

    def test_palavreado_style(self):
        slots = IntentPipeline._extract_slots({
            "keywords": {
                "media:play_song__PlayKw": ["play"],
                "media:play_song__song": ["bohemian rhapsody"],
            },
            "conf": 1.0,
            "utterance": "play bohemian rhapsody",
            "name": "media:play_song",
        })
        assert slots == {"song": "bohemian rhapsody"}

    def test_nebulento_style_list_values(self):
        slots = IntentPipeline._extract_slots({"song": ["africa"]})
        assert slots == {"song": "africa"}

    def test_entities_dict_wins(self):
        slots = IntentPipeline._extract_slots(
            {"entities": {"song": "africa"}, "other": "x"}
        )
        assert slots == {"song": "africa"}

    def test_empty(self):
        assert IntentPipeline._extract_slots({}) == {}


class TestNormalise:
    def test_skill_prefix_stripped(self):
        assert IntentPipeline._normalise("arena:media:play_song") == "media:play_song"
        assert IntentPipeline._normalise("arena.media:play_song") == "media:play_song"

    def test_bare_intent_untouched(self):
        assert IntentPipeline._normalise("media:play_song") == "media:play_song"


class _ToyLabelPipeline:
    """Stand-in for a label-only classifier (e.g. m2v): it names the intent
    but never extracts slots — exactly the case intent transformers exist to
    cover."""

    def __init__(self, bus, config):
        self.bus = bus
        self.config = config

    def match_high(self, utterances, lang, message):
        from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch

        utterance = utterances[0]
        if "play" in utterance:
            return IntentHandlerMatch(
                match_type="play_song", match_data={"conf": 1.0},
                skill_id="arena", utterance=utterance,
            )
        return None


TRAIN_ROWS = {
    "template": [{
        "intent_id": "play_song",
        "template": "play {song}",
        "slots": [{"name": "song", "examples": ["africa"]}],
    }],
}


def _toy_pipeline(with_transformer: bool) -> IntentPipeline:
    config = {"pipeline": ["toy-label-pipeline-high"]}
    if with_transformer:
        config["intent_transformers"] = {"ovos-keyword-template-matcher": {}}
    return IntentPipeline(config, lang="en-us")


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec(
        "kw_template_matcher") is None,
    reason="keyword-template-matcher plugin not installed",
)
class TestIntentTransformers:
    @pytest.fixture(autouse=True)
    def _toy_engine_registered(self):
        from runner import intent_pipeline as ip_mod

        ip_mod.ENGINE_REGISTRY["toy-label-pipeline"] = ip_mod.EngineSpec(
            f"{__name__}:_ToyLabelPipeline", "template", None, "toy", "toy")
        yield
        del ip_mod.ENGINE_REGISTRY["toy-label-pipeline"]

    def test_slots_filled_with_transformer_configured(self):
        pipeline = _toy_pipeline(with_transformer=True)
        pipeline.train(TRAIN_ROWS)

        intent_id, slots, confidence, _latency, stage = pipeline.predict(
            "play africa"
        )

        assert intent_id == "play_song"
        assert stage == "toy-label-pipeline-high"
        assert slots == {"song": "africa"}

    def test_register_templates_emits_spec_topic(self):
        # OVOS-INTENT-4 §6: ovos-workshop's Skill.register_template()
        # dual-emits padatious:register_intent (legacy) and
        # ovos.intent.register.template (spec) — the runner's own
        # registration must do the same so transformers that only bind
        # the spec topic (or key templates by its skill-prefixed form,
        # like kw-template-matcher) still see the training data.
        pipeline = _toy_pipeline(with_transformer=True)
        seen = []
        pipeline.xformer_bus.on(
            "ovos.intent.register.template", lambda m: seen.append(m)
        )

        pipeline.train(TRAIN_ROWS)

        assert len(seen) == 1
        assert seen[0].data["skill_id"] == "arena"
        assert seen[0].data["intent_name"] == "play_song"
        assert seen[0].data["lang"] == "en-us"
        assert "play africa" in seen[0].data["samples"]

    def test_transform_slots_falls_back_to_skill_prefixed_key(self):
        # kw-template-matcher keys templates it learned from the spec
        # topic as "skill_id:intent_name" (m2v's IntentHandlerMatch.
        # match_type convention) rather than the bare name the legacy
        # padatious topic uses. Register a template ONLY via the spec
        # topic (bypassing train()'s legacy emission) and confirm the
        # runner's lookup still finds it by trying the skill-prefixed
        # form after the bare form misses.
        from ovos_bus_client.message import Message

        pipeline = _toy_pipeline(with_transformer=True)
        pipeline.xformer_bus.emit(Message("ovos.intent.register.template", {
            "skill_id": "arena",
            "intent_name": "spec_only_intent",
            "samples": ["play {song}"],
            "lang": "en-us",
        }))

        slots = pipeline._transform_slots(
            "spec_only_intent", {"conf": 1.0}, "play africa"
        )

        assert slots == {"song": "africa"}

    def test_no_slots_without_transformer_configured(self):
        pipeline = _toy_pipeline(with_transformer=False)
        pipeline.train(TRAIN_ROWS)

        intent_id, slots, confidence, _latency, stage = pipeline.predict(
            "play africa"
        )

        assert intent_id == "play_song"
        assert stage == "toy-label-pipeline-high"
        assert slots == {}
