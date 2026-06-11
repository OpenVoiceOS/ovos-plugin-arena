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
        }

    def test_paradigms(self):
        assert ENGINE_REGISTRY["ovos-adapt-pipeline-plugin"].paradigm == "keyword"
        assert ENGINE_REGISTRY["ovos-padatious-pipeline-plugin"].paradigm == "template"

    def test_unknown_plugin_raises(self):
        with pytest.raises(KeyError):
            IntentPipeline("not-a-plugin")

    def test_bad_tier_raises(self):
        with pytest.raises(ValueError):
            IntentPipeline("ovos-adapt-pipeline-plugin", tier="extreme")


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
