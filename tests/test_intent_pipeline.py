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
            "ovos-m2v-prototype-pipeline",
            "ovos-nebulento-hierarchical-pipeline-plugin",
            "ovos-linha-fina-domain-pipeline-plugin",
            "ovos-linha-fina-hierarchical-pipeline-plugin",
            "ovos-markov-domain-pipeline-plugin",
            "ovos-adapt-domain-pipeline-plugin",
            "ovos-adapt-hierarchical-pipeline-plugin",
            "ovos-palavreado-hierarchical-pipeline",
        }

    def test_every_engine_has_traits(self):
        """Loading, paradigm and regime are three views of one engine list:
        an engine the arena can run but the schema cannot classify would be
        unfileable in any league."""
        from registry.schemas import ENGINE_TRAITS

        assert set(ENGINE_REGISTRY) == set(ENGINE_TRAITS)

    def test_paradigms(self):
        from registry.schemas import engine_paradigm

        assert engine_paradigm("ovos-adapt-pipeline-plugin") == "keyword"
        assert engine_paradigm("ovos-padatious-pipeline-plugin") == "template"

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


class _RecordingBus:
    def __init__(self):
        self.emitted = []

    def emit(self, message):
        self.emitted.append(message)


class TestRegisterTemplatesTypedSlots:
    """OVOS-INTENT-1 §3.4/§4.1: {type:name} slots must reach engines as
    bare {name}, exactly like ovos-workshop's own loader strips them
    before registration."""

    def _register(self, rows):
        bus = _RecordingBus()
        pipeline = object.__new__(IntentPipeline)
        pipeline.lang = "en-us"
        IntentPipeline._register_templates(pipeline, bus, rows)
        [msg] = [m for m in bus.emitted if m.msg_type == "padatious:register_intent"]
        return msg.data["samples"]

    def test_typed_slot_stripped(self):
        samples = self._register([{
            "intent_id": "lights",
            "template": "turn the lights {color:shade}",
            "slots": [{"name": "shade", "examples": ["red"]}],
        }])
        assert "turn the lights {shade}" in samples
        assert not any("{color:shade}" in s for s in samples)

    def test_untyped_template_byte_identical(self):
        samples = self._register([{
            "intent_id": "lights",
            "template": "turn the lights {shade}",
            "slots": [{"name": "shade", "examples": ["red"]}],
        }])
        assert "turn the lights {shade}" in samples

    def test_unregistered_type_degrades_to_untyped(self):
        # strip_type_prefixes strips ANY {type:name} slot, registered or
        # not, degrading unregistered types to the bare slot rather than
        # forwarding the typed syntax to an engine.
        samples = self._register([{
            "intent_id": "lights",
            "template": "turn the lights {foo:bar}",
            "slots": [{"name": "bar", "examples": ["red"]}],
        }])
        assert "turn the lights {bar}" in samples
        assert not any("{foo:bar}" in s for s in samples)

    def test_registration_carries_skill_id_in_context(self):
        # OVOS-INTENT-4 §3.1: engines key registration on
        # message.context['skill_id'], not message.data — ovos-padatious
        # 2.1.2a2 rejects (and warns) a registration message that carries
        # skill_id only in data, same as adapt and m2v. Run the runner's
        # actual registration path through a real PadatiousPipeline so a
        # regression here is caught by the engine's own acceptance rule,
        # not by inspecting the message we happen to have built.
        pipeline = IntentPipeline(
            {"pipeline": ["ovos-padatious-pipeline-plugin-high"]},
            lang="en-us",
        )
        pipeline.train({"template": [{
            "intent_id": "play_song",
            "template": "play {song}",
            "slots": [{"name": "song", "examples": ["africa"]}],
        }]})
        engine = pipeline.plugins["ovos-padatious-pipeline-plugin"]
        # Value, not presence: the engine files the intent under the
        # skill_id it read from context, so it must land under "arena" and
        # nowhere else.
        assert dict(engine._skill2intent) == {"arena": ["play_song"]}


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
        from registry import schemas
        from runner import intent_pipeline as ip_mod

        ip_mod.ENGINE_REGISTRY["toy-label-pipeline"] = ip_mod.EngineSpec(
            f"{__name__}:_ToyLabelPipeline", None, "toy")
        schemas.ENGINE_TRAITS["toy-label-pipeline"] = schemas.EngineTraits(
            paradigm="template", regime=schemas.TrainingRegime.ONLINE,
            short_name="toy")
        yield
        del ip_mod.ENGINE_REGISTRY["toy-label-pipeline"]
        del schemas.ENGINE_TRAITS["toy-label-pipeline"]

    def test_slots_filled_with_transformer_configured(self):
        pipeline = _toy_pipeline(with_transformer=True)
        pipeline.train(TRAIN_ROWS)

        intent_id, slots, confidence, _latency, stage = pipeline.predict(
            "play africa"
        )

        assert intent_id == "play_song"
        assert stage == "toy-label-pipeline-high"
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


class TestTrainingCompletes:
    def test_padatious_matches_on_first_query_after_train(self, tmp_path, monkeypatch):
        # Padatious compiles on a background thread and serves "no match"
        # until the first pass lands; a query waits at most
        # FIRST_COMPILE_WAIT_S for it. On a corpus that takes longer to
        # train than that wait (snips, banking77), every row predicted
        # before the pass landed scored as an abstention. Setting the wait
        # to zero reproduces that on a corpus small enough for a unit test,
        # and a fresh intent cache keeps the engine from loading a model a
        # previous run trained.
        from ovos_padatious.intent_container import IntentContainer

        monkeypatch.setattr(IntentContainer, "FIRST_COMPILE_WAIT_S", 0.0)
        pipeline = IntentPipeline(
            {
                "pipeline": ["ovos-padatious-pipeline-plugin-medium"],
                "ovos-padatious-pipeline-plugin": {
                    "intent_cache": str(tmp_path / "intent_cache"),
                },
            },
            lang="en-US",
        )
        pipeline.train({"template": [
            {"intent_id": "clock:time", "template": "what time is it"},
            {"intent_id": "clock:time", "template": "tell me the time"},
            {"intent_id": "fun:joke", "template": "tell me a joke"},
            {"intent_id": "fun:joke", "template": "make me laugh"},
        ]})

        intent_id, _slots, _conf, _latency, stage = pipeline.predict("what time is it")

        assert intent_id == "clock:time"
        assert stage == "ovos-padatious-pipeline-plugin-medium"


class _NeverTrains:
    """An engine whose background training never lands."""

    def __init__(self, bus, config):
        self.trained = __import__("threading").Event()

    def wait_until_trained(self, timeout=None):
        return self.trained.wait(timeout)

    def match_high(self, utterances, lang, message):
        return None


class _NoTimeoutWait(_NeverTrains):
    """A wait helper that takes no timeout argument and reports success."""

    def wait_until_trained(self):
        return True


class TestTrainingWaitTimeout:
    @pytest.fixture
    def engine(self, monkeypatch):
        from registry import schemas
        from runner import intent_pipeline as ip_mod

        def register(cls_name):
            monkeypatch.setitem(ip_mod.ENGINE_REGISTRY, "toy-train-pipeline",
                                ip_mod.EngineSpec(f"{__name__}:{cls_name}", None, "toy"))
            monkeypatch.setitem(schemas.ENGINE_TRAITS, "toy-train-pipeline",
                                schemas.EngineTraits(paradigm="template",
                                                     regime=schemas.TrainingRegime.ONLINE,
                                                     short_name="toy"))
            return IntentPipeline({"pipeline": ["toy-train-pipeline-high"]},
                                  lang="en-US", train_timeout=0.2)
        return register

    def test_training_that_never_lands_raises_within_the_timeout(self, engine):
        import time

        pipeline = engine("_NeverTrains")
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="did not finish training within 0.2s"):
            pipeline.train(TRAIN_ROWS)
        assert time.monotonic() - start < 2

    def test_wait_helper_without_timeout_argument_is_called(self, engine):
        pipeline = engine("_NoTimeoutWait")
        pipeline.train(TRAIN_ROWS)  # no TypeError, no RuntimeError
        assert pipeline.predict("play africa")[0] is None


class _SlowLoadingM2V:
    """Mirrors ovos-m2v-pipeline's cold start (``_ensure_model``): the model
    loads on a background thread and a match waits ``model_load_budget``
    (default 0.5 s) before it skips the utterance, unless ``preload_model``
    loads it synchronously at construction."""

    LOAD_SECS = 0.8

    def __init__(self, bus, config):
        import threading
        import time

        self.budget = float(config.get("model_load_budget", 0.5))
        self.loaded = threading.Event()

        def load():
            time.sleep(self.LOAD_SECS)
            self.loaded.set()

        if config.get("preload_model", False):
            load()
        else:
            threading.Thread(target=load, daemon=True).start()

    def _match(self, utterances, lang, message):
        from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch

        if not self.loaded.wait(self.budget):
            return None
        return IntentHandlerMatch(match_type="play_song", match_data={"conf": 1.0},
                                  skill_id="arena", utterance=utterances[0])

    match_high = match_medium = match_low = _match


def _m2v_fighter_blocks():
    from registry.loaders import load_all_competitors

    for comp in load_all_competitors():
        intents = comp.config.get("intents") or {}
        for plugin_id in ("ovos-m2v-pipeline", "ovos-m2v-prototype-pipeline"):
            if any(stage.startswith(f"{plugin_id}-") for stage in intents.get("pipeline", [])):
                yield pytest.param(plugin_id, dict(intents.get(plugin_id) or {}),
                                   id=f"{comp.competitor_id}:{plugin_id}")


class TestM2VColdStart:
    @pytest.mark.parametrize("plugin_id,block", list(_m2v_fighter_blocks()))
    def test_first_query_after_train_is_answered(self, monkeypatch, plugin_id, block):
        # A fighter's m2v block must load its model before the first query,
        # or the first rows of a cold run score as abstentions.
        from runner import intent_pipeline as ip_mod

        monkeypatch.setitem(ip_mod.ENGINE_REGISTRY, plugin_id,
                            ip_mod.EngineSpec(f"{__name__}:_SlowLoadingM2V", None, "toy"))
        pipeline = IntentPipeline({"pipeline": [f"{plugin_id}-high"], plugin_id: block},
                                  lang="en-US")
        pipeline.train(TRAIN_ROWS)

        assert pipeline.predict("play africa")[0] == "play_song"

    def test_the_m2v_fighters_are_collected(self):
        ids = {param.id.split(":")[0] for param in _m2v_fighter_blocks()}
        assert {"ovos-stock", "m2v-prototype", "m2v-v5-en-2m"} <= ids
