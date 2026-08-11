"""
Arena-native executor for OVOS intent pipelines.

A fighter's ``config`` is a valid ``mycroft.conf`` fragment: the
``intents`` section carries an ordered ``pipeline`` list of
``<plugin>-<tier>`` stages plus per-plugin config blocks.  This module
instantiates the real OPM ``ConfidenceMatcherPipeline`` plugins over a
``FakeBus``, registers the benchmark intents the same way OVOS skills do
(bus messages), triggers each plugin's own training, and runs utterances
through the cascade — the first stage whose own ``match_<tier>`` gate
fires wins, exactly like ovos-core dispatches its pipeline.

Two registration paradigms exist (mirroring ``intents-for-eval``):

- **template** engines (Padatious, Padacioso, Nebulento) consume
  ``train_templates.jsonl`` rows — ``padatious:register_intent`` messages
  with raw ``{slot}`` templates plus expanded sample utterances, and
  ``padatious:register_entity`` for slot example values.
- **keyword** engines (Adapt, Palavreado) consume ``train_keywords.jsonl``
  rows — ``register_vocab`` messages plus an ``IntentBuilder`` rule per
  intent.

The arena owns no confidence numbers; the plugins gate themselves.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import time
from dataclasses import dataclass
from typing import Any

from registry.schemas import split_pipeline_stage

logger = logging.getLogger(__name__)

# Keys in IntentHandlerMatch.match_data that are not slot entities
_META_KEYS = {
    "name", "intent_name", "conf", "confidence", "utterance", "utterances",
    "lang", "skill_id", "intent_type", "target", "__tags__",
}


@dataclass(frozen=True)
class EngineSpec:
    """How to load, train and version one OPM intent pipeline plugin."""

    import_path: str  # "module:ClassName"
    paradigm: str  # "template" | "keyword"
    train_message: str | None  # bus message that triggers training
    dist: str  # python distribution name, for plugin_version
    short_name: str  # legacy mycroft.conf config key (e.g. "adapt")


ENGINE_REGISTRY: dict[str, EngineSpec] = {
    "ovos-padatious-pipeline-plugin": EngineSpec(
        "ovos_padatious.opm:PadatiousPipeline", "template",
        "mycroft.skills.train", "ovos-padatious", "padatious"),
    "ovos-padacioso-pipeline-plugin": EngineSpec(
        "padacioso.opm:PadaciosoPipeline", "template",
        None, "padacioso", "padacioso"),
    "ovos-nebulento-pipeline-plugin": EngineSpec(
        "nebulento.opm:NebulentoPipeline", "template",
        "mycroft.skills.train", "nebulento", "nebulento"),
    "ovos-adapt-pipeline-plugin": EngineSpec(
        "ovos_adapt.opm:AdaptPipeline", "keyword",
        None, "ovos-adapt-parser", "adapt"),
    "ovos-palavreado-pipeline-plugin": EngineSpec(
        "palavreado.opm:PalavreadoPipeline", "keyword",
        None, "palavreado", "palavreado"),
    "ovos-jurebes-pipeline-plugin": EngineSpec(
        "jurebes.opm:JurebesPipeline", "template",
        "mycroft.ready", "jurebes", "jurebes"),
    "ovos-linha-fina-pipeline-plugin": EngineSpec(
        "linha_fina.opm:LinhaFinaPipeline", "template",
        "mycroft.ready", "linha-fina", "linha_fina"),
    "ovos-markov-pipeline-plugin": EngineSpec(
        "ovos_markov_pipeline:MarkovPipeline", "template",
        "mycroft.skills.train", "ovos-markov-pipeline-plugin", "markov"),
    "ovos-m2v-pipeline": EngineSpec(
        "ovos_m2v_pipeline:Model2VecIntentPipeline", "template",
        "mycroft.ready", "ovos-m2v-pipeline", "m2v"),
    "ovos-hierarchical-knn-pipeline": EngineSpec(
        "ovos_hierarchical_knn_pipeline:HierarchicalKNNIntentPipeline",
        "template",
        "mycroft.ready", "ovos-hierarchical-knn-pipeline", "hknn"),
    # Domain/hierarchical two-stage variants — separate OPM entry points
    # (opm.pipeline) from the same distributions as their flat siblings
    # above; wired in here so the arena can dispatch to them.
    "ovos-nebulento-hierarchical-pipeline-plugin": EngineSpec(
        "nebulento.opm:HierarchicalNebulentoPipeline", "template",
        "mycroft.skills.train", "nebulento", "nebulento_hierarchical"),
    "ovos-linha-fina-domain-pipeline-plugin": EngineSpec(
        "linha_fina.opm:DomainLinhaFinaPipeline", "template",
        "mycroft.ready", "linha-fina", "linha_fina_domain"),
    "ovos-linha-fina-hierarchical-pipeline-plugin": EngineSpec(
        "linha_fina.hierarchical_opm:HierarchicalLinhaFinaPipeline", "template",
        "mycroft.ready", "linha-fina", "linha_fina_hierarchical"),
    "ovos-markov-domain-pipeline-plugin": EngineSpec(
        "ovos_markov_pipeline:DomainMarkovPipeline", "template",
        "mycroft.skills.train", "ovos-markov-pipeline-plugin", "markov_domain"),
}


def plugin_version(plugin_id: str) -> str:
    """``<plugin_id>==<installed dist version>`` for §3.2 reproducibility."""
    spec = ENGINE_REGISTRY[plugin_id]
    try:
        return f"{plugin_id}=={importlib.metadata.version(spec.dist)}"
    except importlib.metadata.PackageNotFoundError:
        return plugin_id


def expand_template(template: str, slots: list[dict], cap: int = 6) -> list[str]:
    """Expand ``{slot}`` placeholders with example values.

    Engines that natively parse ``{slot}`` / ``(a|b)`` syntax also receive
    the expanded forms — harmless duplication that keeps the adapter
    engine-agnostic, while engines without template support still see
    concrete utterances.
    """
    used = [s for s in (slots or [])
            if s.get("examples") and "{" + s["name"] + "}" in template]
    if not used:
        return [template]
    n = min(cap, max(len(s["examples"]) for s in used))
    out = []
    for i in range(n):
        utterance = template
        for slot in used:
            examples = slot["examples"]
            utterance = utterance.replace(
                "{" + slot["name"] + "}", examples[i % len(examples)]
            )
        out.append(utterance)
    return out


class IntentPipeline:
    """One instantiated and trained intent pipeline (single- or multi-stage).

    Parameters
    ----------
    intents_config:
        The ``intents`` section of a mycroft.conf fragment:
        ``{"pipeline": ["<plugin>-<tier>", …], "<plugin>": {…}, …}``.
    lang:
        BCP-47 language to train and match in.
    """

    def __init__(self, intents_config: dict[str, Any], lang: str = "en-US"):
        # Validate the pipeline before touching the OVOS runtime so config
        # errors raise even where the engines are not installed.
        pipeline = intents_config.get("pipeline") or []
        if not pipeline:
            raise ValueError("intents config carries no pipeline stages")

        self.lang = lang
        self.stages: list[tuple[str, str]] = []  # (plugin_id, tier), ordered
        for stage in pipeline:
            plugin_id, tier = split_pipeline_stage(stage)
            if plugin_id not in ENGINE_REGISTRY:
                raise KeyError(
                    f"unknown intent plugin {plugin_id!r} in pipeline; "
                    f"known: {sorted(ENGINE_REGISTRY)}"
                )
            self.stages.append((plugin_id, tier))

        # The OPM pipeline plugins read the active language from the global
        # ovos-config singleton, not from their plugin config — set it before
        # instantiation or every non-default language trains zero containers.
        from ovos_config.config import Configuration
        from ovos_utils.fakebus import FakeBus
        Configuration()["lang"] = lang

        # One plugin instance per unique plugin id, shared across tiers,
        # each with its own per-plugin config block from the intents section
        self.plugins: dict[str, Any] = {}
        self.buses: dict[str, Any] = {}
        for plugin_id, _tier in self.stages:
            if plugin_id in self.plugins:
                continue
            spec = ENGINE_REGISTRY[plugin_id]
            plugin_config = dict(
                intents_config.get(plugin_id)
                or intents_config.get(spec.short_name)
                or {}
            )
            module_name, class_name = spec.import_path.split(":")
            plugin_cls = getattr(importlib.import_module(module_name), class_name)
            bus = FakeBus()
            self.plugins[plugin_id] = plugin_cls(bus, plugin_config)
            self.buses[plugin_id] = bus

    @property
    def stage_names(self) -> list[str]:
        return [f"{plugin_id}-{tier}" for plugin_id, tier in self.stages]

    # -- training ----------------------------------------------------------

    def train(self, train_data: dict[str, list[dict]]) -> None:
        """Train every stage plugin from its paradigm's training corpus.

        *train_data* maps paradigm → rows in that paradigm's datashape
        (``"template"`` rows vs ``"keyword"`` rows are different datasets —
        see the registry's ``role: train`` entries).
        """
        from ovos_bus_client.message import Message

        for plugin_id, plugin in self.plugins.items():
            spec = ENGINE_REGISTRY[plugin_id]
            bus = self.buses[plugin_id]
            rows = train_data.get(spec.paradigm) or []
            if not rows:
                raise ValueError(
                    f"No {spec.paradigm!r}-paradigm training rows for "
                    f"{plugin_id} — check the dataset's train_datasets links"
                )
            if spec.paradigm == "template":
                self._register_templates(bus, rows)
            else:
                self._register_keywords(bus, rows)

            if spec.train_message:
                bus.emit(Message(spec.train_message, {}))
            if hasattr(plugin, "train"):
                try:
                    plugin.train()
                except TypeError:
                    pass  # train() signatures vary; bus message already fired

    def _register_templates(self, bus, rows: list[dict]) -> None:
        from ovos_bus_client.message import Message

        by_intent: dict[str, list[dict]] = {}
        entities: dict[str, list[str]] = {}
        for row in rows:
            by_intent.setdefault(row["intent_id"], []).append(row)
            for slot in row.get("slots") or []:
                if slot.get("examples"):
                    merged = entities.setdefault(slot["name"], [])
                    merged.extend(
                        e for e in slot["examples"] if e not in merged
                    )

        for intent_id, intent_rows in by_intent.items():
            # Raw templates keep the {slot} placeholders for engines that
            # parse them natively (slot capture); the expanded forms give
            # engines without template support concrete utterances.
            samples: list[str] = []
            for row in intent_rows:
                template = row.get("template", "")
                samples.append(template)
                samples.extend(expand_template(template, row.get("slots")))
            samples = list(dict.fromkeys(s for s in samples if s))
            if not samples:
                continue
            bus.emit(Message("padatious:register_intent", {
                "name": intent_id,
                "samples": samples,
                "lang": self.lang,
                "skill_id": "arena",
            }))

        # Entities registered once per name — merged example values across
        # intents (some engines raise on re-registration)
        for name, samples in entities.items():
            bus.emit(Message("padatious:register_entity", {
                "name": name,
                "samples": samples,
                "lang": self.lang,
                "skill_id": "arena",
            }))

    def _register_keywords(self, bus, rows: list[dict]) -> None:
        from ovos_adapt.intent import IntentBuilder
        from ovos_bus_client.message import Message

        for row in rows:
            intent_id = row["intent_id"]
            builder = IntentBuilder(intent_id)
            for group, vocab_map in (
                ("required", row.get("required_vocab") or {}),
                ("optional", row.get("optional_vocab") or {}),
            ):
                for label, words in vocab_map.items():
                    if not words:
                        continue
                    entity = f"{intent_id}__{label}"
                    for word in words:
                        bus.emit(Message("register_vocab", {
                            "entity_value": word,
                            "entity_type": entity,
                            "lang": self.lang,
                            "skill_id": "arena",
                        }))
                    builder = (builder.require(entity) if group == "required"
                               else builder.optionally(entity))
            intent = builder.build()
            data = dict(intent.__dict__)
            data["skill_id"] = "arena"
            bus.emit(Message("register_intent", data))

    # -- prediction ----------------------------------------------------------

    def predict(
        self, utterance: str
    ) -> tuple[str | None, dict[str, str], float | None, float, str | None]:
        """Run one utterance through the cascade.

        Returns ``(intent_id, slots, confidence, latency_ms, fired_stage)``;
        intent_id and fired_stage are None when no stage's gate fires.
        """
        from ovos_bus_client.message import Message

        message = Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": self.lang},
        )
        start = time.perf_counter()
        for plugin_id, tier in self.stages:
            match_fn = getattr(self.plugins[plugin_id], f"match_{tier}")
            try:
                match = match_fn([utterance], self.lang, message)
            except Exception as exc:
                logger.warning("%s-%s failed on %r: %s",
                               plugin_id, tier, utterance, exc)
                match = None
            if match is None:
                continue

            latency_ms = (time.perf_counter() - start) * 1000
            intent_id = self._normalise(str(match.match_type))
            raw_data = match.match_data
            data = dict(raw_data) if isinstance(raw_data, dict) else {}
            confidence = data.get("conf", data.get("confidence"))
            slots = self._extract_slots(data)
            return intent_id, slots, confidence, latency_ms, f"{plugin_id}-{tier}"

        latency_ms = (time.perf_counter() - start) * 1000
        return None, {}, None, latency_ms, None

    @staticmethod
    def _normalise(match_type: str) -> str:
        """Strip skill-id prefixes some engines prepend to the intent name."""
        for sep in (":", "."):
            prefix = f"arena{sep}"
            if match_type.startswith(prefix):
                return match_type[len(prefix):]
        return match_type

    @staticmethod
    def _extract_slots(match_data: dict) -> dict[str, str]:
        """Best-effort slot extraction from IntentHandlerMatch.match_data.

        Vocabulary groups named ``*Kw`` are keyword triggers, not slots
        (``intents-for-eval`` convention) and are skipped.
        """
        entities = match_data.get("entities")
        if isinstance(entities, dict):
            return {str(k): str(v) for k, v in entities.items()}

        slots: dict[str, str] = {}
        # palavreado style: {"keywords": {"<intent>__<group>": [values]}}
        keywords = match_data.get("keywords")
        if isinstance(keywords, dict):
            for key, values in keywords.items():
                name = key.split("__", 1)[1] if "__" in key else key
                if name.endswith("Kw") or not values:
                    continue
                slots[name] = str(values[0] if isinstance(values, list) else values)
            return slots

        # adapt style: {"<intent>__<group>": "value"}; nebulento style:
        # {"<slot>": ["value", …]}
        for key, value in match_data.items():
            if key in _META_KEYS:
                continue
            if isinstance(value, list):
                value = value[0] if value and isinstance(value[0], str) else None
            if not isinstance(value, str):
                continue
            name = key.split("__", 1)[1] if "__" in key else key
            if name.endswith("Kw"):
                continue
            slots[name] = value
        return slots
