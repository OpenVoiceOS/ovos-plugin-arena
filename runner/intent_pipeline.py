"""
Arena-native adapter for OVOS intent pipeline plugins.

Instantiates a real OPM ``ConfidenceMatcherPipeline`` plugin over a
``FakeBus``, registers the benchmark intents the same way OVOS skills do
(bus messages), triggers the plugin's own training, and exposes a single
``predict`` call used by the bench scripts.

Two registration paradigms exist (mirroring ``intents-for-eval``):

- **template** engines (Padatious, Padacioso, Nebulento) consume
  ``train_templates.jsonl`` rows — ``padatious:register_intent`` messages
  with expanded sample utterances plus ``padatious:register_entity`` for
  slot example values.
- **keyword** engines (Adapt, Palavreado) consume ``train_keywords.jsonl``
  rows — ``register_vocab`` messages plus an ``IntentBuilder`` rule per
  intent.

The plugin's own ``match_<tier>`` confidence gates decide whether a stage
fires; the arena owns no threshold numbers.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TIERS = ("high", "medium", "low")

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
    train_message: Optional[str]  # bus message that triggers training
    dist: str  # python distribution name, for plugin_version


ENGINE_REGISTRY: Dict[str, EngineSpec] = {
    "ovos-padatious-pipeline-plugin": EngineSpec(
        "ovos_padatious.opm:PadatiousPipeline", "template",
        "mycroft.skills.train", "ovos-padatious"),
    "ovos-padacioso-pipeline-plugin": EngineSpec(
        "padacioso.opm:PadaciosoPipeline", "template",
        None, "padacioso"),
    "ovos-nebulento-pipeline-plugin": EngineSpec(
        "nebulento.opm:NebulentoPipeline", "template",
        "mycroft.skills.train", "nebulento"),
    "ovos-adapt-pipeline-plugin": EngineSpec(
        "ovos_adapt.opm:AdaptPipeline", "keyword",
        None, "ovos-adapt-parser"),
    "ovos-palavreado-pipeline-plugin": EngineSpec(
        "palavreado.opm:PalavreadoPipeline", "keyword",
        None, "palavreado"),
}


def plugin_version(plugin_id: str) -> str:
    """``<plugin_id>==<installed dist version>`` for §3.2 reproducibility."""
    spec = ENGINE_REGISTRY[plugin_id]
    try:
        return f"{plugin_id}=={importlib.metadata.version(spec.dist)}"
    except importlib.metadata.PackageNotFoundError:
        return plugin_id


def expand_template(template: str, slots: List[dict], cap: int = 6) -> List[str]:
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
    """One instantiated and trained intent pipeline plugin."""

    def __init__(self, plugin_id: str, config: Optional[dict] = None,
                 lang: str = "en-US", tier: str = "medium"):
        from ovos_utils.fakebus import FakeBus

        if plugin_id not in ENGINE_REGISTRY:
            raise KeyError(
                f"unknown intent plugin {plugin_id!r}; "
                f"known: {sorted(ENGINE_REGISTRY)}"
            )
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

        self.plugin_id = plugin_id
        self.spec = ENGINE_REGISTRY[plugin_id]
        self.lang = lang
        self.tier = tier
        self.config = dict(config or {})
        self.config.pop("tier", None)  # adapter-level setting, not plugin's

        # The OPM pipeline plugins read the active language from the global
        # ovos-config singleton, not from their plugin config — set it before
        # instantiation or every non-default language trains zero containers.
        from ovos_config.config import Configuration
        Configuration()["lang"] = lang

        module_name, class_name = self.spec.import_path.split(":")
        plugin_cls = getattr(importlib.import_module(module_name), class_name)
        self.bus = FakeBus()
        self.plugin = plugin_cls(self.bus, self.config)

    # -- training ----------------------------------------------------------

    def train(self, template_rows: List[dict], keyword_rows: List[dict]) -> None:
        if self.spec.paradigm == "template":
            self._register_templates(template_rows)
        else:
            self._register_keywords(keyword_rows)

        from ovos_bus_client.message import Message

        if self.spec.train_message:
            self.bus.emit(Message(self.spec.train_message, {}))
        if hasattr(self.plugin, "train"):
            try:
                self.plugin.train()
            except TypeError:
                pass  # train() signatures vary; bus message already fired

    def _register_templates(self, rows: List[dict]) -> None:
        from ovos_bus_client.message import Message

        by_intent: Dict[str, List[dict]] = {}
        entities: Dict[str, List[str]] = {}
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
            samples: List[str] = []
            for row in intent_rows:
                template = row.get("template", "")
                samples.append(template)
                samples.extend(expand_template(template, row.get("slots")))
            samples = list(dict.fromkeys(s for s in samples if s))
            if not samples:
                continue
            self.bus.emit(Message("padatious:register_intent", {
                "name": intent_id,
                "samples": samples,
                "lang": self.lang,
                "skill_id": "arena",
            }))

        # Entities registered once per name — merged example values across
        # intents (some engines raise on re-registration)
        for name, samples in entities.items():
            self.bus.emit(Message("padatious:register_entity", {
                "name": name,
                "samples": samples,
                "lang": self.lang,
                "skill_id": "arena",
            }))

    def _register_keywords(self, rows: List[dict]) -> None:
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
                        self.bus.emit(Message("register_vocab", {
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
            self.bus.emit(Message("register_intent", data))

    # -- prediction ----------------------------------------------------------

    def predict(
        self, utterance: str
    ) -> Tuple[Optional[str], Dict[str, str], Optional[float], float]:
        """Match one utterance.

        Returns ``(intent_id, slots, confidence, latency_ms)`` — intent_id is
        None when the plugin's own ``match_<tier>`` gate does not fire.
        """
        from ovos_bus_client.message import Message

        match_fn = getattr(self.plugin, f"match_{self.tier}")
        message = Message(
            "recognizer_loop:utterance",
            {"utterances": [utterance], "lang": self.lang},
        )
        start = time.perf_counter()
        try:
            match = match_fn([utterance], self.lang, message)
        except Exception as exc:
            logger.warning("%s failed on %r: %s", self.plugin_id, utterance, exc)
            match = None
        latency_ms = (time.perf_counter() - start) * 1000

        if match is None:
            return None, {}, None, latency_ms

        intent_id = self._normalise(str(match.match_type))
        data = dict(match.match_data or {})
        confidence = data.get("conf", data.get("confidence"))
        slots = self._extract_slots(data)
        return intent_id, slots, confidence, latency_ms

    @staticmethod
    def _normalise(match_type: str) -> str:
        """Strip skill-id prefixes some engines prepend to the intent name."""
        for sep in (":", "."):
            prefix = f"arena{sep}"
            if match_type.startswith(prefix):
                return match_type[len(prefix):]
        return match_type

    @staticmethod
    def _extract_slots(match_data: dict) -> Dict[str, str]:
        """Best-effort slot extraction from IntentHandlerMatch.match_data.

        Vocabulary groups named ``*Kw`` are keyword triggers, not slots
        (``intents-for-eval`` convention) and are skipped.
        """
        entities = match_data.get("entities")
        if isinstance(entities, dict):
            return {str(k): str(v) for k, v in entities.items()}

        slots: Dict[str, str] = {}
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
