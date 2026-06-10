"""
Intent adapter for the OVOS Plugin Arena.

Runs an OVOS intent engine over a set of test utterances and computes
exact-match accuracy (F1 / precision / recall against reference labels).

Input format
------------
``input_ref`` is a JSON string (or path to a JSON file) with the schema::

    {
      "utterance": "turn on the living room lights",
      "expected_intent": "LightsOnIntent",
      "expected_entities": {"location": "living room"}
    }

The adapter accepts plugins implementing the ``OVOSIntentMatcherInterface``
(``ovos_plugin_manager.templates.intent``).

For padatious the adapter trains on the intents_dir if provided.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.arena.adapters.base import BaseAdapter
from app.arena.models import PluginFamily

logger = logging.getLogger(__name__)


class IntentAdapter(BaseAdapter):
    """
    Concrete adapter for OVOS intent plugins.

    Parameters
    ----------
    plugin_name  : e.g. "padatious" or "ovos-adapt-plugin"
    config       : plugin configuration dict
    intents_dir  : directory containing .intent files (padatious / padaos style)
    lang         : language tag
    """

    family = PluginFamily.INTENT

    def __init__(
        self,
        plugin_name: str,
        config: Optional[Dict[str, Any]] = None,
        intents_dir: Optional[str] = None,
        lang: str = "en-us",
    ) -> None:
        super().__init__(plugin_name, config)
        self.intents_dir = intents_dir
        self.lang = lang
        self._engine: Any = None

    def _load_plugin(self) -> None:
        if self.plugin_name == "padatious":
            self._load_padatious()
        elif self.plugin_name in ("ovos-adapt-plugin", "adapt"):
            self._load_adapt()
        else:
            self._load_opm_intent()

    def _load_padatious(self) -> None:
        from padatious import IntentContainer

        self._engine = IntentContainer(cache_dir=self.config.get("cache_dir", "/tmp/padatious_cache"))
        if self.intents_dir:
            idir = Path(self.intents_dir)
            for intent_file in idir.glob("*.intent"):
                intent_name = intent_file.stem
                utterances = [
                    line.strip()
                    for line in intent_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                self._engine.load_intent(intent_name, utterances)
            self._engine.train(debug=False)
        logger.info("Loaded padatious with intents_dir=%s", self.intents_dir)

    def _load_adapt(self) -> None:
        try:
            from ovos_adapt_plugin import AdaptExtractor
            self._engine = AdaptExtractor(self.config)
            logger.info("Loaded adapt engine")
        except ImportError:
            from adapt.intent import IntentBuilder
            from adapt.engine import IntentDeterminationEngine
            self._engine = IntentDeterminationEngine()
            logger.info("Loaded adapt (standalone)")

    def _load_opm_intent(self) -> None:
        from ovos_plugin_manager.intent import OVOSIntentFactory

        try:
            self._engine = OVOSIntentFactory.create(
                {"module": self.plugin_name, **self.config}
            )
            logger.info("Loaded intent plugin: %s", self.plugin_name)
        except Exception as e:
            logger.error("Failed to load intent plugin %s: %s", self.plugin_name, e)
            raise

    def _unload_plugin(self) -> None:
        self._engine = None

    def _run_one(
        self,
        input_ref: str,
        output_dir: Optional[str],
    ) -> tuple[Optional[str], Dict[str, float]]:
        """
        Predict intent for a single *input_ref*.

        *input_ref* is either:
        - a raw utterance string, or
        - a JSON string with keys ``utterance``, ``expected_intent``, ``expected_entities``

        Returns
        -------
        (None, metrics_dict)  — intent adapters produce no audio artifacts
        """
        # Parse input
        try:
            data = json.loads(input_ref)
            utterance = data.get("utterance", input_ref)
            expected_intent = data.get("expected_intent")
            expected_entities: Dict[str, str] = data.get("expected_entities", {})
        except (json.JSONDecodeError, TypeError):
            utterance = input_ref
            expected_intent = None
            expected_entities = {}

        # Run prediction
        predicted_intent = None
        predicted_entities: Dict[str, str] = {}
        confidence = 0.0

        try:
            result = self._predict(utterance)
            if result:
                predicted_intent = result.get("intent_type") or result.get("name")
                predicted_entities = {
                    k: v
                    for k, v in result.items()
                    if k not in ("intent_type", "name", "confidence", "utterance")
                }
                confidence = float(result.get("confidence", 0.0))
        except Exception as e:
            logger.warning("Intent prediction failed for '%s': %s", utterance[:60], e)

        metrics: Dict[str, float] = {"confidence": round(confidence, 4)}

        if expected_intent is not None:
            intent_match = 1.0 if predicted_intent == expected_intent else 0.0
            metrics["intent_exact_match"] = intent_match

            # Entity F1
            if expected_entities:
                f1 = _entity_f1(predicted_entities, expected_entities)
                metrics["entity_f1"] = round(f1, 4)
            else:
                metrics["entity_f1"] = 1.0 if intent_match else 0.0

        return None, metrics

    def _predict(self, utterance: str) -> Optional[Dict[str, Any]]:
        """Dispatch prediction to the loaded engine."""
        if hasattr(self._engine, "calc_intent"):
            result = self._engine.calc_intent(utterance)
            if result and hasattr(result, "__dict__"):
                return result.__dict__
            return result
        if hasattr(self._engine, "determine_intent"):
            for r in self._engine.determine_intent(utterance):
                return r
        return None


def _entity_f1(predicted: Dict[str, str], expected: Dict[str, str]) -> float:
    """Compute token-level entity F1 between predicted and expected entity dicts."""
    if not expected:
        return 1.0
    pred_set = set(predicted.items())
    exp_set = set(expected.items())
    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(exp_set) if exp_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
