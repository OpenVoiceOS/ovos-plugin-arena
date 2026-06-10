"""
Intent prediction runner — evaluate a pipeline plugin over a labelled
utterance dataset and write per-utterance prediction rows to a JSONL file.

Implements §3.1/§3.2 for the intent modality:

- Competitor is identified by a ``registry/competitors/intent/<id>.json`` file.
- Dataset is identified by a ``registry/datasets/intent/<id>.json`` file.
- Output: ``predictions/<competitor_id>.jsonl``, one row per utterance, shape::

      {
        "competitor_id": "adapt-default-en",
        "sample_id": "clinc150_test_0042",
        "dataset_id": "clinc150-en",
        "lang": "en-US",
        "plugin_id": "ovos-adapt-pipeline-plugin",
        "plugin_version": "ovos-adapt-pipeline-plugin/default",
        "utterance": "what is the weather like today",
        "reference_intent": "get_weather",
        "prediction": "WeatherIntent",
        "exact_match": false,
        "entity_f1": 0.0,
        "runner_version": "0.1.0",
        "created_at": "2026-06-11T00:00:00"
      }

Fairness note (§3 intent para):
  Intent modality predictions run end-to-end through the OVOS pipeline plugin
  (full pipeline routing via ``plugin.match_intent``).  Full ovoscope e2e
  routing is gated on TigreGotico/ovoscope#64; this runner uses the plugin's
  ``match_intent`` method directly which exercises the same matching logic.

Usage (CLI)::

    python -m runner.intent_runner \\
        --competitor adapt-default-en \\
        --dataset clinc150-en \\
        --output-dir predictions/ \\
        [--max-samples 50]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

RUNNER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Schema for one intent prediction row (§3.2 contract)
# ---------------------------------------------------------------------------


def _make_row(
    competitor_id: str,
    sample_id: str,
    dataset_id: str,
    lang: str,
    plugin_id: str,
    plugin_version: str,
    utterance: str,
    reference_intent: str,
    prediction: Optional[str],
    entity_f1: float = 0.0,
) -> dict:
    exact = (
        prediction is not None
        and prediction.lower().rstrip("intent") == reference_intent.lower().rstrip("intent")
    ) or (
        prediction is not None
        and prediction == reference_intent
    )
    return {
        "competitor_id": competitor_id,
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "lang": lang,
        "plugin_id": plugin_id,
        "plugin_version": plugin_version,
        "utterance": utterance,
        "reference_intent": reference_intent,
        "prediction": prediction,
        "exact_match": exact,
        "entity_f1": entity_f1,
        "runner_version": RUNNER_VERSION,
        "created_at": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Dataset streaming — HuggingFace, parquet-only (mirrors plugin_runner style)
# ---------------------------------------------------------------------------


def _stream_intent_dataset(
    hf_id: str,
    split: str,
    subset: Optional[str],
    utterance_key: str,
    intent_key: str,
    max_samples: int = 0,
    revision: str = "main",
) -> Iterator[Tuple[str, str, str]]:
    """Yield (sample_id, utterance, reference_intent) tuples.

    Uses the datasets library for intent datasets (no audio decoding needed).
    Falls back to direct parquet streaming if datasets is unavailable.
    """
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load(
            hf_id,
            name=subset,
            split=split,
            streaming=True,
            revision=revision,
            trust_remote_code=False,
        )
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            utterance = row.get(utterance_key) or row.get("text") or ""
            intent = row.get(intent_key)
            if not utterance:
                continue
            # intent may be an int (class index) — convert to str
            if isinstance(intent, int):
                # Try to get label names from dataset features
                try:
                    features = ds.features if hasattr(ds, "features") else None
                    if features and intent_key in features:
                        feat = features[intent_key]
                        if hasattr(feat, "names"):
                            intent = feat.names[intent]
                except Exception:
                    intent = str(intent)
            intent = str(intent) if intent is not None else "unknown"
            yield f"sample_{i:06d}", utterance, intent
    except Exception as exc:
        raise RuntimeError(f"Cannot stream intent dataset {hf_id}: {exc}") from exc


# ---------------------------------------------------------------------------
# Plugin loading and matching
# ---------------------------------------------------------------------------


def _load_pipeline_plugin(plugin_name: str, config: dict, lang: str):
    """Load an OVOS pipeline plugin and return the instance."""
    from ovos_plugin_manager.pipeline import load_pipeline_plugin as _load
    from ovos_utils.fakebus import FakeBus

    bus = FakeBus()
    plugin_cls = _load(plugin_name)
    if plugin_cls is None:
        raise RuntimeError(f"Pipeline plugin not found: {plugin_name}")
    instance = plugin_cls(bus=bus, config=config)
    return instance, bus


def _match_utterance(
    plugin,
    bus,
    utterance: str,
    lang: str,
) -> Optional[str]:
    """Run one utterance through the plugin and return the matched intent name or None."""
    from ovos_bus_client.message import Message

    msg = Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": lang},
    )
    result = plugin.match_intent((utterance,), lang, msg.serialize())
    if result is None:
        return None
    # match_type is "skill_id:IntentName" — return full string
    return result.match_type


# ---------------------------------------------------------------------------
# Adapt vocabulary / intent registration helpers
# ---------------------------------------------------------------------------


def _register_adapt_vocab_from_clinc(bus, lang: str = "en-US") -> None:
    """Register a minimal set of Adapt vocabulary for the CLINC-150 intents.

    This is a demonstration set sufficient to resolve a small sample of
    CLINC-150 utterances.  A production competitor would ship a full
    ``.voc`` / ``.intent`` skill instead.
    """
    from ovos_bus_client.message import Message
    from adapt.intent import IntentBuilder
    import time

    # (vocab_word, EntityType)
    vocab = [
        ("weather", "Weather"),
        ("forecast", "Weather"),
        ("temperature", "Weather"),
        ("translate", "Translate"),
        ("translation", "Translate"),
        ("timer", "Timer"),
        ("alarm", "Alarm"),
        ("reminder", "Reminder"),
        ("remind", "Reminder"),
        ("news", "News"),
        ("headlines", "News"),
        ("joke", "Joke"),
        ("funny", "Joke"),
        ("directions", "Navigation"),
        ("navigate", "Navigation"),
        ("traffic", "Traffic"),
        ("balance", "BankBalance"),
        ("account", "BankAccount"),
        ("transfer", "BankTransfer"),
        ("flight", "Flight"),
        ("book", "Booking"),
        ("hotel", "Hotel"),
        ("restaurant", "Restaurant"),
        ("recipe", "Recipe"),
        ("calorie", "Nutrition"),
        ("calories", "Nutrition"),
        ("nutrition", "Nutrition"),
        ("exercise", "Exercise"),
        ("workout", "Exercise"),
        ("meaning", "Definition"),
        ("definition", "Definition"),
        ("synonym", "Definition"),
        ("currency", "Currency"),
        ("exchange", "Currency"),
        ("convert", "Convert"),
    ]

    for word, entity_type in vocab:
        bus.emit(Message("register_vocab", {"entity_value": word, "entity_type": entity_type}))
    time.sleep(0.2)

    # (intent_name, required_entity)
    intents = [
        ("get_weather", "Weather"),
        ("translate", "Translate"),
        ("timer", "Timer"),
        ("alarm", "Alarm"),
        ("reminder", "Reminder"),
        ("news", "News"),
        ("joke", "Joke"),
        ("navigate", "Navigation"),
        ("traffic", "Traffic"),
        ("check_balance", "BankBalance"),
        ("account_info", "BankAccount"),
        ("transfer", "BankTransfer"),
        ("flight_status", "Flight"),
        ("book_hotel", "Hotel"),
        ("find_restaurant", "Restaurant"),
        ("get_recipe", "Recipe"),
        ("nutrition_info", "Nutrition"),
        ("exercise_info", "Exercise"),
        ("word_definition", "Definition"),
        ("currency_exchange", "Currency"),
        ("convert_units", "Convert"),
    ]

    for intent_name, entity in intents:
        full_name = f"arena_adapt_demo:{intent_name}"
        builder = IntentBuilder(full_name).require(entity)
        bus.emit(Message("register_intent", builder.build().__dict__))

    time.sleep(0.5)
    logger.info("Registered %d vocab entries and %d intents", len(vocab), len(intents))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_intent_job(
    competitor_id: str,
    dataset_id: str,
    output_dir: Path,
    registry_root: Optional[Path] = None,
    max_samples: int = 0,
) -> Path:
    """Run an intent prediction job.

    Loads competitor and dataset definitions from the registry, instantiates
    the pipeline plugin, streams utterances from the dataset, runs each
    utterance through the plugin, and writes rows to a JSONL file.

    Returns the path to the output JSONL file.
    """
    import sys

    # Load registry definitions
    rr = registry_root or (Path(__file__).parent.parent / "registry")
    sys.path.insert(0, str(rr.parent))

    from registry.loaders import load_competitor, load_dataset

    comp = load_competitor("intent", competitor_id)
    ds_def = load_dataset("intent", dataset_id)

    # Resolve output path
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{competitor_id}.jsonl"

    # Load existing rows to enable resume
    done_ids: set = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    done_ids.add(obj.get("sample_id", ""))
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming: %d rows already done", len(done_ids))

    # Plugin version string
    plugin_version = f"{comp.plugin}/{comp.config.get('model', 'default')}"
    lang = comp.langs[0] if comp.langs else "en-US"

    # Load plugin
    logger.info("Loading plugin: %s (config=%s)", comp.plugin, comp.config)
    plugin, bus = _load_pipeline_plugin(comp.plugin, comp.config, lang)

    # Register vocabulary for Adapt (demo competency)
    if "adapt" in comp.plugin.lower():
        _register_adapt_vocab_from_clinc(bus, lang)

    # Resolve source fields
    src = ds_def.source
    ref = ds_def.reference_fields
    utterance_key = ref.get("utterance", "text")
    intent_key = ref.get("intent", "intent")
    hf_id = src.hf_id
    split = src.split
    subset = src.subset if hasattr(src, "subset") else None
    revision = src.revision if hasattr(src, "revision") else "main"

    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for sample_id, utterance, reference_intent in _stream_intent_dataset(
            hf_id=hf_id,
            split=split,
            subset=subset,
            utterance_key=utterance_key,
            intent_key=intent_key,
            max_samples=max_samples or ds_def.source.dict().get("max_samples", 0) if hasattr(ds_def.source, "dict") else max_samples,
            revision=revision,
        ):
            if sample_id in done_ids:
                continue

            try:
                prediction = _match_utterance(plugin, bus, utterance, lang)
            except Exception as exc:
                logger.warning("Error on sample %s: %s", sample_id, exc)
                prediction = None

            row = _make_row(
                competitor_id=competitor_id,
                sample_id=sample_id,
                dataset_id=dataset_id,
                lang=lang,
                plugin_id=comp.plugin,
                plugin_version=plugin_version,
                utterance=utterance,
                reference_intent=reference_intent,
                prediction=prediction,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1

    logger.info("Intent job done: %s  rows_written=%d", competitor_id, written)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Intent prediction runner")
    p.add_argument("--competitor", required=True, help="Competitor ID from registry")
    p.add_argument("--dataset", required=True, help="Dataset ID from registry")
    p.add_argument("--output-dir", default="predictions", help="Output directory for JSONL files")
    p.add_argument("--max-samples", type=int, default=0, help="Cap on samples (0=all)")
    p.add_argument("--registry-root", default=None, help="Path to registry/ dir (default: auto)")
    args = p.parse_args(argv)

    registry_root = Path(args.registry_root) if args.registry_root else None
    out = run_intent_job(
        competitor_id=args.competitor,
        dataset_id=args.dataset,
        output_dir=Path(args.output_dir),
        registry_root=registry_root,
        max_samples=args.max_samples,
    )
    print(f"Output: {out}")


if __name__ == "__main__":
    _main()
