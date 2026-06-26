"""Migrate legacy ``ovos-intent-benchmark`` predictions into the arena §3.2
contract, to bootstrap intent boards/ELO for fighters the arena has not yet
swept itself.

The legacy repo (https://github.com/OpenVoiceOS/ovos-intent-benchmark) writes
one JSONL per ``predictions/<pipeline>/<dataset>/<lang>.jsonl`` with rows::

    {"utterance", "bucket", "gold", "gold_slots", "pred", "conf",
     "pred_slots", "latency_ms"}

We map each single-engine legacy pipeline to the arena competitor that runs the
same engine, drop the cascades (their stage composition does not match the
arena's named fusions one-to-one), and emit §3.2 rows. By default we only write
for competitors that have **no** native arena prediction file for that language
yet, so migrated smoke data never overwrites a real sweep.

Migrated rows carry ``migrated: true`` and a ``source_pipeline`` for
provenance; ``stage`` is null (the legacy data never recorded the firing
stage) and ``sample_id`` lives in a ``m<hash>`` id-space derived from the
utterance, so migrated fighters pair with each other in battles without
colliding with the integer sample ids of native runs.

Run::

    python -m runner.migrate.migrate_intent \
        --source /path/to/ovos-intent-benchmark/benchmark/predictions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from registry.loaders import load_all_competitors

log = logging.getLogger("migrate-intent")

# Legacy single-engine pipeline dir -> arena competitor_id. Only unambiguous
# one-engine mappings; cascades are intentionally excluded.
PIPELINE_TO_COMPETITOR: Dict[str, str] = {
    "padacioso_only": "padacioso-medium",
    "padatious_only": "padatious-medium",
    "nebulento_only": "nebulento-medium",
    "markov_only": "markov-medium",
    "linha_fina_only": "linha-fina-medium",
    "jurebes_only": "jurebes-medium",
    "adapt_only": "adapt-medium",
    "palavreado_only": "palavreado-medium",
    "m2v_only": "m2v-medium",
    "hknn_only": "hierarchical-knn-medium",
}

# Legacy dataset dir name -> arena dataset_id.
DATASET_MAP: Dict[str, str] = {
    "intents_for_eval": "intents-for-eval",
    "massive_templates": "massive-templates",
}

SOURCE_NAME = "ovos-intent-benchmark"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_id(lang: str, utterance: str) -> str:
    """Stable per-utterance id, shared across engines that ran the same set."""
    digest = hashlib.sha1(utterance.encode("utf-8")).hexdigest()[:10]
    return f"{lang}/m{digest}"


def _clean_slots(slots: Optional[dict]) -> Optional[dict]:
    """Drop null/empty slot values (legacy rows pad every slot key with null)."""
    if not slots:
        return None
    cleaned = {k: v for k, v in slots.items() if v not in (None, "")}
    return cleaned or None


def migrate_row(competitor, dataset_id: str, lang: str, legacy: dict,
                source_pipeline: str) -> dict:
    reference_intent = legacy.get("gold")
    prediction = legacy.get("pred")
    if reference_intent is None:
        exact = prediction is None  # OOD: correct behaviour is no match
    else:
        exact = prediction == reference_intent
    latency = legacy.get("latency_ms")
    return {
        "competitor_id": competitor.competitor_id,
        "sample_id": _sample_id(lang, legacy["utterance"]),
        "dataset_id": dataset_id,
        "dataset_revision": "migrated",
        "lang": lang,
        "modality": competitor.modality.value,
        "plugin_id": competitor.plugin or "ensemble",
        "plugin_version": "migrated",
        "pipeline": competitor.pipeline,
        "stage": None,
        "utterance": legacy["utterance"],
        "reference_intent": reference_intent,
        "reference_slots": _clean_slots(legacy.get("gold_slots")),
        "prediction": prediction,
        "predicted_slots": _clean_slots(legacy.get("pred_slots")),
        "exact_match": exact,
        "confidence": legacy.get("conf"),
        "bucket": legacy.get("bucket"),
        "latency_ms": round(latency, 3) if isinstance(latency, (int, float)) else None,
        "runner_version": f"{SOURCE_NAME} (migrated)",
        "created_at": _now_iso(),
        "migrated": True,
        "source_pipeline": source_pipeline,
    }


def migrate(source: Path, output_dir: Path, only_new: bool = True,
            force: bool = False) -> int:
    competitors = {c.competitor_id: c for c in load_all_competitors()}
    written_files = 0

    for source_pipeline, competitor_id in PIPELINE_TO_COMPETITOR.items():
        competitor = competitors.get(competitor_id)
        if competitor is None:
            log.warning("no registry competitor %s — skipping %s",
                        competitor_id, source_pipeline)
            continue
        modality = competitor.modality.value
        pipe_dir = source / source_pipeline
        if not pipe_dir.is_dir():
            continue

        for legacy_ds_dir in sorted(p for p in pipe_dir.iterdir() if p.is_dir()):
            dataset_id = DATASET_MAP.get(legacy_ds_dir.name)
            if dataset_id is None:
                continue
            for legacy_file in sorted(legacy_ds_dir.glob("*.jsonl")):
                lang = legacy_file.stem
                if lang.endswith(".meta"):
                    continue
                out_path = (output_dir / dataset_id / modality / lang
                            / f"{competitor.competitor_id}.jsonl")
                if out_path.exists() and only_new and not force:
                    log.info("skip %s/%s — native arena predictions exist",
                             competitor.competitor_id, lang)
                    continue

                rows = [json.loads(ln) for ln in
                        legacy_file.read_text().splitlines() if ln.strip()]
                if not rows:
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as fh:
                    for legacy in rows:
                        row = migrate_row(competitor, dataset_id, lang, legacy,
                                          source_pipeline)
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.info("%-22s %s/%s: %d rows -> %s",
                         source_pipeline, competitor.competitor_id, lang,
                         len(rows), out_path)
                written_files += 1
    return written_files


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True,
        help="ovos-intent-benchmark predictions dir "
             "(.../benchmark/predictions)")
    parser.add_argument("--output-dir", default="predictions",
                        help="arena predictions root (default: predictions)")
    parser.add_argument(
        "--all", action="store_true",
        help="also write where native arena predictions already exist "
             "(default: only competitors with no native file for that lang)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files unconditionally")
    args = parser.parse_args(argv)

    n = migrate(Path(args.source), Path(args.output_dir),
                only_new=not args.all, force=args.force)
    log.info("migrated %d prediction file(s)", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
