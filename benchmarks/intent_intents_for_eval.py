#!/usr/bin/env python3
"""
Intent benchmark over ``OpenVoiceOS/intents-for-eval``.

Every benchmark in the arena is one dedicated, reproducible script.  This
one runs each intent fighter from the registry — a fighter's config is a
valid mycroft.conf fragment (an ``intents`` section with a tier-suffixed
``pipeline`` plus per-plugin config blocks) — over the eval corpus, and
publishes fresh per-sample predictions.  Single-stage pipelines benchmark
one engine; multi-stage pipelines are ensemble fighters.

Training data is paradigm-specific: the eval dataset entry links one
``role: train`` dataset per paradigm (template rows vs keyword rules are
different datashapes), and each stage plugin trains from the corpus
matching its paradigm.

Per (fighter, lang) it:

1. downloads the eval split and both training corpora from HF (revision
   pinned at run start and recorded in every row);
2. instantiates + trains the pipeline (``runner.intent_pipeline``);
3. writes one §3.2 prediction row per test utterance to
   ``predictions/<competitor_id>.jsonl`` (resumable — already-done
   (lang, sample) pairs are skipped); rows record which pipeline stage
   fired;
4. optionally uploads the JSONL files to the HF results dataset repo.

Usage::

    python benchmarks/intent_intents_for_eval.py                  # full run
    python benchmarks/intent_intents_for_eval.py --langs en-US \\
        --competitors padatious-medium --max-samples 20           # smoke run
    python benchmarks/intent_intents_for_eval.py --upload         # + publish

The default results repo is ``OpenVoiceOS/ovos-intent-bench-intents-for-eval``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.version import __version__ as ARENA_VERSION  # noqa: E402
from registry.loaders import list_competitors, load_dataset  # noqa: E402
from runner.intent_pipeline import (  # noqa: E402
    ENGINE_REGISTRY,
    IntentPipeline,
    plugin_version,
)

log = logging.getLogger("intent-bench")

DATASET_ID = "intents-for-eval"
DEFAULT_RESULTS_REPO = "OpenVoiceOS/ovos-intent-bench-intents-for-eval"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def resolve_revision(hf_id: str, revision: str) -> str:
    """Pin a branch name to the commit sha it points at right now."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(hf_id, revision=revision)
    return info.sha or revision


def fetch_rows(dataset_def, lang: str, revision: str) -> list[dict]:
    """Fetch one language's rows of a file-pattern dataset."""
    from huggingface_hub import hf_hub_download

    pattern = dataset_def.source.file_pattern
    if not pattern:
        raise ValueError(
            f"{dataset_def.dataset_id}: source.file_pattern is required"
        )
    path = hf_hub_download(
        dataset_def.source.hf_id,
        pattern.format(lang=lang),
        repo_type="dataset",
        revision=revision,
    )
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Prediction rows
# ---------------------------------------------------------------------------


def make_row(
    competitor,
    lang: str,
    sample_index: int,
    test_row: dict,
    prediction: str | None,
    slots: dict,
    confidence: float | None,
    latency_ms: float,
    stage: str | None,
    dataset_revision: str,
) -> dict:
    reference_intent = test_row.get("expected_intent")
    if reference_intent is None:
        exact = prediction is None  # OOD: correct behaviour is no match
    else:
        exact = prediction == reference_intent
    versions = ";".join(
        plugin_version(p) for p in competitor.pipeline_plugins
    )
    return {
        "competitor_id": competitor.competitor_id,
        "sample_id": f"{lang}/{sample_index:05d}",
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "lang": lang,
        "plugin_id": competitor.plugin or "ensemble",
        "plugin_version": versions,
        "pipeline": competitor.pipeline,
        "stage": stage,
        "utterance": test_row["utterance"],
        "reference_intent": reference_intent,
        "reference_slots": test_row.get("expected_slots") or None,
        "prediction": prediction,
        "predicted_slots": slots or None,
        "exact_match": exact,
        "confidence": confidence,
        "bucket": test_row.get("split"),
        "latency_ms": round(latency_ms, 3),
        "runner_version": f"ovos-plugin-arena=={ARENA_VERSION}",
        "created_at": _now_iso(),
    }


def done_samples(out_path: Path) -> set[str]:
    """sample_ids already present in a (resumable) output file."""
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def run_competitor_lang(
    competitor,
    lang: str,
    eval_def,
    train_defs: dict,
    revision: str,
    out_path: Path,
    max_samples: int = 0,
) -> int:
    """Train one fighter for one language and predict the eval split."""
    test_rows = fetch_rows(eval_def, lang, revision)
    if max_samples:
        test_rows = test_rows[:max_samples]

    done = done_samples(out_path)
    todo = [
        (i, row) for i, row in enumerate(test_rows)
        if f"{lang}/{i:05d}" not in done
    ]
    if not todo:
        log.info("  %s/%s: already complete", competitor.competitor_id, lang)
        return 0

    train_data = {
        paradigm: fetch_rows(train_def, lang, revision)
        for paradigm, train_def in train_defs.items()
    }

    pipeline = IntentPipeline(competitor.config["intents"], lang=lang)
    log.info("  training %s for %s (stages: %s)",
             competitor.competitor_id, lang, ", ".join(pipeline.stage_names))
    pipeline.train(train_data)

    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, test_row in todo:
            prediction, slots, confidence, latency_ms, stage = (
                pipeline.predict(test_row["utterance"])
            )
            row = make_row(
                competitor, lang, i, test_row,
                prediction, slots, confidence, latency_ms, stage, revision,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 500 == 0:
                fh.flush()
                log.info("    %s/%s: %d/%d", competitor.competitor_id, lang,
                         written, len(todo))
    log.info("  %s/%s: wrote %d rows", competitor.competitor_id, lang, written)
    return written


def upload_predictions(output_dir: Path, results_repo: str) -> None:
    """Upload every predictions JSONL (plus a dataset card) to HF."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(results_repo, repo_type="dataset", exist_ok=True)

    card = REPO_ROOT / "benchmarks" / "cards" / "intent_intents_for_eval.md"
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=results_repo,
            repo_type="dataset",
        )
    for path in sorted(output_dir.glob("*.jsonl")):
        log.info("Uploading %s → %s", path.name, results_repo)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"predictions/{path.name}",
            repo_id=results_repo,
            repo_type="dataset",
            commit_message=f"bench: refresh {path.stem} predictions",
        )


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--competitors", default="",
                        help="Comma-separated competitor ids (default: all intent)")
    parser.add_argument("--langs", default="",
                        help="Comma-separated languages (default: all in dataset)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap test rows per language (smoke runs)")
    parser.add_argument("--output-dir", default="predictions",
                        help="Local output directory for prediction JSONLs")
    parser.add_argument("--upload", action="store_true",
                        help="Upload predictions to the HF results repo")
    parser.add_argument("--results-repo", default=DEFAULT_RESULTS_REPO)
    args = parser.parse_args(argv)

    eval_def = load_dataset("intent", DATASET_ID)
    train_defs = {
        paradigm: load_dataset("intent", dataset_id)
        for paradigm, dataset_id in (eval_def.train_datasets or {}).items()
    }
    revision = resolve_revision(eval_def.source.hf_id, eval_def.source.revision)
    log.info("Dataset %s @ %s (train sets: %s)",
             eval_def.source.hf_id, revision[:12], ", ".join(train_defs))

    competitors = [
        comp for comp in list_competitors("intent")
        if all(p in ENGINE_REGISTRY for p in comp.pipeline_plugins)
    ]
    if args.competitors:
        wanted = {c.strip() for c in args.competitors.split(",") if c.strip()}
        competitors = [c for c in competitors if c.competitor_id in wanted]
        missing = wanted - {c.competitor_id for c in competitors}
        if missing:
            log.error("Unknown competitors: %s", ", ".join(sorted(missing)))
            return 1

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()] or (
        eval_def.langs or [eval_def.lang]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for competitor in competitors:
        out_path = output_dir / f"{competitor.competitor_id}.jsonl"
        log.info("Fighter %s → %s", competitor.competitor_id, out_path)
        for lang in langs:
            if competitor.langs and lang not in competitor.langs:
                log.info("  skipping %s (not in competitor langs)", lang)
                continue
            try:
                run_competitor_lang(
                    competitor, lang, eval_def, train_defs, revision, out_path,
                    max_samples=args.max_samples,
                )
            except Exception:
                log.exception("  %s/%s failed", competitor.competitor_id, lang)

    if args.upload:
        upload_predictions(output_dir, args.results_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
