#!/usr/bin/env python3
"""
Intent benchmark over ``OpenVoiceOS/intents-for-eval``.

Every benchmark in the arena is one dedicated, reproducible script.  This
one trains each intent competitor from the registry on the dataset's own
training files and runs the full test split through the plugin's
``match_<tier>`` gate — predictions are produced fresh by this script,
never copied from elsewhere.

Per (competitor, lang) it:

1. downloads ``<lang>/train_templates.jsonl``, ``<lang>/train_keywords.jsonl``
   and ``<lang>/test.jsonl`` from the dataset repo (revision pinned at run
   start and recorded in every row);
2. instantiates + trains the OPM pipeline plugin (``runner.intent_pipeline``);
3. writes one §3.2 prediction row per test utterance to
   ``predictions/<competitor_id>.jsonl`` (resumable — already-done
   (lang, sample) pairs are skipped);
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


def fetch_jsonl(hf_id: str, filename: str, revision: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        hf_id, filename, repo_type="dataset", revision=revision
    )
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Prediction rows
# ---------------------------------------------------------------------------


def make_row(
    competitor_id: str,
    plugin_id: str,
    lang: str,
    sample_index: int,
    test_row: dict,
    prediction: str | None,
    slots: dict,
    confidence: float | None,
    latency_ms: float,
    dataset_revision: str,
) -> dict:
    reference_intent = test_row.get("expected_intent")
    if reference_intent is None:
        exact = prediction is None  # OOD: correct behaviour is no match
    else:
        exact = prediction == reference_intent
    return {
        "competitor_id": competitor_id,
        "sample_id": f"{lang}/{sample_index:05d}",
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "lang": lang,
        "plugin_id": plugin_id,
        "plugin_version": plugin_version(plugin_id),
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
    hf_id: str,
    revision: str,
    out_path: Path,
    max_samples: int = 0,
) -> int:
    """Train one competitor for one language and predict the test split."""
    templates = fetch_jsonl(hf_id, f"{lang}/train_templates.jsonl", revision)
    keywords = fetch_jsonl(hf_id, f"{lang}/train_keywords.jsonl", revision)
    test_rows = fetch_jsonl(hf_id, f"{lang}/test.jsonl", revision)
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

    tier = competitor.config.get("tier", "medium")
    pipeline = IntentPipeline(
        competitor.plugin, config=competitor.config, lang=lang, tier=tier
    )
    log.info("  training %s for %s (%d templates, %d keyword rules)",
             competitor.plugin, lang, len(templates), len(keywords))
    pipeline.train(templates, keywords)

    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, test_row in todo:
            prediction, slots, confidence, latency_ms = pipeline.predict(
                test_row["utterance"]
            )
            row = make_row(
                competitor.competitor_id, competitor.plugin, lang, i,
                test_row, prediction, slots, confidence, latency_ms, revision,
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

    dataset = load_dataset("intent", DATASET_ID)
    hf_id = dataset.source.hf_id
    revision = resolve_revision(hf_id, dataset.source.revision)
    log.info("Dataset %s @ %s", hf_id, revision[:12])

    competitors = [
        comp for comp in list_competitors("intent")
        if comp.plugin in ENGINE_REGISTRY
    ]
    if args.competitors:
        wanted = {c.strip() for c in args.competitors.split(",") if c.strip()}
        competitors = [c for c in competitors if c.competitor_id in wanted]
        missing = wanted - {c.competitor_id for c in competitors}
        if missing:
            log.error("Unknown competitors: %s", ", ".join(sorted(missing)))
            return 1

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()] or (
        dataset.langs or [dataset.lang]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for competitor in competitors:
        out_path = output_dir / f"{competitor.competitor_id}.jsonl"
        log.info("Competitor %s → %s", competitor.competitor_id, out_path)
        for lang in langs:
            if competitor.langs and lang not in competitor.langs:
                log.info("  skipping %s (not in competitor langs)", lang)
                continue
            try:
                run_competitor_lang(
                    competitor, lang, hf_id, revision, out_path,
                    max_samples=args.max_samples,
                )
            except Exception:
                log.exception("  %s/%s failed", competitor.competitor_id, lang)

    if args.upload:
        upload_predictions(output_dir, args.results_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
