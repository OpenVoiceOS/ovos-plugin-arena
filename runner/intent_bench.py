"""
Shared engine for the intent benchmark scripts.

Every benchmark stays one dedicated script under ``benchmarks/`` (P4); this
module is the machinery they share.  A benchmark run:

1. loads the eval dataset definition + its paradigm-specific training
   corpora from the registry, pinning the dataset revision;
2. selects the eligible fighters — every stage's paradigm must have a
   training corpus in this benchmark (a keyword engine cannot train where
   only templates exist);
3. trains each fighter per language and predicts the eval split, writing
   resumable §3.2 rows to
   ``<output_dir>/<dataset_id>/<modality>/<lang>/<competitor_id>.jsonl``;
4. on ``--upload``, publishes **one HF dataset repo per benchmark
   modality** — ``OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`` — with
   files at ``predictions/<lang>/<competitor_id>.jsonl`` and a generated
   dataset card declaring one split per language.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from arena.metrics import domain_of
from arena.version import __version__ as ARENA_VERSION
from registry.loaders import load_all_competitors, load_dataset
from runner.perf import hw_fingerprint, measure_call
from runner.intent_pipeline import (
    ENGINE_REGISTRY,
    IntentPipeline,
    plugin_version,
)

log = logging.getLogger("intent-bench")

HF_OWNER = "OpenVoiceOS"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def results_repo_for(modality: str, dataset_id: str, owner: str = HF_OWNER) -> str:
    """One dedicated HF repo per benchmark modality."""
    return f"{owner}/ovos-{modality.replace('_', '-')}-bench-{dataset_id}"


def split_name(lang: str) -> str:
    """HF split names allow only word characters."""
    return lang.replace("-", "_")


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def resolve_revision(hf_id: str, revision: str) -> str:
    """Pin a branch name to the commit sha it points at right now."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(hf_id, revision=revision)
    return info.sha or revision


def fetch_rows(dataset_def, lang: str, revision: str) -> list:
    """Fetch one language's rows for this dataset.

    Two source shapes exist. ``source.file_pattern`` datasets are plain
    JSONL files under the HF repo (one per language) whose rows already
    carry the canonical row shape (``utterance``/``expected_intent`` for
    eval, ``intent_id``/``template`` for train) — used by datasets the
    arena curated itself (e.g. massive-templates). Datasets absorbed
    straight from a public HF *classification* dataset (SNIPS, BANKING77,
    CLINC150) have no such file and are read through :mod:`datasets`
    instead, with ``reference_fields`` mapping the source columns onto
    the same canonical row shape.
    """
    pattern = dataset_def.source.file_pattern
    if pattern:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            dataset_def.source.hf_id,
            pattern.format(lang=lang),
            repo_type="dataset",
            revision=revision,
        )
        return [json.loads(line) for line in Path(path).read_text().splitlines()
                if line.strip()]
    return fetch_hf_classification_rows(dataset_def, lang, revision)


def fetch_hf_classification_rows(dataset_def, lang: str, revision: str) -> list:
    """Read a plain HF text-classification dataset into canonical rows.

    ``reference_fields`` must provide ``utterance`` (the text column) and
    ``intent`` (the label column — an int ``ClassLabel`` or a plain string
    column, either works). Integer labels are decoded through the source
    dataset's own feature metadata, never guessed. When ``source.id_field``
    is set, rows are deduplicated on that column, keeping the first
    occurrence — some HF mirrors ship the same row more than once;
    deduping is a no-op on clean sources. ``lang`` is accepted for
    interface parity with ``fetch_rows`` (a future per-language-column
    source would filter on it here) — every currently-registered
    classification source is single-language or split per language via
    ``source.split``/``source.subset`` instead.
    """
    from datasets import load_dataset

    src = dataset_def.source
    fields = dataset_def.reference_fields
    text_col = fields.get("utterance")
    intent_col = fields.get("intent")
    domain_label = dataset_def.domain_label
    if not text_col or (not intent_col and domain_label is None):
        raise ValueError(
            f"{dataset_def.dataset_id}: reference_fields must map "
            "'utterance' and 'intent' for a plain HF classification source "
            "(or set domain_label for a corpus with no label column)"
        )
    ds = load_dataset(src.hf_id, name=src.subset, split=src.split, revision=revision)

    intent_feature = ds.features.get(intent_col)
    def _decode(value):
        if intent_feature is not None and hasattr(intent_feature, "int2str"):
            return intent_feature.int2str(value)
        return value

    id_col = src.id_field
    oos = dataset_def.oos_label
    is_train = dataset_def.role == "train"
    rows = []
    seen_ids: set = set()
    for r in ds:
        if id_col is not None:
            row_id = r.get(id_col)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
        label = domain_label if intent_col is None else _decode(r[intent_col])
        text = r[text_col]
        if oos and label == oos:
            if is_train:
                continue  # engines don't train on the negative class
            rows.append({"utterance": text, "expected_intent": None, "split": "far_ood"})
            continue
        if is_train:
            rows.append({"intent_id": label, "template": text})
        else:
            rows.append({"utterance": text, "expected_intent": label, "split": "test"})
    return rows


# ---------------------------------------------------------------------------
# Fighter selection
# ---------------------------------------------------------------------------


def needed_paradigms(competitor) -> set:
    return {ENGINE_REGISTRY[p].paradigm for p in competitor.pipeline_plugins}


def check_league(competitor) -> None:
    """Paradigm leagues are pure: every stage engine must match the league.

    The open ``intent`` league accepts any mix.
    """
    league = competitor.modality.value
    if league == "intent":
        return
    expected = league.removeprefix("intent_")
    for plugin_id in competitor.pipeline_plugins:
        paradigm = ENGINE_REGISTRY[plugin_id].paradigm
        if paradigm != expected:
            raise ValueError(
                f"{competitor.competitor_id}: {plugin_id} is a "
                f"{paradigm}-paradigm engine but the fighter is in the "
                f"{league} league"
            )


def eligible_competitors(train_paradigms: set) -> list:
    """Runnable intent fighters: known engines, pure leagues, trainable here."""
    eligible = []
    for comp in load_all_competitors():
        if not comp.modality.value.startswith("intent"):
            continue
        if not all(p in ENGINE_REGISTRY for p in comp.pipeline_plugins):
            continue
        check_league(comp)
        if needed_paradigms(comp) - train_paradigms:
            log.info("Skipping %s — needs %s training data this benchmark "
                     "does not provide", comp.competitor_id,
                     ", ".join(sorted(needed_paradigms(comp) - train_paradigms)))
            continue
        eligible.append(comp)
    return eligible


# ---------------------------------------------------------------------------
# Prediction rows
# ---------------------------------------------------------------------------


def make_row(
    competitor,
    dataset_id: str,
    lang: str,
    sample_index: int,
    test_row: dict,
    prediction: str | None,
    slots: dict,
    confidence: float | None,
    latency_ms: float,
    stage: str | None,
    dataset_revision: str,
    granularity: str = "intent",
    peak_rss_mb: float | None = None,
) -> dict:
    """Build one §3.2 prediction row.

    ``granularity`` mirrors the eval dataset's ``reference_granularity``:
    'intent' (default) requires an exact string match; 'domain' compares
    only the text before the first ':' on both sides, for corpora that
    only carry a domain-level reference (e.g. meteocat).
    """
    reference_intent = test_row.get("expected_intent")
    if reference_intent is None:
        exact = prediction is None  # OOD: correct behaviour is no match
    elif granularity == "domain":
        exact = domain_of(prediction) == domain_of(reference_intent)
    else:
        exact = prediction == reference_intent
    versions = ";".join(
        plugin_version(p) for p in competitor.pipeline_plugins
    )
    return {
        "competitor_id": competitor.competitor_id,
        "sample_id": f"{lang}/{sample_index:05d}",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "lang": lang,
        "modality": competitor.modality.value,
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
        # performance-metrics campaign M1 (runner.perf) — additive, optional;
        # no audio_secs here, intent rows are text-in/text-out.
        "elapsed_ms": round(latency_ms, 3),
        "peak_rss_mb": round(peak_rss_mb, 3) if peak_rss_mb is not None else None,
        "hw": hw_fingerprint(),
        "runner_version": f"ovos-plugin-arena=={ARENA_VERSION}",
        "created_at": _now_iso(),
    }


def done_samples(out_path: Path) -> set:
    """sample_ids already present in a (resumable) output file."""
    done = set()
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
    dataset_id: str,
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

    needed = needed_paradigms(competitor)
    # The eval revision sha belongs to the EVAL repo. Train corpora may live
    # in a different HF repo (meteocat trains from intents-for-eval), so each
    # train dataset pins its own branch to its own repo's sha.
    train_data = {
        paradigm: fetch_rows(
            train_def, lang,
            resolve_revision(train_def.source.hf_id,
                             train_def.source.revision))
        for paradigm, train_def in train_defs.items()
        if paradigm in needed
    }

    pipeline = IntentPipeline(competitor.config["intents"], lang=lang)
    log.info("  training %s for %s (stages: %s)",
             competitor.competitor_id, lang, ", ".join(pipeline.stage_names))
    pipeline.train(train_data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, test_row in todo:
            (prediction, slots, confidence, latency_ms, stage), _, peak_rss_mb = (
                measure_call(lambda: pipeline.predict(test_row["utterance"]))
            )
            row = make_row(
                competitor, dataset_id, lang, i, test_row,
                prediction, slots, confidence, latency_ms, stage, revision,
                granularity=eval_def.reference_granularity,
                peak_rss_mb=peak_rss_mb,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 500 == 0:
                fh.flush()
                log.info("    %s/%s: %d/%d", competitor.competitor_id, lang,
                         written, len(todo))
    log.info("  %s/%s: wrote %d rows", competitor.competitor_id, lang, written)
    return written


# ---------------------------------------------------------------------------
# Publishing — one repo per benchmark modality, splits per language
# ---------------------------------------------------------------------------


def _dataset_card(modality: str, dataset_id: str, eval_def, langs: list[str]) -> str:
    configs = "\n".join(
        f"  - split: {split_name(lang)}\n"
        f"    path: predictions/{lang}/*.jsonl"
        for lang in sorted(langs)
    )
    league = {
        "intent": "open intent league (mixed-paradigm pipeline fusions)",
        "intent_template": "template-paradigm intent league",
        "intent_keyword": "keyword-paradigm intent league",
    }.get(modality, modality)
    return f"""---
license: apache-2.0
tags:
  - openvoiceos
  - intent-classification
  - benchmark
  - predictions
pretty_name: OVOS {modality} bench — {dataset_id}
configs:
- config_name: default
  data_files:
{configs}
---

# OVOS `{modality}` bench — `{dataset_id}`

Per-sample predictions of the {league} fighters of the
[OVOS Plugin Arena](https://github.com/OpenVoiceOS/ovos-plugin-arena) over
[`{eval_def.source.hf_id}`](https://huggingface.co/datasets/{eval_def.source.hf_id}).

One dedicated repo per benchmark modality; one dataset split per language;
one JSONL file per fighter under `predictions/<lang>/<competitor_id>.jsonl`.
Rows follow the arena §3.2 contract (pinned `dataset_revision`,
`plugin_version`, fired pipeline `stage`, `exact_match` with correct-OOD
semantics). Produced by the reproducible benchmark script in the arena repo;
the arena's `assemble` workflow turns these rows into benchmark boards,
blind battle pools and a benchmark-seeded ELO ladder.

Funded by the [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme.
"""


def upload_predictions(
    bench_dir: Path,
    dataset_id: str,
    eval_def,
    owner: str = HF_OWNER,
) -> None:
    """Upload ``<bench_dir>/<modality>/…`` to the per-modality HF repos."""
    from huggingface_hub import HfApi

    api = HfApi()
    for modality_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
        modality = modality_dir.name
        repo = results_repo_for(modality, dataset_id, owner)
        langs = sorted(d.name for d in modality_dir.iterdir() if d.is_dir())
        if not langs:
            continue
        try:
            api.create_repo(repo, repo_type="dataset", exist_ok=True)
        except Exception as exc:
            # Restricted tokens may write to existing repos but not create
            # new ones — proceed and let the upload itself decide.
            log.warning("create_repo(%s) refused (%s) — uploading anyway",
                        repo, exc)
        api.upload_file(
            path_or_fileobj=_dataset_card(
                modality, dataset_id, eval_def, langs).encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        log.info("Uploading %s → %s (%d langs)", modality_dir, repo, len(langs))
        api.upload_folder(
            folder_path=str(modality_dir),
            path_in_repo="predictions",
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=["**/*.jsonl"],
            commit_message=f"bench: refresh {modality} predictions",
        )


# ---------------------------------------------------------------------------
# Entry point shared by the benchmark scripts
# ---------------------------------------------------------------------------


def run_benchmark(dataset_id: str, description: str, argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--competitors", default="",
                        help="Comma-separated competitor ids (default: all eligible)")
    parser.add_argument("--langs", default="",
                        help="Comma-separated languages (default: all in dataset)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap test rows per language (smoke runs)")
    parser.add_argument("--output-dir", default="predictions",
                        help="Local root for prediction JSONLs")
    parser.add_argument("--upload", action="store_true",
                        help="Upload predictions to the per-modality HF repos")
    parser.add_argument("--hf-owner", default=HF_OWNER)
    args = parser.parse_args(argv)

    eval_def = load_dataset("intent", dataset_id)
    train_defs = {
        paradigm: load_dataset(f"intent_{paradigm}", train_id)
        for paradigm, train_id in (eval_def.train_datasets or {}).items()
    }
    revision = resolve_revision(eval_def.source.hf_id, eval_def.source.revision)
    log.info("Dataset %s @ %s (train paradigms: %s)",
             eval_def.source.hf_id, revision[:12],
             ", ".join(train_defs) or "none")

    competitors = eligible_competitors(set(train_defs))
    if args.competitors:
        wanted = {c.strip() for c in args.competitors.split(",") if c.strip()}
        competitors = [c for c in competitors if c.competitor_id in wanted]
        missing = wanted - {c.competitor_id for c in competitors}
        if missing:
            log.error("Unknown/ineligible competitors: %s",
                      ", ".join(sorted(missing)))
            return 1

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()] or (
        eval_def.langs or [eval_def.lang]
    )

    bench_dir = Path(args.output_dir) / dataset_id
    for competitor in competitors:
        modality = competitor.modality.value
        log.info("Fighter %s [%s]", competitor.competitor_id, modality)
        for lang in langs:
            if competitor.langs and lang not in competitor.langs:
                continue
            out_path = (bench_dir / modality / lang
                        / f"{competitor.competitor_id}.jsonl")
            try:
                run_competitor_lang(
                    competitor, dataset_id, lang, eval_def, train_defs,
                    revision, out_path, max_samples=args.max_samples,
                )
            except Exception:
                log.exception("  %s/%s failed", competitor.competitor_id, lang)

    if args.upload:
        upload_predictions(bench_dir, dataset_id, eval_def, owner=args.hf_owner)
    return 0
