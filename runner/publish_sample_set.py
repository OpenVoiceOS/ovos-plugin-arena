"""Publish deterministic sample-set manifests for policy-capped datasets.

A dataset's registry ``sample_policy`` (see ``registry/schemas.py``) fixes
which rows a sweep draws, but the sweep runner itself never records WHICH
rows those were. Two fighters can end up scored over different sample
populations — one swept before a cap existed (or against a bigger cap), one
against the current policy — and a benchmark board built from both is not
comparable, even though every row it holds is individually correct.

This module makes the selected subset an artifact of its own: for one
(dataset, lang), it streams only the id-bearing columns of the parquet
shards (:func:`runner.audio_io._iter_parquet_rows` — no audio decode),
applies :func:`runner.audio_io.select_sample_positions` with the dataset's
policy, and writes the resulting sample ids to
``sample_sets/<lang>.json`` in the dataset's predictions repo (the same
``OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`` repo predictions already
publish to). ``arena.cli assemble`` downloads this manifest alongside a
dataset's predictions and filters every fighter's rows to it before scoring
a board — see ``arena/cli.py::cmd_assemble`` and
``arena/metrics.py::build_benchmark_board``.

Sample-id derivation is byte-identical to a real sweep: this module calls
the exact same :func:`runner.audio_io._sample_id` function
:func:`runner.audio_io.stream_audio_dataset` uses, over the exact same
:func:`runner.audio_io._iter_parquet_rows` row-walk, so a manifest's ids
always match what streaming that dataset would actually produce.
"""
from __future__ import annotations

import argparse
import json
import logging

log = logging.getLogger("publish-sample-set")


def compute_sample_set(dataset_def, revision: str) -> dict:
    """Build the manifest for one dataset (one lang, its own registry entry).

    Raises ``ValueError`` if the dataset has no ``sample_policy`` — there is
    nothing to publish for an uncapped dataset (it already streams every
    row for every fighter, no manifest needed for comparability).
    """
    from runner.audio_io import (
        _iter_parquet_rows,
        _sample_id,
        resolve_parquet_locals,
        resolve_selected_positions,
    )

    policy = dataset_def.sample_policy
    if policy is None:
        raise ValueError(
            f"{dataset_def.dataset_id}: no sample_policy — nothing to publish"
        )

    fields = dataset_def.reference_fields or {}
    audio_key = fields.get("audio", "audio")
    locals_ = resolve_parquet_locals(dataset_def.source, revision)
    selected = resolve_selected_positions(locals_, policy.max_samples, policy.seed)

    total = 0
    sample_ids: list[str] = []
    for pos, row, audio_cell in _iter_parquet_rows(locals_, audio_key):
        total = pos + 1
        if selected is not None and pos not in selected:
            continue
        if audio_cell is None:
            continue  # matches stream_audio_dataset's null-audio skip
        sample_ids.append(_sample_id(row, audio_cell, audio_key, None, pos))

    return {
        "dataset_id": dataset_def.dataset_id,
        "lang": dataset_def.lang,
        "seed": policy.seed,
        "max_samples": policy.max_samples,
        "total_rows": total,
        "sample_ids": sample_ids,
    }


def publish_sample_set(dataset_def, revision: str, owner: str, dry_run: bool) -> dict:
    """Compute one dataset's manifest and (unless *dry_run*) upload it."""
    from runner.intent_bench import results_repo_for

    manifest = compute_sample_set(dataset_def, revision)
    repo = results_repo_for(dataset_def.modality.value, dataset_def.dataset_id, owner)
    lang_file = manifest["lang"].replace("-", "_")
    path_in_repo = f"sample_sets/{lang_file}.json"

    log.info("%s/%s: %d/%d rows selected (seed=%s, cap=%s)%s",
             dataset_def.dataset_id, dataset_def.lang,
             len(manifest["sample_ids"]), manifest["total_rows"],
             manifest["seed"], manifest["max_samples"],
             " [dry-run]" if dry_run else "")

    if not dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            api.create_repo(repo, repo_type="dataset", exist_ok=True)
        except Exception as exc:
            log.warning("create_repo(%s) refused (%s) — uploading anyway", repo, exc)
        api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2).encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"sample-set: {dataset_def.dataset_id}/{dataset_def.lang}",
        )
        log.info("  uploaded %s → %s", path_in_repo, repo)

    return manifest


def main(argv=None) -> int:
    from registry.loaders import list_datasets
    from runner.intent_bench import resolve_revision

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Publish sample-set manifests for every policy-capped "
                    "registry dataset (or a chosen subset)."
    )
    parser.add_argument("--modality", default="",
                        help="Only this modality (default: every modality)")
    parser.add_argument("--dataset", default="",
                        help="Only this dataset_id (default: every dataset "
                             "with a sample_policy)")
    parser.add_argument("--upload", action="store_true",
                        help="Publish to HF (default: --dry-run, counts only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts only, never upload (default "
                             "unless --upload is given)")
    parser.add_argument("--hf-owner", default="OpenVoiceOS")
    args = parser.parse_args(argv)
    dry_run = not args.upload or args.dry_run

    targets = [
        d for d in list_datasets(modality=args.modality or None)
        if d.sample_policy is not None
        and (not args.dataset or d.dataset_id == args.dataset)
    ]

    if not targets:
        log.warning("No datasets with a sample_policy matched the given filters")
        return 0

    log.info("Publishing sample sets for %d dataset(s)%s", len(targets),
             " (dry-run)" if dry_run else "")
    failed = 0
    for dataset_def in targets:
        try:
            revision = resolve_revision(dataset_def.source.hf_id,
                                        dataset_def.source.revision)
            publish_sample_set(dataset_def, revision, args.hf_owner, dry_run)
        except Exception:
            log.exception("  %s/%s failed", dataset_def.dataset_id, dataset_def.lang)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
