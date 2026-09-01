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

A hung per-shard HTTP request against a source corpus (mid-listing, the log
goes silent for 25+ minutes even with the HF_HUB_ETAG_TIMEOUT/
HF_HUB_DOWNLOAD_TIMEOUT env vars set — those cover file DOWNLOADS, not
every metadata call a run makes) took the whole run down twice in
production, and a restart redid every already-published manifest from
scratch. Two things fix that: ``--skip-existing`` skips a dataset whose
manifest is already published under the SAME policy, and every dataset's
work runs under a bounded per-dataset timeout (``--timeout-secs``, default
15 minutes) so one wedged corpus can never wedge the whole run — it's
logged and skipped, the run moves on, and the failures are listed at the
end so a rerun with ``--skip-existing`` only redoes the stragglers.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os

log = logging.getLogger("publish-sample-set")

DEFAULT_TIMEOUT_SECS = 15 * 60
DEFAULT_REQUEST_TIMEOUT_SECS = 30


def compute_sample_set(
    dataset_def, revision: str, etag_timeout: float | None = None,
) -> dict:
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
    locals_ = resolve_parquet_locals(dataset_def.source, revision,
                                     etag_timeout=etag_timeout)
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


def _manifest_repo_and_path(dataset_def, owner: str) -> tuple[str, str]:
    from runner.intent_bench import results_repo_for

    repo = results_repo_for(dataset_def.modality.value, dataset_def.dataset_id, owner)
    lang_file = dataset_def.lang.replace("-", "_")
    return repo, f"sample_sets/{lang_file}.json"


def existing_sample_set(dataset_def, owner: str) -> dict | None:
    """Read a dataset's already-published ``sample_sets/<lang>.json``, if
    any — used by ``--skip-existing`` to decide whether recomputation is
    needed at all. Returns ``None`` on any error (no repo yet, no file yet,
    a transient network failure, malformed JSON) — ``--skip-existing``
    simply proceeds to recompute in that case, same as if nothing had ever
    been published.
    """
    repo, path_in_repo = _manifest_repo_and_path(dataset_def, owner)
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo, path_in_repo, repo_type="dataset",
                                revision="main")
        with open(local, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def sample_set_is_current(existing: dict, dataset_def) -> bool:
    """True when an already-published manifest (``existing``) was built
    from the dataset's CURRENT ``sample_policy`` — the cheap check
    ``--skip-existing`` needs. ``seed``, ``max_samples`` and ``total_rows``
    are all already IN the manifest (:func:`existing_sample_set` already
    read the one small file that holds them) — no corpus access needed to
    compare them, which matters because re-touching the source corpus is
    exactly what ``--skip-existing`` exists to avoid after a hang. A
    manifest missing the fields needed to compare (an older/foreign schema)
    falls back to existence-only — ``True``, since something published
    under this exact (dataset, lang) key already exists and there is
    nothing cheap left to check it against.
    """
    policy = dataset_def.sample_policy
    if policy is None:
        return False
    if "seed" not in existing or "max_samples" not in existing:
        return True  # can't compare — existence is all we can cheaply know
    return (existing["seed"] == policy.seed
            and existing["max_samples"] == policy.max_samples)


def publish_sample_set(
    dataset_def, revision: str, owner: str, dry_run: bool,
    etag_timeout: float | None = None,
) -> dict:
    """Compute one dataset's manifest and (unless *dry_run*) upload it."""
    manifest = compute_sample_set(dataset_def, revision, etag_timeout=etag_timeout)
    repo, path_in_repo = _manifest_repo_and_path(dataset_def, owner)

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


def _publish_one(dataset_def, owner: str, dry_run: bool,
                 request_timeout: float | None) -> dict:
    """One dataset's full unit of work — revision resolution AND manifest
    computation/upload — run together under the caller's per-dataset
    timeout, since either half can independently hang on a slow/wedged
    corpus.
    """
    from runner.intent_bench import resolve_revision

    revision = resolve_revision(dataset_def.source.hf_id,
                                dataset_def.source.revision,
                                timeout=request_timeout)
    return publish_sample_set(dataset_def, revision, owner, dry_run,
                              etag_timeout=request_timeout)


def run_with_timeout(fn, timeout_secs: float):
    """Run ``fn()`` in a worker thread, bounded by ``timeout_secs``.

    Raises ``concurrent.futures.TimeoutError`` when ``fn`` doesn't finish in
    time. The worker thread itself is NOT killed — Python cannot safely
    interrupt a thread blocked in a C-level socket read — it is simply
    abandoned so the caller can move on to the next dataset. A leaked
    hung worker is why ``main()`` force-exits via ``os._exit`` instead of
    returning normally whenever a timeout actually fired: a plain
    ``ThreadPoolExecutor`` registers every worker it ever spawns with an
    ``atexit`` hook that joins them all, which would hang interpreter exit
    on exactly the wedged thread this function exists to escape.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout_secs)
    finally:
        executor.shutdown(wait=False)


def main(argv=None) -> int:
    from registry.loaders import list_datasets

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
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a dataset whose sample_sets manifest is "
                             "already published under the SAME policy "
                             "(seed/max_samples) — resumes a run that died "
                             "partway through without redoing prior work")
    parser.add_argument("--timeout-secs", type=float, default=DEFAULT_TIMEOUT_SECS,
                        help="Per-dataset wall-clock budget (default: "
                             f"{DEFAULT_TIMEOUT_SECS}s / 15 min) — a dataset "
                             "that exceeds it is logged and skipped, not "
                             "allowed to wedge the whole run")
    parser.add_argument("--request-timeout-secs", type=float,
                        default=DEFAULT_REQUEST_TIMEOUT_SECS,
                        help="Explicit per-request timeout passed to the "
                             "HF metadata/download calls that accept one "
                             f"(default: {DEFAULT_REQUEST_TIMEOUT_SECS}s)")
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
    failed: list[str] = []
    timed_out: list[str] = []
    for dataset_def in targets:
        label = f"{dataset_def.dataset_id}/{dataset_def.lang}"

        if args.skip_existing:
            existing = existing_sample_set(dataset_def, args.hf_owner)
            if existing is not None and sample_set_is_current(existing, dataset_def):
                log.info(
                    "%s: sample_sets manifest already published for the "
                    "current policy (seed=%s, cap=%s) — skipping",
                    label, dataset_def.sample_policy.seed,
                    dataset_def.sample_policy.max_samples,
                )
                continue

        try:
            run_with_timeout(
                lambda dd=dataset_def: _publish_one(
                    dd, args.hf_owner, dry_run, args.request_timeout_secs),
                args.timeout_secs,
            )
        except concurrent.futures.TimeoutError:
            log.error(
                "%s: timed out after %.0fs — skipping. Rerun with "
                "--skip-existing to retry only the stragglers.",
                label, args.timeout_secs,
            )
            timed_out.append(label)
        except Exception:
            log.exception("%s: failed", label)
            failed.append(label)

    if timed_out:
        log.error("Timed out (%d): %s", len(timed_out), ", ".join(timed_out))
    if failed:
        log.error("Failed (%d): %s", len(failed), ", ".join(failed))

    if timed_out or failed:
        if timed_out:
            # a leaked hung worker thread would otherwise block interpreter
            # exit — see run_with_timeout's docstring.
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
