"""
24/7 prediction runner daemon.

Processes all jobs in the queue YAML sequentially, then sleeps and
polls for queue-file changes (mtime).  If the queue changes, the new
jobs are picked up on the next cycle.

Worker isolation: each job runs in a subprocess spawned from a
``ProcessPoolExecutor`` (max_workers from config).  This keeps model
state isolated and lets us set ORT/BLAS thread counts before any
import happens.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker entry-point (runs in subprocess)
# ---------------------------------------------------------------------------


def _worker(
    job_dict: dict,
    base_dir: str,
    per_sample_timeout: int,
    flush_every: int,
    ort_threads: int,
) -> dict:
    """Run one job in a worker process."""
    # Pin ORT / BLAS threads before any heavy import
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "ORT_NUM_THREADS"):
        os.environ[var] = str(ort_threads)

    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        stream=sys.stdout,
    )

    # Reconstruct JobSpec from plain dict (avoids pickle issues with dataclasses)
    from runner.queue_config import DatasetSpec, JobSpec, PluginSpec
    plugin = PluginSpec(**job_dict["plugin"])
    dataset = DatasetSpec(**job_dict["dataset"])
    job = JobSpec(plugin=plugin, dataset=dataset,
                  hf_output_dataset=job_dict["hf_output_dataset"])

    from runner.plugin_runner import run_job
    out = run_job(
        job=job,
        base_dir=Path(base_dir),
        per_sample_timeout=per_sample_timeout,
        flush_every=flush_every,
    )
    return {"output": str(out), "job_key": job_dict.get("job_key", "")}


def _job_to_dict(job) -> dict:
    """Serialise a JobSpec to a plain dict for cross-process transfer."""
    from dataclasses import asdict
    return {
        "plugin": {
            "plugin_name": job.plugin.plugin_name,
            "model_name": job.plugin.model_name,
            "lang": job.plugin.lang,
            "extra_config": job.plugin.extra_config,
        },
        "dataset": {
            "hf_repo": job.dataset.hf_repo,
            "split": job.dataset.split,
            "subset": job.dataset.subset,
            "ground_truth_key": job.dataset.ground_truth_key,
            "audio_key": job.dataset.audio_key,
            "entry_id_key": job.dataset.entry_id_key,
            "trust_remote_code": job.dataset.trust_remote_code,
            "max_samples": job.dataset.max_samples,
        },
        "hf_output_dataset": job.hf_output_dataset,
        "job_key": (f"{job.plugin.plugin_name}|{job.plugin.model_name}"
                    f"|{job.dataset.dataset_id}"),
    }


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------


def run_daemon(
    queue_path: str,
    base_dir: str,
    max_workers: int = 12,
    per_sample_timeout: int = 60,
    flush_every: int = 100,
    sleep_seconds: int = 300,
    ort_threads: int = 1,
    hf_token: Optional[str] = None,
    publish: bool = True,
) -> None:
    """Main daemon loop.  Never returns under normal operation."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(base / "runner.log"),
        ],
    )

    logger.info("arena-runner daemon starting  queue=%s  base=%s", queue_path, base)
    logger.info("max_workers=%d  ort_threads=%d  per_sample_timeout=%ds",
                max_workers, ort_threads, per_sample_timeout)

    queue_mtime: float = 0.0

    while True:
        # -----------------------------------------------------------------
        # Load / reload queue if file changed
        # -----------------------------------------------------------------
        try:
            current_mtime = Path(queue_path).stat().st_mtime
        except FileNotFoundError:
            logger.error("queue file not found: %s — sleeping %ds", queue_path, sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        if current_mtime != queue_mtime:
            from runner.queue_config import load_queue
            jobs = load_queue(queue_path)
            queue_mtime = current_mtime
            logger.info("queue loaded/reloaded: %d jobs", len(jobs))

        if not jobs:
            logger.info("queue is empty — sleeping %ds", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        # -----------------------------------------------------------------
        # Submit all jobs to the bounded process pool (one at a time to
        # avoid OOM on the 24-core box)
        # -----------------------------------------------------------------
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _worker,
                    _job_to_dict(job),
                    base_dir,
                    per_sample_timeout,
                    flush_every,
                    ort_threads,
                ): job
                for job in jobs
            }

            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    output_file = Path(result["output"])
                    logger.info("job complete: %s  output=%s",
                                result["job_key"], output_file)

                    if publish and output_file.exists():
                        from runner.publish import publish_output
                        uploaded = publish_output(
                            output_file=output_file,
                            hf_repo=job.hf_output_dataset,
                            token=hf_token,
                        )
                        if uploaded:
                            logger.info("published to %s: %s",
                                        job.hf_output_dataset, uploaded)
                            # Rotate output file so next cycle appends cleanly
                            output_file.rename(
                                output_file.with_suffix(".published.jsonl")
                            )
                        else:
                            logger.warning("publish returned no files for %s",
                                           output_file)
                except Exception as exc:
                    logger.error("job failed (%s/%s): %s",
                                 job.plugin.plugin_name, job.plugin.model_name, exc,
                                 exc_info=True)

        logger.info("cycle complete — sleeping %ds before next cycle", sleep_seconds)
        time.sleep(sleep_seconds)
