"""
24/7 prediction runner daemon.

Processes all jobs in the queue YAML sequentially in the main process.
ORT/BLAS thread counts are set via environment variables before the first
plugin import.  Models are unloaded between jobs to avoid OOM.

Sequential execution is intentional: the box has 24 cores but Jellyfin
claims ~16, so we run one STT job at a time with 1 ORT thread to stay
well within the 8-core budget.  The queue loops continuously with a
configurable sleep between full cycles.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_daemon(
    queue_path: str,
    base_dir: str,
    max_workers: int = 4,   # kept for API compat; sequential mode ignores it
    per_sample_timeout: int = 60,
    flush_every: int = 100,
    sleep_seconds: int = 300,
    ort_threads: int = 1,
    hf_token: Optional[str] = None,
    publish: bool = True,
) -> None:
    """Main daemon loop.  Never returns under normal operation."""
    # Pin ORT / BLAS threads before any heavy import
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "ORT_NUM_THREADS"):
        os.environ[var] = str(ort_threads)

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
    logger.info("sequential mode  ort_threads=%d  per_sample_timeout=%ds",
                ort_threads, per_sample_timeout)

    queue_mtime: float = 0.0
    jobs = []

    while True:
        # -----------------------------------------------------------------
        # Load / reload queue if file changed
        # -----------------------------------------------------------------
        try:
            current_mtime = Path(queue_path).stat().st_mtime
        except FileNotFoundError:
            logger.error("queue file not found: %s — sleeping %ds",
                         queue_path, sleep_seconds)
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
        # Run each job sequentially
        # -----------------------------------------------------------------
        for job in jobs:
            from runner.plugin_runner import run_job
            try:
                output_file = run_job(
                    job=job,
                    base_dir=base,
                    per_sample_timeout=per_sample_timeout,
                    flush_every=flush_every,
                )
                logger.info("job complete: %s/%s  output=%s",
                            job.plugin.plugin_name, job.plugin.model_name,
                            output_file)

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
                        output_file.rename(
                            output_file.with_suffix(".published.jsonl")
                        )
                    else:
                        logger.warning("publish returned no files for %s",
                                       output_file)

            except Exception as exc:
                logger.error("job failed (%s/%s): %s",
                             job.plugin.plugin_name, job.plugin.model_name,
                             exc, exc_info=True)

        logger.info("cycle complete — sleeping %ds before next cycle",
                    sleep_seconds)
        time.sleep(sleep_seconds)
