"""
Usage:
    python -m runner --queue /path/to/queue.yaml --base-dir /path/to/workdir

Options:
    --queue          Path to the job queue YAML file (default: runner/queue.yaml)
    --base-dir       Working directory for manifests, output, logs
                     (default: /tmp/arena-runner)
    --max-workers    Bound on concurrent jobs (default: 12)
    --no-publish     Disable HuggingFace publish step (write local JSONL only)
    --sleep          Seconds to sleep between full queue cycles (default: 300)
    --timeout        Per-sample inference timeout in seconds (default: 60)
    --ort-threads    ORT/BLAS threads per worker process (default: 1)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OVOS Plugin Arena — STT prediction runner daemon"
    )
    parser.add_argument(
        "--queue",
        default=str(Path(__file__).parent / "queue.yaml"),
        help="Path to queue YAML",
    )
    parser.add_argument(
        "--base-dir",
        default="/tmp/arena-runner",
        help="Working directory (manifests, output, logs)",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip HuggingFace upload step")
    parser.add_argument("--sleep", type=int, default=300,
                        help="Seconds between queue cycles")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-sample timeout in seconds")
    parser.add_argument("--ort-threads", type=int, default=1,
                        help="ORT/BLAS threads per worker")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    from runner.daemon import run_daemon
    run_daemon(
        queue_path=args.queue,
        base_dir=args.base_dir,
        max_workers=args.max_workers,
        per_sample_timeout=args.timeout,
        sleep_seconds=args.sleep,
        ort_threads=args.ort_threads,
        hf_token=hf_token,
        publish=not args.no_publish,
    )


if __name__ == "__main__":
    main()
