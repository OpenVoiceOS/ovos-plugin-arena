#!/usr/bin/env python3
"""
Wake-word benchmark over the 'computer' phrase (real community recordings).

Runs every registry wake-word fighter configured for 'computer' over the
real recordings in OpenVoiceOS/ovos-community-wakewords-dataset (other phrases
as adversarial hard negatives) and publishes §3.2 detection rows. Each clip is
primed with leading silence so streaming engines activate as they would on a
live mic. See ``runner/ww_bench.py`` and ``runner/media_bench.py``.

Usage::

    python benchmarks/ww_computer.py --max-samples 30
    python benchmarks/ww_computer.py --competitors vosk-ww-computer
    python benchmarks/ww_computer.py --upload
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.media_bench import run_benchmark  # noqa: E402
from runner.ww_bench import WakeWordBench  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark(
        WakeWordBench(), "community-computer", __doc__.split("\n")[1]))
