#!/usr/bin/env python3
"""
Wake-word benchmark over the held-out 'hey mycroft' ww-bench set.

Runs every registry wake-word fighter over the ww-bench evaluation manifest
(eval-only donor voices) and publishes §3.2 detection rows to
``OpenVoiceOS/ovos-wake-word-bench-<dataset_id>``; the arena's ``assemble``
workflow turns them into detection-error / false-accept / false-reject boards
and a benchmark-seeded ELO ladder.  See ``runner/ww_bench.py`` for the adapter
and ``runner/media_bench.py`` for the shared engine.

Usage::

    python benchmarks/ww_hey_mycroft.py                    # all fighters
    python benchmarks/ww_hey_mycroft.py --competitors openwakeword-hey-mycroft \
        --max-samples 50                                   # smoke run
    python benchmarks/ww_hey_mycroft.py --upload           # + publish to HF
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
        WakeWordBench(), "ww-bench-hey-mycroft", __doc__.split("\n")[1]))
