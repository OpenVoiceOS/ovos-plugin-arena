#!/usr/bin/env python3
"""
Wake-word benchmark over the 'hey jarvis' phrase — openWakeWord vs microWakeWord.

Both engines ship a pretrained 'hey jarvis' model, so this is a head-to-head
neural comparison. Positives are synthetic 'hey jarvis' clips; negatives are
pooled across the not-wake-word collection (speech, sounds, music, noise) so
false-accept rate spans many scenarios. Each clip is primed with leading
silence so streaming engines activate. See ``runner/ww_bench.py``.

Usage::

    python benchmarks/ww_hey_jarvis.py --max-samples 30
    python benchmarks/ww_hey_jarvis.py --upload
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
        WakeWordBench(), "synthetic-wakewords-hey_jarvis", __doc__.split("\n")[1]))
