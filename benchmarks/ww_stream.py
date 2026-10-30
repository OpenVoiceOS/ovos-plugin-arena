#!/usr/bin/env python3
"""
Streaming false wake-up benchmark over one ``ww-negatives-<lang>`` set (§A3.2).

Runs every registry wake-word fighter that declares ``"stream"`` in its
``capabilities`` (see ``runner/ww_bench.py:WakeWordStreamBench``) over the
continuous recordings with no wake word of one language, and publishes §3.2
detection-event rows to ``OpenVoiceOS/ovos-ww-stream-bench-<dataset_id>``; the
arena's ``assemble`` workflow turns them into a false-wake-ups-per-hour board
separate from the isolated-clip `wake_word` league
(``scripts/publish_ww_stream_bench.py`` builds the corpus).

Every fighter streams every recording, so name the fighters with ``--competitors``.

Usage::

    python benchmarks/ww_stream.py --dataset ww-negatives-en-US \\
        --competitors openwakeword-alexa,microwakeword-alexa-v2
    python benchmarks/ww_stream.py --competitors precise-onnx-hey-mycroft \\
        --max-samples 1                                           # smoke run
    python benchmarks/ww_stream.py --dataset ww-negatives-en-US \\
        --competitors openwakeword-alexa --upload                  # + publish to HF
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.media_bench import run_benchmark  # noqa: E402
from runner.ww_bench import WakeWordStreamBench  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark(
        WakeWordStreamBench(), "ww-negatives-en-US",
        __doc__.split("\n")[1]))
