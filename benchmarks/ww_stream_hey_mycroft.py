#!/usr/bin/env python3
"""
Streaming wake-word benchmark over the 'hey mycroft' continuous-audio set (§A3.2).

Runs every registry wake-word fighter that declares ``"stream"`` in its
``capabilities`` (see ``runner/ww_bench.py:WakeWordStreamBench``) over long
continuous ground-truth-event clips, and publishes §3.2 detection-event rows
to ``OpenVoiceOS/ovos-ww-stream-bench-<dataset_id>``; the arena's ``assemble``
workflow turns them into a false-reject / false-accept-per-hour board separate
from the isolated-clip `wake_word` league. See ``runner/ww_bench.py`` for the
adapter and ``runner/media_bench.py`` for the shared engine.

The ``ww_stream_hey_mycroft`` corpus itself is not yet published — this
script is the scaffolding for the eventual sweep; running it before the
corpus exists fails at dataset load, exactly like any other unbuilt eval set.

Usage::

    python benchmarks/ww_stream_hey_mycroft.py                    # all stream-capable fighters
    python benchmarks/ww_stream_hey_mycroft.py --competitors openwakeword-hey-mycroft \\
        --max-samples 5                                           # smoke run
    python benchmarks/ww_stream_hey_mycroft.py --upload           # + publish to HF
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
        WakeWordStreamBench(), "ww_stream_hey_mycroft",
        __doc__.split("\n")[1]))
