#!/usr/bin/env python3
"""
VAD benchmark: speech vs non-speech detection.

Runs every registry VAD fighter over a speech / non-speech evaluation set and
publishes §3.2 decision rows to ``OpenVoiceOS/ovos-vad-bench-<dataset_id>``; the
arena's ``assemble`` workflow turns them into error-rate / false-accept /
false-reject boards and a benchmark-seeded ELO ladder. See ``runner/vad_bench.py``
for the adapter and ``runner/media_bench.py`` for the shared engine.

Usage::

    python benchmarks/vad_speech.py                                  # en-US, all fighters
    python benchmarks/vad_speech.py --dataset speech-vs-nonspeech-de-DE
    python benchmarks/vad_speech.py --competitors silero-vad \
        --max-samples 50                                             # smoke run
    python benchmarks/vad_speech.py --upload                         # + publish to HF
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.media_bench import run_benchmark  # noqa: E402
from runner.vad_bench import VADBench  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark(
        VADBench(), "speech-vs-nonspeech-en-US",
        __doc__.split("\n")[1]))
