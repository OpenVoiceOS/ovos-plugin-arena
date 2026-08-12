#!/usr/bin/env python3
"""
STT benchmark over the mTEDx Brazilian Portuguese corpus.

Transcribes every registry STT fighter over the ``mtedx-pt-BR`` eval set and
publishes §3.2 prediction rows to ``OpenVoiceOS/ovos-stt-bench-<dataset_id>``;
the arena's ``assemble`` workflow turns them into WER benchmark boards, blind
A/B battles and a WER-seeded ELO ladder.  See ``runner/stt_bench.py`` for the
adapter and ``runner/media_bench.py`` for the shared engine.

Usage::

    python benchmarks/stt_mtedx.py                          # pt-BR, all fighters
    python benchmarks/stt_mtedx.py --competitors whispercpp-base \
        --max-samples 20                                     # smoke run
    python benchmarks/stt_mtedx.py --upload                  # + publish to HF
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.media_bench import run_benchmark  # noqa: E402
from runner.stt_bench import STTBench  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark(STTBench(), "mtedx-pt-BR", __doc__.split("\n")[1]))
