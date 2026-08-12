#!/usr/bin/env python3
"""
Intent benchmark over ``benayas/snips``.

SNIPS voice-assistant queries (7 intents, English). Absorbed from
ovos-intent-benchmark's dataset list — registered directly against the
public HF mirror rather than copied. Text-only (no slot spans in this
mirror), template-paradigm training only (``snips-train``).

Predictions publish to one HF repo per modality
(``OpenVoiceOS/ovos-<modality>-bench-snips``) with one dataset split per
language. See ``runner/intent_bench.py`` for the shared engine and the row
contract.

Usage::

    python benchmarks/intent_snips.py                 # full run
    python benchmarks/intent_snips.py --max-samples 5 # smoke run
    python benchmarks/intent_snips.py --upload         # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("snips", __doc__.split("\n")[1]))
