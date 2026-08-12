#!/usr/bin/env python3
"""
Intent benchmark over ``mteb/banking77``.

BANKING77 (Casanueva et al. 2020) — 77 fine-grained banking intents,
English. Absorbed from ovos-intent-benchmark's dataset list. Stresses
high-class-count discrimination. Registered against the mteb/banking77
mirror (the original PolyAI/banking77 repo ships a deprecated loading
script the datasets library can no longer execute). Template-paradigm
training only (``banking77-train``).

Predictions publish to one HF repo per modality
(``OpenVoiceOS/ovos-<modality>-bench-banking77``) with one dataset split
per language. See ``runner/intent_bench.py`` for the shared engine and the
row contract.

Usage::

    python benchmarks/intent_banking77.py                 # full run
    python benchmarks/intent_banking77.py --max-samples 5 # smoke run
    python benchmarks/intent_banking77.py --upload         # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("banking77", __doc__.split("\n")[1]))
