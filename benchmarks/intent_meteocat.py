#!/usr/bin/env python3
"""
Intent benchmark over ``crodri/meteocat``.

Synthetic Catalan weather queries (Projecte AINA / BSC) with domain-level
references only: every row's reference is the constant domain ``weather``
and scoring compares just the domain part of the prediction
(``reference_granularity: domain``).  This makes it a recall-only probe of
the domain stage — a fighter that always answers ``weather:x`` scores
perfectly, so pair it with a mixed-domain dataset for precision.  It is
most informative for the domain/hierarchical fighters, whose first stage
is exactly a domain classifier.

Predictions publish to ``OpenVoiceOS/ovos-intent-bench-meteocat`` at
``predictions/<lang>/<competitor_id>.jsonl``.  See ``runner/intent_bench.py``
for the shared engine and the row contract.

Usage::

    python benchmarks/intent_meteocat.py                       # full run
    python benchmarks/intent_meteocat.py --competitors \
        adapt-domain-medium --max-samples 20                   # smoke run
    python benchmarks/intent_meteocat.py --upload              # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("meteocat", __doc__.split("\n")[1]))
