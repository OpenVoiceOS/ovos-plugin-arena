#!/usr/bin/env python3
"""
Intent benchmark over ``clinc/clinc_oos`` (plus config).

CLINC150 (Larson et al. 2019) — 150 in-scope intents across 10 domains,
English, plus an explicit out-of-scope class. Absorbed from
ovos-intent-benchmark's dataset list. Uses the 'plus' config, whose 1000
out-of-scope test utterances are tagged ``bucket='far_ood'`` with
``expected_intent=None`` so the scorer treats a non-fire as correct — a
genuine "don't guess" regression test. Template-paradigm training only
(``clinc150-train``).

Predictions publish to one HF repo per modality
(``OpenVoiceOS/ovos-<modality>-bench-clinc150``) with one dataset split
per language. See ``runner/intent_bench.py`` for the shared engine and the
row contract.

Usage::

    python benchmarks/intent_clinc150.py                 # full run
    python benchmarks/intent_clinc150.py --max-samples 5 # smoke run
    python benchmarks/intent_clinc150.py --upload         # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("clinc150", __doc__.split("\n")[1]))
