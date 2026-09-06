#!/usr/bin/env python3
"""
Intent benchmark over ``OpenVoiceOS/intents-for-eval``.

The OVOS canonical paradigm-neutral intent benchmark: 50 intents, between
1,354 and 1,384 test rows per language across template / in_distribution /
paraphrase / far_ood / asr_noise / typos buckets, 12 languages, slot
annotations.  Both training paradigms ship in-dataset, so every league
competes — template engines, keyword engines, and the open-league fusions.

Predictions publish to one HF repo per modality
(``OpenVoiceOS/ovos-<modality>-bench-intents-for-eval``) with one dataset
split per language.  See ``runner/intent_bench.py`` for the shared engine
and the row contract.

Usage::

    python benchmarks/intent_intents_for_eval.py                  # full run
    python benchmarks/intent_intents_for_eval.py --langs en-US \
        --competitors padatious-medium --max-samples 20           # smoke run
    python benchmarks/intent_intents_for_eval.py --upload         # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("intents-for-eval", __doc__.split("\n")[1]))
