#!/usr/bin/env python3
"""
Intent benchmark over ``OpenVoiceOS/ovos-intents-v5-eval``.

The OVOS skill fleet's own intent corpus: 221 canonical skill-intent labels
over 40 locales, with a matching template training corpus
(``ovos-intents-v5-templates-train``). Two buckets: ``id_test`` holds
phrasings from the same generator as the training templates and is scored as
in-distribution, ``ood`` holds held-out paraphrases of the same intents and
is what ``generalization_accuracy`` is computed over.

This is the corpus the pretrained ``intent_offline`` fighters were trained
on, so it is the only one they are eligible for (``label_set``).

Predictions publish to one HF repo per training datashape
(``OpenVoiceOS/ovos-intent-<paradigm>-bench-ovos-intents-v5``) with one
dataset split per language. See ``runner/intent_bench.py`` for the shared
engine and the row contract.

Usage::

    python benchmarks/intent_ovos_intents_v5.py                 # full run
    python benchmarks/intent_ovos_intents_v5.py --max-samples 5 # smoke run
    python benchmarks/intent_ovos_intents_v5.py --upload        # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("ovos-intents-v5", __doc__.split("\n")[1]))
