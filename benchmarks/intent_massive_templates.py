#!/usr/bin/env python3
"""
Intent benchmark over ``OpenVoiceOS/massive-templates``.

Amazon MASSIVE recast into the paradigm-neutral template benchmark format:
52 languages with held-out test utterances (intent + slot annotations).
Only a template-paradigm training corpus exists, so keyword engines and
keyword-bearing fusions are automatically ineligible — the template league
and template-pure fusions compete.

The training corpus is large (~13k templates per language).  Engines whose
training is super-linear in corpus size (Padatious-class neural training)
do not finish in practical time here; run them with an explicit
``--competitors`` selection only if you can afford multi-day training.
Sample-lookup engines (Padacioso, Nebulento) handle the corpus fine.

Predictions publish to one HF repo per modality
(``OpenVoiceOS/ovos-<modality>-bench-massive-templates``) with one dataset
split per language.  See ``runner/intent_bench.py`` for the shared engine
and the row contract.

Usage::

    python benchmarks/intent_massive_templates.py                 # full run
    python benchmarks/intent_massive_templates.py --langs en-US \
        --max-samples 20                                          # smoke run
    python benchmarks/intent_massive_templates.py --upload        # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark("massive-templates", __doc__.split("\n")[1]))
