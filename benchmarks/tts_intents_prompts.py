#!/usr/bin/env python3
"""
TTS synthesis benchmark over the multilingual voice-assistant prompt set.

Every registry TTS fighter synthesises each prompt; the clip is stored and its
URL recorded as the §3.2 prediction, published to
``OpenVoiceOS/ovos-tts-bench-<dataset_id>``.  TTS has no objective metric — the
arena assembles the clips into blind A/B listening battles (human votes only;
no benchmark board, no ELO seed).  See ``runner/tts_bench.py`` for the adapter
and ``runner/media_bench.py`` for the shared engine.

Usage::

    python benchmarks/tts_intents_prompts.py --langs en-US --max-samples 30
    python benchmarks/tts_intents_prompts.py --competitors edge-tts-aria \
        --langs en-US --max-samples 30
    python benchmarks/tts_intents_prompts.py --langs en-US --upload  # publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.media_bench import run_benchmark  # noqa: E402
from runner.tts_bench import TTSBench  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_benchmark(
        TTSBench(), "intents-for-eval-prompts", __doc__.split("\n")[1]))
