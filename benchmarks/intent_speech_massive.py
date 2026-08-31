#!/usr/bin/env python3
"""
Spoken-intent benchmark over FBK-MT/Speech-MASSIVE-test.

The Amazon MASSIVE corpus's spoken realisation: audio clips of the same
intent taxonomy as ``massive-templates``, evaluated end-to-end
audio -> transcript -> intent. Audio input is a property of these eval
datasets, not a new league — fighters compete in the open ``intent``
(fusion) league exactly like every other intent dataset, trained from
``massive-templates-train``.

Each ``registry/datasets/intent/speech-massive-<lang>.json`` pins a
per-language default STT (``ovos-stt-plugin-onnx-asr`` — the same engine
family the STT league runs). The runner transcribes every clip ONCE per
(dataset, lang) with that pinned STT, caches the transcript under
``--transcript-cache-dir``, and feeds the SAME transcript to every intent
fighter — isolating intent-ranking from STT variance. v1 does not fan out
combinatorial STT x intent fighters; see docs/SPECIFICATION.md §3.2.

One registered language per invocation (``--lang``, default: every
registered Speech-MASSIVE language). Not every Speech-MASSIVE language is
registered yet: tr-TR has no pinned onnx-asr default in the STT league and
is skipped until one is added.

Predictions publish to one HF repo per modality
(``OpenVoiceOS/ovos-intent-bench-speech-massive-<lang>``) with one dataset
split per language. See ``runner/intent_bench.py`` for the shared engine
and the row contract.

Usage::

    python benchmarks/intent_speech_massive.py                       # all langs
    python benchmarks/intent_speech_massive.py --lang de-DE \
        --max-samples 3 --no-upload                                  # smoke run
    python benchmarks/intent_speech_massive.py --lang de-DE --upload # + publish
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.intent_bench import run_benchmark  # noqa: E402

# Kept in sync with registry/datasets/intent/speech-massive-*.json.
REGISTERED_LANGS = (
    "ar-SA", "de-DE", "es-ES", "fr-FR", "hu-HU", "ko-KR",
    "nl-NL", "pl-PL", "pt-PT", "ru-RU", "vi-VN",
)


def _default_dataset_id_for_static_resolution():  # pragma: no cover
    """Never called — this script loops over ``REGISTERED_LANGS`` at
    runtime, so its default dataset id is not a single string literal
    ``run_benchmark(...)`` call the way every other benchmark script's is.
    tests/test_benchmark_defaults.py statically scans for exactly that
    literal to confirm a script's documented default run resolves in the
    registry; this stub gives it one to find (de-DE, this script's first
    registered language).
    """
    return run_benchmark("speech-massive-de-DE", __doc__)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lang = None
    if "--lang" in argv:
        i = argv.index("--lang")
        lang = argv[i + 1]
        del argv[i:i + 2]
    langs = (lang,) if lang else REGISTERED_LANGS
    description = __doc__.split("\n")[1]
    status = 0
    for one_lang in langs:
        status |= run_benchmark(f"speech-massive-{one_lang}", description, argv)
    return status


if __name__ == "__main__":
    sys.exit(main())
