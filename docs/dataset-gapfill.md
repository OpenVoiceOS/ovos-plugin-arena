# Dataset gap-fill

This extends `docs/dataset-review.md` with the pattern used to add a new
STT eval dataset when a language has fighters registered but no dataset to
score them against, and the coverage gaps that pattern has not closed yet.

## Registering a FLEURS-backed STT dataset

`google/fleurs` covers over a hundred languages as separate configs, so a
language that already has a FLEURS config needs only a JSON registration,
no new loader code. `registry/datasets/stt/fleurs-<lang>.json` follows the
schema of the existing `fleurs-ca-ES.json` / `fleurs-gl-ES.json` entries:
`source.type: huggingface`, `hf_id: google/fleurs`, `subset: <fleurs_config>`,
`split: test`, `reference_fields: {audio, ground_truth: transcription}`,
`license: cc-by-4.0`, `role: eval`.

`ar`, `as`, `bn`, `cs`, `da`, `fa`, `gu`, `hi`, `or`, and `sd` are
registered this way. Check the live FLEURS config list
(`huggingface_hub.HfApi.dataset_info("google/fleurs")`) before assuming a
language is or isn't covered, since it's the authoritative source, not a
list to remember.

A dataset registration alone doesn't put fighters on a board: a
`predictions_hf` repo (`OpenVoiceOS/ovos-stt-bench-fleurs-<lang>`) has to
exist and a `benchmarks/*.py` script has to score fighters against it before
the dataset stops being registered-but-idle.

## Known gaps

Basque (`eu`) has no FLEURS config. `docs/dataset-review.md` names the
alternative source for it. Esperanto, Dogri, Fon, and Bodo have fighters
with no FLEURS config and no other mined loader precedent, so closing
that gap needs a different source, such as an MMS-language Common Voice
subset or OpenSLR.

Belarusian (`be`) has a FLEURS config (`be_by`) and two registered
fighters (`onnx-asr-conformer-transducer-be`,
`onnx-asr-nvidia-be-conformer-ctc-large`), but no
`registry/datasets/stt/fleurs-be-BY.json` yet. Both fighters are idle
until that JSON-only registration lands.

TTS prompt sets need text-only prompt corpora, not audio+transcript pairs,
so FLEURS and Common Voice don't register the same way. Aragonese,
Asturian, Frisian, and Occitan fighters have no prompt set.

Asturian and Occitan do have FLEURS transcripts (`ast_es`, `oc_fr`) that
could become a `fleurs-derived-prompts-<lang>` prompt source, but that
needs a loader that extracts the transcription field and drops the
audio, which is a `benchmarks/` code change rather than a
schema-identical JSON addition. Aragonese and Frisian have no known HF
source at all yet.

The wake-word league has Spanish fighters with no matching eval dataset.
The only Spanish recording found, a 48-sample community set, is too small
and not packaged the way the registry's wake-word datasets expect.

---
[← Dataset review](dataset-review.md) · [Home](index.md) · [Visual site tour →](site-tour.md)
