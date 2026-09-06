# Registry audit

Notes on conventions used across `registry/competitors/`, the eight
leagues (`stt`, `tts`, `wake_word`, `vad`, `intent_zero_shot`,
`intent_online`, `intent_offline`, `intent_keyword`), and a few schema
quirks worth knowing before adding or diffing a fighter file.

Run `for d in registry/competitors/*; do echo "$d $(ls "$d" | wc -l)";
done` for the current per-modality fighter count. The registry grows as
fighters are added, so a number written here would go stale.

## The intent leagues

Each of these leagues registers exactly one fighter per meaningfully
different pipeline configuration, not one fighter per confidence tier of
the same configuration.

A single-engine pipeline's confidence gate
(`ovos-adapt-pipeline-plugin-high` vs. `...-medium` vs. `...-low`) selects
which of the plugin's own built-in gates OVOS routes through. It does not
change the plugin's configuration, so only the tier that actually appears
on a published board is kept. `docs/ensemble-rationale.md` covers why
each multi-engine fusion fighter is composed the way it is.

Jurebes-based fighters set `"exact_match": false` in their
`ovos-jurebes-pipeline-plugin` config block. Jurebes 0.4.0 added an
`exact_match` key (plugin default: `true`) that short-circuits
classification with a byte-identical utterance lookup, run before the
sklearn classifier.

Leaving `exact_match` at its default would make fighters built to
compare sklearn recipes (featurizer × classifier) produce identical
predictions whenever the lookup fires, instead of exercising the
classifier under test.

## wake_word

`wakewordlab-*` fighters gate detection with a Silero VAD pre-filter set
inside the hotword's own config block (`config.hotwords.<name>.vad`),
rather than as a separate `listener.VAD` pipeline stage.

Fighters that gate this way carry `"vad-gated"` in `types` alongside
fighters that add an explicit `listener.VAD` stage
(`openwakeword-*-silero`, `microwakeword-*-silero`,
`wakeforge-*-silero`, `precise-onnx-*-silero`), so boards can tell a
gated detector from an ungated one regardless of which mechanism did the
gating. Each detector that supports it also ships `-speaker`
(speaker-verification gate) and `-silero-speaker` (both gates) variants
for the same phrase.

`vosk-ww-*` registers one fighter per phrase rather than one per
configuration, because Vosk's keyword-spotting grammar is defined by the
phrase itself. A different phrase is a different fighter, not a config
variation of an existing one.

Don't count that spread as engine-diversity coverage when reasoning
about roster gaps: it is one engine's per-dataset presence.

## vad

Every VAD fighter's config repeats its module-specific keys twice: once
flat under `listener.VAD` and once nested under the module's own key
inside `listener.VAD` (`silero-vad.json`, for instance, sets `threshold`
in both places).

Both copies are load-bearing for different consumers. OVOS reads
per-module config from the nested block, keyed by module name, while the
flat keys are read directly by the listener's VAD manager for
user-facing settings like `threshold`. A fighter's `notes` field says as
much so the duplication reads as intentional rather than copy-paste
residue.

## stt

Language-pinned STT plugins that ship one checkpoint per language
(`fasterwhisper-base-*`, `vosk-small-*`, `onnx-asr-conformer-transducer-*`)
register one fighter per language rather than a single fighter listing
every language in `langs`, because boards are keyed per language and
per-language scoring needs a specific fighter to compare against.

Multilingual, auto-detecting checkpoints (`onnx-asr-canary`,
`onnx-asr-parakeet-tdt-06b-v3`) instead enumerate the specific languages
the pinned checkpoint supports.

## tts

TTS fighters follow the same one-fighter-per-plugin-configuration rule
as the other leagues: `types`, `langs`, and `config` are expected to
stay internally consistent.

A language or voice variant that changes what is actually being scored
gets its own fighter file rather than folding into an existing one's
`langs` list.

---
[← Operations](operations.md) · [Home](index.md) · [Dataset review →](dataset-review.md)
