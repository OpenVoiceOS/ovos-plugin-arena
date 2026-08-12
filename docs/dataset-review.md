# Dataset review

This document reviews `registry/datasets/`, the corpora the arena scores
fighters against, and proposes additions to close coverage gaps. It covers
five modalities: `stt`, `wake_word`, `tts`, `vad`, `intent`.

A dataset entry (`registry/datasets/<modality>/<id>.json`) only becomes a
usable benchmark once two more things exist:

- a **published board**, a `predictions_hf` repo on the `OpenVoiceOS` HF
  org holding prediction rows for at least one fighter
- a **default in a benchmark script** under `benchmarks/*.py`, so the
  dataset actually gets exercised without a caller having to name it
  explicitly

A dataset with neither is registered but idle: nothing runs against it and
no board exists to show results.

## stt

26 datasets: 14 `minds14-*` (telephone-domain, intent-classification
speech, source `PolyAI/minds14`) and 12 `speech-massive-*` (voice-assistant
utterances, source `FBK-MT/Speech-MASSIVE-test`).

| dataset id | lang | HF source | board published | script default |
|---|---|---|---|---|
| minds14-en-US | en-US | PolyAI/minds14 | yes | yes (`stt_minds14.py`... see note) |
| minds14-en-GB | en-GB | PolyAI/minds14 | yes | no |
| minds14-en-AU | en-AU | PolyAI/minds14 | yes | no |
| minds14-fr-FR | fr-FR | PolyAI/minds14 | yes | no |
| minds14-de-DE | de-DE | PolyAI/minds14 | yes | no |
| minds14-it-IT | it-IT | PolyAI/minds14 | yes | no |
| minds14-es-ES | es-ES | PolyAI/minds14 | yes | no |
| minds14-pt-PT | pt-PT | PolyAI/minds14 | no | yes (default) |
| minds14-nl-NL | nl-NL | PolyAI/minds14 | no | no |
| minds14-cs-CZ | cs-CZ | PolyAI/minds14 | no | no |
| minds14-ko-KR | ko-KR | PolyAI/minds14 | no | no |
| minds14-pl-PL | pl-PL | PolyAI/minds14 | no | no |
| minds14-ru-RU | ru-RU | PolyAI/minds14 | no | no |
| minds14-zh-CN | zh-CN | PolyAI/minds14 | no | no |
| speech-massive-ar-SA | ar-SA | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-de-DE | de-DE | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-es-ES | es-ES | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-fr-FR | fr-FR | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-hu-HU | hu-HU | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-ko-KR | ko-KR | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-nl-NL | nl-NL | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-pl-PL | pl-PL | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-pt-PT | pt-PT | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-ru-RU | ru-RU | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-tr-TR | tr-TR | FBK-MT/Speech-MASSIVE-test | no | no |
| speech-massive-vi-VN | vi-VN | FBK-MT/Speech-MASSIVE-test | no | no |

`benchmarks/stt_minds14.py` hardcodes `minds14-pt-PT` as its default
dataset (`run_benchmark(STTBench(), "minds14-pt-PT", ...)`). Every other
`minds14-*` id must be passed explicitly. No script in `benchmarks/` names
any `speech-massive-*` dataset, the whole corpus family is unreachable
through the default entry points.

Boards exist on the `OpenVoiceOS` HF org for exactly 7 `minds14-*` locales
(en-US, en-GB, en-AU, fr-FR, de-DE, it-IT, es-ES), the `pt-PT` locale that
the script defaults to has **no** board (`ovos-stt-bench-minds14-pt-PT`
does not exist). A board named `OpenVoiceOS/ovos-stt-bench-pt-PT` does
exist on HF but matches no registered dataset id (the registry's naming
convention is `ovos-stt-bench-<dataset_id>`, i.e.
`ovos-stt-bench-minds14-pt-PT` or `ovos-stt-bench-speech-massive-pt-PT`), this is an orphaned board left over from before the `minds14-`/
`speech-massive-` id prefixes were introduced, or a one-off run that was
never wired to a registry entry.

### Fighters with no matching eval dataset

Four STT fighters target languages the `stt` registry has no dataset for
at all:

- `citrinet-512-ca` (lang `ca`)
- `onnx-asr-conformer-transducer-ca` (lang `ca`)
- `onnx-asr-conformer-ctc-gl` (lang `gl`)
- `onnx-asr-conformer-transducer-eu` (lang `eu`)
- `whisper-lm-eu` (lang `eu`)

These fighters cannot be scored on the arena today. There is no failure to
fix, just missing data.

## wake_word

27 datasets across four families: `picovoice-*` (real recordings, source
`Picovoice/wake-word-benchmark`), `community-*` (community-recorded
wake words), `synthetic-wakewords-*` (TTS-generated, per-wakeword), and
`sam-wake-word`.

Only 3 of 27 have a published board:

| dataset id | board published |
|---|---|
| synthetic-wakewords-hey_mycroft | yes |
| synthetic-wakewords-hey_jarvis | yes |
| community-computer | yes |
| all other 24 | no |

`community-computer`'s board (`ovos-wake-word-bench-community-computer`)
exists as an HF dataset but its `predictions/` directory is empty, no
fighter has actually published a prediction row to it, so the board is
live in name only.

`benchmarks/` ships three wake-word scripts, `ww_computer.py`,
`ww_hey_jarvis.py`, `ww_hey_mycroft.py`, each defaulting to one dataset.
That accounts for the 3 populated (or nominally populated) boards. The
other 24 datasets, including every `picovoice-*` entry except none, all
`community-*` entries except `community-computer`, and `sam-wake-word`,
have neither a board nor a default script pointing at them.

## tts

Only 2 datasets, both `lang: multi` prompt sets used to generate synthesis
audio for subjective comparison (TTS has no automatic reference metric in
this arena, scoring is listen-and-judge):

| dataset id | lang | HF source | board published | script default |
|---|---|---|---|---|
| intents-for-eval-prompts | multi | OpenVoiceOS/intents-for-eval | yes | yes (`tts_intents_prompts.py`) |
| massive-prompts | multi | OpenVoiceOS/massive-templates | yes | no |

Both existing boards are populated. The gap here is not idle
infrastructure, it's coverage: 2 prompt sets have to stand in for 31 TTS
fighters spanning per-language voices (many are single-language,
single-speaker models, the phoonnx and piper voice families in
particular). A `multi` prompt set gives every fighter the same prompt
text regardless of the language it actually synthesizes in, which is a
weak signal for language-specific pronunciation and prosody quality.

## vad

19 datasets: one legacy bare `speech-vs-nonspeech` (lang `en`, source
`TigreGotico/not-wake-words-speech-en`) plus 18 per-locale
`speech-vs-nonspeech-<lang>` files covering ar-SA, cs-CZ, de-DE, en-AU,
en-GB, en-US, es-ES, fr-FR, hu-HU, it-IT, ko-KR, nl-NL, pl-PL, pt-PT,
ru-RU, tr-TR, vi-VN, zh-CN.

All 19 have published boards on the `OpenVoiceOS` HF org, and
`benchmarks/vad_speech.py` runs the modality, this is the healthiest
modality in the registry by board coverage.

The one structural issue: the bare `speech-vs-nonspeech` dataset and
`speech-vs-nonspeech-en-US` (and to a lesser extent `en-GB`, `en-AU`) all
draw positives from the same English-language source pool and score the
same fighters, producing two (effectively three) boards for what is
functionally one English VAD eval. `speech-vs-nonspeech` predates the
per-locale split and was never retired once `speech-vs-nonspeech-en-US`
existed as the locale-tagged equivalent.

## intent

12 datasets across two paradigms (template, keyword) plus one raw-corpus
train set:

| dataset id | role | board published |
|---|---|---|
| intents-for-eval | eval | yes (`ovos-intent-bench-intents-for-eval`) |
| intents-for-eval-templates | train (template) | n/a, train corpus |
| intents-for-eval-keywords | train (keyword) | n/a, train corpus |
| massive-templates | eval | yes (`ovos-intent-template-bench-massive-templates`) |
| massive-templates-train | train (template) | n/a, train corpus |
| hass-intent-templates | train (template) | n/a, train corpus |
| snips | eval | no |
| snips-train | train (template) | n/a, train corpus |
| banking77 | eval | no |
| banking77-train | train (template) | n/a, train corpus |
| clinc150 | eval | no |
| clinc150-train | train (template) | n/a, train corpus |

### Absorbed from `ovos-intent-benchmark`

SNIPS, BANKING77, and CLINC150 close most of the gap with the sibling
`ovos-intent-benchmark` repo's dataset list. None had a prior
`registry/datasets/intent/*.json` entry. Full raw MASSIVE (that repo's
fourth loader) was deliberately **not** absorbed, the arena already
carries MASSIVE via `massive-templates`, and the owner decided that
paradigm-neutral template recast is coverage enough. A second, raw-corpus
MASSIVE registration was judged not worth the duplication.

These three are plain HF text-classification corpora, not the JSONL
file-per-language shape the registry previously assumed for intent
datasets. `runner/intent_bench.fetch_rows` gained a second path
(`fetch_hf_classification_rows`) for sources without a `file_pattern`: it
loads the HF dataset directly, decodes `ClassLabel` columns through the
source's own feature metadata, and maps `reference_fields` onto the
runner's canonical row shape. CLINC150's explicit out-of-scope class uses
the new `oos_label` field, eval rows get `expected_intent=None` +
`bucket=far_ood` (correct behaviour is not firing), train rows are
dropped. The loader also supports a general `source.id_field` dedup
(first-occurrence-wins on that column), general hygiene against HF
mirrors that ship duplicate rows. None of the three registered sources
need it, but it's exercised by a synthetic-source regression test.

All three are text-only (no slot spans in these mirrors), so, like
`massive-templates`, they register `template`-paradigm training only. Keyword-paradigm and keyword-bearing fusion fighters are automatically
ineligible for them, same as the existing precedent. None have a
published board yet (no `benchmarks/*.py` script drives them). Adding
one is a follow-up, not in this change's scope.

`intents-for-eval` and `massive-templates` are properly wired: each
declares `train_datasets` pointing at its own template/keyword training
corpora, and both have live boards driven by `benchmarks/intent_intents_for_eval.py`
and `benchmarks/intent_massive_templates.py`.

`hass-intent-templates` is an orphan. It is a large corpus, Home
Assistant intents recast as phrase templates across 62 locales, but no
eval dataset's `train_datasets` references it, and grepping the full
`registry/datasets/intent/` tree confirms it is the only train-role entry
with zero inbound references. It has a `predictions_hf` field
(`OpenVoiceOS/ovos-intent-bench-hass-intent-templates`) as if it were
meant to back a board, but no eval dataset drives predictions into it.

## Gap analysis summary

| modality | datasets | boards published | idle datasets |
|---|---|---|---|
| stt | 26 | 7 | 19 |
| wake_word | 27 | 3 (1 empty) | 24 |
| tts | 2 | 2 | 0 (coverage gap instead) |
| vad | 19 | 19 | 0 (duplicate-board issue instead) |
| intent | 12 (6 eval-eligible) | 4 | 3 (1 orphaned train corpus + 3 newly-absorbed eval corpora with no board yet) |

stt and wake_word are the two modalities where most registered datasets
are pure dead weight: registered, schema-valid, never run, no board. The
other three modalities have their own distinct issues, tts is
under-provisioned relative to fighter count, vad carries a duplicate
English board, and intent has one orphaned training corpus.

## Proposals

### 1. Wire the speech-massive family into a benchmark script

Add a `benchmarks/stt_speech_massive.py` mirroring `stt_minds14.py`'s
structure, defaulting to one `speech-massive-*` locale (suggest `pt-PT` to
pair with the existing minds14 default, or `de-DE` since German has the
most STT fighter density). Loop the remaining 11 locales through CI the
same way `minds14` locales beyond the default are presumably run (check
how the 7 populated `minds14-*` boards got populated if not through the
default arg, likely a matrix invocation outside `stt_minds14.py`'s
hardcoded default. Replicate that invocation pattern for
`speech-massive-*`).

### 2. Publish boards for the 7 un-boarded minds14 locales

nl-NL, cs-CZ, ko-KR, pl-PL, ru-RU, zh-CN, and pt-PT (despite being the
script default) have no board yet. Running the existing
`stt_minds14.py` with each dataset id explicit closes this without any
new code.

### 3. Retire or rename the orphaned `ovos-stt-bench-pt-PT` HF board

It matches no `registry/datasets/stt/*.json` `predictions_hf` value under
the current naming convention. Either delete it or confirm whether it's
actually feeding `minds14-pt-PT` under a legacy name and rename to
`ovos-stt-bench-minds14-pt-PT`.

### 4. New STT eval datasets for ca / gl / eu

Five fighters (`citrinet-512-ca`, `onnx-asr-conformer-transducer-ca`,
`onnx-asr-conformer-ctc-gl`, `onnx-asr-conformer-transducer-eu`,
`whisper-lm-eu`) have no matching dataset. Propose Common Voice-derived
eval sets, following the existing HF layout convention
(`ovos-stt-bench-<dataset>-<lang>`):

- `registry/datasets/stt/commonvoice-ca.json`, source
  `mozilla-foundation/common_voice_17_0`, subset `ca`, test split.
  Unlocks `citrinet-512-ca` and `onnx-asr-conformer-transducer-ca`.
  Board: `OpenVoiceOS/ovos-stt-bench-commonvoice-ca`. Script:
  `benchmarks/stt_commonvoice.py` (new, parameterized by locale like
  `stt_minds14.py`).
- `registry/datasets/stt/commonvoice-gl.json`, source
  `mozilla-foundation/common_voice_17_0`, subset `gl`, test split.
  Unlocks `onnx-asr-conformer-ctc-gl`. Board:
  `OpenVoiceOS/ovos-stt-bench-commonvoice-gl`. Script: same
  `stt_commonvoice.py`.
- `registry/datasets/stt/commonvoice-eu.json`, source
  `mozilla-foundation/common_voice_17_0`, subset `eu`, test split.
  Unlocks `onnx-asr-conformer-transducer-eu` and `whisper-lm-eu`. Board:
  `OpenVoiceOS/ovos-stt-bench-commonvoice-eu`. Script: same
  `stt_commonvoice.py`.

All three locales are present in Common Voice with usable test splits and
CC0/CC-BY licensing, matching the `license` field convention already used
elsewhere in `registry/datasets/stt/`.

### 5. Wire up 1-2 wake-word scripts for idle picovoice/community sets

Every `picovoice-*` dataset except `picovoice-computer` (partially
covered via `community-computer`, itself unpopulated) is idle, and every
`community-*` dataset except `community-computer` is idle. Prioritize
`picovoice-alexa` and `picovoice-jarvis`, they have the largest positive
sample pools in the Picovoice benchmark corpus, by adding
`benchmarks/ww_alexa.py` / confirming `ww_hey_jarvis.py` also covers
`picovoice-jarvis` alongside `synthetic-wakewords-hey_jarvis` (check
whether one script can take a dataset-id argument instead of one script
per wakeword, to avoid an 24-file sprawl in `benchmarks/`).

### 6. Populate `community-computer`'s empty predictions directory

The board exists but holds no fighter predictions. Run any wake-word
fighter against it once to confirm the pipeline actually writes to this
board, an empty board that predates the pipeline's current write path is
a sign the dataset was registered before the publishing code existed, or
that publishing silently failed.

### 7. Per-language TTS prompt sets

Only 2 prompt sets exist, both `lang: multi`, against 31 TTS fighters.
Propose per-language prompt sets pulled from the existing
`intents-for-eval` and `massive-templates` corpora, filtered to fighters'
actual synthesis languages, particularly for single-language voice
families that currently only get generic multi-language prompts:

- `registry/datasets/tts/intents-for-eval-prompts-pt-PT.json` and
  `-pt-BR.json`, unlocks meaningful language-specific scoring for the
  `pipertts_pt-PT_*`, `pipertts_pt-BR_*`, and `phoonnx_pt-PT_*` voice
  fighters (Portugal has the densest single-language TTS fighter pool in
  the registry).
- `registry/datasets/tts/intents-for-eval-prompts-gl-ES.json`,
  `-an.json`, `-ast.json`, `-oc.json`, matches the phoonnx Galician,
  Aragonese, Asturian, and Occitan voices, none of which have any
  language-appropriate prompt text today.
- Script: extend `tts_intents_prompts.py` to accept a `--lang` filter
  instead of adding one script per language, since TTS scoring is
  judge-based rather than metric-based and the harness logic doesn't
  change per language.

### 8. Retire the legacy bare `speech-vs-nonspeech` VAD dataset

`speech-vs-nonspeech` and `speech-vs-nonspeech-en-US` overlap in source
material and scored fighters. Once confirmed that no external consumer
depends on the bare id, remove `registry/datasets/vad/speech-vs-nonspeech.json`
and its board, keeping the per-locale `-en-US` entry as the single English
VAD reference. This removes one duplicate board without losing any
distinct positive/negative material, both already draw from the same
`TigreGotico/not-wake-words-speech-en` positives and the same negative
pools.

### 9. Retire or wire up `hass-intent-templates`

62-locale train corpus with zero eval datasets referencing it. Either:

- **retire**: drop the registry entry and its `predictions_hf` field if
  no eval dataset is planned to consume it, or
- **wire up**: add an `intents-for-eval`-style eval dataset per locale (or
  a `multi` eval set) with `train_datasets: {"template": "hass-intent-templates"}`,
  giving intent-classification fighters an eval path through Home
  Assistant-style commands distinct from the MASSIVE-derived sets already
  in place. Given the corpus already spans 62 locales, this is the single
  addition with the biggest payoff for intent-modality language coverage
  if pursued.

### 10. Candidate datasets on the `OpenVoiceOS` HF org not yet registered

The `OpenVoiceOS` HF org hosts several datasets that look benchmark-ready
but have no corresponding `registry/datasets/` entry:

- `OpenVoiceOS/ovos-localize-intents` and
  `OpenVoiceOS/ovos-localize-intents-translated`, localized intent
  utterances. Worth checking against the intent registry's language
  coverage gap in the same way as proposal 9.
- `OpenVoiceOS/ovos-localize-synthetic-multilingual`, synthetic
  multilingual speech. A candidate STT or TTS-prompt source depending on
  its actual content (audio+transcript pairs would be an STT eval set,
  text-only would be a TTS prompt set, needs inspection before writing
  a registry entry).
- `OpenVoiceOS/MT-intents-dataset-pt-PT`, a Portuguese intent dataset
  with no registry entry in `registry/datasets/intent/`. Likely an
  additional pt-PT intent eval or train source.
- `OpenVoiceOS/tts-vc-mcv-scripted-v24.0-fy-nl-dii` and
  `-miro`, Frisian (fy-NL) scripted voice-conversion/TTS material,
  speaker-tagged (`dii`, `miro`). No `fy-NL` entry exists in
  `registry/datasets/tts/`. This looks purpose-built for exactly the
  per-language TTS prompt gap in proposal 7.
- `OpenVoiceOS/tts-test-sentence`, looks like a minimal TTS smoke-test
  set, distinct from the two registered prompt corpora. Worth checking
  whether it's meant to replace or supplement them.
- `OpenVoiceOS/ovos-intents-ilenia-testset-ca`,
  `-es`, `-nl`, held-out intent test sets for Catalan, Spanish, and
  Dutch, named after the ILENIA project. None appear in
  `registry/datasets/intent/`. These would give the `ca` locale both an
  intent eval path (via these sets) and an STT eval path (via proposal 4)
  where today it has neither.

Each of these needs a content inspection pass (row shape, split, license)
before it can get a registry entry, this review only confirms they exist
on the org and are unclaimed by the current registry, it does not
validate their internal schema.

## `stt-sampler-v1`: breadth before depth

`registry/datasets/stt/stt-sampler-v1.json` adds a single multi-lang STT
eval corpus (`lang: multi`) covering every locale the arena's STT fighters
are registered against today: the 14 MInDS-14 locales, the 12 Speech-MASSIVE
locales, and FLEURS `ca_es`/`gl_es` (20 locales total, some overlapping
between sources). It targets 100 clips per language, banking calls from
MInDS-14, voice-assistant utterances from Speech-MASSIVE, read speech
from FLEURS. Where a locale is covered by two sources (de-DE, es-ES,
fr-FR, ko-KR, nl-NL, pl-PL, pt-PT, ru-RU) each source contributes 50, so
neither dominates a shared locale. Single-source locales draw all 100
from that source. Clips are selected deterministically by sorting each
source split by its sample id and striding through it (`index = i *
floor(n/k)`, k=50 or 100), not by taking the first k rows, which would
cluster on a single speaker. The 20 language subsets live as one HF
dataset repo (`OpenVoiceOS/stt-sampler-v1`, parquet, embedded audio,
16kHz mono float32, one file per `lang` config) so a single registry
entry and a single predictions repo
(`OpenVoiceOS/ovos-stt-bench-stt-sampler-v1`) cover all of them.

The point is breadth before depth: a new STT fighter or a new locale gets
an immediate, balanced leaderboard across every registered language from
day one, at 1/N the compute of a full run over MInDS-14 + Speech-MASSIVE +
FLEURS. The full per-locale corpora (`minds14-*`, `speech-massive-*`,
`fleurs-*`) remain the deep benchmarks, this sampler is the fast board
that runs first and most often. It does not replace them, and its
100-clip-per-language sample is not a substitute for the per-locale WER
numbers the full corpora produce over time as compute allows.

## Schema note: no sample-count field

No `DatasetDef` in `registry/datasets/` carries a row/sample count, so
there is no way to compare dataset size at a glance from the registry
alone, audit relies on reading the actual HF split. `registry/schemas.py`
declares `DatasetDef` with `model_config = ConfigDict(extra="forbid")`, so
adding a `num_samples: int | None` field is a real schema change, not a
free-form annotation, every existing JSON file would need the field
added (or the field would need a default of `None` to stay backward
compatible with the 80 existing entries), and any code that validates or
diffs `DatasetDef` instances against a fixed field set would need to
account for it. This review flags the gap and the schema-impact shape. It does not implement the change.

---
[← Registry audit](registry-audit.md) · [Home](index.md) · [Dataset gapfill →](dataset-gapfill.md)
