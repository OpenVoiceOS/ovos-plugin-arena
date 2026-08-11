# Registry audit

This is a per-fighter review of `registry/competitors/`: 156 fighter JSON files across seven
modalities (`stt`, `tts`, `wake_word`, `vad`, `intent`, `intent_keyword`, `intent_template`).
Each fighter got a verdict:

- **KEEP** — correctly modeled, stays as-is.
- **FIX** — stays, but the JSON or its tagging needs a concrete change.
- **DROP** — remove, with the reason.
- **ADD** — a fighter that should exist and does not.

Findings below were checked against the actual JSON files (config diffs, `langs` fields,
`types` tags) rather than assumed from naming. Where a claim rests on an outside source
(the OpenVoiceOS Hugging Face org, or a plugin repo's own README) that source is named.

## Summary table

| Modality | Fighters | KEEP | FIX | DROP | ADD proposed |
|---|---|---|---|---|---|
| stt | 41 | 38 | 3 | 0 | 3 plugins + 3 onnx-asr language families |
| tts | 31 | 31 | 0 | 0 | 0 |
| wake_word | 45 | 43 | 2 | 0 | 12 (stack variants) |
| vad | 7 | 0 | 7 (schema-wide) | 0 | 1 (precise) |
| intent | 8 | 6 | 0 | 0 | 0 (2 relocate to new league) |
| intent_keyword | 6 | 2 | 0 | 4 | 0 |
| intent_template | 18 | 6 | 0 | 12 | 0 |
| intent_embedding (new) | 0 | — | — | — | 2 (relocated from `intent`) |
| **Total** | **156** | **126** | **12** | **16** | **~19 concrete + Indic follow-up batch** |

FIX counts a fighter once even when it needs more than one change. The vad row's "7 FIX"
means the duplicate-key schema/documentation problem touches all 7 existing VAD files —
none are functionally broken, so "KEEP 0 / FIX 7" means all seven need the same
documentation touch-up, not that any of them should be dropped.

## intent, intent_keyword, intent_template

### DROP: the `-high` / `-low` variants of every tiered single-stage pipeline

`intent_keyword` and `intent_template` ship each single-engine pipeline three times, once
per confidence gate: `adapt-{high,medium,low}`, `palavreado-{high,medium,low}`,
`padatious-{high,medium,low}`, `jurebes-{high,medium,low}`, `linha-fina-{high,medium,low}`,
`markov-{high,medium,low}`, `nebulento-{high,medium,low}`, `padacioso-{high,medium,low}`.

Diffing each `-high` and `-low` file against its `-medium` sibling (`config`, `langs`,
`species`, `types`, `description`, `size` all included) shows the **per-plugin config block
is byte-identical** in every pair. The only difference anywhere in the file is the string
inside `config.intents.pipeline`: `ovos-adapt-pipeline-plugin-high` vs
`...-medium` vs `...-low`. That string selects which of the plugin's own three built-in
confidence gates OVOS routes through — it is not a different configuration of the plugin,
it's the same plugin at a different pipeline stage name. The `-high`/`-low` files' own
`notes` field admits this ("per-plugin config is identical to the -medium fighter").

Only the `-medium` variant of each of these eight engines ever appears on a published
benchmark board — the `-high`/`-low` triplicate exists in the registry but contributes no
board coverage. That's 16 dead files (8 engines × 2 unused tiers) doing no work:

**DROP:** `intent_keyword/adapt-high.json`, `adapt-low.json`, `palavreado-high.json`,
`palavreado-low.json`; `intent_template/jurebes-high.json`, `jurebes-low.json`,
`linha-fina-high.json`, `linha-fina-low.json`, `markov-high.json`, `markov-low.json`,
`nebulento-high.json`, `nebulento-low.json`, `padacioso-high.json`, `padacioso-low.json`,
`padatious-high.json`, `padatious-low.json`.

**KEEP** the `-medium` variant of each: `adapt-medium`, `palavreado-medium`,
`jurebes-medium`, `linha-fina-medium`, `markov-medium`, `nebulento-medium`,
`padacioso-medium`, `padatious-medium` — these are the ones that actually differentiate
the roster (7 template/keyword engines, one config each) and the ones boards use.

If a genuine confidence-gate comparison is wanted later, it needs a fighter whose config
*visibly* changes the gate (e.g. an explicit `conf_high`/`conf_med`/`conf_low` override per
tier) rather than three copies of the same numbers wearing different pipeline-stage labels.

### FIX: `hierarchical-knn-medium` and `m2v-medium` are misfiled

Both are single-stage embedding classifiers (`ovos-hierarchical-knn-pipeline` and
`ovos-m2v-pipeline`) sitting in the `intent` league, which per the roster's own convention
is reserved for multi-engine fusion/ensemble pipelines (`frankenparse`, `nebulapt`,
`nebulatious`, `padapt`, `palavadapt` are all two-or-more-engine combinations; `padatioso`
was subsequently DROPped, see `docs/ensemble-rationale.md`).
`hierarchical-knn-medium` and `m2v-medium` each run exactly one engine — they compete on a
completely different axis (embedding similarity vs. rule/template fusion) and their own
`notes` field flags a modeling caveat (pretrained on the legacy intent-benchmark corpus,
not native OVOS skill intents) that doesn't apply to the fusion fighters around them.

**FIX:** create an `intent_embedding` league (new `registry/competitors/intent_embedding/`
directory) and move `hierarchical-knn-medium.json` and `m2v-medium.json` there unchanged.
Leave the genuine fusion pipelines (`frankenparse`, `nebulapt`, `nebulatious`, `padapt`,
`palavadapt`) as the `intent` league. `padatioso` was DROPped in the 2026-08-11
ensemble-justification pass — see `docs/ensemble-rationale.md`.

### `frankenparse` — SUPERSEDED, see `docs/ensemble-rationale.md`

The paragraph below reflects the pre-2026-08-11 config and is kept for history; the
justification pass in `docs/ensemble-rationale.md` reached the opposite conclusion on the
double-`padatious` stage and slimmed the pipeline. See that doc for the current rationale
and config.

`frankenparse`'s pre-justification `pipeline` list was:
`padacioso-high → adapt-high → padatious-high → palavreado-medium → padatious-medium →
nebulento-medium → adapt-low`.

`ovos-padatious-pipeline-plugin` did appear twice, at `-high` and `-medium`. That was
argued at the time to be intentional pipeline design: OVOS pipeline routing tries stages in
order and stops at the first one that fires, so listing the same plugin at its high gate
early (favor precision) and again at its medium gate later (fallback for recall) reads as a
legitimate two-chance strategy for that one engine, mirroring what the rest of the chain
does for adapt (`-high` then `-low`). The 2026-08-11 ensemble-justification pass found this
insufficient on its own — a second tier of the *same* trained classifier is not an
independent paradigm — and additionally found the leading `padacioso-high` stage redundant:
Padatious runs `padaos` (the same regex/exact-match engine padacioso wraps) internally
unless `disable_padaos=true` (it is not, here), and returns `conf=1.0` for perfect matches
before its neural score is even consulted (verified in
`ovos_padatious/intent_container.py::IntentContainer.calc_intents`). Both stages were
dropped; see `docs/ensemble-rationale.md` for the resulting 5-stage config and hypothesis.

## wake_word

### FIX: `wakewordlab-hey-jarvis` silently competes VAD-gated against un-gated fighters

`wakewordlab-hey-jarvis` sets `vad: true` / `vad_threshold: 0.5` **inside the hotword's own
config block** (`config.hotwords.hey_jarvis.vad`), which enables the plugin's built-in
Silero pre-filter. `openwakeword-hey-mycroft-silero` gets the same behavior — a VAD gate in
front of the detector — but does it by attaching `listener.VAD` as a separate pipeline
stage, and it's tagged `"types": ["neural-net", "vad-gated"]` for it. `wakewordlab-hey-jarvis`
is tagged only `["neural-net"]` — no `vad-gated` tag — despite gating being on.

On `synthetic-wakewords-hey_jarvis`, this means `wakewordlab-hey-jarvis` is quietly
benchmarked against `microwakeword-hey-jarvis` and `wakeforge-hey-jarvis` — both genuinely
un-gated — as if all three were the same kind of detector, when wakewordlab already gets
the false-accept suppression the other two would need a separate VAD stage to get. The
`-thr07` sibling has the same issue.

**FIX:** add `"vad-gated"` to `types` on `wakewordlab-hey-jarvis.json` and
`wakewordlab-hey-jarvis-thr07.json`, and note in `notes` that the VAD gate is built into
the plugin config rather than a separate `listener.VAD` stage (so readers don't expect a
`listener.VAD` block when diffing it against the openWakeWord combo fighters).

### ADD: WW stacks exist only for openWakeWord + hey_mycroft

The three "stack" variants — VAD-gated (`-silero`), speaker-verified (`-speaker`), and
both (`-silero-speaker`) — exist only for `openwakeword-hey-mycroft`. `microwakeword`,
`wakewordlab`, `wakeforge`, and `precise-onnx` each have only their bare detector fighter.
That's a real roster gap: the benchmark can currently answer "does a VAD gate or speaker
verifier help openWakeWord" but not whether the same holds for the other four engines,
which is exactly the kind of engine-vs-engine comparison the arena exists to run.

Proposed adds (one config-variation fighter per engine × stack, following the existing
openWakeWord combo files as the template — each is a complete mycroft.conf-style `config`
block, not a diff):

- `microwakeword-hey-jarvis-silero.json` / `-speaker.json` / `-silero-speaker.json`
- `wakewordlab-hey-jarvis-silero.json` / `-speaker.json` / `-silero-speaker.json` — note
  this one needs the *external* `listener.VAD` stage added on top of the plugin's own
  `vad: true`, or `vad: false` set first so the comparison isolates the same gate mechanism
  as the openWakeWord fighters; otherwise it would double-gate.
- `wakeforge-hey-jarvis-silero.json` / `-speaker.json` / `-silero-speaker.json`
- `precise-onnx-hey-mycroft-silero.json` / `-speaker.json` / `-silero-speaker.json` (picked
  because `hey-mycroft` is the one phrase precise-onnx shares with openWakeWord, keeping
  the comparison apples-to-apples on `synthetic-wakewords-hey_mycroft`)

That's 12 new fighters. `wav2vec2`/`vosk-ww-*`/`openwakeword-alexa` are excluded from this
list since they either don't build a VAD/verifier stack pattern or already have no
`hey_mycroft`/`hey_jarvis` peer to compare against.

### `vosk-ww-*` (22 fighters) — assessed, KEEP

22 fighters, one per phrase, each the sole competitor on its own synthetic dataset
(`vosk-ww-alexa`, `-amelia`, `-athena`, … `-view-glass`). This isn't roster breadth in the
sense of "many ways to run one engine" — it's one fighter per phrase because Vosk's
keyword-spotting grammar is defined by the phrase itself, so a different phrase really is a
different fighter, not a config variation of an existing one. Legitimate as configured.
**KEEP all 22**, but don't count them as engine-diversity coverage when reasoning about
roster gaps — they're a single engine's per-dataset presence, not competing configurations.

## vad

### FIX: every VAD fighter duplicates its config keys flat and nested

All 7 files (`silero-vad`, `silero-vad-thr03`, `silero-vad-thr07`, `webrtcvad`,
`webrtcvad-mode1`, `noise-vad`, `noise-vad-strict`) place identical keys both directly
under `listener.VAD` and again nested under the module's own name inside `listener.VAD`.
For example `silero-vad.json`:

```json
"listener": {
  "VAD": {
    "module": "ovos-vad-plugin-silero",
    "threshold": 0.5,
    "ovos-vad-plugin-silero": {
      "threshold": 0.5
    }
  }
}
```

`threshold: 0.5` is written twice — once flat, once under the `ovos-vad-plugin-silero` key.
`webrtcvad.json` duplicates four keys the same way (`vad_mode`, `padding_duration_ms`,
`frame_duration_ms`, `thresh`, all repeated verbatim under `ovos-vad-plugin-webrtcvad`).
This isn't a per-fighter typo, it's the schema every VAD file was generated from — it's not
wrong in the sense of contradicting itself (both copies always agree), but it's redundant
and doubles the risk that a future edit updates one copy and not the other, silently
desyncing "the config a user pastes into mycroft.conf" from "the config actually read by
the nested module block." mycroft.conf itself only needs one of the two forms (OVOS reads
per-module config from the nested block, keyed by module name; the flat keys are consumed
directly by the VAD manager for user-facing settings like `threshold`) — so this isn't
purely cosmetic, it reflects two different consumers reading the same value from different
places.

**FIX:** keep both keyed locations (both are load-bearing for different consumers) but add
a one-line comment/note in each VAD fighter's `notes` field stating that the flat key is
read by the listener's VAD manager and the nested key is read by the plugin itself, so the
duplication reads as intentional rather than copy-paste residue when someone next edits
these files.

### ADD: `ovos-vad-plugin-precise`

Confirmed present and unarchived at `~/AgentWorkspaces/ovos/plugins/vad/ovos-vad-plugin-precise`.
No fighter exists for it. **ADD** `precise-vad.json` following the existing VAD schema
(flat + nested keys per the FIX above), at the plugin's default threshold, as a baseline;
a `-thr` variant can follow the `silero-vad-thr03`/`-thr07` pattern once the default is on
a board.

## stt

### FIX: `whisper-large-v3-turbo`, `whispercpp-base`, `whispercpp-small` have `langs: []`

Every other STT fighter pins one or more specific `langs`. These three ship `langs: []`,
which — per how boards select fighters — makes them eligible to appear against every
language board rather than only the ones they were actually evaluated on. That may be
intentional (Whisper is genuinely multilingual and both plugins auto-detect language), but
as written it's indistinguishable from an omitted field. Two other multilingual STT
fighters in the registry, `onnx-asr-canary` and `onnx-asr-parakeet-tdt-06b-v3`, handle this
correctly: they enumerate the specific languages the pinned checkpoint actually supports
(`['en','de','fr','es']` and an 11-language list respectively) instead of leaving `langs`
empty.

**FIX:** either (a) enumerate the language set actually exercised in scoring for
`whisper-large-v3-turbo`/`whispercpp-base`/`whispercpp-small`, matching the
`onnx-asr-canary` pattern, or (b) if `langs: []` is meant as documented any-language
semantics, add that as an explicit convention (e.g. a `"langs": "any"` sentinel or a note
in the registry schema doc) so it isn't read as three fighters that simply forgot to fill
in the field.

### `fasterwhisper-base-{de,en,fr,it,pt}` — assessed, no schema change recommended

Five files, one model (`ovos-stt-plugin-fasterwhisper`, `model: base`,
`compute_type: int8`, `beam_size: 3`, all else identical), differing only in `langs` and
the top-level `lang`/`config.lang` field. A `langs: [de, en, fr, it, pt]` list on one file
would collapse this to one fighter, but that would break the modality's own convention:
every other lang-pinned STT fighter (`vosk-small-*`, `onnx-asr-conformer-transducer-*`,
`citrinet-512-ca`) is also one file per language, and boards are keyed per-language, so a
multi-lang fighter would need special-casing in the board-selection logic to know which
language's audio it's being scored against. **KEEP as 5 separate fighters** — this is the
registry's normal per-language pattern applied consistently, not a duplication defect.

### ADD: unarchived plugins with local repos and no fighter

Checked each candidate against its own repo README under `~/AgentWorkspaces/ovos/plugins/`:

| Plugin | Repo status | Verdict |
|---|---|---|
| `ovos-stt-plugin-HiTZ` | **archived** (README banner) | do not add |
| `ovos-stt-plugin-nos` | **archived** (README banner) | do not add |
| `ovos-stt-plugin-mms` | **archived** (README banner) | do not add |
| `ovos-stt-plugin-MyNorthAI` | active | **ADD** |
| `ovos-stt-plugin-coreml` | active | **ADD** |
| `ovos-stt-plugin-wav2vec2` | active, distinct from the already-registered `ovos-stt-plugin-wav2vec` (`wav2vec2-xlsr-en.json`) | **ADD** |

**ADD** `mynorthai-<lang>.json`, `coreml-<model>.json`, `wav2vec2-<lang>.json` — one
fighter per meaningful config axis (language for MyNorthAI/wav2vec2, quantization tier for
coreml, matching the pattern the existing `fasterwhisper`/`parakeet` coreml family on HF
uses: fp16/int8/4bit/6bit are meaningfully different fighters, not just packaging).

### ADD: onnx-asr models on the OpenVoiceOS Hugging Face org not yet in the registry

Listed the org via `huggingface_hub.list_models(author='OpenVoiceOS')` (269 models total,
covering STT, TTS, phonemization, and non-arena media-classifier models). Filtered to
ASR/onnx checkpoints usable by `ovos-stt-plugin-onnx-asr` and compared against the 16
`onnx-asr-*` fighters already in the registry (`canary`, `conformer-ctc-{en,gl,it}`,
`conformer-transducer-{ca,de,es,eu,fr,ru}`, `parakeet-{nl,pl,pt}`,
`parakeet-tdt-{06b-v3,11b}`, `parakeet-tdt-ctc-110m`, `whisper-small-pt`).

Not covered at all today, by language/family:

- **`ai4bharat-indicconformer-*-onnx`** — 22 separate Indic-language checkpoints (as, bn,
  brx, doi, gu, hi, kn, kok, ks, mai, ml, mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur).
  Zero Indic-language coverage exists in the STT league today. Proposed ids:
  `onnx-asr-indicconformer-hi.json`, `-ta.json`, `-te.json`, `-bn.json`, `-mr.json` as a
  first slice (the five largest speaker populations); the remaining 17 as a follow-up batch
  rather than all 22 at once.
- **`artpark-iisc-vaani-fastconformer-{hi,kn,ml,or,te,multi}-onnx`** — a second, distinct
  Indic model family (Vaani, not IndicConformer) for hi/kn/ml/or/te plus a multilingual
  checkpoint — worth adding at least `-multi` (`onnx-asr-vaani-multi.json`) as a second
  Indic engine to compare against IndicConformer, not just a second checkpoint.
- **`nvidia-{be,hr,eo,kab,rw}-conformer-{ctc,transducer}-large-onnx`** — Belarusian,
  Croatian, Esperanto, Kabyle, Kinyarwanda: languages nvidia ships conformer checkpoints
  for that have zero registry presence (the registry currently only pulls ca/de/en/es/fr/it/ru
  from the nvidia conformer family). Proposed: `onnx-asr-conformer-transducer-hr.json`,
  `-be.json`, `-rw.json`, `-eo.json`, `-kab.json`.
- **`stt-eu-conformer-ctc-large-onnx`** and **`hitz-eu-conformer-transducer-large-v2-onnx`**
  — a CTC Basque checkpoint (registry only has the transducer variant,
  `onnx-asr-conformer-transducer-eu`) and a v2 HiTZ Basque model; both are meaningful
  variations on an already-covered language (architecture and version respectively) and fit
  the "multiple fighters per plugin for meaningful config variation" rule directly.
- **`nvidia-parakeet-ctc-1.1b-onnx`**, **`nvidia-parakeet-rnnt-110m-da-dk-onnx`** — English
  CTC-head Parakeet (registry only has TDT-head Parakeet variants) and Danish, a language
  with zero registry presence.
- **`stt-ca-es-conformer-transducer-large-onnx`** — Catalan-Spanish code-switching
  conformer, distinct from the already-registered mono-Catalan `nvidia-ca` checkpoint.

Not proposed for addition: the `bbs-*-coreml`, `parakeet-*-coreml*`, and `stt-*-coreml*`
families (these are Apple CoreML packagings of models already representable via the onnx
path, and CoreML isn't what `ovos-stt-plugin-onnx-asr` consumes — they'd need the
not-yet-added `ovos-stt-plugin-coreml` plugin instead, see the ADD row above); the
`misterkissi-*`, `carlosdanielhernandezmena-*`, and other single-contributor low-resource
models (real, but out of scope for a first pass — flag for a dedicated low-resource-STT
sweep rather than folding into this audit).

## tts

All 31 TTS fighters checked out clean: `types`, `langs`, and `config` are internally
consistent, no duplicate-variant pattern like the intent leagues, no misfiled fighters.
**KEEP all 31.** No ADD candidates surfaced beyond the plugin coverage already tracked
elsewhere (`ovos-tts-plugin-matxa-multispeaker-cat` is confirmed **archived** — see its
README banner — so it is correctly absent, not a gap).
