# Wake word league (`wake_word`)

The wake word league measures one decision per clip. A clip either holds the
wake phrase (a positive) or does not (a negative). The fighter must fire on
the positives and stay silent on the negatives. Each clip is short, so this
board measures detection and rejection, not false wake-ups per hour. For
false wake-ups per hour, read the [false wake-up league](ww_stream.md).

## How a fighter runs

`runner/ww_bench.py` (`WakeWordBench`) loads the fighter's real OVOS
`HotWordEngine` through OPM (`load_wake_word_plugin`). It feeds each clip
frame by frame through `update()` and `found_wake_word()`.
`ovoscope.wakeword_probe.WakeWordProbe` drives the engine. It calls
`reset()`, then primes the clip with `PRIME_SECONDS` of leading silence, so
that streaming feature buffers warm up as they do on a live microphone. It
adds `TAIL_SECONDS` of trailing silence, so that a late activation can
settle. The engine uses its own threshold. The arena records only the
decision and the latency. The first activation ends the clip.

The frame size depends on the engine:

- **Frame engines.** Every engine except a windowed one gets frames of
  1280 samples (80 ms at 16 kHz, `FRAME_SAMPLES`).
- **Windowed engines.** An engine that has a `time_between_checks`
  attribute buffers audio and checks it once per window. In the registry,
  the vosk plugin is such an engine. It gets the microphone chunk of
  `ovos-dinkum-listener`, 2048 samples (`LISTENER_CHUNK_SAMPLES`). The
  runner builds a fresh engine for each clip in an untimed `prepare()` step,
  because the plugin's `reset()` leaves the check clock and the unread audio
  of the previous clip in place. It adds a check phase to the leading
  silence: an offset inside one check window, from the SHA-256 of the clip's
  `audio_url`, or of its samples when it has no URL. A listener's check clock
  runs on while the user is silent, so a phrase starts at a random point of
  the window. The trailing silence is `TAIL_SECONDS` plus one check window,
  so the check after the phrase always runs.

A fighter is the engine as the listener stacks it:

- `config.VAD` adds a pre-wake VAD. When the VAD hears no speech in the
  clip, the detector does not run.
- `config.hotword_verifier` adds a verifier, for example a speaker check. An
  activation counts only when the verifier confirms it. When the verifier
  raises an error, the activation counts.

Each `(engine, VAD, verifier)` combination is its own fighter, with its own
false accept and false reject trade-off. A threshold is not a fighter. A
different activation threshold is one more point on the same model's curve,
so a threshold variant counts one model twice. Each model runs at its own
reference threshold.

## Metrics

`arena.metrics.score_wake_word` computes these metrics for each
`(dataset, lang)`. A row's `label` is the truth and its `prediction` is the
decision. The tokens `positive`, `wake`, `wakeword`, `detected`, `hit`,
`true`, `yes`, `1`, `speech` and `voice` mean present. Any other non-empty token means absent.

| Metric | Definition | Better |
|---|---|---|
| **`error_rate`** (primary) | (false accepts + false rejects) divided by all scored clips. | lower |
| `accuracy` | 1 − `error_rate`. | higher |
| `false_accept_rate` | Negatives that fired, divided by all negatives. | lower |
| `false_reject_rate` | Positives that did not fire, divided by all positives. | lower |
| `latency_ms_median` | The median decision time of one clip. | lower |
| `fa_per_hour` | False accepts per hour of negative audio. Only rows that record `audio_secs` count. Absent below 0.25 h (`MIN_FA_HOURS`) of such audio. | lower |
| `fa_per_hour_hours` | The hours of negative audio behind `fa_per_hour`. | n/a |

`error_rate` weights the two error types by the number of clips in each
class. Read `false_accept_rate` and `false_reject_rate` beside it.

The confidence interval is a seeded bootstrap of the 0/1 error of each
scored clip: 1000 rounds, seed 0, 2.5th to 97.5th percentile.

A board ranks an entry only when it has both classes. An entry with no
negatives or no positives is unranked with the reason `one_class`, because
its `error_rate` then measures one error type only. An entry with fewer than
30 scored clips in all, both classes counted together, is unranked with the
reason `too_few_samples`.

## Datasets

[Datasets](../datasets.md) lists every clip set. The families are:

| Family | Positives | Negatives |
|---|---|---|
| `synthetic-wakewords-<phrase>` | TTS clips of the phrase in many voices, from `OpenVoiceOS/synthetic-wakewords`. | The shared pool, plus `mlsw-negatives-<lang>`. |
| `community-<phrase>` | Real recordings from `OpenVoiceOS/ovos-community-wakewords-dataset`. | The shared pool, plus the spoken-word set of the language. |
| `picovoice-<phrase>` | Real recordings from `Picovoice/wake-word-benchmark`. | The shared pool, plus the spoken-word set of the language. |
| `sam-wake-word` | The `positive/` folder of `quo-labs/sam-wake-word`. | The `negative/` folder of the same corpus, plus the spoken-word set of the language. |
| `tigregotico-<phrase>[-<lang>]` | Synthetic clips from the TigreGotico wake word repos, one set per phrase and language. `scripts/gen_tigregotico_ww_sets.py` writes them, and skips a set with fewer than 50 clips of the phrase (`MIN_POSITIVES`). | The shared pool, plus the spoken-word set of the language where one exists. |
| `mlsw-negatives-<lang>` | None. These are pools of single spoken words from `MLCommons/ml_spoken_words` (`role: unrestricted`). | Every clip. |

The shared pool (`negatives_sources`) holds five TigreGotico corpora:
`not-wake-words-speech-en`, `public_domain_sounds_3secs`, `ESC-50`,
`FMA_3secs` and `ambient_noises`. A set lists its spoken-word pool in
`negatives_dataset_ids`.

For the `tigregotico-*` sets:

| Repos | Clips with the phrase | Languages |
|---|---|---|
| English phrase repos | `train/`, without `openvoiceos-synthetic-wakewords/` (the source of the `synthetic-wakewords-*` sets) and `original/`. `unchecked/` is never read. | en-US |
| Localised repos with a `test/` split | The held-out `test/` clips, one set per `lang` of its `metadata.jsonl` (`positives_filter`). | pt-PT, de-DE, ca-ES, es-ES, an-ES, eu-ES, it-IT, da-DK, nl-NL |
| Train-only localised and Portuguese repos | `train/` | ca-ES, fr-FR, es-ES, pt-PT |

A repo whose clips say several phrases gives one set per phrase, selected by
the `text` of its `metadata.jsonl`. `exclude_dirs` leaves out folders that
another set already holds, so no clip is scored twice under two names.
`positives_dir` names the folder of positives when it is not named after the
phrase, as in `sam-wake-word`.

## Samples and manifests

Every clip set has `sample_policy.max_samples: 100`. On a clip set this cap
applies per class: up to 100 positives, up to 100 clips from the listed
negatives, and up to 100 clips from the pooled spoken-word sets, shared
evenly between them. A run thus holds about three times the cap. The
positives and the listed negatives are picked at even steps through the
sorted file listing, so the seed has no effect on them.

A `mlsw-negatives-<lang>` set has its own policy: `max_samples: 5000`, seed
1337. When a clip set pools it, its share of clips is a seeded shuffle of
the spoken-word corpus, capped by that policy. The seed thus decides which
spoken words a run holds.

`python -m runner.publish_sample_set` publishes the selected clip ids as
`sample_sets/<ll_RR>.json` in the prediction repo of the set. `assemble`
scores every fighter on that manifest. An entry that covers less than 90% of
the manifest is unranked with the reason `sample_set_partial`.

## Fairness rules

**No fighter is scored on its own training data.** A fighter lists in
`trained_on` the datasets whose recordings were in its training data. The
benchmark skips those pairs, the assembler drops published rows for them,
and the fleet runner never schedules them. Examples:

- The Precise ONNX fighters run five models: hey mycroft (the default
  model and the Mark 1 model), hey robin, marvin and sheila. They list no set, because their training data is not a
  registered eval corpus. The other published Precise models are mostly
  single-speaker personal models, and the registry holds no fighter for them.
- Every English WakeHuBERT fighter (`ovos-ww-plugin-wakeforge`) lists the
  `tigregotico-*` set of its phrase. When its model card names the whole
  repo, it also lists the `synthetic-wakewords-*` set of the phrase.
- A few-shot fighter, such as one of `ovos-ww-plugin-local-wake`, names its
  reference clips with `samples_from`. That dataset must be in `trained_on`.
  `validate-registry` refuses the entry otherwise.

**Pairs match on phrase and language.** `registry.loaders.is_same_phrase`
compares the fighter's `hotwords` key with the set's `wakeword`, after
lowercase and with spaces and dashes as underscores. A set with no
`wakeword`, such as a pool of negatives, accepts every fighter. The rule
applies at two points:

- `runner.queue_tools.is_compatible` uses it, so the fleet runner and the
  queue never schedule a fighter on a set of another phrase. It also
  requires a language overlap, on the primary subtag.
- `arena.predictions.group_rows` uses it, so `assemble` drops the rows of a
  fighter on a set of another phrase. The assemble summary counts them in
  `cross_phrase_rows_excluded`, keyed `<fighter>/<dataset>`.

The benchmark scripts do not check the phrase. They run every fighter that
`--competitors` names, less the `trained_on` sets and the plugins that are
not installed. A cross-phrase row from a script never reaches a board.

## Battles and ELO

The wake word battle group is the league itself. `assemble` builds a blind
battle pool per `(dataset, lang)`, an ELO seed per language and a ladder. A
voter hears the source clip and sees two decisions.

On a wake word set, only the fighters that the benchmark board ranks battle
or seed a rating. A fighter that the board leaves unranked, for too few
rows, one class or a failed run, takes no part. A set with no ranked
fighter gives no battles and no seed. For each clip and each pair of
ranked fighters where exactly one is right, the right one wins an
auto-battle. A pair whose confidence intervals overlap gets no auto-battles
on that set, and the weight of one pair is capped at 5 human votes.
[Rating](../rating.md#auto-battles-and-the-seed) gives the seed rules.

The free-form matchup pool (`battles-wake_word-freeform-<lang>.json`) pairs
only fighters of the same phrase.

## Which boards the site lists

`arena/ww_listing.py` decides which wake word boards `export-index` lists in
`data/index.json`. A board that is not listed stays on disk under
`frontend-static/public/data/`. Only its entry in `index.json` is left out,
for the benchmark board and for the battle pool.

The rules use the registry `types` of each fighter. A fighter is
**generic** when its `types` hold `zero-shot`, `few-shot` or
`keyword-match`: it works for any phrase without training on it. Every
other fighter is a **trained model**.

1. A clip board **qualifies** when its test set holds 50 or more positive
   clips (`MIN_POSITIVES`) and it ranks 2 or more trained models
   (`MIN_TRAINED`). Rows of a fighter trained on the set never reach a
   board, so "ranks" counts after that exclusion.
2. A phrase is **featured** when a qualifying board ranks a trained model
   whose plugin is not `ovos-ww-plugin-wakeforge` (WakeHuBERT).
3. A clip board is **listed** when it qualifies and its phrase is featured.
4. A false wake-up board `ww-negatives-<lang>` is listed when a listed clip
   board has the same primary language. It ranks only the fighters of
   featured phrases. Every other `ww_stream` board is hidden.
5. A `mlsw-negatives-<lang>` clip set holds no positives, so its board is
   never listed.

The fighter generators apply the same rules before any board exists.
`registry_featured_phrases` predicts the featured phrases from the registry
alone, and `scripts/gen_ww_fewshot_fighters.py` writes generic fighters only
for those phrases. It removes the generated fighters of the other phrases.
`scripts/gen_ww_model_fighters.py` writes a model fighter only for a phrase
that has a wake word test set.

The `positives` field of a clip board holds the positive count that rule 1
reads: the most positives that any one fighter scored.

## Known limits

- **WakeHuBERT false accept rates on clip boards are optimistic.** The shared
  negative pool holds `not-wake-words-speech-en`, `ESC-50`,
  `public_domain_sounds_3secs`, `FMA_3secs` and `ambient_noises`. The
  wakeforge trainer, which makes the WakeHuBERT models, uses these corpora as
  training negatives by default. A WakeHuBERT fighter has therefore heard
  many clip-board negatives in training. The `ww-negatives-<lang>` false wake
  up boards of the [false wake-up league](ww_stream.md) use none of these
  sources. Use those boards to compare false wake-ups.
- **A script run can hold cross-phrase rows.** A benchmark script runs
  every fighter it is given. Name the fighters of the set's phrase with
  `--competitors` to save the run time. `assemble` drops the other rows.
- **A short clip does not measure time.** `false_accept_rate` is a share of
  clips. It does not tell how often a detector wakes on a live microphone.
- **Synthetic positives have no room.** TTS clips carry no room acoustics and
  no microphone variation.

---
[← Intent](intent.md) · [Home](../index.md) · [False wake-ups →](ww_stream.md)
