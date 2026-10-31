# Intent leagues

The four intent leagues share one task. A fighter reads one written
utterance and names the intent it belongs to, or names nothing when the
utterance is out of scope. Where the corpus annotates slots, the fighter also
fills them. The leagues differ only in what the fighter needs before it can
answer.

| League | What the fighter needs | Engines |
|---|---|---|
| `intent_zero_shot` | Nothing but the templates a skill registers, used as they arrive. | Prototype-mode embedding matchers. |
| `intent_online` | A training pass over those templates when the device boots. | Padatious, Padacioso, Nebulento, Jurebes, Linha Fina, Markov, and fusions of them. |
| `intent_offline` | A pretrained artefact, shipped as a model id, that never sees the device's templates. | Classifier-mode embedding models. |
| `intent_keyword` | Adapt-style `required_vocab` and `optional_vocab` rules. | Adapt, Palavreado. |

A fighter's league follows from its stages, and the registry rejects a
fighter filed in another league. A fusion takes the league of its heaviest
stage: a cascade whose first stage carries a pretrained head competes in
`intent_offline`. A fighter built only from keyword engines competes in
`intent_keyword` whatever its regime.

## How a fighter runs

A fighter is the `intents` section of a `mycroft.conf`: an ordered
`pipeline` list of `<plugin>-<tier>` stages, plus a config block per plugin.
`runner/intent_bench.py` runs each fighter through
`runner.intent_pipeline.IntentPipeline` in these steps:

1. **Load.** The runner imports each pipeline plugin from the hard-coded
   `ENGINE_REGISTRY` in `runner/intent_pipeline.py`, by module and class
   name. It does not look the plugin up through OPM entry points. A stage
   whose plugin is not in `ENGINE_REGISTRY` raises `KeyError`. The runner
   sets the global `ovos-config` language to the language of the run before
   it builds the plugins, because the plugins read the language from there.
2. **One bus per plugin.** Each distinct plugin gets its own in-process
   `ovos_utils` `FakeBus` and one instance, shared by all its tiers. No
   plugin sees the registration or training messages of another plugin.
3. **Register like a skill.** The training rows come from the corpus that
   the eval set names in `train_datasets`, for the datashape of the plugin.
   A template plugin gets one `padatious:register_intent` message per intent,
   with the raw templates and up to 6 expanded utterances per template, and
   one `padatious:register_entity` message per slot. A keyword plugin gets
   `register_vocab` messages and one `register_intent` message per intent,
   built with an Adapt `IntentBuilder`. Every message carries the skill id
   `arena`.
4. **Train.** The runner emits the training message that `ENGINE_REGISTRY`
   names for the plugin (`mycroft.skills.train` or `mycroft.ready`), when it
   names one. It then calls the plugin's `train()`, when the plugin has one.
   When the plugin has `wait_until_trained()`, the runner waits on it before
   the first match. Padatious trains on a background thread and answers no
   match until that pass ends, and a deployed assistant takes its first
   utterance long after boot training ends. The wait uses the training
   budget of the cell, 3600 s by default (`DEFAULT_TRAIN_TIMEOUT_SECS`).
   When training does not end in that time, the cell fails and writes no
   rows for that fighter and language.
5. **Match.** For each test utterance, the runner calls `match_<tier>` of
   each stage in pipeline order. The first stage that returns a match wins,
   and the row records it in `stage`. When no stage matches, the
   prediction is `null`. This is how `ovos-core` dispatches its pipeline.

The arena owns no confidence threshold. Each stage gates itself through its
own `match_high`, `match_medium` or `match_low`. When the fighter config has
an `intent_transformers` section, the runner builds the transformer chain
on its own bus and adds the slots it extracts.

An offline fighter that pins `model_revision` gets that model commit from
the Hub through `snapshot_download`. The runner gives the plugin the local
path, so the run uses the pinned commit.

`ovos-m2v-pipeline` loads its model on a background thread, and it skips an
utterance that arrives before the load ends. Every m2v fighter therefore
sets `preload_model: true`, which loads the model when the plugin is built.

The runner reads the label and text columns of a test set through its
`reference_fields` and renames them onto the canonical row shape. The
`ovos-intents-v5` test split stores its label under `label`, for example.

A sample whose `reference_intent` is `null` is out of scope (OOD). The
correct answer to it is no intent.

Each training corpus has one datashape. A template corpus
(`registry/datasets/intent_template/`) holds phrase templates with
`{slot}` placeholders. A keyword corpus (`registry/datasets/intent_keyword/`)
holds vocabulary rules. A fighter runs on an eval set only when the eval set
provides every datashape its stages need. For example, a keyword engine does
not run on a corpus that has only templates.

## Metrics

`arena.metrics.score_intent` computes these metrics for each
`(league, dataset, lang)`. A row is correct when its prediction equals
`reference_intent`, or, for an OOD row, when the prediction is `null`. The
`exact_match` field of the row holds this result. The runner computes it with
the dataset's `reference_granularity`: `domain` compares only the text before
the first `:` on both sides.

| Metric | Definition | Better |
|---|---|---|
| **`generalization_accuracy`** (primary) | Correct rows divided by all rows that are not in an in-distribution bucket. The rows are pooled, so a bucket with more rows has more weight. | higher |
| `generalization_n` | The number of rows behind `generalization_accuracy`. | n/a |
| `accuracy` | Correct rows divided by all rows, OOD rows and in-distribution buckets included. | higher |
| `macro_f1` | The mean F1 over every intent that appears as a reference or as a prediction. An answer on an OOD row counts as a false positive for the intent that fired. | higher |
| `ood_fpr` | OOD rows that got any intent, divided by all OOD rows. | lower |
| `slot_exact_match` | Of the rows with a correct intent and gold slots, the share where every gold slot value equals the predicted value, after trim and lowercase. | higher |
| `latency_ms_median` | The median match time of one utterance. | lower |
| `ece` | Expected calibration error over 10 equal bins, on rows that carry a confidence. Absent when no row carries one. | lower |
| `acc_<bucket>` | Accuracy inside one test bucket. The bucket `near_ood` is published as `acc_in_distribution`. | higher |

A bucket is in-distribution when the dataset declares it so in
`bucket_roles`. A dataset that declares no roles falls back to the bucket
names `template`, `in_distribution` and `near_ood`. Every other bucket enters
`generalization_accuracy`. A corpus with one plain `test` bucket therefore has
`generalization_accuracy` equal to `accuracy`.

On `intents-for-eval`, the `template` and `in_distribution` buckets are
in-distribution. `paraphrase`, `far_ood`, `asr_noise` and `typos` enter the
ranked metric. The train and test splits are separate at the template level,
so no test row is an expansion of a training template.
[Methodology](../methodology.md#traintest-separation-in-the-intent-corpora)
gives the check behind that claim.

The confidence interval on the board is a seeded bootstrap of the primary
metric: 1000 rounds, seed 0, over the 0/1 correctness of the rows that
`generalization_accuracy` counts. It is the 2.5th to 97.5th percentile.

## Boards, battles and ELO

Each league has its own benchmark board per dataset and language:
`benchmark-<league>-<dataset>-<lang>.json`. All four leagues share one blind
battle pool and one ELO ladder per language: `battles-intent-<dataset>-<lang>`,
`elo-seed-intent-<lang>` and `leaderboard-intent-<lang>`. A voter judges two
answers to the same utterance and cannot see how each engine was prepared.

The ELO seed uses only the rows that `generalization_accuracy` counts. For
each such row and each pair of fighters where exactly one is correct, the
correct one wins an auto-battle at a quarter of the K-factor. A pair whose
confidence intervals on `generalization_accuracy` overlap on a dataset gets
no auto-battles there. The weight of the auto-battles of one pair is capped
at 5 human votes (`MAX_AUTO_WEIGHT_PER_PAIR`).
[Rating](../rating.md#auto-battles-and-the-seed) gives the seed rules.

## Datasets

[Datasets](../datasets.md) lists every intent set with its source, pin,
licence and languages. The families are:

| Family | What it holds |
|---|---|
| `intents-for-eval` | The arena's own test set: 50 intents with slots, 6 buckets. It has template and keyword training sets. |
| `ovos-intents-v5` | The test split of `OpenVoiceOS/ovos-intents`: utterances of the OVOS skill fleet, labelled with skill ids. The test split and its template training split pin the same commit, the `v6.1` tag. |
| `massive-templates` | Amazon MASSIVE in the template format. |
| `banking77`, `clinc150`, `snips`, `mtop-<lang>` | Public intent benchmarks, each with a template training set. `clinc150` has OOD rows. |

## Fairness rules

**Offline fighters run only on their label set.** An `intent_offline`
fighter lists in `label_set` the corpora whose labels its artefact can emit.
`eligible_competitors` skips it on every other corpus, where every answer
would be wrong for a reason that says nothing about the engine. Before it
scores, the runner reads the class list of the loaded model and intersects it
with the labels of the corpus. When the overlap is zero, the runner writes no
rows. Each row records the measured overlap in `extras.label_overlap`. An
offline fighter also pins its model commit (`model_revision`), and each row
carries that commit.

**No fighter is scored on its own training data.** A fighter that lists a
dataset in `trained_on` does not run on it. The assembler also drops
published rows for that pair.

**Rows from another corpus revision are dropped.** When the registry pins a
dataset to a 40-character commit, the board drops rows made against any other
revision and counts them in `rows_other_revision`.

**Each prediction repo holds one datashape.** Rows go to
`OpenVoiceOS/ovos-intent-template-bench-<dataset>` for template-only
fighters, `OpenVoiceOS/ovos-intent-keyword-bench-<dataset>` for keyword-only
fighters, and `OpenVoiceOS/ovos-intent-bench-<dataset>` for fighters that mix
both. The assembler reads the league of each row from the registry, so a
fighter that changes league keeps its rows.

## Samples

The intent sets have no `sample_policy`, so a sweep reads every test row.
`--max-samples` on a benchmark script makes a smaller run for a smoke
test.

## Known limits

- An intent set has no sample-set manifest. Each fighter is scored on the
  rows it has, and the board marks the entries `sample_set: "unmanaged"`. A
  fighter that ran with a smaller `--max-samples` is then scored on fewer
  rows than the others.
- The fleet runner (`runner.autorun`) does not run intent fighters. Intent
  rows come from the benchmark scripts (see [Benchmarks](../benchmarks.md)).
- On a corpus with one plain `test` bucket, `generalization_accuracy` is
  plain accuracy. Its guarantee is the train and test split of that corpus.
- `ovos-intents-v5` labels follow the skill intent names of its source tag.
  An offline head trained on older labels scores a renamed label as a miss.

---
[← Leagues](../leagues.md) · [Home](../index.md) · [Wake word →](wake_word.md)
