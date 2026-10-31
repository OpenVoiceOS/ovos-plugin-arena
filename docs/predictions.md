# Prediction datasets

Every number on a benchmark board comes from prediction rows. A prediction row
is one answer of one fighter to one sample. The arena keeps the rows in public
Hugging Face dataset repos, never in this repository. Anybody can download
them, check a board, or score the rows another way.

This page gives the repo names, the file layout inside a repo, the fields of
a row, how revisions pin a board to exact data, and what the assembler does
with the rows. The last section shows how to read rows with `huggingface_hub`,
`datasets` and the arena's own loader.

## One repo per league and dataset

`registry.loaders.results_repo_for` names the repo of a dataset:

```text
OpenVoiceOS/ovos-<league>-bench-<dataset_id>
```

An underscore in the league name becomes a hyphen. Each registry dataset
also holds its repo name in `predictions_hf`, and `assemble` reads that field.

| League | Repo name |
|---|---|
| `stt` | `OpenVoiceOS/ovos-stt-bench-<dataset_id>` |
| `tts` | `OpenVoiceOS/ovos-tts-bench-<dataset_id>` |
| `vad` | `OpenVoiceOS/ovos-vad-bench-<dataset_id>` |
| `wake_word` | `OpenVoiceOS/ovos-wake-word-bench-<dataset_id>` |
| `ww_stream` | `OpenVoiceOS/ovos-ww-stream-bench-<dataset_id>` |
| intent, template fighters | `OpenVoiceOS/ovos-intent-template-bench-<dataset_id>` |
| intent, keyword fighters | `OpenVoiceOS/ovos-intent-keyword-bench-<dataset_id>` |
| intent, fighters that mix both | `OpenVoiceOS/ovos-intent-bench-<dataset_id>` |

The intent repos follow the training data a fighter needs, not its league
(`runner.intent_bench.predictions_store`). A fighter built only from template
engines writes to the template repo, whatever its league. The assembler
reads the league of each intent row from the registry. A fighter that moves
to another league keeps its published rows.

The `dataset_id` in the repo name is always the test set, never its training
set. Two test sets that share one training set each have their own repos.

## Files inside a repo

| Path | What it holds | Written by |
|---|---|---|
| `README.md` | The dataset card. It declares one split per language and names the funder. | the benchmark upload |
| `predictions/<lang>/<competitor_id>.jsonl` | Every row of one fighter in one language. | the benchmark upload, the fleet runner, the queue daemon |
| `audio/<lang>/<competitor_id>/<text>.wav` | TTS only: the clip a fighter synthesised for a prompt. | the TTS benchmark upload |
| `sample_sets/<lang>.json` | The sample ids a capped dataset scores. The hyphen in `<lang>` becomes an underscore, for example `sample_sets/en_US.json`. | `python -m runner.publish_sample_set` |

The card declares a split per language. The split name is the language tag
with `_` in place of `-`, because a split name allows only word characters.
Each split reads `predictions/<lang>/*.jsonl`.

`runner.dataset_cards.check_funding_block` refuses to upload a card that does
not name the NGI0 Commons Fund, NLnet and the grant agreement number.

The assembler also accepts the flat legacy path
`predictions/<competitor_id>.jsonl`. It does not read any other file. A file
at the repo root is not a prediction file for the assembler.

### How a file grows

A benchmark run writes rows to a local file first:
`<output-dir>/<dataset_id>/<league>/predictions/<lang>/<competitor_id>.jsonl`.
The default `--output-dir` is `predictions`. A second run skips every sample
id that the file already holds, so a run can stop and start again.

An upload sends the whole local file to the same path in the repo, and the
new file replaces the old one. The local file is the source of truth, so the
remote file is always a superset of the previous upload.

When the registry pins a dataset to a commit, the runner first removes rows
of any other revision from the local file
(`runner.intent_bench.prune_other_revisions`). A row counts as done only when
it carries the pinned revision. A changed pin thus makes the runner produce
those rows again.

## The row

`arena.models.PredictionRow` defines a row. A JSONL line is one row. The
loader (`arena.predictions.parse_row`) moves every key that the model does
not define into `extras`.

### Fields of every row

| Field | Type | What it holds |
|---|---|---|
| `competitor_id` | string | The fighter, as in `registry/competitors/`. |
| `sample_id` | string | The sample. Two fighters with the same `sample_id` answered the same input. |
| `dataset_id` | string | The registry dataset. |
| `lang` | string | The language of the run. |
| `plugin_id` | string | The `plugin` of the fighter. An intent fusion writes `ensemble`. |
| `modality` | string | The league the row is scored in. A streaming row of a `wake_word` fighter writes `ww_stream`. |
| `schema_version` | integer | 2 for a row in this shape. A legacy STT row gets 1 when the loader converts it. |
| `plugin_version` | string | `<distribution>==<version>` of the plugin that ran. An intent row joins one entry per engine with `;`, plus `<model>@<commit>` for a pretrained model. |
| `runner_version` | string | `ovos-plugin-arena==<version>`. |
| `created_at` | string | UTC time of the answer, ISO 8601. |
| `dataset_revision` | string | The commit sha of the source dataset that the run read. |
| `prediction` | string or null | The answer. Its meaning depends on the league, see below. |
| `latency_ms` | number | The time of the call that produced the answer. |
| `elapsed_ms` | number | Wall time of the inference call. |
| `peak_rss_mb` | number | A sample of the process memory around the call. It is not a true peak and not limited to the call. |
| `audio_secs` | number | The length of the input clip, or of the synthesised clip for TTS. |
| `hw` | object | The machine: `host_class`, `cpu_model`, `threads`, `accelerator`, `hostname`. |
| `extras` | object | League fields that the model does not define. |

Old rows can lack the newer fields. Every field except `competitor_id`,
`sample_id`, `dataset_id`, `lang` and `plugin_id` has a default.

### Fields per league

| League | Fields | Meaning |
|---|---|---|
| intent | `utterance`, `reference_intent`, `prediction`, `exact_match`, `confidence`, `bucket`, `reference_slots`, `predicted_slots`, `model_revision` | `prediction` is the intent that fired, or null. `exact_match` is the runner's verdict: for an out-of-scope row (`reference_intent` null) it is true when nothing fired. `bucket` is the test bucket. `model_revision` is the model commit of a pretrained fighter. |
| intent, extras | `pipeline`, `stage`, `label_overlap` | The pipeline stages, the stage that answered, and for an offline fighter the number of corpus labels its model can emit. |
| intent, spoken test sets | `stt_plugin`, `stt_config`, `stt_revision` | The STT that made the transcript every fighter read. The loader moves them into `extras`. |
| `stt` | `reference_text`, `prediction`, `confidence`, `audio_url` | `prediction` is the transcript. The scorer computes WER from `reference_text` and `prediction`. It reads a stored `wer` only when one of them is missing. |
| `tts` | `input_text`, `prediction`, `audio_url` | `prediction` and `audio_url` are the URL of the synthesised clip. Both are null when synthesis failed. |
| `tts`, extras | `utmos`, `utmos_judge`, `utmos_judge_revision`, `sigmos.<dim>`, `dnsmos.<dim>`, `nisqa.<dim>` and their judge fields, `intelligibility_wer`, `intelligibility_cer`, `intelligibility_judge`, `intelligibility_judge_revision`, `intelligibility_judges`, `intelligibility_consensus`, `intelligibility_agreement`, `intelligibility_rover`, `synthesis_error` | The naturalness score, the other quality scores, and the ASR round trip. A language with no ASR judge has `intelligibility: "not_available"` and null error rates. |
| `wake_word` | `label`, `prediction`, `audio_url` | `label` is `positive` or `negative`. `prediction` is `detected` or `not_detected`. `audio_url` points at the clip in the source corpus. |
| `ww_stream` | `label`, `prediction`, `audio_url` | `label` is `negative`, because a false wake-up recording holds no wake word. `prediction` is always `WW_STREAM`. |
| `ww_stream`, extras | `events`, `truth_onsets`, `truth_ends`, `duration_s`, `category` | `events` lists every activation as `[time_s, score]`. The score is 1.0 when the engine reports no number. `truth_onsets` and `truth_ends` are empty. |
| `vad` | `label`, `prediction`, `audio_url` | `label` is `speech` or `non_speech`. `prediction` is `speech` or `silence`. |

### Sample ids

A sample id must be the same for every fighter. The runners derive it from the
source, never from the fighter:

- Intent and TTS: `<lang>/<index>`, with the index as five digits, for example
  `en-US/00042`. The index is the position of the row in the source.
- Wake word clip sets: the path of the clip in the source repo. A clip from a
  pooled negative set has the prefix `<dataset_id>/`.
- Parquet audio corpora: the id column when the dataset names one, else the
  file name of the audio, else `sample_<index>` with six digits.

### Legacy STT rows

Some `ovos-stt-bench-*` repos hold rows in an older column layout with
`dataset_entry_id` and `plugin_name`. The loader converts each one
(`arena.legacy_schema.STTRow`). It maps `plugin_name` to a fighter through the
registry `alias` list. When no fighter matches, the file name gives the
`competitor_id`.

## Revisions

A board names the exact data behind it in three places.

**The dataset revision.** Each row carries `dataset_revision`, the commit of
the source corpus that the run read. When the registry `source.revision` is a
40-character commit sha, the board scores only rows with that revision
(`arena.metrics.drop_rows_off_pinned_revision`). Each entry counts the rows it
lost in `rows_other_revision`. An entry left with no rows is unranked with the
reason `no rows on the pinned dataset revision`. The board field
`dataset_revision` holds the pin, or null when the registry names a branch.
On a branch, the board scores the rows of every revision.

**The prediction repo revision.** `assemble` resolves every prediction repo
to a commit sha before it downloads anything
(`arena.predictions.resolve_predictions_revision`). It uses the registry
`predictions_revision` when a dataset sets one, else the `--revision`
argument, which defaults to `main`. The board field `predictions_revisions`
maps each repo to the sha it was built from. Download that commit to get the
exact rows of the board.

**The plugin and model versions.** Each entry lists the `plugin_version`
values of its rows in `plugin_versions`. `version_blended: true` warns that
the rows mix versions. A pretrained intent fighter also pins its model commit
in the registry `model_revision`, and each row records the commit the
download resolved to.

## What the assembler does with the rows

`python -m arena.cli assemble` turns the rows into boards. With no
`--predictions` list, it reads the repos of every `role: eval` dataset
(`registry.loaders.list_prediction_repos`). `--modality` limits it to one
league. Any intent league reads every intent repo, because the four intent
leagues share one battle group.

**Fetch.** It downloads only `predictions/**/*.jsonl` of each repo at the
resolved commit. A repo that does not exist yet holds no data, and the run
goes on. A gated repo, a private repo or a pinned revision that no longer
exists stops that source with an error. A transient failure, such as a rate
limit, gets two retries, after 5 and 15 seconds.

**Language.** For a dataset with one language, it loads
`predictions/<lang>/` and the flat legacy files. Rows that another run
published under another language folder do not enter the board.

**Partial input.** It works one language at a time. When any source of a
language fails to load, it writes no artifact for that language and leaves
the committed files as they are. The run then fails.

**Filter.** `arena.predictions.group_rows` groups the rows by league, dataset,
language, sample and fighter. It drops these rows before any board sees them:

| Rows dropped | Why |
|---|---|
| A row whose league it cannot tell | No board can score it. |
| A row of a `competitor_id` that is not in the registry | A fighter removed from the registry must leave the boards, even when its old files stay on the Hub. |
| A row of an offline intent fighter on a dataset outside its `label_set` | Its model cannot emit the labels of that corpus. |
| A row of a fighter on a dataset in its `trained_on` | The fighter heard those recordings in training. |
| A wake word or stream row of a fighter whose hotword is another phrase | A detector that fires on another phrase measures nothing about either phrase. |

When two rows have the same sample and fighter, the last one read wins.
`assemble-summary.json` lists the dropped counts per fighter in
`unregistered_competitors_excluded`, `trained_on_rows_excluded` and
`cross_phrase_rows_excluded`.

**Sample set.** For a dataset with a `sample_policy`, the assembler downloads
`sample_sets/<lang>.json` and scores each fighter only on those ids. The entry
records `sample_set: "manifest"` and its `sample_set_coverage`. An entry under
90% coverage is unranked. With no manifest, the entry records
`sample_set: "unmanaged"`.

**Revision.** Last, the board drops rows off the pinned dataset revision, as
the [Revisions](#revisions) section tells.

## Read the rows yourself

The commands below use the Python environment of [Local testing](local-testing.md),
with the `hf` extra. Public repos need no token. The examples read the
`synthetic-wakewords-hey_mycroft` wake word repo because its files are small.

### With `huggingface_hub`

Download one fighter's file at a fixed commit:

```python
import json

from huggingface_hub import HfApi, hf_hub_download

repo = "OpenVoiceOS/ovos-wake-word-bench-synthetic-wakewords-hey_mycroft"
api = HfApi()
sha = api.dataset_info(repo).sha  # pin the commit you read
files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha)
         if f.startswith("predictions/")]
print(len(files), files[0])

path = hf_hub_download(repo, "predictions/en-US/microwakeword-hey-mycroft-v2.jsonl",
                       repo_type="dataset", revision=sha)
with open(path, encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh if line.strip()]
print(len(rows), rows[0]["sample_id"], rows[0]["label"], rows[0]["prediction"])
```

To check a board, use the sha from its `predictions_revisions` in place of
`dataset_info(repo).sha`.

### With `datasets`

Load one file, or one language split with every fighter in it:

```python
from datasets import load_dataset

repo = "OpenVoiceOS/ovos-wake-word-bench-synthetic-wakewords-hey_mycroft"
one = load_dataset(repo, split="train",
                   data_files="predictions/en-US/microwakeword-hey-mycroft-v2.jsonl")
print(one.num_rows, one.column_names[:6])

split = load_dataset(repo, split="en_US")  # every fighter of one language
print(split.num_rows, len(set(split["competitor_id"])))
```

A split holds every row in the repo, before any of the filters above. It can
hold fighters that a board does not score, for example wake word fighters of
another phrase.

### With the arena loader

Run this from the repository root. It applies the same parsing and the same
filters as `assemble`:

```python
from arena.predictions import group_rows, load_predictions, resolve_predictions_revision

repo = "OpenVoiceOS/ovos-wake-word-bench-synthetic-wakewords-hey_mycroft"
sha = resolve_predictions_revision(repo)
rows = load_predictions(repo, revision=sha, lang="en-US")

dropped = {}
grouped = group_rows(rows, unregistered=dropped, cross_phrase_dropped=dropped)
for (league, dataset_id, lang), samples in grouped.items():
    fighters = {c for by_fighter in samples.values() for c in by_fighter}
    print(league, dataset_id, lang, len(samples), "samples", len(fighters), "fighters")
print("rows left out:", sum(dropped.values()))
```

`grouped` is the input of the board builder. The sample-set and revision
filters run later, in `arena.metrics.build_benchmark_board`.

## Known limits

- The TTS clip URL points at the `main` branch of the prediction repo, not at
  a commit. A later upload of the same prompt by the same fighter replaces the
  clip behind an old row.
- The assembler reads `sample_sets/<lang>.json` from `main`, not from the
  commit in `predictions_revisions`.
- `peak_rss_mb` samples the memory of the whole process. Read it as a rough
  size, not as the memory of one call.

---
[← Registry](registry.md) · [Home](index.md) · [Frontend →](frontend.md)
