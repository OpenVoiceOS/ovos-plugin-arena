# Benchmarks

A benchmark run loads an eval dataset and the registry fighters, runs the
real OVOS plugins offline, and writes one prediction row per sample (§3.2).
The arena has no CI workflow that runs predictions. Runs happen on a laptop
or on your own box, and a person uploads the rows. The `assemble` workflow
then turns the published rows into benchmark boards, battle pools and ELO
seeds.

There are two ways to run a benchmark:

- The **benchmark notebooks** in the `benchmarks/` folder of
  [OpenVoiceOS/ngi0-notebooks](https://github.com/OpenVoiceOS/ngi0-notebooks)
  run a whole league over many test sets and print the boards.
- The **benchmark scripts** in `benchmarks/` run one dataset, or one family
  of datasets, from the command line.

Both call the same arena code: `registry/` for fighters and datasets,
`runner/` to run a fighter, and `arena/metrics.py` to score it.

## Where rows are published

Each dataset has its own prediction repo on Hugging Face:
`OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`, with `_` in the modality
written as `-` (`registry.loaders.results_repo_for`). For example, the rows of
`community-computer` go to `OpenVoiceOS/ovos-wake-word-bench-community-computer`.
By the same rule, the repo of `ww-negatives-en-US` is
`OpenVoiceOS/ovos-ww-stream-bench-ww-negatives-en-US`. The `predictions_hf`
field of each dataset entry names its repo, and `assemble` reads every such
repo when it gets no `--predictions` list.

An intent dataset has one repo per training datashape:
`ovos-intent-template-bench-<dataset>` for template-only fighters,
`ovos-intent-keyword-bench-<dataset>` for keyword-only fighters, and
`ovos-intent-bench-<dataset>` for fighters that mix both.

Inside a repo, the rows of one fighter and language are one file:
`predictions/<lang>/<competitor_id>.jsonl`. A policy-capped dataset also has
`sample_sets/<ll_RR>.json`, the manifest of the sample ids that a run selects.

## The benchmark notebooks

The notebooks are in the `benchmarks/` folder of the ngi0-notebooks
repository, not in this one.
[`ww_benchmark.ipynb`](https://github.com/OpenVoiceOS/ngi0-notebooks/blob/dev/benchmarks/ww_benchmark.ipynb)
runs the wake word fighters on the clip test sets and, with
`RUN_STREAM = True`, on the false wake-up sets. The intent leagues have their own
notebook,
[`intent_benchmark.ipynb`](https://github.com/OpenVoiceOS/ngi0-notebooks/blob/dev/benchmarks/intent_benchmark.ipynb).
Each notebook pins one commit of this repository. The rest of this section
describes the wake word notebook.

The notebook keeps its settings in one Settings cell. An environment
variable of the same name, or a line `NAME=value` in a `.env` file,
overrides a setting.

**The arena commit is pinned.** The setting `ARENA_REF` holds one 40-character
commit of this repository. The notebook fetches exactly that commit, puts it
on `sys.path`, and imports `registry`, `runner` and `arena` from it. The
fighters, the test sets, the code that runs a fighter and the scoring code
are thus those of `ARENA_REF`. To repeat a run, keep `ARENA_REF`. To run with
newer fighters or datasets, set it to a newer commit.

**The steps.** The notebook installs the plugins from PyPI, runs the
fighters with the arena's runner, and scores the rows with
`arena.metrics.build_benchmark_board`. The board is the same board the arena
builds. The notebook also counts the errors itself, and the column
`own count agrees` says whether its count gives the same error rate. The
columns `published …` show what the arena published for the same fighter. A
run that stops can start again: the rows already written stay, and only the
missing clips run.

The main settings are:

| Setting | What it does |
|---|---|
| `DATASETS` | The arena test sets to run. |
| `FIGHTERS` | The fighters to run. |
| `PLUGINS` | The plugins to run. |
| `SMOKE` | A small run with no upload. |
| `UPLOAD`, `HF_TOKEN` | Send the rows to the prediction repos. The token needs write access to the OpenVoiceOS organisation. |
| `ARENA_REF` | The arena commit to use. |
| `OUTPUT_DIR` | The folder for the rows and the tables. |
| `MAX_PER_CLASS` | Clips per class per test set. |
| `RUN_STREAM`, `STREAM_RECORDINGS` | Also stream recordings, and how many per test set. |
| `SAME_PHRASE_ONLY` | Run a detector only on the test sets of its own phrase. |

**How `UPLOAD` merges.** A local file replaces the published file of the
same path, so the notebook merges first. For each local file
`predictions/<lang>/<competitor_id>.jsonl`, it downloads the published file
of the same fighter and language, and appends every published row whose
`sample_id` is not in the local file. No published row is lost. Then the
notebook uploads. It uploads a clip test set only when every fighter on it
has 30 or more clips (`MIN_BOARD_SAMPLES`). The rows of the audio with no
wake word that it streams stay on your machine.

## The benchmark scripts

Each script runs one registry dataset, or one family. Every script
resumes: a new run skips the `sample_id`s that the output file already holds,
and loads the plugin only when a language still has work. Rows go to
`<output-dir>/<dataset_id>/...`, by default under `predictions/`.

### Intent scripts

| Script | Dataset |
|---|---|
| `intent_banking77.py` | `banking77` |
| `intent_clinc150.py` | `clinc150` |
| `intent_intents_for_eval.py` | `intents-for-eval` |
| `intent_massive_templates.py` | `massive-templates` |
| `intent_ovos_intents_v5.py` | `ovos-intents-v5` |
| `intent_snips.py` | `snips` |

The dataset of an intent script is fixed. These scripts have no `--dataset`
flag. `python benchmarks/intent_snips.py --help` prints:

```
usage: intent_snips.py [-h] [--competitors COMPETITORS] [--langs LANGS]
                       [--max-samples MAX_SAMPLES] [--output-dir OUTPUT_DIR]
                       [--upload] [--hf-owner HF_OWNER]
                       [--train-timeout-secs TRAIN_TIMEOUT_SECS]
                       [--train-max-rss-mb TRAIN_MAX_RSS_MB]
                       [--transcript-cache-dir TRANSCRIPT_CACHE_DIR]
```

| Flag | Meaning |
|---|---|
| `--competitors a,b` | Run only these fighters. Default: every eligible fighter. |
| `--langs a,b` | Run only these languages. Default: every language of the dataset. |
| `--max-samples N` | Cap the test rows per language, for a smoke run. |
| `--output-dir DIR` | The local root for the prediction files. |
| `--upload` | Upload the rows to the prediction repos of the dataset. |
| `--hf-owner OWNER` | The Hugging Face owner of the prediction repos. Default: `OpenVoiceOS`. |
| `--train-timeout-secs S` | Stop a fighter whose training and prediction take longer than S seconds. It is also the time the runner waits for an engine to report itself trained. Default: 3600. 0 turns the guard off. |
| `--train-max-rss-mb MB` | Stop a fighter whose memory goes above MB. Default: 8192. 0 turns the guard off. |
| `--transcript-cache-dir DIR` | For audio-input sets: where the cached STT transcripts are. |

### Audio scripts

| Script | Default dataset | Other datasets with `--dataset` |
|---|---|---|
| `ww_hey_mycroft.py` | `synthetic-wakewords-hey_mycroft` | Any `wake_word` dataset id. |
| `ww_hey_jarvis.py` | `synthetic-wakewords-hey_jarvis` | Any `wake_word` dataset id. |
| `ww_computer.py` | `community-computer` | Any `wake_word` dataset id. |
| `ww_stream.py` | `ww-negatives-en-US` | Any `ww-negatives-<lang>` dataset id. |
| `stt_minds14.py`, `stt_fleurs.py`, `stt_mls.py`, `stt_voxpopuli.py`, `stt_ami.py`, `stt_edacc.py`, `stt_speech_massive.py`, `stt_catalan_domain.py` | One set of the family, for example `minds14-pt-PT` for `stt_minds14.py` | Any `stt` dataset id. |
| `tts_intents_prompts.py` | `intents-for-eval-prompts` | Any `tts` dataset id. |
| `vad_speech.py` | `speech-vs-nonspeech-en-US` | Any `vad` dataset id. |

The `--dataset` value is a registry id of the script's modality. The script
of a wake word phrase is a default, not a limit. `python
benchmarks/ww_hey_mycroft.py --help` prints:

```
usage: ww_hey_mycroft.py [-h] [--dataset DATASET] [--competitors COMPETITORS]
                         [--langs LANGS] [--max-samples MAX_SAMPLES]
                         [--output-dir OUTPUT_DIR] [--upload]
                         [--hf-owner HF_OWNER]
```

| Flag | Meaning |
|---|---|
| `--dataset ID` | The registry dataset to run. |
| `--competitors a,b` | Run only these fighters. Default: every fighter of the modality. |
| `--langs a,b` | Run only these languages. Default: the languages of the dataset. |
| `--max-samples N` | Cap the samples per language, for a smoke run. On a clip set, the cap is per class. On a false wake-up set, it counts recordings. |
| `--output-dir DIR` | The local root for the prediction files and audio. |
| `--upload` | Upload the rows to the prediction repo of the dataset. |
| `--hf-owner OWNER` | The Hugging Face owner of the prediction repo. Default: `OpenVoiceOS`. |

A wake word script runs every fighter of the language unless
`--competitors` names some, so name the fighters of the set's phrase. With
`ovos-ww-plugin-vosk` installed, this smoke run writes 15 rows to
`predictions/synthetic-wakewords-hey_mycroft/wake_word/predictions/en-US/vosk-ww-hey-mycroft.jsonl`:
5 clips with the phrase, 5 from the shared negative pool and 5 from
`mlsw-negatives-en-US`.

```bash
python benchmarks/ww_hey_mycroft.py --competitors vosk-ww-hey-mycroft --max-samples 5
```

When a fighter's plugin is not installed, the script logs an error for that
fighter, goes on with the next one, and exits 0.

This run streams the first recording of the English false wake-up set, 30
minutes of read speech, through the same fighter. It writes one row with
`prediction: "WW_STREAM"`, the `category` of the recording and the events
of the fighter:

```bash
python benchmarks/ww_stream.py --dataset ww-negatives-en-US \
    --competitors vosk-ww-hey-mycroft --max-samples 1
```

### The sample cap

A dataset with a `sample_policy` sets the cap of a run. `--max-samples` can
make a run smaller than the policy cap, for a smoke test. A larger value is
cut to the policy cap. Without a policy, `--max-samples` keeps the first rows
in corpus order. [Datasets](datasets.md) shows the policy of every set.

### Upload from a script

`--upload` sends the local folder of the dataset as it is. A local file
replaces the published file of the same fighter and language. A script does
not merge the published rows first. To add rows to a published fighter,
download its published file into the output folder before the run, or use the
wake word notebook, which merges. The fleet runner (`runner.autorun`) also starts from
the published file. See [Runner](runner.md).

## Adding a benchmark

1. Add the dataset JSON under `registry/datasets/<modality>/`.
2. Add the fighter JSONs under `registry/competitors/<modality>/` (see
   [Add a fighter](add-a-fighter.md)).
3. For a new audio dataset, reuse the existing adapter. For a new corpus
   shape, point the dataset `source` at it: a parquet split, a per-language
   `file_pattern`, or a `manifest.jsonl` sidecar for wake word.
4. Add a thin `benchmarks/<modality>_<corpus>.py` that calls
   `run_benchmark(<Adapter>(), "<dataset_id>", __doc__...)`.
5. Run `python scripts/gen_datasets_doc.py` to update
   [Datasets](datasets.md).
6. Run with `--max-samples` first, then with `--upload`.

---
[← Reproduce a row](reproduce-a-row.md) · [Home](index.md) · [Local testing →](local-testing.md)
