# Registry reference

The registry is the list of fighters and datasets that the arena knows. It is
a folder of JSON files. A pull request that adds or changes a file is how a
plugin joins the arena or a dataset joins a league.

This page lists every field of the two kinds of file. The script
`scripts/gen_registry_doc.py` writes it from the pydantic models in
`registry/schemas.py`. Do not edit it by hand: change the schema or the
`DESCRIPTIONS` table in the script, then run the script again.

## Where the files go

| Kind | Path | Model |
|---|---|---|
| Fighter (competitor) | `registry/competitors/<league>/<competitor_id>.json` | `CompetitorDef` |
| Dataset | `registry/datasets/<league>/<dataset_id>.json` | `DatasetDef` |

The fighter folders are `stt`, `tts`, `vad`, `wake_word`, `intent_zero_shot`,
`intent_online`, `intent_offline` and `intent_keyword`. There is no
`ww_stream` folder for fighters: the false wake-up league runs the `wake_word`
fighters that list `stream` in `capabilities`.

The dataset folders are one per league, plus `intent` for the intent test
sets that the four intent leagues share, and `intent_template` and
`intent_keyword` for the intent training sets.

## How the registry is checked

Both models forbid unknown keys, so a typo in a field name fails. The command
below checks every file and prints one line per error:

```bash
python -m arena.cli validate-registry --registry registry
```

The `unit_tests` workflow runs the same command on each pull request to
`dev`. The check fails when any file is wrong. The schema checks each file alone.
`validate-registry` adds these checks across files:

- Every dataset has a `display_name`.
- Every `role: eval` dataset has a `summary`.
- Each `label_set` entry is a registered dataset.
- Each `negatives_dataset_ids` entry is a registered `wake_word` dataset. A
  language that differs from the set gives a warning, not an error.
- Each `trained_on` entry is a registered dataset of a league that scores the
  fighter. For a `wake_word` fighter that is `wake_word` or `ww_stream`.
- A `samples_from` block in a hotword config is `{dataset_id, count}`. The
  dataset is registered and is in `trained_on`. `count` is an integer of 1 or
  more.

The schema itself also refuses these entries:

- A language tag that is not a full BCP-47 tag, such as `en`. The bare
  subtags `arb` and `kab` are the only exceptions. `multi` is valid on a
  dataset.
- An intent fighter with no `config.intents.pipeline`, or with a stage name
  that does not end in `-high`, `-medium` or `-low`.
- An intent fighter with no `training_regime`, or with a regime that differs
  from the heaviest stage of its pipeline.
- An intent fighter filed in the wrong league. A fighter built only from
  keyword engines belongs in `intent_keyword`. Every other fighter belongs in
  `intent_<training_regime>`.
- An offline intent fighter with no `label_set`, or any other fighter with
  one.
- A `positives_filter` on a dataset that is not an audiofolder `wake_word`
  corpus.
- A `primary_metric` other than `fa_per_hour`, or on a dataset that is not
  `ww_stream`.
- A `test_set_kind` on a dataset that is not `ww_stream`.
- An `input: audio` dataset outside the intent leagues, or without
  `stt_plugin`, `stt_config` and `reference_fields.audio`.
- A `role: train` dataset with a `paradigm` whose `modality` is not
  `intent_<paradigm>`.

## How to read the tables

Each table has these columns:

- **Field**: the JSON key.
- **Type**: the type that the schema accepts. `X | None` means the key can be
  left out or set to `null`.
- **Default**: the value when the key is left out. "required" means the file
  must set it.
- **What it does**: what the arena does with the value.

## Competitor fields (`CompetitorDef`)

| Field | Type | Default | What it does |
|---|---|---|---|
| `competitor_id` | `str` | `required` | The stable id of the fighter. Boards, battles, votes and prediction files use it as the key. The file name is `<competitor_id>.json`. |
| `schema_version` | `int` | `1` | The version of the entry shape. Only 1 is defined. |
| `modality` | `"stt" \| "tts" \| "wake_word" \| "vad" \| "intent" \| "intent_template" \| "intent_zero_shot" \| "intent_online" \| "intent_offline" \| "intent_keyword" \| "ww_stream"` | `required` | The league of the fighter. It must match the folder under `registry/competitors/`. Wake word fighters use `wake_word` and also run on the `ww_stream` league. |
| `plugin` | `str \| None` | `null` | The OPM entry point name of the plugin. A row stores it as `plugin_id`. An intent fighter can leave it out. The schema then takes it from a pipeline of one engine, and keeps it empty for a fusion. |
| `config` | `dict[str, any]` | `{}` | A `mycroft.conf` fragment that the runner gives to the plugin. STT reads `stt`, TTS reads `tts`, VAD reads `VAD` or `listener.VAD`, wake word reads `hotwords`, `VAD` and `hotword_verifier`. An intent fighter needs `intents.pipeline`, a list of `<plugin>-<tier>` stages. |
| `langs` | `list[str]` | `[]` | The BCP-47 tags the fighter supports. A run pairs the fighter with each dataset language that shares the primary subtag. An empty list means every language. |
| `training_regime` | `"zero_shot" \| "online" \| "offline" \| None` | `null` | Intent fighters only, and required there: `zero_shot`, `online` or `offline`. It must equal the regime of the heaviest stage of the pipeline. |
| `label_set` | `list[str] \| None` | `null` | Offline intent fighters only, and required there: the `dataset_id` values whose labels the pretrained model can emit. The fighter runs only on those datasets. Any other regime must leave it out. |
| `alias` | `list[str] \| None` | `null` | Old `plugin_id` values that map to this fighter when a legacy row has no `competitor_id`. The schema always adds `plugin` to the list. |
| `trained_on` | `list[str]` | `[]` | The `dataset_id` values whose recordings were in the training data of the fighter. The fighter never gets a score on these datasets. |
| `capabilities` | `list["clip" \| "stream"]` | `["clip"]` | Wake word fighters only. `clip` means the fighter runs on clip sets. `stream` means it also runs on the `ww_stream` recordings. |
| `display_name` | `str \| None` | `null` | The name the site shows for the fighter. |
| `species` | `str \| None` | `null` | The plugin class the fighter is an instance of, for example `PadatiousPipeline`. |
| `family` | `str \| None` | `null` | The grouping key for the fighter cards. Do not write it by hand. In an intent league the schema sets it from `species` through `FAMILY_ALIASES`. In every other league it equals `competitor_id`. |
| `types` | `list[str]` | `[]` | Architecture tags, for example `neural-net`, `fuzzy-match` or `embedding`. |
| `description` | `str \| None` | `null` | A short text about how the fighter works. |
| `model` | `str \| None` | `null` | The name of the model the fighter runs, when it has one. |
| `model_revision` | `str \| None` | `null` | The commit sha of the model repo. The intent runner downloads that commit and gives the plugin the local path. The schema also writes it as `revision` into each intent stage block that has a `model`. Each row records the commit the download resolved to. |
| `model_hf_repo` | `str \| None` | `null` | The Hugging Face model repo of the weights. `arena.model_size` reads its download size for the `model_mb` column. Without it, the column stays empty. |
| `size` | `"micro" \| "tiny" \| "small" \| "base" \| "medium" \| "large" \| "x-large" \| "giant" \| "titan" \| None` | `null` | The installed footprint class: micro under 5 MB, tiny 5 to 50 MB, small 50 to 200 MB, base 200 to 500 MB, medium 500 MB to 2 GB, large 2 to 8 GB, x-large 8 to 20 GB, giant 20 to 80 GB, titan over 80 GB. |
| `links` | `dict[str, str]` | `{}` | Named URLs, for example `source`, `pypi` or `paper`. |
| `notes` | `str \| None` | `null` | Free text for maintainers. |

## Dataset fields (`DatasetDef`)

| Field | Type | Default | What it does |
|---|---|---|---|
| `dataset_id` | `str` | `required` | The stable id of the dataset. The file name is `<dataset_id>.json`. Prediction rows, boards and repo names use it. |
| `display_name` | `str \| None` | `null` | The name the site shows for the dataset. `validate-registry` requires it on every dataset. |
| `summary` | `str \| None` | `null` | Two to four plain sentences about the data and the task. `validate-registry` requires it on every `role: eval` dataset. |
| `schema_version` | `int` | `1` | The version of the entry shape. Only 1 is defined. |
| `modality` | `"stt" \| "tts" \| "wake_word" \| "vad" \| "intent" \| "intent_template" \| "intent_zero_shot" \| "intent_online" \| "intent_offline" \| "intent_keyword" \| "ww_stream"` | `required` | The folder of the entry under `registry/datasets/`. Intent test sets use `intent`. Intent training sets use `intent_template` or `intent_keyword`. |
| `source` | `HuggingFaceSource \| PathSource` | `required` | Where the data lives. The `type` field selects the shape: `huggingface` or `path`. The tables below list the fields of each. |
| `reference_fields` | `dict[str, str]` | `{}` | A map from a role to a column name of the source, for example `{"audio": "audio", "ground_truth": "text"}` for STT. |
| `lang` | `str` | `required` | A full BCP-47 tag such as `en-US`, or `multi` for a dataset with several languages. A bare subtag such as `en` fails, except `arb` and `kab`. |
| `langs` | `list[str] \| None` | `null` | The languages of a `lang: multi` dataset. Each tag follows the same rule as `lang`. |
| `license` | `str \| None` | `null` | The licence of the source data. |
| `role` | `"eval" \| "train" \| "unrestricted"` | `"eval"` | `eval` is a test set that a board scores. `train` is an intent training set. `unrestricted` is open data that other sets pool, such as the spoken-word negatives of the wake word league. |
| `paradigm` | `"template" \| "keyword" \| None` | `null` | For a `role: train` intent set: `template` or `keyword`. The `modality` must then be `intent_<paradigm>`. |
| `train_datasets` | `dict[str, str] \| None` | `null` | For an intent test set: a map from paradigm to the `dataset_id` of its training set, for example `{"template": "intents-for-eval-templates"}`. |
| `wakeword` | `str \| None` | `null` | Wake word clip sets: the phrase of the positives, and the folder that holds them. The assembler drops the rows of a fighter whose hotword is another phrase, and the fleet runner does not schedule that pair. |
| `positives_dir` | `str \| None` | `null` | Wake word: the folder of positives when it does not have the name of the phrase, for example `positive`. |
| `negative_dirs` | `list[str] \| None` | `null` | Wake word: the folders of the same corpus that give negatives. The default is every folder except the positives. |
| `exclude_dirs` | `list[str] \| None` | `null` | Wake word: folders that this set leaves out, for example a folder that another set already holds. |
| `positives_filter` | `dict[str, str \| list[str]] \| None` | `null` | Wake word: keep only the positives whose row in `metadata.jsonl` has each listed value. A list allows any of its values. Only an audiofolder `wake_word` corpus accepts it. |
| `primary_metric` | `str \| None` | `null` | `ww_stream` only: the ranked metric when it is not the league default. The only value the schema accepts is `fa_per_hour`. |
| `negatives_hf` | `str \| None` | `null` | Wake word: one other Hugging Face dataset that gives the negatives. `negatives_sources` replaces it when both are set. |
| `negatives_dir` | `str \| None` | `null` | Wake word: the folder inside `negatives_hf` that holds the clips. |
| `negatives_sources` | `list[str] \| None` | `null` | Wake word: several corpora pooled as negatives, each `hf_id` or `hf_id/subdir`. Pin one with `@<sha>` after the repo name. An entry without a pin reads `main`. |
| `negatives_dataset_ids` | `list[str] \| None` | `null` | Wake word: registry datasets pooled as more negatives. Each must be a `wake_word` dataset, and each uses its own `sample_policy`. |
| `predictions_hf` | `str \| None` | `null` | The Hugging Face dataset repo of the prediction rows, `OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`. `assemble` reads it when no `--predictions` list is given. |
| `sample_rate_hz` | `int \| None` | `null` | `ww_stream`: the sample rate of the recordings. The stream publish script writes it. No runner or scorer code reads it. |
| `event_manifest_fields` | `dict[str, str] \| None` | `null` | `ww_stream`: the column names of the event manifest. The stream publish script writes it. The runner reads `reference_fields` instead. |
| `min_negative_hours` | `float \| None` | `null` | `ww_stream`: the hours of audio with no wake word in the corpus. The stream publish script writes it. No runner or scorer code reads it. |
| `event_tolerance_s` | `float \| None` | `null` | `ww_stream`: the match window of an event, in seconds. The scorer uses `arena.metrics.EVENT_TOLERANCE_S` and does not read this field. |
| `test_set_kind` | `"real" \| "synthetic" \| None` | `null` | `ww_stream` only: `real` or `synthetic` positives. The site shows it with the facts of the board. |
| `notes` | `str \| None` | `null` | Free text for maintainers. |
| `sample_policy` | `SamplePolicy \| None` | `null` | A fixed cap on the rows a run reads. Leave it out to read every row. The `SamplePolicy` table gives its fields. |
| `predictions_revision` | `str \| None` | `null` | A commit sha of `predictions_hf`. When it is set, `assemble` reads that commit and not the `--revision` it got. |
| `bucket_roles` | `dict[str, "in_distribution" \| "generalization"] \| None` | `null` | Intent test sets: a map from a bucket name to `in_distribution` or `generalization`. In-distribution rows stay out of `generalization_accuracy`. |
| `oos_label` | `str \| None` | `null` | Plain Hugging Face classification sets: the label of an out-of-scope row. A training set drops those rows. A test set keeps them with no intent, in the bucket `far_ood`. |
| `reference_granularity` | `"intent" \| "domain"` | `"intent"` | `intent` compares the whole intent name. `domain` compares only the text before the first `:` on both sides. |
| `domain_label` | `str \| None` | `null` | Plain Hugging Face classification sets with no label column: one label for every row. Use it with `reference_granularity: domain`. |
| `input` | `"text" \| "audio"` | `"text"` | `text` or `audio`. An `audio` intent test set holds clips. The runner transcribes each clip once and gives the same text to every fighter. |
| `stt_plugin` | `str \| None` | `null` | Required when `input` is `audio`: the STT plugin that makes the transcript. |
| `stt_config` | `dict[str, any] \| None` | `null` | Required when `input` is `audio`: the config of that STT plugin. Each row records it. |

## Dataset `source` with `type: huggingface` (`HuggingFaceSource`)

| Field | Type | Default | What it does |
|---|---|---|---|
| `type` | `"huggingface"` | `"huggingface"` | Always `huggingface` for this shape. |
| `hf_id` | `str` | `required` | The Hugging Face dataset id, for example `PolyAI/minds14`. |
| `revision` | `str` | `"main"` | A branch, a tag or a commit sha. Only a 40-character sha is a pin: the board then drops rows made on any other revision. |
| `split` | `str` | `"train"` | The split to read. |
| `subset` | `str \| None` | `null` | The config name of the dataset, when it has several. |
| `file_pattern` | `str \| None` | `null` | A file path in the repo with a `{lang}` placeholder, for data stored as plain files, for example `{lang}/test.jsonl`. |
| `id_field` | `str \| None` | `null` | Plain classification sets: the column of a stable row id. The loader keeps the first row of each id. |
| `lang_field` | `str \| None` | `null` | Plain classification sets: the column of a language tag, for a split that mixes languages. |
| `lang_value` | `str \| None` | `null` | The value of `lang_field` to keep, for example `de_XX`. The loader drops every other row. |

## Dataset `source` with `type: path` (`PathSource`)

| Field | Type | Default | What it does |
|---|---|---|---|
| `type` | `"path"` | `"path"` | Always `path` for this shape. No runner reads a `path` source, and no registry dataset uses one. |
| `path` | `str` | `required` | A path on the local file system. |
| `format` | `str` | `"jsonl"` | `jsonl`, `csv` or `parquet`. |

## Dataset `sample_policy` (`SamplePolicy`)

| Field | Type | Default | What it does |
|---|---|---|---|
| `max_samples` | `int \| None` | `null` | The cap on rows per language. Empty means every row. On a wake word clip set, the cap applies to each class and to each pooled negative channel. |
| `seed` | `int` | `1337` | The seed of the shuffle that picks the rows. A wake word clip set ignores it and picks clips at even steps through the sorted list. |

## Example: a competitor

An STT fighter. The `stt` block is what the runner gives to the plugin. [`registry/competitors/stt/fasterwhisper-base-en.json`](../registry/competitors/stt/fasterwhisper-base-en.json):

```json
{
  "competitor_id": "fasterwhisper-base-en",
  "modality": "stt",
  "plugin": "ovos-stt-plugin-fasterwhisper",
  "config": {
    "lang": "en",
    "stt": {
      "module": "ovos-stt-plugin-fasterwhisper",
      "ovos-stt-plugin-fasterwhisper": {
        "model": "base",
        "compute_type": "int8",
        "cpu_threads": 1,
        "beam_size": 3,
        "use_cuda": false,
        "vad_filter": false
      }
    },
    "model": "base",
    "compute_type": "int8",
    "cpu_threads": 1,
    "beam_size": 3
  },
  "langs": [
    "en-US",
    "en-GB",
    "en-AU",
    "de-DE",
    "fr-FR",
    "it-IT",
    "pt-PT"
  ],
  "notes": "faster-whisper base model, int8 quantised; flat config keys mirror the stt fragment for the prediction runner",
  "display_name": "FasterWhisper (base)",
  "species": "FasterWhisperSTT",
  "types": [
    "neural-net",
    "transformer"
  ],
  "description": "OpenAI Whisper base model served through faster-whisper (CTranslate2, int8). Encoder-decoder speech transformer; the small-footprint profile of the Whisper family.",
  "model": "whisper-base (int8)",
  "links": {
    "source": "https://github.com/OpenVoiceOS/ovos-stt-plugin-fasterwhisper"
  },
  "size": "small"
}
```

## Example: a dataset

An STT test set that reads a branch, so it is not pinned to a commit. [`registry/datasets/stt/ami-en-GB.json`](../registry/datasets/stt/ami-en-GB.json):

```json
{
  "dataset_id": "ami-en-GB",
  "display_name": "AMI meeting corpus (British English)",
  "summary": "Recordings of real business meetings, captured on each speaker's own headset microphone. One row is one segment of meeting speech with its transcript, and a fighter has to transcribe it. The speech is spontaneous, overlapping and full of interruptions and false starts, none of which a read-speech corpus contains. The whole registered split is scored, uncapped.",
  "modality": "stt",
  "source": {
    "type": "huggingface",
    "hf_id": "edinburghcstr/ami",
    "revision": "main",
    "split": "test",
    "subset": "ihm"
  },
  "reference_fields": {
    "audio": "audio",
    "ground_truth": "text"
  },
  "lang": "en-GB",
  "license": "cc-by-4.0",
  "role": "eval",
  "predictions_hf": "OpenVoiceOS/ovos-stt-bench-ami-en-GB",
  "notes": "AMI Meeting Corpus, IHM (individual headset mic) test split, en-GB business-meeting speech. The far-field SDM config exists in the same repo (edinburghcstr/ami, subset 'sdm') and is a natural follow-up once single-mic results establish a baseline."
}
```

---
[← Architecture](architecture.md) · [Home](index.md) · [Predictions →](predictions.md)
