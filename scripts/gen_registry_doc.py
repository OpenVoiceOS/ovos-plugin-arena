#!/usr/bin/env python3
"""Write ``docs/registry.md``: a field reference for the registry JSON files.

The field names, types, defaults and the required flag come from the pydantic
models in ``registry/schemas.py``. The meaning of each field comes from
``DESCRIPTIONS`` below, one entry per model field. The two examples are read
from real registry files. ``tests/test_registry_doc.py`` fails when the
committed page differs from this script's output, and when a schema field has
no entry in ``DESCRIPTIONS``.

Usage, from the repository root::

    python scripts/gen_registry_doc.py           # rewrite docs/registry.md
    python scripts/gen_registry_doc.py --check   # exit 1 when the page is stale
"""

from __future__ import annotations

import argparse
import enum
import json
import sys
import types
import typing
from pathlib import Path

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from registry.schemas import (  # noqa: E402
    CompetitorDef,
    DatasetDef,
    HuggingFaceSource,
    PathSource,
    SamplePolicy,
)

OUTPUT = ROOT / "docs" / "registry.md"

#: The models the page documents, in page order, with their section titles.
MODELS: tuple[tuple[type[BaseModel], str], ...] = (
    (CompetitorDef, "Competitor fields (`CompetitorDef`)"),
    (DatasetDef, "Dataset fields (`DatasetDef`)"),
    (HuggingFaceSource, "Dataset `source` with `type: huggingface` (`HuggingFaceSource`)"),
    (PathSource, "Dataset `source` with `type: path` (`PathSource`)"),
    (SamplePolicy, "Dataset `sample_policy` (`SamplePolicy`)"),
)

EXAMPLE_COMPETITOR = "registry/competitors/stt/fasterwhisper-base-en.json"
EXAMPLE_DATASET = "registry/datasets/stt/ami-en-GB.json"

#: What each field means, keyed ``<Model>.<field>``. Every field of every
#: model in MODELS must have an entry: the test fails on a missing one.
DESCRIPTIONS: dict[str, str] = {
    # CompetitorDef
    "CompetitorDef.competitor_id": (
        "The stable id of the fighter. Boards, battles, votes and prediction "
        "files use it as the key. The file name is `<competitor_id>.json`."
    ),
    "CompetitorDef.schema_version": "The version of the entry shape. Only 1 is defined.",
    "CompetitorDef.modality": (
        "The league of the fighter. It must match the folder under "
        "`registry/competitors/`. Wake word fighters use `wake_word` and also run "
        "on the `ww_stream` league."
    ),
    "CompetitorDef.plugin": (
        "The OPM entry point name of the plugin. A row stores it as `plugin_id`. "
        "An intent fighter can leave it out. The schema then takes it from a "
        "pipeline of one engine, and keeps it empty for a fusion."
    ),
    "CompetitorDef.config": (
        "A `mycroft.conf` fragment that the runner gives to the plugin. STT reads "
        "`stt`, TTS reads `tts`, VAD reads `VAD` or `listener.VAD`, wake word reads "
        "`hotwords`, `VAD` and `hotword_verifier`. An intent fighter needs "
        "`intents.pipeline`, a list of `<plugin>-<tier>` stages."
    ),
    "CompetitorDef.langs": (
        "The BCP-47 tags the fighter supports. A run pairs the fighter with each "
        "dataset language that shares the primary subtag. An empty list means "
        "every language."
    ),
    "CompetitorDef.training_regime": (
        "Intent fighters only, and required there: `zero_shot`, `online` or "
        "`offline`. It must equal the regime of the heaviest stage of the pipeline."
    ),
    "CompetitorDef.label_set": (
        "Offline intent fighters only, and required there: the `dataset_id` values "
        "whose labels the pretrained model can emit. The fighter runs only on "
        "those datasets. Any other regime must leave it out."
    ),
    "CompetitorDef.alias": (
        "Old `plugin_id` values that map to this fighter when a legacy row has no "
        "`competitor_id`. The schema always adds `plugin` to the list."
    ),
    "CompetitorDef.trained_on": (
        "The `dataset_id` values whose recordings were in the training data of the "
        "fighter. The fighter never gets a score on these datasets."
    ),
    "CompetitorDef.capabilities": (
        "Wake word fighters only. `clip` means the fighter runs on clip sets. "
        "`stream` means it also runs on the `ww_stream` recordings."
    ),
    "CompetitorDef.display_name": "The name the site shows for the fighter.",
    "CompetitorDef.species": (
        "The plugin class the fighter is an instance of, for example "
        "`PadatiousPipeline`."
    ),
    "CompetitorDef.family": (
        "The grouping key for the fighter cards. Do not write it by hand. In an "
        "intent league the schema sets it from `species` through `FAMILY_ALIASES`. "
        "In every other league it equals `competitor_id`."
    ),
    "CompetitorDef.types": (
        "Architecture tags, for example `neural-net`, `fuzzy-match` or `embedding`."
    ),
    "CompetitorDef.description": "A short text about how the fighter works.",
    "CompetitorDef.model": "The name of the model the fighter runs, when it has one.",
    "CompetitorDef.model_revision": (
        "The commit sha of the model repo. The intent runner downloads that commit "
        "and gives the plugin the local path. The schema also writes it as "
        "`revision` into each intent stage block that has a `model`. Each row "
        "records the commit the download resolved to."
    ),
    "CompetitorDef.model_hf_repo": (
        "The Hugging Face model repo of the weights. `arena.model_size` reads its "
        "download size for the `model_mb` column. Without it, the column stays empty."
    ),
    "CompetitorDef.size": (
        "The installed footprint class: micro under 5 MB, tiny 5 to 50 MB, small 50 "
        "to 200 MB, base 200 to 500 MB, medium 500 MB to 2 GB, large 2 to 8 GB, "
        "x-large 8 to 20 GB, giant 20 to 80 GB, titan over 80 GB."
    ),
    "CompetitorDef.links": "Named URLs, for example `source`, `pypi` or `paper`.",
    "CompetitorDef.notes": "Free text for maintainers.",
    # DatasetDef
    "DatasetDef.dataset_id": (
        "The stable id of the dataset. The file name is `<dataset_id>.json`. "
        "Prediction rows, boards and repo names use it."
    ),
    "DatasetDef.display_name": (
        "The name the site shows for the dataset. `validate-registry` requires it "
        "on every dataset."
    ),
    "DatasetDef.summary": (
        "Two to four plain sentences about the data and the task. "
        "`validate-registry` requires it on every `role: eval` dataset."
    ),
    "DatasetDef.schema_version": "The version of the entry shape. Only 1 is defined.",
    "DatasetDef.modality": (
        "The folder of the entry under `registry/datasets/`. Intent test sets use "
        "`intent`. Intent training sets use `intent_template` or `intent_keyword`."
    ),
    "DatasetDef.source": (
        "Where the data lives. The `type` field selects the shape: `huggingface` "
        "or `path`. The tables below list the fields of each."
    ),
    "DatasetDef.reference_fields": (
        "A map from a role to a column name of the source, for example "
        "`{\"audio\": \"audio\", \"ground_truth\": \"text\"}` for STT."
    ),
    "DatasetDef.lang": (
        "A full BCP-47 tag such as `en-US`, or `multi` for a dataset with several "
        "languages. A bare subtag such as `en` fails, except `arb` and `kab`."
    ),
    "DatasetDef.langs": (
        "The languages of a `lang: multi` dataset. Each tag follows the same rule "
        "as `lang`."
    ),
    "DatasetDef.license": "The licence of the source data.",
    "DatasetDef.role": (
        "`eval` is a test set that a board scores. `train` is an intent training "
        "set. `unrestricted` is open data that other sets pool, such as the "
        "spoken-word negatives of the wake word league."
    ),
    "DatasetDef.paradigm": (
        "For a `role: train` intent set: `template` or `keyword`. The `modality` "
        "must then be `intent_<paradigm>`."
    ),
    "DatasetDef.train_datasets": (
        "For an intent test set: a map from paradigm to the `dataset_id` of its "
        "training set, for example `{\"template\": \"intents-for-eval-templates\"}`."
    ),
    "DatasetDef.wakeword": (
        "Wake word clip sets: the phrase of the positives, and the folder that "
        "holds them. The assembler drops the rows of a fighter whose hotword is "
        "another phrase, and the fleet runner does not schedule that pair."
    ),
    "DatasetDef.positives_dir": (
        "Wake word: the folder of positives when it does not have the name of the "
        "phrase, for example `positive`."
    ),
    "DatasetDef.negative_dirs": (
        "Wake word: the folders of the same corpus that give negatives. The default "
        "is every folder except the positives."
    ),
    "DatasetDef.exclude_dirs": (
        "Wake word: folders that this set leaves out, for example a folder that "
        "another set already holds."
    ),
    "DatasetDef.positives_filter": (
        "Wake word: keep only the positives whose row in `metadata.jsonl` has "
        "each listed value. A list allows any of its values. Only an audiofolder "
        "`wake_word` corpus accepts it."
    ),
    "DatasetDef.primary_metric": (
        "`ww_stream` only: the ranked metric when it is not the league default. "
        "The only value the schema accepts is `fa_per_hour`."
    ),
    "DatasetDef.negatives_hf": (
        "Wake word: one other Hugging Face dataset that gives the negatives. "
        "`negatives_sources` replaces it when both are set."
    ),
    "DatasetDef.negatives_dir": "Wake word: the folder inside `negatives_hf` that holds the clips.",
    "DatasetDef.negatives_sources": (
        "Wake word: several corpora pooled as negatives, each `hf_id` or "
        "`hf_id/subdir`. Pin one with `@<sha>` after the repo name. An entry "
        "without a pin reads `main`."
    ),
    "DatasetDef.negatives_dataset_ids": (
        "Wake word: registry datasets pooled as more negatives. Each must be a "
        "`wake_word` dataset, and each uses its own `sample_policy`."
    ),
    "DatasetDef.predictions_hf": (
        "The Hugging Face dataset repo of the prediction rows, "
        "`OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`. `assemble` reads it "
        "when no `--predictions` list is given."
    ),
    "DatasetDef.sample_rate_hz": (
        "`ww_stream`: the sample rate of the recordings. The stream publish script "
        "writes it. No runner or scorer code reads it."
    ),
    "DatasetDef.event_manifest_fields": (
        "`ww_stream`: the column names of the event manifest. The stream publish "
        "script writes it. The runner reads `reference_fields` instead."
    ),
    "DatasetDef.min_negative_hours": (
        "`ww_stream`: the hours of audio with no wake word in the corpus. The "
        "stream publish script writes it. No runner or scorer code reads it."
    ),
    "DatasetDef.event_tolerance_s": (
        "`ww_stream`: the match window of an event, in seconds. The scorer uses "
        "`arena.metrics.EVENT_TOLERANCE_S` and does not read this field."
    ),
    "DatasetDef.test_set_kind": (
        "`ww_stream` only: `real` or `synthetic` positives. The site shows it "
        "with the facts of the board."
    ),
    "DatasetDef.notes": "Free text for maintainers.",
    "DatasetDef.sample_policy": (
        "A fixed cap on the rows a run reads. Leave it out to read every row. "
        "The `SamplePolicy` table gives its fields."
    ),
    "DatasetDef.predictions_revision": (
        "A commit sha of `predictions_hf`. When it is set, `assemble` reads that "
        "commit and not the `--revision` it got."
    ),
    "DatasetDef.bucket_roles": (
        "Intent test sets: a map from a bucket name to `in_distribution` or "
        "`generalization`. In-distribution rows stay out of "
        "`generalization_accuracy`."
    ),
    "DatasetDef.oos_label": (
        "Plain Hugging Face classification sets: the label of an out-of-scope row. "
        "A training set drops those rows. A test set keeps them with no intent, "
        "in the bucket `far_ood`."
    ),
    "DatasetDef.reference_granularity": (
        "`intent` compares the whole intent name. `domain` compares only the text "
        "before the first `:` on both sides."
    ),
    "DatasetDef.domain_label": (
        "Plain Hugging Face classification sets with no label column: one label "
        "for every row. Use it with `reference_granularity: domain`."
    ),
    "DatasetDef.input": (
        "`text` or `audio`. An `audio` intent test set holds clips. The runner "
        "transcribes each clip once and gives the same text to every fighter."
    ),
    "DatasetDef.stt_plugin": (
        "Required when `input` is `audio`: the STT plugin that makes the "
        "transcript."
    ),
    "DatasetDef.stt_config": (
        "Required when `input` is `audio`: the config of that STT plugin. Each "
        "row records it."
    ),
    # HuggingFaceSource
    "HuggingFaceSource.type": "Always `huggingface` for this shape.",
    "HuggingFaceSource.hf_id": "The Hugging Face dataset id, for example `PolyAI/minds14`.",
    "HuggingFaceSource.revision": (
        "A branch, a tag or a commit sha. Only a 40-character sha is a pin: the "
        "board then drops rows made on any other revision."
    ),
    "HuggingFaceSource.split": "The split to read.",
    "HuggingFaceSource.subset": "The config name of the dataset, when it has several.",
    "HuggingFaceSource.file_pattern": (
        "A file path in the repo with a `{lang}` placeholder, for data stored as "
        "plain files, for example `{lang}/test.jsonl`."
    ),
    "HuggingFaceSource.id_field": (
        "Plain classification sets: the column of a stable row id. The loader "
        "keeps the first row of each id."
    ),
    "HuggingFaceSource.lang_field": (
        "Plain classification sets: the column of a language tag, for a split "
        "that mixes languages."
    ),
    "HuggingFaceSource.lang_value": (
        "The value of `lang_field` to keep, for example `de_XX`. The loader drops "
        "every other row."
    ),
    # PathSource
    "PathSource.type": (
        "Always `path` for this shape. No runner reads a `path` source, and no "
        "registry dataset uses one."
    ),
    "PathSource.path": "A path on the local file system.",
    "PathSource.format": "`jsonl`, `csv` or `parquet`.",
    # SamplePolicy
    "SamplePolicy.max_samples": (
        "The cap on rows per language. Empty means every row. On a wake word clip "
        "set, the cap applies to each class and to each pooled negative channel."
    ),
    "SamplePolicy.seed": (
        "The seed of the shuffle that picks the rows. A wake word clip set "
        "ignores it and picks clips at even steps through the sorted list."
    ),
}

HEADER = """\
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
"""

FOOTER = (
    "\n---\n[← Architecture](architecture.md) · [Home](index.md)"
    " · [Predictions →](predictions.md)\n"
)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def type_name(annotation: typing.Any) -> str:
    """A short, stable spelling of a field annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(type_name(a) for a in args)
    if origin is typing.Literal:
        return " | ".join(json.dumps(a) for a in args)
    if origin is list:
        return f"list[{type_name(args[0])}]"
    if origin is dict:
        return f"dict[{type_name(args[0])}, {type_name(args[1])}]"
    if annotation is type(None):
        return "None"
    if annotation is typing.Any:
        return "any"
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return " | ".join(json.dumps(m.value) for m in annotation)
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def default_text(field) -> str:
    if field.is_required():
        return "required"
    if field.default_factory is not None:
        return json.dumps(field.default_factory())
    if field.default is PydanticUndefined:
        return "required"
    return json.dumps(field.default)


def _table(model: type[BaseModel]) -> str:
    lines = ["| Field | Type | Default | What it does |", "|---|---|---|---|"]
    for name, field in model.model_fields.items():
        description = DESCRIPTIONS[f"{model.__name__}.{name}"]
        lines.append(
            "| "
            + " | ".join(
                _cell(c)
                for c in (
                    f"`{name}`",
                    f"`{type_name(field.annotation)}`",
                    f"`{default_text(field)}`",
                    description,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _example(title: str, rel_path: str, intro: str) -> str:
    text = (ROOT / rel_path).read_text(encoding="utf-8").rstrip()
    return f"\n## {title}\n\n{intro} [`{rel_path}`](../{rel_path}):\n\n```json\n{text}\n```\n"


def undocumented_fields() -> list[str]:
    """Schema fields with no entry in DESCRIPTIONS."""
    return [
        f"{model.__name__}.{name}"
        for model, _ in MODELS
        for name in model.model_fields
        if f"{model.__name__}.{name}" not in DESCRIPTIONS
    ]


def stale_descriptions() -> list[str]:
    """DESCRIPTIONS entries that name no schema field."""
    fields = {f"{model.__name__}.{name}" for model, _ in MODELS for name in model.model_fields}
    return sorted(set(DESCRIPTIONS) - fields)


def render() -> str:
    """The full text of ``docs/registry.md``."""
    missing = undocumented_fields()
    if missing:
        raise SystemExit(f"no description for {missing}; add them to DESCRIPTIONS")
    parts = [HEADER]
    for model, title in MODELS:
        parts.append(f"\n## {title}\n\n")
        parts.append(_table(model))
    parts.append(
        _example(
            "Example: a competitor",
            EXAMPLE_COMPETITOR,
            "An STT fighter. The `stt` block is what the runner gives to the plugin.",
        )
    )
    parts.append(
        _example(
            "Example: a dataset",
            EXAMPLE_DATASET,
            "An STT test set that reads a branch, so it is not pinned to a commit.",
        )
    )
    parts.append(FOOTER)
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 when docs/registry.md differs from the schema",
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT.relative_to(ROOT)} is stale: run python scripts/gen_registry_doc.py")
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} matches the schema")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
