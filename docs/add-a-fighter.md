# Add a fighter

A fighter (competitor) is one plugin configuration that a user could paste
into `mycroft.conf` and run. The same plugin with another model, another
threshold or another language is another fighter, in its own file. To put
your OVOS plugin on the boards, add one JSON file per configuration to the
registry and open a pull request. Once a fighter has published prediction
rows, it appears on the boards and earns an
[embeddable rank badge](leagues.md#rank-badges).

## 1. Write the competitor file

Put the file at `registry/competitors/<modality>/<competitor-id>.json`.
`<modality>` is one of `stt`, `tts`, `wake_word`, `vad`, or an intent league:
`intent_zero_shot`, `intent_online`, `intent_offline`, `intent_keyword`.
The `ww_stream` league has no folder: its fighters are the `wake_word`
fighters that declare the `stream` capability.

`config` must be a complete `mycroft.conf` fragment: the module name, the
plugin's own block, and every value a user needs. Nothing comes from a
default outside the file.

### Example: a single-engine STT fighter

```json
{
  "competitor_id": "my-stt-en",
  "modality": "stt",
  "plugin": "ovos-stt-plugin-example",
  "config": {
    "lang": "en-US",
    "stt": {
      "module": "ovos-stt-plugin-example",
      "ovos-stt-plugin-example": {
        "lang": "en-US"
      }
    }
  },
  "langs": ["en-US"],
  "display_name": "Example STT (en-US)",
  "species": "ExampleSTT",
  "types": ["neural-net"],
  "size": "small",
  "description": "Streams recorded audio through the Example ASR model and returns the transcript.",
  "links": { "source": "https://github.com/OpenVoiceOS/ovos-stt-plugin-example" }
}
```

### Example: a real STT fighter

`registry/competitors/stt/vosk-small-de.json`:

```json
{
  "competitor_id": "vosk-small-de",
  "modality": "stt",
  "plugin": "ovos-stt-plugin-vosk",
  "config": {
    "lang": "de",
    "stt": {
      "module": "ovos-stt-plugin-vosk",
      "ovos-stt-plugin-vosk": {
        "model": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
        "verbose": false
      }
    }
  },
  "langs": [
    "de-DE"
  ],
  "display_name": "Vosk (small, de)",
  "species": "VoskKaldiSTT",
  "types": ["neural-net"],
  "description": "Offline Kaldi recognizer through the Vosk API. This fighter runs the compact German model vosk-model-small-de-0.15; transcription happens after recording finishes.",
  "model": "vosk-model-small-de-0.15",
  "links": {
    "source": "https://github.com/OpenVoiceOS/ovos-stt-plugin-vosk"
  },
  "notes": "Model download URL from the README advanced configuration; archive auto-downloaded from alphacephei."
}
```

Read it against the [field reference](#2-field-reference):

- **`config`** is what a user puts in `mycroft.conf` to run this model:
  `stt.module` and the plugin's own block. Nothing is left to a default
  outside the file.
- **`langs`** is `["de-DE"]`, a full BCP-47 tag. The inner `config.lang: "de"`
  is the value the plugin expects. Some plugins take the bare language and
  some take the full tag, so copy what the plugin's own schema wants. The
  top-level `langs` is always full BCP-47, because the arena uses it to pair
  the fighter with datasets. `de` alone does not tell `de-DE` from `de-AT`.
- **`model`** and **`notes`** name the exact checkpoint. "Vosk" alone does
  not tell a reader which published model the fighter runs.
- The larger German Vosk model would be a second file, such as
  `vosk-big-de.json`. Same plugin, another configuration, another fighter.

### Example: an intent-pipeline fighter

Intent fighters describe an ordered `intents.pipeline` of `<plugin>-<tier>`
stages. A single-stage pipeline benchmarks one engine; a multi-stage pipeline
is an ensemble fighter. `plugin` is derived automatically for single-engine
pipelines, so you can omit it.

An intent fighter also declares its `training_regime`, and the league in
`modality` follows from it: `zero_shot` for a fighter that answers straight
from the templates a skill registers, `online` for one that trains on them at
boot, `offline` for one shipping a pretrained artefact. A fusion takes the
regime of its heaviest stage, and a fighter built only from keyword engines
goes in `intent_keyword` whatever its regime. The registry rejects a fighter
filed anywhere else, so you never have to guess.

Each stage name is an OPM `opm.pipeline` entry point plus a confidence
tier, not a package name: `ovos-adapt-parser` provides the
`ovos-adapt-pipeline-plugin` entry point, `padacioso` provides
`ovos-padacioso-pipeline-plugin`, and `ovos-core` itself only ships the
`ovos-converse-pipeline-plugin`, `ovos-fallback-pipeline-plugin` and
`ovos-stop-pipeline-plugin` entry points, none of them an intent matcher.

A multi-stage pipeline, such as Padatious then Adapt, is still one fighter
file. The whole cascade is the configuration a user ships.

```json
{
  "competitor_id": "example-padatious-medium",
  "modality": "intent_online",
  "config": {
    "intents": {
      "pipeline": ["ovos-padatious-pipeline-plugin-medium"],
      "ovos-padatious-pipeline-plugin-medium": {
        "fuzz": true
      }
    }
  },
  "langs": ["en-US"],
  "training_regime": "online",
  "display_name": "Padatious (medium)",
  "species": "PadatiousPipeline",
  "types": ["neural-net", "template-match"],
  "size": "small",
  "description": "Federated per-intent neural matcher trained on the intent templates."
}
```

### Example: a wake word fighter

`registry/competitors/wake_word/openwakeword-hey-mycroft.json`:

```json
{
  "competitor_id": "openwakeword-hey-mycroft",
  "modality": "wake_word",
  "plugin": "ovos-ww-plugin-openwakeword",
  "config": {
    "listener": {
      "wake_word": "hey_mycroft"
    },
    "hotwords": {
      "hey_mycroft": {
        "module": "ovos-ww-plugin-openwakeword",
        "listen": true,
        "threshold": 0.5,
        "custom_verifier_models": {},
        "custom_verifier_threshold": 0.1
      }
    }
  },
  "langs": [
    "en-US",
    "en-GB",
    "en-AU"
  ],
  "display_name": "openWakeWord (hey mycroft)",
  "species": "OwwHotwordPlugin",
  "types": [
    "neural-net"
  ],
  "description": "openWakeWord models running on a shared mel-spectrogram and embedding frontend. The pretrained model matching the hotword name is loaded automatically, here the bundled hey_mycroft model.",
  "model": "hey_mycroft (openWakeWord pretrained)",
  "links": {
    "source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-openWakeWord"
  },
  "notes": "Pretrained model auto-selected by hotword name; threshold is the plugin default.",
  "capabilities": [
    "clip",
    "stream"
  ]
}
```

The key under `hotwords` is the phrase id, and `listener.wake_word` names it.
`capabilities: ["clip", "stream"]` lets the fighter run on the clip boards and
on the [false wake-up boards](leagues/ww_stream.md). Every OVOS wake word plugin
runs on the live microphone stream, so a wake word fighter declares both. A
fighter of a phrase competes on the clip sets of its language. It is not
scored on a set it lists in `trained_on`. See the
[wake word league](leagues/wake_word.md) for the fairness rules.

### Note: VAD fighters duplicate config keys on purpose

Every VAD competitor places its tunables both flat under `listener.VAD` and
again nested under the module's own key inside `listener.VAD`, e.g.:

```jsonc
"listener": {
  "VAD": {
    "module": "ovos-vad-plugin-silero",
    "threshold": 0.5,
    "ovos-vad-plugin-silero": { "threshold": 0.5 }
  }
}
```

This is not copy-paste residue. The listener's VAD manager reads the flat key
(the value a user pastes straight into `mycroft.conf`), and the plugin itself
reads the nested key. Both copies must stay in sync when you add or edit a
VAD fighter. Do not deduplicate them.

### Note: few-shot wake-word fighters name their recordings

A plugin that learns a phrase from a few recordings, such as
`ovos-ww-plugin-local-wake`, needs a `samples` folder. A registry entry cannot
hold a path on one machine, so it names a registry dataset instead:

```jsonc
"hotwords": {
  "alexa": {
    "module": "ovos-ww-plugin-local-wake",
    "listen": true,
    "samples_from": { "dataset_id": "synthetic-wakewords-alexa", "count": 4 }
  }
}
```

When the fighter loads, the runner writes `count` positive clips of that
dataset, in the dataset's own sampling order, into a cache folder and gives the plugin that folder as
`samples`. The cache is `~/.cache/ovos-plugin-arena/ww-references`, or
`$ARENA_CACHE/ww-references`. The dataset must also be in `trained_on`, so the
fighter is never scored on its own recordings; `validate-registry` refuses the
entry otherwise. `scripts/gen_ww_fewshot_fighters.py` writes the zero-shot and
few-shot wake-word fighters for the featured phrases only, the phrases the site
shows (`arena/ww_listing.py`). It removes the generated fighters of a phrase
that is not featured.

### Note: wake-word fighters for every published model

`scripts/gen_ww_model_fighters.py` writes one fighter for every published
model of the model-based wake-word plugins: every WakeHuBERT model in
`OpenVoiceOS/wakehubert-wakewords` (batch and streaming), the five Precise ONNX
models for hey mycroft (default and Mark 1), hey robin, marvin and sheila, every ESPHome microWakeWord model
and every pretrained openWakeWord model, when a wake-word test set of its phrase
exists. The other Precise ONNX models are mostly single-speaker personal models.
It leaves an existing fighter file as it is. Then it gives every wake-word
fighter the `stream` capability, so it runs on the false wake-up boards.
Both generators run from the repository root with `PYTHONPATH=.`.

A wake-word dataset whose folder of clips with the phrase is not named after
the phrase sets `positives_dir` to that folder, as `sam-wake-word` does
(`positive/` and `negative/`).

## 2. Field reference

| Field | Required | Notes |
|-------|----------|-------|
| `competitor_id` | yes | Stable unique id. It becomes the file name and the badge path. |
| `modality` | yes | `stt`, `tts`, `wake_word`, `vad`, or an intent league: `intent_zero_shot`, `intent_online`, `intent_offline`, `intent_keyword`. It must match the folder. |
| `config` | yes | A valid `mycroft.conf` fragment. Intent fighters need `config.intents.pipeline`. |
| `plugin` | for non-intent | The OPM entry point name. Derived from the pipeline for single-engine intent fighters. |
| `langs` | recommended | Full BCP-47 tags the fighter supports. The arena uses them to pair the fighter with datasets. An empty list means every language. |
| `training_regime` | for intent | `zero_shot`, `online` or `offline`. It must match the fighter's stages. |
| `label_set` | for offline intent | The dataset ids whose labels the pretrained artefact can emit. The fighter runs on those corpora only. |
| `model_revision` | for offline intent | The model commit the runner downloads and stamps on every row. |
| `trained_on` | when it applies | The dataset ids whose recordings were in the fighter's training data. The fighter is never scored on them. |
| `capabilities` | wake word | `["clip"]` (the default) or `["clip", "stream"]`. Only `stream` fighters run in `ww_stream`. |
| `alias` | optional | Legacy `plugin_id` values from rows made before the registry, mapped to this fighter on ingest. |
| `display_name`, `species`, `family`, `types`, `description`, `model`, `model_hf_repo`, `size`, `links`, `notes` | optional | The card in the fighter browser. |

`size` is the installed footprint class: `micro` under 5MB, `tiny` 5 to
50MB, `small` 50 to 200MB, `base` 200 to 500MB, `medium` 500MB to 2GB,
`large` 2 to 8GB, `x-large` 8 to 20GB, `giant` 20 to 80GB, `titan` over 80GB.

## 3. Check the file

Run both checks from the repository root before you open the pull request:

```bash
python -m arena.cli validate-registry --registry registry
python -m pytest -q tests/test_registry.py tests/test_add_a_fighter_guide.py
```

`validate-registry` is the check CI runs on every push. It validates every
file in `registry/` against the schema, and it checks the references between
files: a `label_set`, `trained_on`, `samples_from` or `negatives_dataset_ids`
entry must name a registered dataset. A clean registry prints:

```
INFO  Registry OK — every competitor/dataset file validated cleanly.
```

## 4. How a merged fighter gets onto the boards

A merged file makes the fighter eligible. Prediction rows put it on the
boards. They come from one of these paths:

- **Audio leagues** (`stt`, `wake_word`, `tts`, `vad`): the fleet runner
  `python -m runner.autorun` reads the registry on each sweep and runs every
  eligible `(fighter, dataset, lang)` pair. It needs no queue file. A new
  fighter joins the rotation on the next sweep. See [Runner](runner.md).
- **False wake-ups** (`ww_stream`): the fleet runner does not run this
  league. Run `benchmarks/ww_stream.py`, or the wake word benchmark notebook
  with `RUN_STREAM = True`. See [Benchmarks](benchmarks.md).
- **Intent leagues**: the fleet runner does not run them. A benchmark script
  (`benchmarks/intent_*.py`) or the intent benchmark notebook reads every
  eligible fighter from the registry. See [Benchmarks](benchmarks.md).

A run with `--upload` publishes the rows to the prediction repo of the
dataset. Then:

1. `assemble.yml` runs daily (or `python -m arena.cli assemble` locally). It
   adds the fighter to the benchmark board, and puts it in the blind battle
   pools against every other fighter of the same battle group and language.
2. `pages.yml` builds the site from the new data and deploys it. The fighter
   gets a card at `/fighter/<competitor_id>` and an entry on the benchmark
   board. It gets an ELO rating once its auto-battles seed one.
3. Each `tally` run writes its [rank badge](leagues.md#rank-badges).

A fighter needs no human vote to appear on a benchmark board. To get an ELO
rating, it needs at least one auto-battle against another fighter of the same
battle group and language (§4 R5, [SPECIFICATION.md](SPECIFICATION.md)).

---
[← Local testing](local-testing.md) · [Home](index.md) · [Runner →](runner.md)
