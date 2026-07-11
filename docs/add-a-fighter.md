# Add your plugin as a fighter

Want your OVOS plugin ranked on the leaderboards — and an
[embeddable rank badge](leagues.md#rank-badges) for its README? Add a
*competitor* to the registry. A competitor is **a configuration you could
ship**: the same plugin under a different config is a different fighter.

You do not need to run any benchmarks yourself. Once your competitor JSON is
merged, the arena's sweep runs your config against the pinned datasets and it
appears on the boards.

## 1. Write the competitor file

Drop a JSON file at `registry/competitors/<modality>/<competitor-id>.json`.
`<modality>` is one of `stt`, `tts`, `ww`, `vad`, `intent`.

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

### Example: an intent-pipeline fighter

Intent fighters describe an ordered `intents.pipeline` of `<plugin>-<tier>`
stages. A single-stage pipeline benchmarks one engine; a multi-stage pipeline
is an ensemble fighter. `plugin` is derived automatically for single-engine
pipelines, so you can omit it.

```json
{
  "competitor_id": "example-padatious-medium",
  "modality": "intent",
  "config": {
    "intents": {
      "pipeline": ["ovos-padatious-pipeline-plugin-medium"],
      "ovos-padatious-pipeline-plugin-medium": {
        "fuzz": true
      }
    }
  },
  "langs": ["en-US"],
  "display_name": "Padatious (medium)",
  "species": "PadatiousPipeline",
  "types": ["neural-net", "template-match"],
  "size": "small",
  "description": "Federated per-intent neural matcher trained on the intent templates."
}
```

## 2. Field reference

| Field | Required | Notes |
|-------|----------|-------|
| `competitor_id` | ✅ | Stable unique id; becomes the filename and badge path. |
| `modality` | ✅ | `stt` / `tts` / `ww` / `vad` / `intent`. |
| `config` | ✅ | A valid `mycroft.conf` fragment. Intent fighters need `config.intents.pipeline`. |
| `plugin` | for non-intent | OPM entry-point name. Derived from the pipeline for single-engine intent fighters. |
| `langs` | recommended | BCP-47 tags the fighter supports. |
| `alias` | optional | Legacy `plugin_id` values from pre-registry prediction rows, re-keyed on ingest. |
| `display_name`, `species`, `types`, `description`, `model`, `size`, `links`, `notes` | optional | Bestiary card shown in the fighter browser. |

`size` is the installed footprint class: `micro` <5MB · `tiny` 5–50MB ·
`small` 50–200MB · `base` 200–500MB · `medium` 500MB–2GB · `large` 2–8GB ·
`x-large` 8–20GB · `giant` 20–80GB · `titan` >80GB.

## 3. Validate before you open the PR

```bash
uv run pytest tests/test_registry.py
```

This validates every competitor file against the schema. If it passes, open a
PR — the sweep and the boards do the rest.
