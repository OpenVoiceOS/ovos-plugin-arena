# Adding a fighter: a worked walkthrough

[`add-a-fighter.md`](add-a-fighter.md) is the field reference for a
competitor JSON file. This page walks through one real fighter end to end,
from the file on disk to a benched leaderboard entry, so you can see how the
pieces the reference describes fit together.

## The rule: one fighter, one shippable config

A competitor is not "a plugin". It is **a configuration you could paste into
your own `mycroft.conf` and run**. `config` in the fighter JSON must be a
valid `mycroft.conf` fragment, in full: the module name, the plugin's own
config block, and everything a user would need. If the same plugin can run
two meaningfully different ways (a different model, a different confidence
gate, a different language), that is **two fighter files**. It is not one
file with a variants list. Each one benches, ranks, and appears on the
boards separately.

## Worked example: `vosk-small-de`

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

Reading it against [`add-a-fighter.md`](add-a-fighter.md)'s field table:

- **`config`** is exactly what you'd drop into `mycroft.conf` to make your
  own OVOS instance use this model: `stt.module` plus the plugin's own
  block, both present, nothing implied or left to a default outside the
  file. This is what "shippable configuration" means in practice.
- **`langs`** is `["de-DE"]`, a full BCP-47 tag, not `"de"`. The inner
  `config.lang: "de"` is what the plugin itself expects. Some OVOS plugins
  take the bare language, and some take the full tag; the fighter copies
  whatever the plugin's own config schema wants. `langs` at the top level is
  what the arena uses to route the fighter to eval datasets, and it is always
  full BCP-47, so it composes with every other fighter's `langs` list without
  ambiguity (`de` alone does not tell you `de-DE` from `de-AT`).
- **`model`** and **`notes`** exist because this is a *neural* fighter with a
  specific model checkpoint. That is bestiary metadata (optional per the
  field table), useful because "Vosk" alone does not tell a reader which of
  several published models this fighter actually runs.
- There is exactly **one** `vosk-small-de.json`. The larger German Vosk model
  would be a sibling file (`vosk-big-de.json`, if or when someone adds it),
  not a second entry inside this one. Same plugin, different shippable
  config, different fighter.

An intent fighter follows the same rule with an `intents.pipeline` list
instead of a `stt` block. See the intent example in
[`add-a-fighter.md`](add-a-fighter.md#1-write-the-competitor-file). A
multi-stage `pipeline` (for example Padatious then Adapt) is still **one**
fighter file. The "shippable config" is the whole cascade, because that is
what a user would actually configure.

## Validate before opening a PR

```bash
uv run pytest tests/test_registry.py
# or, the same check CI runs on every push/PR:
python -m arena.cli validate-registry --registry registry
```

Both commands validate every file in `registry/` against the schema.
Malformed JSON, a missing required field, or a `modality` that does not
match the directory all fail loudly here rather than silently in CI. See
[`local-testing.md`](local-testing.md) for the full local environment setup.

## How a merged fighter becomes benched

Adding the JSON file does not by itself produce a leaderboard row. It makes
the fighter *eligible* to be benched. Two paths get predictions published,
depending on modality:

- **Intent leagues** run directly. `benchmarks/intent_*.py` reads every
  registered competitor for its modality straight from the registry each
  time it runs (see [`benchmarks.md`](benchmarks.md)), with no separate
  queueing step. Re-run the benchmark script (optionally with `--upload`)
  and the new fighter is included.
- **Audio modalities** (`stt`, `wake_word`, `tts`) go through the always-on
  prediction runner on the `ser9` box, driven by `runner/queue.yaml`
  ([`runner.md`](runner.md)). `python -m runner.queue_tools --dry-run
  --modality stt` diffs the registry against what is already published on
  HuggingFace and prints the missing `(competitor × dataset)` job entries.
  A newly merged fighter with no predictions yet shows up there. Someone
  pastes the relevant entries into `runner/queue.yaml`. The runner daemon
  picks them up on its next cycle and publishes rows to the modality's HF
  results repo.

Either way, once predictions exist for the fighter:

1. `assemble.yml` (daily, or `python -m arena.cli assemble` locally, see
   [`local-testing.md`](local-testing.md#5-assemble-battles-and-boards-locally))
   picks up the new rows next time it runs. It adds the fighter to the
   benchmark board and folds it into blind A/B battle pools against every
   other fighter already benched in that league and language.
2. `pages.yml` rebuilds the Astro site from the refreshed
   `frontend-static/public/data/*.json` and deploys it. The fighter now has
   a bestiary card (`/fighter/<competitor_id>`) and appears on the relevant
   benchmark board immediately, and on the ELO leaderboard once its
   benchmark-derived auto-battles seed a rating (`operations.md`'s
   [loop, end to end](operations.md#the-loop-end-to-end) has the full
   sequence).
3. From there it earns a
   [rank badge](leagues.md#rank-badges) and collects human votes the same as
   any other fighter.

No fighter needs a human vote to appear on a *benchmark* board. That board
is pure objective metric. It needs at least one benchmark-derived auto-battle
against another fighter in the same league and language to get an ELO rating at
all (§4 R5, [`SPECIFICATION.md`](SPECIFICATION.md)).

---
[← Add a fighter](add-a-fighter.md) · [Home](index.md) · [Benchmarks →](benchmarks.md)
