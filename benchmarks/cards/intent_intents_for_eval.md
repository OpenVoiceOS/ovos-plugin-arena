---
license: apache-2.0
language:
  - ca
  - da
  - de
  - en
  - es
  - eu
  - fr
  - gl
  - it
  - nl
  - pt
tags:
  - openvoiceos
  - intent-classification
  - benchmark
  - predictions
pretty_name: OVOS Intent Bench — intents-for-eval
---

# OVOS Intent Bench — intents-for-eval

Per-sample predictions of OVOS intent pipeline plugins over the
[`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval)
benchmark (50 intents, 1750 test rows per language, 12 languages, with slot
annotations and template / paraphrase / near-OOD / far-OOD / ASR-noise / typo
buckets).

Produced by the reproducible benchmark script
[`benchmarks/intent_intents_for_eval.py`](https://github.com/OpenVoiceOS/ovos-plugin-arena/blob/dev/benchmarks/intent_intents_for_eval.py)
of the [OVOS Plugin Arena](https://github.com/OpenVoiceOS/ovos-plugin-arena):
each competitor is a shippable `mycroft.conf` fragment (an `intents` section
with a tier-suffixed `pipeline` plus per-plugin configs) declared in the
arena registry. Single-stage pipelines benchmark one engine; multi-stage
pipelines are ensemble fighters. Template engines train from the dataset's
template corpus, keyword engines from its keyword-rule corpus (different
datashapes), and every utterance runs through the cascade — the first stage
whose own `match_<tier>` confidence gate fires wins, as in ovos-core.

## Layout

```
predictions/<competitor_id>.jsonl    one row per (language, test utterance)
```

Row fields: `competitor_id`, `sample_id`, `dataset_id`, `dataset_revision`
(pinned source revision), `lang`, `plugin_id` (`"ensemble"` for multi-engine
pipelines), `plugin_version` (`;`-joined per engine), `pipeline` (ordered
stage list), `stage` (which stage fired; null = no match), `utterance`,
`reference_intent` (null for out-of-scope samples), `reference_slots`,
`prediction` (null = no match fired), `predicted_slots`, `exact_match`,
`confidence`, `bucket`, `latency_ms`, `runner_version`, `created_at`.

For out-of-scope samples the correct behaviour is *no match*:
`exact_match` is true iff the plugin predicted nothing.

## Consumers

The arena's `assemble` workflow turns these rows into benchmark leaderboards,
blind A/B battle pools and a benchmark-seeded ELO ladder — see the
[OVOS Plugin Arena](https://github.com/OpenVoiceOS/ovos-plugin-arena).

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme.
