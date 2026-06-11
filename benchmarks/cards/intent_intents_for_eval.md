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
each competitor (one plugin + one configuration, declared in the arena
registry) is trained on the dataset's own training files and evaluated
end-to-end through the plugin's `match_<tier>` confidence gate.

## Layout

```
predictions/<competitor_id>.jsonl    one row per (language, test utterance)
```

Row fields: `competitor_id`, `sample_id`, `dataset_id`, `dataset_revision`
(pinned source revision), `lang`, `plugin_id`, `plugin_version`, `utterance`,
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
