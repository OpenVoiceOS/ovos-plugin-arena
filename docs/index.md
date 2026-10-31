# OVOS Plugin Arena documentation

The OVOS Plugin Arena benchmarks [OpenVoiceOS](https://github.com/OpenVoiceOS)
plugins and ranks them. It answers one question for a user: which plugin to
use for a task. The repository is the whole arena. It has no server and no
database.

The arena gives two kinds of result:

- **Benchmark boards.** A benchmark script runs the real OVOS plugin offline
  over a labelled dataset and publishes one prediction row per sample to a
  Hugging Face dataset. The `assemble` workflow scores those rows and writes
  one board per league, dataset and language. Anybody can run the same
  script and check a number.
- **Blind battles and ELO ladders.** The site shows a voter two outputs for
  the same input, without the fighter names. The vote is a GitHub issue. A
  scheduled workflow replays the votes on top of a ladder. Auto-battles from
  the benchmark rows seed that ladder. It ranks on a Bradley-Terry rating
  (see [Rating](rating.md)).

A **fighter** (competitor) is one plugin configuration that a user can ship:
one JSON file in `registry/competitors/`. A **dataset** is one JSON file in
`registry/datasets/`. The live site is
[openvoiceos.github.io/ovos-plugin-arena](https://openvoiceos.github.io/ovos-plugin-arena).

## Leagues

| League | What a fighter does | Board ranks on | Page |
|---|---|---|---|
| `intent_zero_shot`, `intent_online`, `intent_offline`, `intent_keyword` | Names the intent of an utterance, or none. | `generalization_accuracy` (higher) | [Intent](leagues/intent.md) |
| `wake_word` | Decides per clip if the wake phrase is present. | `error_rate` (lower) | [Wake word](leagues/wake_word.md) |
| `ww_stream` | Streams hours of audio with no wake word and does not wake up. | `fa_per_hour` (lower) | [False wake-ups](leagues/ww_stream.md) |
| `stt` | Transcribes a clip. | `wer_mean` (lower) | [STT](leagues/stt.md) |
| `tts` | Speaks a prompt. | `utmos` (higher) | [TTS](leagues/tts.md) |
| `vad` | Decides per clip if it holds speech. | `error_rate` (lower) | [VAD](leagues/vad.md) |

[Leagues](leagues.md) gives the overview, the ELO seed rules and the rank
badges.

## How to read a board

A benchmark board is the file
`frontend-static/public/data/benchmark-<league>-<dataset>-<lang>.json`. This
example is the first entry of `benchmark-wake_word-synthetic-wakewords-hey_mycroft-en-US.json`
at one revision of the data, with the `perf` block left out. The numbers
change when the board is rebuilt. To read the live entry, run
`python3 -c "import json; print(json.load(open('frontend-static/public/data/benchmark-wake_word-synthetic-wakewords-hey_mycroft-en-US.json'))['entries'][0])"`
in a fresh clone:

```json
{"rank": 1, "competitor_id": "wakeforge-zeroshot-hey-mycroft",
 "plugin_id": "ovos-ww-plugin-wakeforge-zeroshot", "samples": 300,
 "metrics": {"error_rate": 0.01, "accuracy": 0.99, "false_accept_rate": 0.0,
             "false_reject_rate": 0.03, "latency_ms_median": 162.31,
             "fa_per_hour_hours": 0.105},
 "primary_metric_ci_lower": 0.0,
 "primary_metric_ci_upper": 0.023333333333333334,
 "tied_with_leader": true,
 "plugin_versions": ["ovos-ww-plugin-wakeforge-zeroshot"],
 "version_blended": false, "unranked": false, "unranked_reason": null,
 "model_mb": null, "sample_set": "manifest", "sample_set_coverage": 1.0,
 "rows_other_revision": 0}
```

Read it in this order:

1. **The metric.** `primary_metric` at the top of the board names the
   ranked metric. The league page gives its definition and its direction.
   `metrics` holds every other number the scorer computed.
2. **The interval.** `primary_metric_ci_lower` and `primary_metric_ci_upper`
   are a seeded 95% bootstrap interval of the primary metric (1000 rounds,
   seed 0). `tied_with_leader: true` means that this interval overlaps the
   interval of rank 1. Treat those entries as tied with the leader. Entries
   with equal values share a rank.
3. **The population.** `samples` is the number of scored rows.
   `sample_set: "manifest"` means that every fighter was scored on the same
   published sample ids, and `sample_set_coverage` is the share of them this
   fighter has. `"unmanaged"` means that no manifest applied and each fighter
   was scored on the rows it has.
4. **The provenance.** `dataset_revision` is the corpus commit the registry
   pins, or `null` when the registry pins a branch.
   `predictions_revisions` gives the commit of each prediction repo the board
   was built from. `plugin_versions` lists the plugin versions in the rows,
   and `version_blended: true` warns that the rows mix versions.
5. **Ranked or not.** An entry with `unranked: true` has rank 0 and takes no
   place on the board. The site hides it and shows how many entries it hid.

`arena/metrics.py` (`build_benchmark_board`) gives an unranked entry one of
these reasons. The text of `unranked_reason` starts with the name:

| Reason | What it means |
|---|---|
| `too_few_samples` | The primary metric rests on fewer than 30 rows (`MIN_BOARD_SAMPLES`). On a false wake-up board: less than 4 h of audio. |
| `one_class` | A wake word or VAD entry scored only positives or only negatives, so its error rate measures one error type. |
| `sample_set_partial` | The entry covers less than 90% of the published sample-set manifest. |
| `no <metric>` | The run scored rows, but none of them is in the population of the primary metric. For example, every intent row is in an in-distribution bucket, or a stream entry has less than 1 h of audio. |
| `run failed` | No row was scored. |
| `no rows on the pinned dataset revision` | Every row came from another revision of the corpus. The fighter needs a new sweep. |

A board ranks on the benchmark metric only. The ELO ladder
(`leaderboard-<group>-<lang>.json`) is a separate ranking: it starts from
auto-battles derived from the benchmark rows and moves with human votes.
[Rating](rating.md#the-ladder) gives its maths, and
[Methodology](methodology.md#reading-a-leaderboard) tells how to read it.

## Reading order

**For a reviewer: how a score is made and why it is fair**

1. [Architecture](architecture.md): the data flow from the registry to the
   site, every workflow and what starts it.
2. [Leagues](leagues.md): the leagues, the battle groups, the ELO seed and
   the rank badges.
3. [Intent](leagues/intent.md), [Wake word](leagues/wake_word.md),
   [False wake-ups](leagues/ww_stream.md), [STT](leagues/stt.md),
   [TTS](leagues/tts.md), [VAD](leagues/vad.md): one page per league. Each
   tells how the runner drives the plugin, the metric formulas, the datasets,
   the fairness rules and the known limits.
4. [Rating](rating.md): the board intervals, the auto-battles, the vote
   rules and the Bradley-Terry ladder, with every parameter and its place in
   the code.
5. [Methodology](methodology.md): why the rating works as it does, the
   train and test separation, the sample policy and the dataset revisions.
6. [Datasets](datasets.md): every registry dataset with its source, pin,
   licence and languages. A script writes it from the registry.

**For a plugin author: get a plugin ranked**

7. [Add a fighter](add-a-fighter.md): write the competitor file, check it,
   and get it onto the boards.
8. [Registry](registry.md): every field of a competitor file and a dataset
   file. A script writes it from `registry/schemas.py`.
9. [Predictions](predictions.md): the prediction repos, the row fields, the
   revision pins, and how to read rows.
10. [Reproduce a row](reproduce-a-row.md): run one benchmark and compare
    your number with the published row.
11. [Benchmarks](benchmarks.md): the benchmark notebooks, the scripts and
    their flags, and how an upload merges with published rows.
12. [Local testing](local-testing.md): install, tests, registry check,
    assembler, replay check and the frontend on your own machine.
13. [FAQ](faq.md): what goes wrong, why, and the fix.

**Operate the arena**

14. [Runner](runner.md): the queue daemon, the fleet runner (`runner.autorun`)
    and the sample-set manifests.
15. [Operations](operations.md): the vote loop, the scheduled workflows,
    audits, replay and alarms.
16. [Frontend](frontend.md): the pages of the site, the data files each one
    reads, and how to build and preview it.

**Reference**

17. [Glossary](glossary.md): the terms of the arena.
18. [Specification](SPECIFICATION.md): the normative spec.
19. [Registry audit](registry-audit.md): conventions across the registry.
20. [Dataset review](dataset-review.md): review notes on the eval datasets.
21. [Dataset gap fill](dataset-gapfill.md): how an STT eval set is
    registered, and which languages have none.
22. [Ensemble rationale](ensemble-rationale.md): why the fusion fighters are
    built as they are.
23. [Site tour](site-tour.md): a screenshot of every page of the site.

The [README](../README.md) gives the status of the published data and the
credits.
