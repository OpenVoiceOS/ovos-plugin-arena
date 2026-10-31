# Architecture

The arena has no server and no database. Git holds the registry and the
published boards. Hugging Face holds the prediction rows. GitHub issues hold
the votes. GitHub Actions turns them into JSON files, and GitHub Pages
serves a static site that reads those files.

This page follows one score from the registry to the site, and one vote
from a click to a ladder. It then lists every workflow and what starts it.

## The data flow

```
 registry/competitors/<league>/*.json      registry/datasets/<league>/*.json
                  │                                     │
                  └──────────────┬──────────────────────┘
                                 ▼
          local run: benchmarks/*.py, runner.<league>_bench,
          runner.intent_bench, runner.autorun, python -m runner
          (real OVOS plugin, offline, one row per sample)
                                 │  upload
                                 ▼
          Hugging Face dataset repos
          OpenVoiceOS/ovos-<league>-bench-<dataset_id>
            predictions/<lang>/<competitor_id>.jsonl
            sample_sets/<ll_RR>.json   ◄── publish-sample-sets.yml
                                 │  read at a pinned commit
                                 ▼
          assemble.yml (daily) ── python -m arena.cli assemble
            benchmark-<league>-<dataset>-<lang>.json
            battles-<group>-<dataset>-<lang>.json
            battles-<group>-freeform-<lang>.json
            elo-seed-<group>-<lang>.json
            leaderboard-<group>-<lang>.json
            index.json, competitors.json, evidence.json
                                 │  commit to dev: frontend-static/public/data/
                                 ▼
          pages.yml ── npm run build ── GitHub Pages site
                                 │
            voter opens a battle ▼
          GitHub issue "vote|<battle_id>|<choice>", label "vote"
                                 │
                                 ▼
          tally.yml (hourly) ── python -m arena.cli tally
            votes.jsonl (append only), voter-age-cache.json,
            leaderboard-<group>-<lang>.json, vote-audit.json,
            patch-notes.json, badges/<group>/<competitor_id>.svg
                                 │
                                 ▼
          replay-proof.yml ── python -m arena.cli verify-replay
```

## 1. The registry

The registry is a folder of JSON files. One file in
`registry/competitors/<league>/` is one fighter: a plugin and the config a
user ships with it. One file in `registry/datasets/<league>/` is one test,
training or reference set. The intent leagues share their test sets under
`registry/datasets/intent/`, and keep training sets under
`registry/datasets/intent_template/` and `registry/datasets/intent_keyword/`.

`registry/schemas.py` defines both kinds of file as pydantic models.
`ovos-arena validate-registry` loads every file, prints each error it
finds, and exits with a non-zero code when there is one or more. The `validate_registry` job of
`unit_tests.yml` runs it on every pull request. [Registry](registry.md)
lists every field.

The registry decides everything downstream. A run reads it to know what
to run. `assemble` reads it to know which rows to keep. When a fighter
leaves the registry, `arena.predictions.group_rows` drops its rows, so it
leaves every board on the next `assemble` run, even when its files stay on
Hugging Face.

## 2. The local run

No workflow runs a plugin. A person runs the benchmark on a laptop or a
host, and uploads the rows. [Benchmarks](benchmarks.md) gives the commands.
The entry points are:

| Entry point | Leagues | What it does |
|---|---|---|
| `benchmarks/*.py` | all | One script per dataset family. Each calls the runner code below. |
| `runner/stt_bench.py`, `tts_bench.py`, `vad_bench.py`, `ww_bench.py` | `stt`, `tts`, `vad`, `wake_word`, `ww_stream` | Adapters of `runner/media_bench.py`. They stream the clips, load the plugin through OPM, and write one row per clip. |
| `runner/intent_bench.py` | the four intent leagues | Trains each fighter on the training corpus that the test set names, then sends the test utterances. Intent engines load from `runner.intent_pipeline.ENGINE_REGISTRY`, not through OPM. |
| `python -m runner.autorun` | `stt`, `wake_word`, `tts`, `vad` | Reads the registry on every sweep, finds missing `(fighter, dataset, lang)` pairs, and runs them in turn. |
| `python -m runner` | `stt` | The queue daemon. It works through `runner/queue.yaml`. |

Each run writes `predictions/<lang>/<competitor_id>.jsonl` to the prediction
repo of the dataset. The `predictions_hf` field of the dataset names that
repo. An intent test set also has one repo per training datashape.
[Predictions](predictions.md) gives the layout and the row fields.

A run is a real plugin on real input, with the plugin's own thresholds. The
league pages tell how the runner drives each kind of plugin:
[intent](leagues/intent.md), [wake word](leagues/wake_word.md),
[false wake-ups](leagues/ww_stream.md), [STT](leagues/stt.md),
[TTS](leagues/tts.md) and [VAD](leagues/vad.md).

## 3. Assemble

`assemble.yml` runs `python -m arena.cli assemble` once a day. It has one
matrix leg per league family: `stt`, `tts`, `wake_word`, `vad`, `intent`
and `ww_stream`. The `intent` leg covers all four intent leagues, because
they share one battle group. Each leg reads only the prediction repos of its
own leagues (`registry.loaders.list_prediction_repos`).

For each prediction repo, `assemble` resolves one commit and reads the rows
at that commit. The board records the commit in `predictions_revisions`.
It then works one language at a time:

1. **Group and filter.** `group_rows` keys each row by league, dataset,
   language, sample and fighter. It drops the rows of a fighter that is not
   in the registry, the rows of a fighter on a dataset in its `trained_on`
   list, and, on a wake word dataset, the rows of a fighter of another
   phrase.
2. **Refuse partial input.** When one source of a language fails to load,
   the run writes nothing for that language, keeps the committed files, and
   exits with an error.
3. **Benchmark boards.** One board per league, dataset and language. The
   board ranks on the primary metric of the dataset. It drops rows made
   against another corpus commit when the registry pins one, and it scores
   only the published sample set when one exists.
4. **Battle pools and seeds.** One pool per battle group, dataset and
   language, one free-form pool per group and language, and one ELO seed
   per group and language. [Rating](rating.md) gives the rules.
5. **Replay.** The run rebuilds every ladder from the new seeds and the
   committed vote record.

When a league has no rows at all, the leg ends without a change.

The `commit` job then joins the legs. It runs `prune-data`, which deletes
board, pool, seed and ladder files that the registry no longer produces. It
runs `tally` once more, so the ladders come from the merged seeds. The step
passes no `--repo`, but `--repo` defaults to `$GITHUB_REPOSITORY`, so `tally`
still calls `gh issue list`. The step sets no `GH_TOKEN`, the call fails,
`fetch_vote_issues` returns an empty list, and the step records no vote and
closes no issue. The ladders thus come from the committed vote record only. It runs `export-bestiary` (`competitors.json`), `export-index`
(`index.json`) and `export-evidence` (`evidence.json`), and it merges the
per-leg `assemble-summary-<league>.json` files. A step prints a warning for
each alarm in the summary. The job commits `frontend-static/public/data/`
to `dev` with `[skip ci]`.

`export-index` leaves out the wake word boards that `arena/ww_listing.py`
hides. A hidden board stays on disk. Only `index.json` changes.
[Wake word](leagues/wake_word.md) gives the rules.

## 4. The site

`pages.yml` builds the Astro site in `frontend-static/` and deploys it to
GitHub Pages. The site reads only the files under `data/`. It has no
backend. [Frontend](frontend.md) lists the pages and the files each one
reads.

## 5. Votes

The battle page shows two outputs for one sample with no fighter names. A
click opens a GitHub issue form, `.github/ISSUE_TEMPLATE/vote.yml`, with
the title `vote|<battle_id>|<choice>`. The form adds the label `vote`. The
voter submits the issue under their own account.

`tally.yml` runs `python -m arena.cli tally` every hour. It records each
new vote issue in `votes.jsonl` once, applies the fraud rules, rebuilds the
ladders, writes `vote-audit.json`, the patch notes and the badges, and
closes each open recorded issue with a comment. It commits the data to
`dev` with `[skip ci]`. [Rating](rating.md#votes) gives each step.

The vote record and the issue history are public. Anybody can replay them:
`verify-replay` rebuilds every ladder from committed files and compares
the result with the published ladders.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `assemble.yml` | Daily at 03:00 UTC<br>manual, with input `max_battles` (default 200) | One `assemble` leg per league family, then one `commit` job: prune, replay, export the index, the bestiary and the evidence, merge the summaries, alarm, commit the data to `dev`. |
| `tally.yml` | Hourly at minute 17<br>manual | Records vote issues, rebuilds the ladders, writes the index, commits the data to `dev`, closes the processed issues. |
| `replay-proof.yml` | After each `Assemble battles` or `Tally votes` run that succeeds<br>each push to `dev`<br>daily at 05:11 UTC<br>manual | Runs `verify-replay` on the tip of `dev`. On a push that changes `arena/metrics.py`, `arena/assembler.py` or `arena/cli.py`, it skips, because the committed boards still come from the old logic. It warns when the proof counts no human vote. |
| `pages.yml` | Push to `dev` that changes `frontend-static/**` or the workflow<br>after each `Assemble battles` or `Tally votes` run<br>manual | Builds the site with `npm run build`, runs an accessibility check that does not block, and deploys to Pages. The deploy job skips on a private repository. |
| `publish-sample-sets.yml` | Weekly, Sunday 04:30 UTC<br>manual, with an optional dataset glob | Runs `python -m runner.publish_sample_set --upload --skip-existing`, which publishes `sample_sets/<ll_RR>.json` for each eval dataset with a `sample_policy`. Needs the `HF_TOKEN` secret. |
| `unit_tests.yml` | Pull request to `dev`<br>push to `dev` that changes code, tests, the frontend or `pyproject.toml`<br>manual | `validate-registry`, the test suite with the `test,audio,hf` extras, ruff and mypy with their real exit code, and a headless browser check of the built site (`npm run ui-check`). |

The data workflows `assemble`, `tally`, `replay-proof`, `pages` and
`publish-sample-sets` share the action `.github/actions/notify-failure`. A
failed run opens one tracking issue, and the next successful run of the same
workflow closes it. [Operations](operations.md#alarms) tells how to act on
an alarm.

Two jobs write to `dev`: the `commit` job of `assemble.yml` and the `tally`
job. Both are in the concurrency group `arena-data`, so they never push at
the same time. Each retries a rejected push after a rebase, because both
touch only `frontend-static/public/data/`.

## Why the design is fair to check

- **Every input is public.** The registry and the boards are in git. The
  rows are on Hugging Face at a recorded commit. The votes are GitHub
  issues and lines of `votes.jsonl`.
- **Every output is a function of those inputs.** The board intervals use a
  fixed bootstrap seed. Battle ids are hashes. The rating fit does not
  depend on the order of the votes. `replay-proof.yml` checks the ladders
  on each data change.
- **Anybody can repeat a row.** The runner is the code in this repository.
  A benchmark notebook pins one commit of it.
  [Reproduce a row](reproduce-a-row.md) gives the steps.

---
[← Rating](rating.md) · [Home](index.md) · [Registry →](registry.md)
