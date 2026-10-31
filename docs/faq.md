# FAQ and troubleshooting

Each answer names the code that causes the behaviour and tells what to do.
Most answers start from a log line or a field in a published file. Search
for the quoted text.

## Running a benchmark

### The log says "plugin ... is not installed — skipping"

The STT, TTS, wake word and VAD benchmark scripts check that an OPM entry
point exists for the fighter's `plugin` before they load it
(`runner/media_bench.py` `plugin_is_installed`). The check reads the
entry-point names of the modern group (`opm.stt`, `opm.tts`,
`opm.wake_word`, `opm.VAD`) and of the legacy groups (`mycroft.plugin.*`,
`ovos.plugin.VAD`). It accepts the name with dashes or with underscores. A
fighter that fails the check writes no rows. The run ends with
`run summary: skipped_missing_plugin=<n>`. The fleet runner
(`runner/autorun.py`) does the same check once per plugin, and logs
`fighter(s) skipped: plugin not installed`.

To fix it, install the plugin into the same environment as the arena. The
arena does not install plugins. The `audio` extra installs only the decode
stack and the scoring models.

### The log says "plugin not found"

The entry point exists, but OPM returned no class for any spelling of the
name (`runner/media_bench.py` `load_plugin_class`). This happens when the
plugin module raises an error on import, for example because a dependency
of the plugin is missing. Import the plugin module in the same Python and
read the error.

### My intent fighter does not run, and the log does not name it

`runner/intent_bench.py` `eligible_competitors` leaves out an intent fighter
without a log line when one of its pipeline stages names an engine that is
not in `registry.schemas.ENGINE_TRAITS`. Intent engines do not load through
OPM discovery. The runner imports each engine from the fixed table
`runner/intent_pipeline.py` `ENGINE_REGISTRY`, and a test keeps that table
and `ENGINE_TRAITS` equal. `validate-registry` accepts a fighter with an
unknown engine, so the registry check does not catch it.

When you name the fighter with `--competitors`, the script stops with
`Unknown/ineligible competitors: <id>` and exit code 1.

To fix it, add the engine to `ENGINE_REGISTRY` (import path, training
message, distribution name) and to `ENGINE_TRAITS` (paradigm, regime, short
name) in the same pull request.

### My intent engine is in the table, but every cell fails

The intent runner cannot probe an intent plugin before it loads it:
`plugin_is_installed` knows no OPM group for the intent leagues and returns
true. A missing engine package then fails at `importlib.import_module` in
`runner/intent_pipeline.py`. The cell logs the traceback, writes no rows,
and the run goes on with the next cell. Install the distribution that
`ENGINE_REGISTRY` names for the engine.

### The log says "Skipping ... needs keyword training data this benchmark does not provide"

A fighter runs on an eval set only when the set provides every datashape
its stages need (`eligible_competitors`, `needed_paradigms`). The eval set
names its training corpora in `train_datasets`. A keyword engine needs an
`intent_keyword` training corpus. A template engine needs an
`intent_template` one. To fix it, register the missing training corpus and
add it to the eval set's `train_datasets`.

### The log says "Skipping ... not trained on this label set"

The fighter is in `intent_offline`. Its pretrained head can emit only the
labels of the corpora in its `label_set`, so the runner skips every other
corpus (`runner/intent_bench.py` `trained_on`). To fix it, add the corpus to
`label_set` only when the model was trained on that corpus's labels.

### The log says "the loaded model emits none of ...'s labels"

The `label_set` names the corpus, but the class list of the loaded model
shares no label with it (`runner/intent_bench.py` `model_label_overlap`).
The runner writes no rows, so the fighter does not get a board full of
misses. Each row of a run with some overlap records the count in
`extras.label_overlap`. To fix it, correct `label_set`, or point `model`
and `model_revision` at the artefact that was trained on the corpus.

### The log says "cannot resolve <model>@<revision>"

The fighter sets `model_revision`, and `huggingface_hub.snapshot_download`
cannot fetch that commit (`runner/intent_bench.py` `resolve_model_pin`). The
runner refuses the fighter and writes no rows. A fighter that sets
`model_revision` without `model` fails the same way. Check the repo id and
the commit on the Hub.

### The log says "aborted training ... reason=train_timeout" or "reason=train_memory"

With the guard on, each intent cell trains and predicts in a child process.
The parent kills the child when it runs longer than `--train-timeout-secs`
(3600 by default) or uses more resident memory than `--train-max-rss-mb`
(8192 by default). Rows the child flushed before the kill stay on disk, and
the next run skips them. Raise the limit, or set it to 0 to turn that guard
off.

### The log says "trained on this corpus, not scored"

The fighter lists the dataset in `trained_on` (`runner/queue_tools.py`
`is_trained_on`). The arena never scores a fighter on its own training
data. The benchmark scripts skip the pair, the fleet runner does not
schedule it, and the assembler drops any published row for it. This is
the intended behaviour.

### The log says "Dataset ... requires STT plugin ..., which is not installed"

The eval set has `input: "audio"`. The runner first transcribes each clip
with the pinned STT plugin of the set (`stt_plugin`). When that plugin is
missing, the run skips every cell of the set and exits 0. Install the STT
plugin that the dataset entry names.

## Reading a board

### My rows are on the Hub, but the fighter is not on the board

`arena/predictions.py` `group_rows` is the one place that every board,
battle pool and ELO seed reads rows through. It drops a row when:

- the `competitor_id` is not in the registry for that league. A fighter
  removed from the registry loses its rows, even when its old shards are
  still on the Hub.
- the dataset is in the fighter's `trained_on`.
- the dataset has a `wakeword` and the fighter's hotword is another phrase
  (`registry/loaders.py` `is_same_phrase`). A detector that fires on the
  positives of another phrase measures nothing.

`assemble-summary.json` counts each drop under
`unregistered_competitors_excluded`, `trained_on_rows_excluded` and
`cross_phrase_rows_excluded`. The benchmark scripts themselves do not check
the phrase. Only the fleet runner and the assembler do. Run a wake word
script with `--competitors` set to the fighters of the set's phrase.

### The entry is on the board with rank 0

The entry is unranked. `arena/metrics.py` `build_benchmark_board` sets
`unranked: true` and starts `unranked_reason` with one of these names:

| Reason | Cause | What to do |
|---|---|---|
| `too_few_samples` | The primary metric rests on fewer than 30 rows (`MIN_BOARD_SAMPLES`). On a false wake-up board, the entry has less than 4 h of negative audio (`MIN_NEGATIVES_BOARD_HOURS`). | Run more samples. Do not cap the run with a small `--max-samples`. |
| `one_class` | On a wake word or VAD board, the entry has no positives or no negatives, so `error_rate` measures one error type. | Run the fighter over both classes of the set. |
| `sample_set_partial` | The entry covers less than 90% of the published sample-set manifest. | Run the fighter again with the policy cap, so it covers the manifest. |
| `no <metric>` | The run scored rows, but none is in the population of the primary metric. Example: every intent row is in an in-distribution bucket. | Check the bucket roles of the dataset. |
| `run failed` | No row was scored. | Read the runner log of that fighter. |
| `no rows on the pinned dataset revision` | Every row came from another commit of the corpus. | Run the fighter again against the pinned commit. |

### What is `rows_other_revision`?

When the registry pins a dataset to a 40-character commit, the board drops
every row made against another revision before it scores
(`drop_rows_off_pinned_revision`). `rows_other_revision` counts the rows
dropped for that entry. An entry with no rows left is unranked with the
reason `no rows on the pinned dataset revision`. `assemble-summary.json`
adds up the drops in `rows_dropped_off_revision`. A registry that pins a
branch instead of a commit drops nothing, and the board shows
`dataset_revision: null`.

### What is `version_blended`?

The rows of the entry carry more than one `plugin_version`. The board does
not split them. It sets `version_blended: true`, lists every version in
`plugin_versions`, and logs a warning. The score is then not the score of
one version. Run the fighter again over the whole set with one version.

### What is `sample_set: "unmanaged"`?

No sample-set manifest applied to the board. Either the dataset has no
`sample_policy`, or nobody published its `sample_sets/<ll_RR>.json`. Each
fighter is then scored on the rows it has. When a manifest exists but
cannot be read, the board also falls back to unmanaged, and
`assemble-summary.json` lists it under `boards_unmanaged_on_error`.

### A board has entries, but every entry is unranked

`assemble` lists such a board in `boards_without_ranked_fighters` in
`assemble-summary.json`, and the alarm step of `assemble.yml` reads that
file. Read the `unranked_reason` of each entry.

## The site

### A wake word board is on disk, but the site does not show it

`arena/ww_listing.py` decides which wake word boards the site lists. A clip
board is listed only when all of these are true:

- its test set holds at least 50 positive clips (`MIN_POSITIVES`),
- at least 2 trained models are ranked on it (`MIN_TRAINED`), counted after
  the `trained_on` drop,
- its phrase is featured: some board that meets the two rules above ranks
  a trained model of that phrase whose plugin is not
  `ovos-ww-plugin-wakeforge` (WakeHuBERT).

A trained model is a `wake_word` fighter whose `types` hold none of
`zero-shot`, `few-shot` and `keyword-match`. A false wake-up board is
listed when a listed clip board has the same primary language. Every other
`ww_stream` board is hidden.

The board file and its battle pool stay on disk. Only `export-index`
leaves them out of `index.json`, so the site never offers them. To get a
board listed, add trained models of the phrase until the rules hold.

### A league has no tab on the site

`export-index` offers a league only when it has a listed benchmark board
with at least one entry (`arena/models.py` `leagues`). A league whose
fighters nobody has run yet has no board, so it gets no tab. A league whose
only boards are hidden wake word boards gets no tab either.

### The leaderboard page says there is nothing to show

The page found no `benchmarks` and no `leaderboards` in `index.json`. Run
`export-index` after `assemble` and `tally`. The page reads only what the
index lists.

### My badge shows an old rank

`tally` writes the badges. A badge changes only after a tally run and the
Pages deploy that follows it. `_prune_stale_badges` in `arena/cli.py`
deletes the badge of a fighter that the registry no longer holds, but only in
a folder whose name is a league: it compares `badges/<folder>/` with the
fighters of the league of that name. Badges sit in battle-group folders, and
no league is named `intent`, so the badge of a removed intent fighter stays
in `badges/intent/`. This is a known limit.

## Votes

### My vote was recorded but did not count

The `tally` workflow closes each vote issue with a comment. A vote that did
not count names its reason in the comment and in `vote-audit.json`
(`discarded`):

| Reason | Rule |
|---|---|
| `daily_vote_cap_exceeded` | More than 50 votes by one voter in one league on one UTC day (`arena/fraud.py` `DAILY_VOTE_CAP`). The later votes of that day are discarded. |
| `account_too_new` | The GitHub account was less than 7 days old when the vote was cast (`NEW_ACCOUNT_MIN_DAYS`). The account age is fetched once at ingest and kept in `voter-age-cache.json`. |
| `battle_not_in_pool` | The battle id in the title is in no battle pool at ingest. |
| `competitor_retired` | One of the two fighters has left the ladder of that league and language. |
| `duplicate_record` | The vote record holds the same issue number twice. |

[Rating](rating.md) tells how a counted vote moves the ladder.

A vote can also count at reduced weight. A voter with 20 or more A or B
votes, of which more than 95% pick the same side, has all those votes
counted at weight 0.5 (`ONE_SIDED_WEIGHT`). `vote-audit.json` lists them
under `downweighted`. Ties and "both wrong" do not enter that ratio.

A second vote by the same voter on the same battle is not counted
(`arena/cli.py` `dedupe_votes`). The first vote, by issue number, counts.
An issue whose title does not match `vote|<battle_id>|<a|b|tie|both_wrong>`
is recorded as invalid.

### Can I edit my vote?

No. `tally` records the title as it saw it at ingest in `votes.jsonl`, and
never reads the issue again. A later title edit changes nothing. Cast a
vote on another battle instead.

## Replay proof

### What does `replay-proof.yml` prove?

It runs `python -m arena.cli verify-replay` on the committed data. The
command replays `votes.jsonl` on top of the `elo-seed-*.json` files with
`voter-age-cache.json`, through the same function `tally` uses
(`replay_boards`). It then compares the result with every published
`leaderboard-*.json` and with `vote-audit.json`. It reads no network. Run
it yourself:

```bash
python -m arena.cli verify-replay --data-dir frontend-static/public/data
```

It prints the counted human votes per board, and exits 0 when every board
matches.

### The replay proof failed

The output starts with `REPLAY MISMATCH` and prints a diff per file. The
causes are:

- A hand edit of a `leaderboard-*.json`, `vote-audit.json`, `votes.jsonl`
  or `elo-seed-*.json` file. Revert it and run `tally` again.
- A replayed board with no published file. The diff names it under
  `missing_published_boards`.
- A line of `votes.jsonl` that is not a JSON object with an `issue`. The
  command stops with exit code 1 and names the line.

On a push that changes `arena/metrics.py`, `arena/assembler.py` or
`arena/cli.py`, the workflow skips the replay. The published boards still
hold the output of the old code, so a mismatch is expected there. The
replay runs again when the next assemble or tally run ends.

### The replay proof warns about zero human votes

The step "Alarm on a zero-human-vote proof" warns when `verify-replay`
counts 0 human votes on every board. The proof then shows only that the
seeds replay, not that a real vote goes through the whole loop.

## Registry

### `validate-registry` reports an error

Run the check:

```bash
python -m arena.cli validate-registry
```

It prints one `<path>: <message>` line per error and exits 1, or logs
`Registry OK` and exits 0 (`registry/loaders.py` `validate_registry`). The
common errors:

| Message | Fix |
|---|---|
| `Extra inputs are not permitted` | The schemas forbid unknown keys. Fix the spelling of the field. |
| `display_name is required` | Give every dataset a display name. |
| `summary is required on a role=eval corpus` | Give every eval set a plain summary. The site prints it above the board. |
| `trained_on references unknown dataset_id` | The id in `trained_on` is not registered. Fix the id. |
| `trained_on entry ... has modality=...` | The dataset is in a league that does not score this fighter. |
| `samples_from dataset ... is missing from trained_on` | A few-shot wake word fighter takes its reference clips from a dataset. Add that dataset to `trained_on`. |
| `label_set references unknown dataset_id` | Fix the id in `label_set`. |
| `offline fighters must declare label_set` | An `intent_offline` fighter needs `label_set`. |
| `label_set is only meaningful for offline fighters` | Remove `label_set` from a zero-shot, online or keyword fighter. |
| `declares training_regime=... but its stages need ...` | Set `training_regime` to what the stages need. |
| `belongs in the ... league` | Move the file to the league that its stages put it in. |
| `has no -high/-medium/-low tier suffix` | Each pipeline stage is `<plugin>-<tier>`. |
| `negatives_dataset_ids references unknown dataset_id` | Fix the id of the negatives pool. |
| `is not a valid BCP-47 language tag` | Use a full `lang-REGION` tag, such as `pt-PT`. |

[Registry](registry.md) gives every field of the competitor and dataset
files.

---
[← Frontend](frontend.md) · [Home](index.md) · [Glossary →](glossary.md)
