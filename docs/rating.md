# Rating

This page gives the rating maths of the arena and the place of every
parameter in the code. [Methodology](methodology.md) tells why the arena
uses each method. This page tells what the code does.

The arena makes two kinds of ranking:

- A **benchmark board** ranks fighters on one metric of one dataset in one
  language. The metric comes straight from the prediction rows. No vote
  changes it.
- An **ELO ladder** ranks fighters in one battle group and one language. It
  starts from auto-battles that the benchmark rows decide, and human votes
  then move it. It ranks on a Bradley-Terry rating.

## Battle groups

A league is one competition with its own boards. A battle group is the pool
in which fighters battle and get a ladder. `arena.models.battle_group()`
maps a league to its group:

| League | Battle group |
|---|---|
| `intent_zero_shot`, `intent_online`, `intent_offline`, `intent_keyword` | `intent` |
| `stt` | `stt` |
| `tts` | `tts` |
| `vad` | `vad` |
| `wake_word` | `wake_word` |
| `ww_stream` | `ww_stream` |

The four intent leagues share one group, because a blind voter sees two
answers to the same utterance and cannot see how each engine was prepared.
Every other league is its own group. Nine leagues thus make six battle
groups.

`arena.models.VOTELESS_MODALITIES` names the leagues that get boards but
no battles. The set is empty, so every group gets battles and a ladder.

## Benchmark board

`arena.metrics.build_benchmark_board` sorts the entries on the primary
metric of the dataset. The league page gives the formula of each metric.

| Parameter | Value | Where |
|---|---|---|
| Rows that a ranked entry needs | 30 | `arena/metrics.py` `MIN_BOARD_SAMPLES` |
| Bootstrap rounds of the board interval | 1000 | `arena/metrics.py` `BOOTSTRAP_ROUNDS` |
| Bootstrap seed | 0 | `arena/metrics.py` `BOOTSTRAP_SEED` |
| Interval percentiles | 2.5 and 97.5 | `arena/metrics.py` `CI_LOWER_PCT`, `CI_UPPER_PCT` |
| Board logic version | an integer, part of the board input hash | `arena/metrics.py` `BOARD_LOGIC_VERSION` |

`arena.metrics.primary_metric_ci` computes the interval with one of three
methods:

- **Mean of a value per row.** Intent `generalization_accuracy`, wake word
  and VAD `error_rate` (0/1 values) and TTS `utmos` resample the per-row
  values. Each round takes the mean (`bootstrap_mean_ci`).
- **Ratio of sums.** STT resamples `(word errors, reference words)` pairs
  and computes `sum(errors) / sum(reference words)` in each round
  (`bootstrap_ratio_ci`). This is an interval of the pooled WER. The board
  ranks on `wer_mean`, the mean of the per-row WER, so the interval does not
  bracket the ranked number exactly.
- **None.** A `ww_stream` entry carries no interval. The false wake-up
  recordings hold no wake word, so there is no miss rate to resample.

`tied_with_leader` is true when the interval of an entry overlaps the
interval of rank 1. Read those entries as tied with the leader.

## Human battle pool

`arena.assembler.assemble_battles` writes the pool that the vote page
shows. One pool exists per battle group, dataset and language:
`battles-<group>-<dataset>-<lang>.json`.

1. A battle pairs two predictions of the **same sample** (§4 R1).
2. A pair with the same prediction, and for intent the same slots, gives no
   battle (R2). A voter cannot choose between two equal answers.
3. A pair whose reference text differs for the same `sample_id` gives no
   battle. The pool counts these pairs in `skipped_reference_mismatches`.
4. Within one fighter pair, the candidates sort first on priority, then on
   `battle_id` (R3): both wrong first, then exactly one wrong, then the
   rest.
5. The pool takes one battle from each fighter pair in turn, in sorted pair
   order, until it holds `--max-battles` battles. The default is 200
   (`arena/assembler.py` `DEFAULT_MAX_BATTLES`, and the `max_battles` input
   of `assemble.yml`).

`arena.models.battle_id_for` makes the id: the first 16 hex characters of
the SHA-256 of `group|dataset|lang|sample|low|high`, where `low` and `high`
are the two competitor ids in sorted order. The id is the same on every
run, so a vote stays valid. When the id as an integer is odd, the two
candidates swap places. The A/B position thus comes from the hash and not
from the names.

The input rows of a pool pass three more filters in `arena.cli.cmd_assemble`:

- On a `wake_word` or `ww_stream` dataset, only the fighters that the board
  ranks enter a battle. On any other dataset, the dataset and language need
  30 or more samples (`MIN_BOARD_SAMPLES`), or the run makes no battles and
  no seed for it.
- When the dataset has a sample-set manifest, only the samples of the
  manifest enter.
- When the registry pins the dataset to a commit, only rows made against
  that commit enter.

A second pool, `battles-<group>-freeform-<lang>.json`, holds one battle with
no sample for each fighter pair (`freeform_battles`). A voter who tested the
plugins outside the arena votes there. In the `wake_word` group, only the
fighters of the same phrase make a pair.

## Auto-battles and the seed

`arena.assembler.seed_elo` writes one seed per battle group and language:
`elo-seed-<group>-<lang>.json`. It reads every dataset of that group and
language, in sorted order, and replays one auto-battle for each sample and
each fighter pair where the benchmark picks a winner.

**The significance gate.** For each dataset, the assembler computes the
primary metric interval of each fighter with `primary_metric_ci`. A fighter
pair gives auto-battles on that dataset only when the two intervals do
**not** overlap (`arena.metrics.significant_from_cis`). When one of the two
intervals cannot be computed, the gate lets the pair through. The gate
applies in every battle group, the intent group included.

**The outcome** (`arena.assembler.auto_outcome`):

| Group | The winner |
|---|---|
| `intent` | The fighter whose answer is correct, when exactly one is. A row in an in-distribution bucket gives no auto-battle (`seeds_elo`). |
| `stt` | The fighter with the lower WER on the sample. Equal or missing WER gives no battle. |
| `wake_word`, `vad` | The fighter whose decision is correct, when exactly one is. |
| `tts` | The fighter with the higher composite score `(min(max(utmos, 0), 5) / 5) × (1 − min(cer, 1)) × agreement` (`arena.metrics.tts_seed_score`). When both rows have no CER, the higher UTMOS wins. When one row has a CER and the other does not, or two judges of different models scored the rows, no battle. |
| `ww_stream` | No auto-battle. The false wake-up recordings hold no wake word, so no fighter catches an onset that the other misses (`_replay_ww_stream_dataset`). |

**The weight.** Each auto-battle updates the sequential ELO at a quarter of
the K-factor and adds a Bradley-Terry weight of 0.25 (`arena/elo.py`
`AUTO_K_DIVISOR = 4`, `BT_AUTO_WEIGHT = 1 / AUTO_K_DIVISOR`).

**The cap.** After all datasets, `_cap_auto_pairwise_weight` scales the
total auto-battle weight of each fighter pair down to at most 5
(`MAX_AUTO_WEIGHT_PER_PAIR`), in units of one human vote. The scaling keeps
the win rate of the pair. A large benchmark thus cannot outweigh a few
human votes.

`ovos-arena audit-seeds --data-dir frontend-static/public/data` prints the
weight of each pair and marks the pairs at the cap.

**Secondary ladders.** `seed_secondary_metrics` makes one auto-only ladder
for each other row metric that `arena.metrics.ROW_METRIC_ACCESSORS` lists
for the group. The `stt`, `tts` and `wake_word` groups get an `rtf` ladder.
The `tts` group also gets one ladder per quality dimension in
`TTS_QUALITY_DIMENSION_KEYS`. These ladders skip the significance gate,
take no human votes, and use the same cap. The table keys its intent
entries by league name, and the seed looks it up by the group name
`intent`, so the intent group has no secondary ladder.

## Votes

The site opens a GitHub issue with the form `.github/ISSUE_TEMPLATE/vote.yml`.
The title is `vote|<battle_id>|<choice>`. The choice is `a`, `b`, `tie` or
`both_wrong`. The form adds the label `vote`.

`tally.yml` runs `ovos-arena tally` every hour. The command does these steps:

1. **Fetch.** `fetch_vote_issues` lists every issue with the label `vote`,
   open and closed.
2. **Record.** `ingest_vote_issues` appends each issue that is not yet in
   `votes.jsonl`, once. The record holds the title as seen, the author, the
   time, and the battle context from the pool at that time. A title that
   does not parse is recorded as `invalid`. A battle id that is not in the
   pool is recorded with the reason `battle_not_in_pool`. The record is
   never rewritten, so a later title edit cannot change a vote.
3. **Account age.** The command reads the creation date of each new author
   from the GitHub API once and stores it in `voter-age-cache.json`. When
   the fetch fails, the issue stays out of the record until a later run.
4. **Replay.** `replay_boards` rebuilds every ladder from the seeds, the
   record and the age cache. It uses no network.
5. **Close.** The command comments on each recorded issue that is still
   open and closes it with the label `processed`.

### Fraud rules

`replay_boards` applies these rules in this order. A rule never deletes a
record. It sets a weight or a discard reason, and `vote-audit.json` lists
every discard and every reduced weight.

| Rule | Effect | Parameter | Where |
|---|---|---|---|
| Repeat of an issue number in the record | Discarded: `duplicate_record` | n/a | `arena/cli.py` `replay_boards` |
| More than one vote by one author on one battle | Only the first, by issue number, stays | n/a | `arena/cli.py` `dedupe_votes` |
| More than 50 votes by one voter in one battle group on one UTC day | The later votes: `daily_vote_cap_exceeded` | `DAILY_VOTE_CAP = 50` | `arena/fraud.py` |
| Account younger than 7 days at the time of the vote | Discarded: `account_too_new` | `NEW_ACCOUNT_MIN_DAYS = 7` | `arena/fraud.py` |
| 20 or more A or B votes of one voter, of which more than 95% pick the same literal side | All those votes get weight 0.5 | `ONE_SIDED_MIN_VOTES = 20`, `ONE_SIDED_THRESHOLD = 0.95`, `ONE_SIDED_WEIGHT = 0.5` | `arena/fraud.py` |
| A fighter of the vote is not in the seed of its board | Discarded: `competitor_retired` | n/a | `arena/cli.py` `replay_boards` |

The one-sided rule looks at the literal A or B side, not at the fighter. The
A/B position comes from the battle id hash, so a voter who always clicks the
same side is the signal. Ties and `both_wrong` votes do not count toward
the ratio.

A reduced weight changes only the Bradley-Terry fit. The sequential ELO
column takes the vote at full strength. A discarded vote enters neither.

## The ladder

`arena.cli.build_elo_board` writes `leaderboard-<group>-<lang>.json`. It
starts from the seed and applies the counted human votes in issue order.

### Bradley-Terry rating

Each fighter `i` has a strength `π_i > 0`. The probability that `i` beats
`j` is `π_i / (π_i + π_j)`. `arena.rating.fit_bradley_terry` finds the
strengths with the minorization-maximization update:

```
π_i ← (Σ_j wins[i][j]) / (Σ_j games[i][j] / (π_i + π_j))
```

`wins[i][j]` is the weighted count of wins of `i` over `j`. A tie or a
`both_wrong` vote adds 0.5 to each side. `games[i][j]` is the weighted
count of games between them. Auto-battles and human votes add into the same
totals with their weights.

Each fighter also gets one virtual tie, of weight 1, against a fixed
phantom fighter of strength 1. This prior connects the comparison graph,
and it keeps a fighter with no loss or no win at a finite strength.

`arena.rating.to_rating_scale` maps the strengths to the display rating:

```
bt_rating_i = 1200 + (400 / ln 10) × (ln π_i − mean_j ln π_j)
```

The geometric mean of the strengths is thus at 1200. A gap of 400 points
means that the stronger fighter is 10 times as likely to win.

The ladder ranks on `bt_rating`. Equal ratings sort on `competitor_id`.

### Rating interval

`arena.rating.bootstrap_confidence_intervals` resamples the **human votes
only**, with replacement, in 100 rounds with seed 0. Each round adds the
resampled votes to the fixed seed totals, fits Bradley-Terry again, and
records each rating. `ci_lower` and `ci_upper` are the 2.5th and 97.5th
percentiles. The seed is a fixed function of the benchmark rows, so the
bootstrap does not resample it. With no human vote, each interval is one
point.

A ladder with fewer than 10 counted human votes has `provisional: true`.
Its order comes mostly from the benchmark seed.

### Sequential ELO

`arena/elo.py` also keeps a sequential ELO in the `elo` column. It is a
secondary display. It depends on the order of the votes, and the ladder
does not rank on it.

```
E_a = 1 / (1 + 10^((R_b − R_a) / 400))
R_a ← R_a + K × (S_a − E_a)
```

`S_a` is 1 for a win, 0 for a loss and 0.5 for a tie or `both_wrong`.

| Parameter | Value | Where |
|---|---|---|
| Start rating | 1200 | `arena/elo.py` `INITIAL_ELO` |
| K-factor | 32 | `arena/elo.py` `K_FACTOR` |
| K-factor after 30 battles | 16 | `arena/elo.py` `K_FACTOR_VETERAN`, `VETERAN_THRESHOLD` |
| K-factor of an auto-battle | K / 4 | `arena/elo.py` `AUTO_K_DIVISOR` |

### All ladder parameters

| Parameter | Value | Where |
|---|---|---|
| Display anchor | 1200 | `arena/rating.py` `ANCHOR_RATING` |
| Points per natural-log unit | 400 / ln 10 | `arena/rating.py` `SCALE` |
| Prior weight | 1.0 | `arena/rating.py` `PRIOR_WEIGHT` |
| Fit iterations, at most | 200 | `arena/rating.py` `MM_MAX_ITER` |
| Fit tolerance on log-strength | 1e-9 | `arena/rating.py` `MM_TOLERANCE` |
| Bootstrap rounds of the ladder interval | 100 | `arena/rating.py` `DEFAULT_BOOTSTRAP_ROUNDS` |
| Human votes before a ladder is not provisional | 10 | `arena/rating.py` `PROVISIONAL_MIN_HUMAN_VOTES` |
| Bradley-Terry weight of an auto-battle | 0.25 | `arena/elo.py` `BT_AUTO_WEIGHT` |
| Auto-battle weight cap per pair | 5.0 | `arena/assembler.py` `MAX_AUTO_WEIGHT_PER_PAIR` |

## Replay proof

The ladders are a pure function of committed files: the seeds, `votes.jsonl`
and `voter-age-cache.json`. `ovos-arena verify-replay` runs the same
`replay_boards` function as `tally`. It compares each replayed ladder,
field by field, with the published file, and it ignores `generated_at`. It
also compares `vote-audit.json`. Any difference makes the command fail.

`replay-proof.yml` runs this check after each `assemble` or `tally` run, on
each push to `dev`, and once a day. Run it on your own clone:

```bash
python -m arena.cli verify-replay --data-dir frontend-static/public/data
```

The command prints the counted human votes of each ladder and a total line,
`human-votes: total: <n>`. The workflow warns when that total is 0, because
a proof with no human vote shows only that the seed replays.

---
[← Methodology](methodology.md) · [Home](index.md) · [Architecture →](architecture.md)
