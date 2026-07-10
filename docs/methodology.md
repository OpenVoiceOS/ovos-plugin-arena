# Rating methodology

This page is the normative explanation behind `EloEntry.bt_rating` and its
confidence interval — the primary ranking signal on every leaderboard (§5 of
`SPECIFICATION.md`). It exists so a skeptical reader can check the arena's
math without reading `arena/rating.py`.

## Why not sequential ELO alone

Sequential ELO (`arena/elo.py`, still shown as the secondary `elo` column)
updates two ratings after every single vote, in whatever order the votes
happen to arrive. That has a real problem: **the same set of votes, replayed
in a different order, produces a different final rating.** A competitor that
happens to get its toughest matchups early in the vote log ends up rated
differently than if those same matchups had come later. For an arena whose
entire premise is "the vote log is public and replayable" (§P5), an order
dependency in the headline number undermines the premise — two people
replaying the identical, public vote log by hand in a different order would
get different rankings, with no way to say which one is "right".

**Bradley-Terry**, fit as a batch over the whole vote log at once, does not
have this problem. It asks a single, order-independent question: *what set
of competitor strengths makes the observed vote log most likely?* Shuffle
the votes and refit — you get the identical answer (see
`tests/test_rating.py::TestFitBradleyTerry::test_shuffle_invariant`).

## Why not TrueSkill

TrueSkill (Microsoft, used in Xbox Live matchmaking) is a plausible
alternative, and was considered. It was rejected for two reasons specific to
this arena:

1. **TrueSkill is fundamentally sequential** — it maintains a Gaussian
   belief per player and updates it incrementally per game, exactly the
   property that makes sequential ELO order-dependent in the first place.
   Batch-refitting a TrueSkill model over a full vote log to get an
   order-independent result is possible but loses the closed-form
   incremental-update convenience that is TrueSkill's main selling point —
   at that point a batch Bradley-Terry fit is simpler and has a much longer
   track record of transparent, reproducible use in exactly this setting
   (this is what [Chatbot Arena / lmarena](https://lmarena.ai) uses for
   ranking LLMs by blind human vote, the closest prior art to this arena).
2. **TrueSkill's uncertainty model (a per-player Gaussian variance) answers
   a different question than the one this arena needs.** This arena wants a
   confidence interval that answers "how much could this specific
   competitor's rank move if a few more human votes came in the other
   direction?" — which a nonparametric bootstrap over the actual vote log
   answers directly (see below), without assuming Gaussian belief dynamics.

## The fit: Bradley-Terry via minorization-maximization

Every pairwise vote — a blind battle choice, a free-form vote, or a
benchmark-derived auto vote — is one observation: competitor `a` beat
competitor `b` with score `1.0` (win), `0.5` (tie / both-wrong), or `0.0`
(loss). These are aggregated into weighted win/game totals per ordered pair,
`wins[i][j]` and `games[i][j] == games[j][i]`.

The Bradley-Terry model assigns every competitor a strength `π > 0` such
that the probability `i` beats `j` is `π_i / (π_i + π_j)`. The
maximum-likelihood strengths are found by the classic
[Zermelo/Hunter (2004)](https://sites.stat.washington.edu/fritz/DATAFILES2/MMAlgorithms.pdf)
minorization-maximization iteration:

```
π_i ← (Σ_j wins[i][j]) / (Σ_j games[i][j] / (π_i + π_j))
```

repeated until the log-strengths stop moving (`arena/rating.py:MM_TOLERANCE`,
default `1e-9`, capped at `MM_MAX_ITER = 200` rounds). This is deterministic:
the same input totals always converge to the same strengths (up to floating
point, which is itself deterministic for a fixed Python/platform — the
result is reproduced byte-for-byte by `arena assemble` on every CI run over
the same vote log).

Strengths are on an arbitrary multiplicative scale (only ratios matter for
Bradley-Terry). `to_rating_scale()` anchors them to a familiar ELO-shaped
number: the geometric mean of the fitted strengths is placed at 1200, with
`400 / ln(10)` points per natural-log unit of strength — the same
points-per-decade convention as classic ELO, so a `bt_rating` of 1600 vs
1200 still means "10× as likely to win a random matchup" the way it always
has.

### The convergence prior

A raw Bradley-Terry fit has two failure modes with sparse real-world vote
data: a competitor that has never lost gets an unbounded strength (formally,
the MLE is at infinity), and the fit is undefined if the "who played whom"
graph is disconnected (two competitors that never played each other,
directly or through a chain of shared opponents, have no relative order at
all).

Both are solved the same way: every competitor gets one virtual weighted tie
(`PRIOR_WEIGHT = 1.0`) against a fixed-strength "field average" phantom
opponent that is never itself updated. This connects every competitor to
every other competitor through the phantom (so the graph is always
connected) and guarantees every competitor has a nonzero recorded win
fraction (so no strength collapses to zero or diverges). A brand-new fighter
with zero real battles rates near the phantom's anchor (1200) rather than
being undefined — see
`tests/test_rating.py::TestFitBradleyTerry::test_undefeated_and_winless_fighters_converge`.

### Auto vs. human weighting

Auto (benchmark-derived) votes carry `BT_AUTO_WEIGHT = 0.25` — the same §4
R5 intent as sequential ELO's `K/4`, expressed as a pairwise weight instead
of a K-factor. This is capped further at the board level (§4, seed-battle
bias audit) so a large benchmark dataset can never outweigh a modest number
of real human votes; see that section once implemented for the cap
mechanism.

## The confidence interval: bootstrap over human votes only

`bootstrap_confidence_intervals()` resamples the **human vote list only**,
with replacement, `DEFAULT_BOOTSTRAP_ROUNDS = 100` times, using a seeded
`random.Random` (§P5: same seed ⇒ byte-identical CIs on every rerun). Each
round refits Bradley-Terry over (resampled human votes) + (the unchanged
auto-vote seed) and records where every competitor's rating landed; the
reported interval is the 2.5th–97.5th percentile of that distribution — a
standard nonparametric bootstrap 95% CI.

**The auto-vote seed is deliberately never resampled.** It is a
deterministic function of a fixed benchmark corpus (the same reference
audio/text run through the same plugin gives the same metric every time) —
it has no sampling variability to model. What genuinely varies from one
arena snapshot to the next is *how many human votes have been cast, and by
whom* — that is exactly what resampling the human vote list captures. A
board with zero human votes therefore has every CI collapse to a single
point (the seed-only rating; see
`tests/test_rating.py::TestBootstrapConfidenceIntervals::test_zero_human_votes_collapses_to_seed_point`),
and CIs visibly narrow as more human votes accumulate for a matchup (see
`test_ci_narrows_with_more_votes`).

A board with fewer than `PROVISIONAL_MIN_HUMAN_VOTES = 10` human votes sets
`EloBoard.provisional = true` — its ranking is real (computed the same way)
but should be read as a placeholder ordering from the benchmark seed, not a
settled result.

## Reading a leaderboard

- **`bt_rating`** is the number to look at. Two competitors whose confidence
  intervals overlap are statistically indistinguishable from the current
  vote log — treat them as tied, not as strictly ranked, regardless of what
  order they happen to be listed in.
- **`elo`** is kept for continuity with earlier snapshots of this arena but
  is not the ranking key and can disagree with `bt_rating`'s ordering,
  particularly early in a board's life or after a burst of lopsided votes —
  that disagreement is expected and is exactly the order-dependency problem
  `bt_rating` was built to avoid.
- **`provisional`** boards should be captioned as such in the frontend
  rather than presented with the same confidence as an established board.

## Benchmark boards: significance, not just a point estimate

The objective benchmark boards (`benchmark-<modality>-<dataset>-<lang>.json`,
straight from prediction rows — no votes involved) carry the same discipline.
A 0.3% WER gap on 500 clips, or a 96% vs 95% intent accuracy gap on a few
hundred samples, is very often noise rather than a real capability
difference; showing only the point estimate implies more precision than the
sample size supports.

Every `BenchmarkEntry` carries a seeded bootstrap 95% CI
(`arena/metrics.py:primary_metric_ci`, `BOOTSTRAP_ROUNDS = 1000`) on its
primary metric, using one of two strategies depending on what kind of number
the metric is:

- **Mean of a per-row indicator** (intent `accuracy`, wake-word/VAD
  `error_rate`) — bootstrap the 0/1 indicator list directly and report the
  percentile interval of the resampled mean.
- **Ratio of summed counts** (STT `wer_mean` = total word errors / total
  reference words) — a per-utterance WER is not directly comparable across
  utterances of different length, so this bootstraps **(errors,
  reference-word-count) pairs**, recomputing `sum(errors) / sum(ref_words)`
  each round, rather than averaging per-utterance WER values as if they were
  i.i.d. — a one-word command with one error and a twenty-word sentence with
  one error are not the same signal, and the ratio bootstrap weights them
  correctly by how much reference text they actually contain (see
  `tests/test_metrics_ci.py::test_weighted_by_denominator_not_per_pair_average`).

`BenchmarkEntry.tied_with_leader` marks every entry whose CI overlaps the
#1-ranked entry's CI — the frontend should render those as "≈ tied with #1"
rather than a strict rank ordering, and prefer showing a compact tier
grouping over a false-precision numeric rank for entries that are
statistically indistinguishable from the leader.

## Seed-battle bias audit

A benchmark dataset can have thousands of samples; without a check, its
auto-battles would both (a) dwarf a modest number of real human votes, and
(b) manufacture apparent rating separation out of per-sample noise even
when the two competitors' overall performance is not meaningfully
different. Two independent controls in `arena/assembler.py:seed_elo`:

- **Significance gate** (`pair_metric_significant`, using the same bootstrap
  CIs as the benchmark boards — see above). Before any per-sample
  auto-battle is generated for a competitor pair, their *aggregate* primary
  metric confidence intervals for the dataset are compared. If the
  intervals overlap, the pair contributes **zero** auto-battles for that
  dataset — individual samples may still disagree, but that disagreement is
  benchmark noise, not a real capability gap, so it must not seed the
  rating. A pair with a genuinely wide gap (non-overlapping CIs) seeds
  normally.
- **Weight cap** (`MAX_AUTO_WEIGHT_PER_PAIR = 5.0`). Even a pair that clears
  the significance gate has its total Bradley-Terry pairwise weight capped
  at the equivalent of 5 human votes, scaled down proportionally so the
  observed win rate is preserved. A benchmark run with 10,000 samples and a
  benchmark run with 250 samples contribute the same maximum seed weight to
  a pair once both clear the significance gate — dataset size stops being a
  lever on how much the rating can move. This cap applies only to the
  Bradley-Terry pairwise statistics; it does not change the legacy
  sequential ELO's `auto_vote_count` bookkeeping.

Run `ovos-arena audit-seeds --data-dir <path>` to see, per (modality, lang),
every scored pair's weight and whether it sits at the cap.

## Vote fraud / dedup resistance

The vote log is public (§6: "the vote log **is** the issue history") — and
a public, easy-to-participate voting mechanism is a predictable target for
brigading and low-effort automated voting. `arena/fraud.py` applies a
sequence of deterministic rules, none of which deletes anything: every rule
records a `discarded_reason` or a reduced `weight` rather than silently
dropping a vote, and the full audit trail is written to `vote-audit.json`
alongside the leaderboards on every tally run.

- **One vote per (voter, battle).** Handled upstream by
  `arena.cli.dedupe_votes`, keyed on `(author, battle_id)` — since
  `battle_id` already encodes `(dataset, sample, competitor pair)` (§4 R4),
  this is exactly "one vote per voter per battle-pair-on-a-dataset-entry".
- **Per-voter, per-league, per-day cap** (`DAILY_VOTE_CAP = 50`,
  `apply_daily_cap`). Votes are processed in their deterministic order
  (issue number ascending); once a voter passes the cap for a given
  modality on a given UTC calendar day, further votes that day are
  discarded with reason `daily_vote_cap_exceeded`. The cap is per
  modality, not global, so a voter legitimately evaluating multiple
  leagues in one sitting is not penalized.
- **Account-age gate** (`NEW_ACCOUNT_MIN_DAYS = 7`,
  `apply_account_age_gate`). A vote from an account created less than 7
  days before the vote is discarded with reason `account_too_new`. The
  account creation timestamp is fetched from the GitHub API **once per
  author** and persisted to `voter-age-cache.json` in the committed data
  directory — every subsequent tally run reuses the cached value instead of
  re-fetching, so the pure `resolve_vote_weights` replay never touches the
  network (`tests/test_fraud.py::test_pure_no_network` enforces this
  structurally, and `docs/SPECIFICATION.md` §4 requires it for §P5).
- **One-sided voter down-weight** (`ONE_SIDED_MIN_VOTES = 20`,
  `ONE_SIDED_THRESHOLD = 0.95`, `apply_one_sided_downweight`). A voter whose
  surviving votes are more than 95% for the same literal **A** or **B**
  side across at least 20 votes has every one of those votes down-weighted
  to 0.5 rather than discarded outright. This is deliberately keyed on the
  literal A/B choice, not competitor identity — blind battles randomize
  which competitor is shown as "A" per battle from the battle-id hash (§4
  R4), so "always clicks the left button" is the low-effort/bot-like
  signal; "always prefers competitor X" is not detectable this way and
  would be indistinguishable from a voter with a genuine, consistent
  preference.

**Weighting only affects the Bradley-Terry rating, not the legacy
sequential ELO column.** A down-weighted vote (weight 0.5) still updates
`EloEntry.elo` at full strength — that column is secondary/display-only
(see "Reading a leaderboard" above) and does not need this level of rigor.
A fully discarded vote (weight 0) is excluded from both ratings entirely
and is never passed into `build_elo_board`'s human-vote list — it exists
only in the `vote-audit.json` record.

**The full vote-issue history is refetched every run**
(`fetch_vote_issues` lists both open and closed `vote`-labelled issues), so
every tally run genuinely replays the complete log from scratch rather than
only the issues opened since the last run — the earlier design fetched
`--state open` only, which meant closing a processed issue silently
dropped its vote from every future leaderboard rebuild. Already-closed
issues are never re-commented-on or re-closed; `arena.cli.cmd_tally` only
takes GitHub actions (comment + close) on issues that are still `OPEN` in
the freshly-fetched list.

## Open items

The TTS objective-metric judge-bias disclosure (TTS intelligibility) and
the RTF hardware-disclosure convention (TTS latency/RTF) are placeholders
for work tracked elsewhere in the roadmap and will be filled in as that
work lands.
