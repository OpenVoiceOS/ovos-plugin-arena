# Rating methodology

This page is the normative explanation behind `EloEntry.bt_rating` and its
confidence interval, the primary ranking signal on every leaderboard (§5 of
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
dependency in the headline number undermines the premise, two people
replaying the identical, public vote log by hand in a different order would
get different rankings, with no way to say which one is "right".

**Bradley-Terry**, fit as a batch over the whole vote log at once, does not
have this problem. It asks a single, order-independent question: *what set
of competitor strengths makes the observed vote log most likely?* Shuffle
the votes and refit, you get the identical answer (see
`tests/test_rating.py::TestFitBradleyTerry::test_shuffle_invariant`).

## Why not TrueSkill

TrueSkill (Microsoft, used in Xbox Live matchmaking) is a plausible
alternative, and was considered. It was rejected for two reasons specific to
this arena:

1. **TrueSkill is fundamentally sequential**, it maintains a Gaussian
   belief per player and updates it incrementally per game, exactly the
   property that makes sequential ELO order-dependent in the first place.
   Batch-refitting a TrueSkill model over a full vote log to get an
   order-independent result is possible but loses the closed-form
   incremental-update convenience that is TrueSkill's main selling point, at that point a batch Bradley-Terry fit is simpler and has a much longer
   track record of transparent, reproducible use in exactly this setting
   (this is what [Chatbot Arena / lmarena](https://lmarena.ai) uses for
   ranking LLMs by blind human vote, the closest prior art to this arena).
2. **TrueSkill's uncertainty model (a per-player Gaussian variance) answers
   a different question than the one this arena needs.** This arena wants a
   confidence interval that answers "how much could this specific
   competitor's rank move if a few more human votes came in the other
   direction?", which a nonparametric bootstrap over the actual vote log
   answers directly (see below), without assuming Gaussian belief dynamics.

## The fit: Bradley-Terry via minorization-maximization

Every pairwise vote, a blind battle choice, a free-form vote, or a
benchmark-derived auto vote, is one observation: competitor `a` beat
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
point, which is itself deterministic for a fixed Python/platform, the
result is reproduced byte-for-byte by `arena assemble` on every CI run over
the same vote log).

Strengths are on an arbitrary multiplicative scale (only ratios matter for
Bradley-Terry). `to_rating_scale()` anchors them to a familiar ELO-shaped
number: the geometric mean of the fitted strengths is placed at 1200, with
`400 / ln(10)` points per natural-log unit of strength, the same
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
being undefined, see
`tests/test_rating.py::TestFitBradleyTerry::test_undefeated_and_winless_fighters_converge`.

### Auto vs. human weighting

Auto (benchmark-derived) votes carry `BT_AUTO_WEIGHT = 0.25`, the same §4
R5 intent as sequential ELO's `K/4`, expressed as a pairwise weight instead
of a K-factor. This is capped further at the board level (§4, seed-battle
bias audit) so a large benchmark dataset can never outweigh a modest number
of real human votes. See that section once implemented for the cap
mechanism.

## The confidence interval: bootstrap over human votes only

`bootstrap_confidence_intervals()` resamples the **human vote list only**,
with replacement, `DEFAULT_BOOTSTRAP_ROUNDS = 100` times, using a seeded
`random.Random` (§P5: same seed ⇒ byte-identical CIs on every rerun). Each
round refits Bradley-Terry over (resampled human votes) + (the unchanged
auto-vote seed) and records where every competitor's rating landed. The
reported interval is the 2.5th–97.5th percentile of that distribution, a
standard nonparametric bootstrap 95% CI.

**The auto-vote seed is deliberately never resampled.** It is a
deterministic function of a fixed benchmark corpus (the same reference
audio/text run through the same plugin gives the same metric every time), it has no sampling variability to model. What genuinely varies from one
arena snapshot to the next is *how many human votes have been cast, and by
whom*, that is exactly what resampling the human vote list captures. A
board with zero human votes therefore has every CI collapse to a single
point (the seed-only rating. See
`tests/test_rating.py::TestBootstrapConfidenceIntervals::test_zero_human_votes_collapses_to_seed_point`),
and CIs visibly narrow as more human votes accumulate for a matchup (see
`test_ci_narrows_with_more_votes`).

A board with fewer than `PROVISIONAL_MIN_HUMAN_VOTES = 10` human votes sets
`EloBoard.provisional = true`, its ranking is real (computed the same way)
but should be read as a placeholder ordering from the benchmark seed, not a
settled result.

## Reading a leaderboard

- **`bt_rating`** is the number to look at. Two competitors whose confidence
  intervals overlap are statistically indistinguishable from the current
  vote log, treat them as tied, not as strictly ranked, regardless of what
  order they happen to be listed in.
- **`elo`** is kept for continuity with earlier snapshots of this arena but
  is not the ranking key and can disagree with `bt_rating`'s ordering,
  particularly early in a board's life or after a burst of lopsided votes, that disagreement is expected and is exactly the order-dependency problem
  `bt_rating` was built to avoid.
- **`provisional`** boards should be captioned as such in the frontend
  rather than presented with the same confidence as an established board.

## Dataset sampling policy

A benchmark board is only comparable across fighters if every fighter is
scored against the same rows. Some eval corpora are far larger than a sweep
needs — streaming a whole large corpus every run wastes compute without
improving the estimate, and letting each sweep's sample count depend on
whatever `--max-samples` an operator happened to type that day makes runs
incomparable to each other even for the same fighter. A large dataset's
registry entry declares a `sample_policy`: a row cap and a seed that pin one
deterministic subset, drawn the same way every time a sweep streams that
dataset. Datasets small enough to score in full carry no policy and stream
unrestricted.

Declaring a policy only fixes which rows a sweep *would* draw — it does not
by itself make two already-swept fighters comparable. A fighter swept before
a dataset had a policy, one swept against the policy, and one swept with a
smaller ad hoc `--max-samples` can each hold a different subset of the
corpus's rows, and a board built straight from those predictions would rank
them against each other anyway, silently mixing sample populations. Closing
that gap is a separate, explicit step: publishing the policy's selected row
ids as a manifest (`sample_sets/<lang>.json`, alongside a dataset's
predictions) that board assembly downloads and filters every fighter's rows
against before scoring. A fighter is only ranked once its rows cover most of
the manifest; one that covers too little is marked unranked instead of
folded into the ranking on an incomplete or mismatched sample. A dataset
whose policy exists but has no published manifest yet falls back to scoring
whatever rows each fighter happens to have, flagged as an unmanaged sample
set rather than treated as comparable.

## Per-metric ladders

A league's primary metric is not the only number worth ranking on. A TTS
board scores UTMOS (naturalness), but also SIGMOS noise/coloration/
discontinuity and DNSMOS background-noise — a listener who cares
specifically about background noise gets no answer from "who ranks best
overall". Every `leaderboard-<modality>-<lang>.json` board carries a
`metric_ladders` map, keyed by metric name, so every row-level metric a
league scores is independently browsable as its own full Bradley-Terry
ranking, not just a column in the benchmark table.

**Primary ladder vs. secondary ladders — the one real distinction:**

- **The primary ladder** (`metric_ladders[<primary_metric>]`,
  `auto_only: false`) is the league's main ladder — identical to the
  board's own `entries` above. It is seeded from benchmark auto-battles
  and then refined by human votes, same as always.
- **Every other ladder** (`auto_only: true`) is fit from auto-battles
  only. A pairwise comparison for one of these metrics is derived the same
  way the primary metric's auto-battle seed is (§4 R5: same-sample,
  same-competitor-pair, "whoever scored better on this metric wins the
  auto-battle"), but human votes never touch it — voters cast a vote on
  the overall league, not on "which one had less background noise", so
  there is no vote log to replay for a secondary metric and none of these
  ladders is ever "provisional" the way the primary ladder can be. It is
  either ranked from its auto-battles or, with too few comparisons for a
  connected ranking, effectively tied at the anchor rating.

**Which metrics get a ladder.** Only metrics with a genuine per-row value —
one number computed per sample, per competitor, so two competitors' rows on
the same sample can be compared head-to-head. Dataset-aggregate-only
metrics (ECE, macro-F1, OOD false-positive rate, latency percentiles) have
no such per-row value and are never ladderable: there is no sample-level
"A beat B on this metric" signal to build a battle from. Concretely, today:

| League | Ladders |
| --- | --- |
| `intent` / `intent_template` / `intent_keyword` | `accuracy` (primary), `slot_exact_match` (rows with gold slots and a correct intent) |
| `stt` | `wer_mean` (primary) — no per-row CER is computed yet, so WER is the only ladder |
| `tts` | `utmos` (primary), every SIGMOS/DNSMOS/NISQA quality dimension (`sigmos.noise`, `sigmos.col`, `sigmos.disc`, `sigmos.loud`, `sigmos.reverb`, `sigmos.sig`, `sigmos.ovrl`, `dnsmos.sig`, `dnsmos.bak`, `dnsmos.ovrl`, `nisqa.*`) |
| `wake_word` / `vad` | `error_rate` (primary) only — no other per-row metric exists yet |

**Where this lives on disk.** Deliberately not a new artifact family. The
per-metric auto-battle seeds are nested inside the existing
`elo-seed-<modality>-<lang>.json` (`EloSeed.secondary_metrics`), and the
fitted ladders are nested inside the existing
`leaderboard-<modality>-<lang>.json` (`EloBoard.metric_ladders`) —
one file per board, same as before this feature, so a prune guard reasoning
about artifact prefixes never has to learn a new one.

**Performance.** Computing a Bradley-Terry seed per extra metric multiplies
the per-sample comparison work by however many metrics a league has. This
is deliberately done in a single additional pass over the same sample data
the primary seed already visits — every (sample, competitor-pair) is
compared on every ladderable metric in the same inner loop, rather than
re-looping the dataset once per metric — and skips the legacy sequential-
ELO rating update entirely (secondary ladders never read it), which turned
out to be roughly half the per-comparison cost. On a synthetic stress case
(60 fighters × 300 samples, TTS's full 15-dimension quality-metric set —
larger than any league's current real roster), that pass took ~39s versus
~11s for the existing primary-metric seed alone; real TTS rosters today are
smaller. If a future roster's scale makes this a bottleneck, the sample set
can be capped or subsampled per metric without changing the artifact
shape.

## Per-league ELO pools and cross-league vote replay (§4 R18)

Every modality, including all three intent leagues, `intent`,
`intent_template` and `intent_keyword`, runs its own battle pool and its
own ELO ladder. `arena.models.battle_group()` is an identity mapping: a
league's battle group is itself, so `arena.assembler` only ever pairs two
fighters that already share a league. A template-paradigm engine (e.g.
Padatious) is never blind-battled against a keyword-paradigm engine (e.g.
Adapt), and neither paradigm-pure league pools with the open `intent`
fusion league. This mirrors the leagues already being paradigm-pure on the
benchmark side (§2.1), battles and ELO were the one place paradigms used
to mix, and now don't.

A single-stage embedding classifier (Model2Vec, Hierarchical KNN) is not a
fourth league. It is a *strategy*, trained from one of the two intent
training-data formats, both shipped competitors here train from
template-paradigm corpora (`runner/intent_pipeline.py`'s `EngineSpec.paradigm
== "template"` for both `ovos-m2v-pipeline` and
`ovos-hierarchical-knn-pipeline`), so both live in `registry/competitors/
intent_template/` and compete in the `intent_template` league, same as any
other template engine.

**Historical votes and the league split.** Before this split, all three
intent leagues shared one `intent` battle pool and one ELO ladder, a
battle's id was a content hash of `(battle_group, dataset, lang, sample,
sorted(competitor_a, competitor_b))`, and `battle_group` used to collapse
all three modalities to the literal string `"intent"`. After the split,
`battle_group` no longer collapses anything, so every battle id computed
from here on is scoped to its own league and cannot collide with, or be
confused with, a battle id from a different league.

Any vote already cast against an old, pre-split battle id is handled by the
tally pipeline's existing "battle not in the current pool" check
(`arena/cli.py:cmd_tally`): once the committed battles pools are
regenerated post-split, a pre-split battle id (which necessarily hashed the
collapsed `"intent"` group, not a real per-league group) is absent from
every per-league pool, the fighters it paired may have ended up in the
same league or in different ones, but the *id itself* no longer resolves
anywhere, so the vote is uniformly discarded and reported in
`vote-audit.json` rather than silently dropped or, worse, misattributed to
whichever league happens to load first. This is deliberately conservative:
it never lets a genuinely cross-league historical vote leak into a
post-split league's rating, at the cost of not being able to salvage
same-league historical votes that happened to be cast under the old shared
pool. In this arena's committed vote log that cost is zero, no human vote
has ever been counted yet (`vote-audit.json` has `"counted": 0`), so the
split is a clean cut, not a data-loss event.

Going forward, the same mechanism is what enforces the policy on paper: a
vote counts toward league X's replay if and only if its battle id is
present in league X's currently committed battles pool, and, because
matchmaking never mixes leagues, every battle in that pool necessarily
pairs two league-X fighters. The check is a pure function of the vote log
plus the currently committed battles pools, which are themselves a pure
function of the registry and published predictions (P5), so replay stays
deterministic and network-free per league, exactly like the rest of
`tally`.

## Benchmark boards: significance, beyond a point estimate

The objective benchmark boards (`benchmark-<modality>-<dataset>-<lang>.json`,
straight from prediction rows, no votes involved) carry the same discipline.
A 0.3% WER gap on 500 clips, or a 96% vs 95% intent accuracy gap on a few
hundred samples, is very often noise rather than a real capability
difference. Showing only the point estimate implies more precision than the
sample size supports.

Every `BenchmarkEntry` carries a seeded bootstrap 95% CI
(`arena/metrics.py:primary_metric_ci`, `BOOTSTRAP_ROUNDS = 1000`) on its
primary metric, using one of two strategies depending on what kind of number
the metric is:

- **Mean of a per-row indicator** (intent `accuracy`, wake-word/VAD
  `error_rate`), bootstrap the 0/1 indicator list directly and report the
  percentile interval of the resampled mean.
- **Ratio of summed counts** (STT `wer_mean` = total word errors / total
  reference words), a per-utterance WER is not directly comparable across
  utterances of different length, so this bootstraps **(errors,
  reference-word-count) pairs**, recomputing `sum(errors) / sum(ref_words)`
  each round, rather than averaging per-utterance WER values as if they were
  i.i.d., a one-word command with one error and a twenty-word sentence with
  one error are not the same signal, and the ratio bootstrap weights them
  correctly by how much reference text they actually contain (see
  `tests/test_metrics_ci.py::test_weighted_by_denominator_not_per_pair_average`).

`BenchmarkEntry.tied_with_leader` marks every entry whose CI overlaps the
#1-ranked entry's CI, the frontend should render those as "≈ tied with #1"
rather than a strict rank ordering, and prefer showing a compact tier
grouping over a false-precision numeric rank for entries that are
statistically indistinguishable from the leader.

## Seed-battle bias audit

A benchmark dataset can have thousands of samples. Without a check, its
auto-battles would both (a) dwarf a modest number of real human votes, and
(b) manufacture apparent rating separation out of per-sample noise even
when the two competitors' overall performance is not meaningfully
different. Two independent controls in `arena/assembler.py:seed_elo`:

- **Significance gate** (`pair_metric_significant`, using the same bootstrap
  CIs as the benchmark boards, see above). Before any per-sample
  auto-battle is generated for a competitor pair, their *aggregate* primary
  metric confidence intervals for the dataset are compared. If the
  intervals overlap, the pair contributes **zero** auto-battles for that
  dataset, individual samples may still disagree, but that disagreement is
  benchmark noise, not a real capability gap, so it must not seed the
  rating. A pair with a genuinely wide gap (non-overlapping CIs) seeds
  normally.
- **Weight cap** (`MAX_AUTO_WEIGHT_PER_PAIR = 5.0`). Even a pair that clears
  the significance gate has its total Bradley-Terry pairwise weight capped
  at the equivalent of 5 human votes, scaled down proportionally so the
  observed win rate is preserved. A benchmark run with 10,000 samples and a
  benchmark run with 250 samples contribute the same maximum seed weight to
  a pair once both clear the significance gate, dataset size stops being a
  lever on how much the rating can move. This cap applies only to the
  Bradley-Terry pairwise statistics. It does not change the legacy
  sequential ELO's `auto_vote_count` bookkeeping.

Run `ovos-arena audit-seeds --data-dir <path>` to see, per (modality, lang),
every scored pair's weight and whether it sits at the cap.

## Vote fraud / dedup resistance

The vote log is public (§6: "the vote log **is** the issue history"), and
a public, easy-to-participate voting mechanism is a predictable target for
brigading and low-effort automated voting. `arena/fraud.py` applies a
sequence of deterministic rules, none of which deletes anything: every rule
records a `discarded_reason` or a reduced `weight` rather than silently
dropping a vote, and the full audit trail is written to `vote-audit.json`
alongside the leaderboards on every tally run.

- **One vote per (voter, battle).** Handled upstream by
  `arena.cli.dedupe_votes`, keyed on `(author, battle_id)`, since
  `battle_id` already encodes `(dataset, sample, competitor pair)` (§4 R4),
  this is exactly "one vote per voter per battle-pair-on-a-dataset-entry".
- **Per-voter, per-league, per-day cap** (`DAILY_VOTE_CAP = 50`,
  `apply_daily_cap`). Votes are processed in their deterministic order
  (issue number ascending). Once a voter passes the cap for a given
  modality on a given UTC calendar day, further votes that day are
  discarded with reason `daily_vote_cap_exceeded`. The cap is per
  modality, not global, so a voter legitimately evaluating multiple
  leagues in one sitting is not penalized.
- **Account-age gate** (`NEW_ACCOUNT_MIN_DAYS = 7`,
  `apply_account_age_gate`). A vote from an account created less than 7
  days before the vote is discarded with reason `account_too_new`. The
  account creation timestamp is fetched from the GitHub API **once per
  author** and persisted to `voter-age-cache.json` in the committed data
  directory, every subsequent tally run reuses the cached value instead of
  re-fetching, so the pure `resolve_vote_weights` replay never touches the
  network (`tests/test_fraud.py::test_pure_no_network` enforces this
  structurally, and `docs/SPECIFICATION.md` §4 requires it for §P5).
- **One-sided voter down-weight** (`ONE_SIDED_MIN_VOTES = 20`,
  `ONE_SIDED_THRESHOLD = 0.95`, `apply_one_sided_downweight`). A voter whose
  surviving votes are more than 95% for the same literal **A** or **B**
  side across at least 20 votes has every one of those votes down-weighted
  to 0.5 rather than discarded outright. This is deliberately keyed on the
  literal A/B choice, not competitor identity, blind battles randomize
  which competitor is shown as "A" per battle from the battle-id hash (§4
  R4), so "always clicks the left button" is the low-effort/bot-like
  signal. "always prefers competitor X" is not detectable this way and
  would be indistinguishable from a voter with a genuine, consistent
  preference.

**Weighting only affects the Bradley-Terry rating, not the legacy
sequential ELO column.** A down-weighted vote (weight 0.5) still updates
`EloEntry.elo` at full strength, that column is secondary/display-only
(see "Reading a leaderboard" above) and does not need this level of rigor.
A fully discarded vote (weight 0) is excluded from both ratings entirely
and is never passed into `build_elo_board`'s human-vote list, it exists
only in the `vote-audit.json` record.

**The full vote-issue history is refetched every run**
(`fetch_vote_issues` lists both open and closed `vote`-labelled issues), so
every tally run genuinely replays the complete log from scratch rather than
only the issues opened since the last run, the earlier design fetched
`--state open` only, which meant closing a processed issue silently
dropped its vote from every future leaderboard rebuild. Already-closed
issues are never re-commented-on or re-closed. `arena.cli.cmd_tally` only
takes GitHub actions (comment + close) on issues that are still `OPEN` in
the freshly-fetched list.

## Objective TTS scoring: UTMOS (§4 R14)

TTS has no ground-truth reference, there is no single "correct" waveform
for a prompt, so **blind human A/B votes remain the league's primary
ranking signal**, exactly as for every other league's Bradley-Terry rating.
Alongside those votes, `runner/tts_bench.py` scores every synthesised clip
with **UTMOS**, a reference-free (no reference recording needed) naturalness
MOS predictor from the [`speechonnxmetrics`](https://pypi.org/project/speechonnxmetrics/)
package (`speechonnxmetrics.mos.utmos.UTMOS`), on the same 1-5 scale a human
MOS rater would use, higher is better.

**Provenance.** Each scored row records the judge's identity and pinned
revision in `extras`:

- `utmos`, the clip's score (float, 1.0-5.0). - `utmos_judge`, `"TigreGotico/utmos-onnx"`, the HF repo the ONNX judge
  model came from. - `utmos_judge_revision`, the pinned commit the arena has validated against
  (`ff41b8f440cb12ecda18261f9ff7326d058275ce`).

A judge upgrade is a *new* revision, not a silent swap, comparing scores
across judge revisions is not sound, so the revision travels with every row
rather than being assumed from the package version alone.

**What it feeds.** `arena/metrics.py:score_tts` aggregates per-row `utmos`
into the `tts` benchmark board (mean, with the same seeded-bootstrap 95% CI
every other board's primary metric gets, §4 R11). Rows a clip failed to
score for are excluded from the mean, and the board still reports
`n_scored` so a partial run does not read as a full one. `arena/assembler.py`
feeds a significantly-higher-UTMOS clip into the same benchmark-seeded ELO
machinery as every other league (§4 R5/R5a/R5b), significance-gated,
per-pair weight capped, so a large synthetic benchmark run can never drown
out a modest number of real human votes.

**Known biases, read before trusting a UTMOS gap across languages.**
UTMOS-style MOS predictors are themselves trained models with their own
distributional blind spots:

- The public UTMOS training data skews toward 16 kHz, English-adjacent,
  studio-clean recordings (VoiceMOS-challenge-style corpora). A judge
  trained mostly on that distribution tends to reward *clean, familiar-
  sounding* speech and can under- or over-score accented, non-English, or
  differently-recorded material for reasons that have nothing to do with
  how natural the speech actually sounds to a listener of that language.
- **Cross-language UTMOS comparisons are not sound.** A `pt-PT` fighter
  scoring lower than an `en-US` fighter on this metric is not evidence the
  Portuguese voice is worse, it may just be further from the judge's
  training distribution. This is why boards stay **per-language**
  (`benchmark-tts-<lang>.json`) rather than pooling UTMOS across languages
  into one global TTS number. A UTMOS gap is only meaningful *within* the
  same `(dataset, lang)` board, between fighters the judge is equally
  (un)familiar with.
- Because of the above, UTMOS is deliberately a **secondary, objective
  cross-check**, not a replacement for human votes, a synthetic-MOS
  regression is a signal to go listen, not a verdict on its own.

## Objective TTS scoring: intelligibility (§4 R16)

Alongside UTMOS, `runner/tts_bench.py` scores every synthesised clip for
**intelligibility**: it transcribes the rendered clip back to text with a
pinned STT judge, the best offline `onnx-asr` model for the clip's
language (usually a conformer), resolved by `runner/asr_judges.py` from
ovos-config's offline-STT recommends with a per-language fallback table,
never faster-whisper, and scores the transcript against the
original prompt with the same canonical WER (and a companion CER) the STT
league uses (`arena.metrics.normalize_transcript`, §E), this is the
"round trip" check: can a listener's own ears/ASR actually recover the words
the fighter was asked to say, as distinct from how *natural* it sounded
(UTMOS's job).

**Provenance.** Each scored row records the judge's identity and pinned
revision in `extras`, mirroring the UTMOS convention:

- `intelligibility_wer` / `intelligibility_cer`, word/character error rate
  of the STT judge's transcript against the prompt text (§E normalization,
  lower is better). - `intelligibility_judge`, the per-language onnx-asr judge model id
  (e.g. `nemo-parakeet-tdt-0.6b-v3` for en-US). - `intelligibility_judge_revision`, the HF commit sha pinned at authoring
  time, recorded for provenance (`onnx_asr.load_model` cannot pin a
  revision at load time, see `runner/asr_judges.py`).

**Pitfalls this metric is designed around.** These are production failure
modes, not hypotheticals, each one has a dedicated regression test in
`tests/test_tts_bench.py`:

- **Synthesis must go through the direct `get_tts` path**, never an
  audio-player/playback round trip, playback adds device/driver noise the
  metric has no business measuring.
- **A synthesis crash must still produce a row.** `predict()` catches the
  exception itself (rather than letting it propagate into
  `runner/media_bench.py`'s per-sample handler, which just skips the
  sample) and records `intelligibility_wer = intelligibility_cer = 1.0`
  plus a `synthesis_error` note, a crashing fighter shows up as maximally
  unintelligible on the board, not as a shrinking sample count nobody
  notices.
- **Non-wav output must be transcoded, never read as raw PCM.** Some
  plugins render mp3/opus. Reading those bytes as if they were headerless
  PCM samples produces garbage audio and a meaningless score with no error
  to flag it. The clip is decoded through `runner.audio_io.decode_audio_bytes`
  (soundfile, falling back to `av`), the same decoder the STT/wake-word
  benchmarks use.
- **Audio must always be resampled to 16 kHz mono before transcription.**
  Reading a 44.1 kHz (or stereo) file as if it were already 16 kHz mono is
  a known footgun: the judge effectively hears a stretched, garbled clip
  and returns a false ~1.7 WER, silent corruption, not a crash.
  `decode_audio_bytes` always resamples/downmixes regardless of the source
  container.

**What it feeds.** `arena/metrics.py:score_tts` aggregates
`intelligibility_wer` into the `tts` benchmark board as a **secondary**
metric (mean, with the same seeded-bootstrap 95% CI convention as every
other board metric, §4 R11). UTMOS remains the board's primary metric.

**Warn-only across languages.** Like UTMOS, the STT judge is itself a model
with its own blind spots, it transcribes some low-resource languages far
worse than others. A high `intelligibility_wer` for such a language is not
necessarily evidence the TTS clip was unintelligible to a human listener of
that language. It may just be the judge mishearing it. The real score is
always recorded (never suppressed or clamped) and always reported, but it
never gates a board or blocks a benchmark run, it is a signal to go listen,
exactly like a UTMOS regression.

**UTMOS is the primary TTS quality judge.** Its ONNX export (via
`speechonnxmetrics`) carries an MIT license, and it is fast to run per clip,
which is why it drives the objective TTS board and its benchmark-seeded ELO
votes.

**The arena also runs SIGMOS and NISQA for a finer per-dimension quality
breakdown** (noise, coloration, discontinuity, loudness, reverberation, …).
**SIGMOS** (ITU-T P.804, MIT-licensed Microsoft weights) provides the
headline dimensions surfaced as board columns: `noise`, `col`
(coloration), `disc` (discontinuity), plus `loud`, `reverb`, `sig` and
`ovrl` in the full row data. **NISQA-v2** is recorded alongside it as a
complementary predictor, adding a second, independently-trained opinion on
the same style of dimensions (`mos`, `noi`, `dis`, `col`, `loud`).

NISQA's released weights are CC BY-NC-SA 4.0 (non-commercial). This arena
is a non-commercial project of the OpenVoiceOS non-profit (registered in
the Netherlands), so that license is compatible with how this project uses
it — unlike a for-profit fork or a commercial redistribution of the board
data, which would need to drop NISQA or replace it with an MIT-licensed
alternative first. **DNSMOS** (ITU-T P.835, MIT-licensed) runs alongside
both, predicting signal/background-noise/overall quality.

All three families (SIGMOS, DNSMOS, NISQA) land in full on every scored
row's `extras`; only a curated subset (`sigmos.noise`, `sigmos.col`,
`sigmos.disc`, `dnsmos.bak`) is surfaced as headline columns on the TTS
benchmark board, to keep it from growing unreadably wide. Running four MOS
judges per rendered clip (UTMOS, SIGMOS, DNSMOS, NISQA) roughly
quadruples the per-sample inference cost of TTS scoring relative to UTMOS
alone — acceptable for this arena's batch benchmark runs, which are not
latency-sensitive.

**Running five judges on the same clip raises an obvious question: do they
even agree with each other?** Every TTS benchmark board carries a judge
agreement panel that answers this directly, rather than leaving a reader to
eyeball five separate columns and guess. For each pair of judges (UTMOS,
SIGMOS, DNSMOS, NISQA, and intelligibility, the last inverted so higher is
always better before comparing), the board ranks the fighters by each
judge's mean score and computes the Spearman rank correlation between the
two rankings — implemented directly with numpy rather than pulling in
scipy, since ranking with average ranks for ties and then taking a Pearson
correlation of the rank vectors is the entire algorithm. A rho near 1 means
the two judges rank the fighters the same way; near 0 means they disagree;
near -1 means they rank them in opposite order. Alongside the matrix, each
judge's own top-5 fighters are listed side by side, so a reader can see at
a glance whether the same voices keep coming up regardless of who is
judging, or whether the leaderboard's shape depends on which single judge
happens to be primary. The panel only appears once a board has at least
three scored fighters, because a correlation computed over one or two
points is not a correlation, it is noise dressed up as a number. Even a
strong rho does not mean the judges are individually correct: five models
can agree with each other while sharing the same blind spot relative to a
human listener, and rank agreement says nothing about whether any of them
tracks human perception well in an underrepresented language or accent.
High agreement is evidence a board's ranking is not an artifact of one
judge's idiosyncrasies; it is not evidence the ranking is right.

## Performance data on prediction rows (§3.2, M1)

Every row a benchmark script writes now carries a small, optional set of
performance columns, in addition to `latency_ms` (see §3.2 of
`SPECIFICATION.md` for the exact field list): `elapsed_ms` (wall time of the
single inference call), `peak_rss_mb` (a rough process RSS sample, not a
true per-call peak), `audio_secs` (input clip duration for STT/wake word,
produced clip duration for TTS — RTF is `elapsed_ms / 1000 / audio_secs`)
and `hw` (a hardware fingerprint captured once per run and stamped on every
row of it: CPU model, thread count, accelerator if any, host class,
hostname).

Two things this is deliberately *not*:

- **Not a profiler.** `peak_rss_mb` is bracketed by one RSS sample before
  and one after the call (`runner.perf.measure_call`) — a spike that rises
  and falls entirely inside the call is invisible, and the number is
  process-wide, so it is not attributable to the one call alone. It is a
  coarse, per-row signal for spotting gross regressions across runs, not a
  memory profile.
- **Not a per-battle vote signal.** RTF/peak-memory/model-size never feed
  auto-battle outcomes or the ELO ladder (§ per-metric ladders above
  covers *quality* ladders only) — they are benchmark-board columns and the
  Pareto view below, not a rating axis.

**Backward compatibility is load-bearing here.** The overwhelming majority
of already-published prediction rows predate this capture. `arena.models.
PredictionRow` defaults every one of these fields to `None`/absent, and
`arena.predictions.parse_row` never requires them — an old row loads
exactly as it did before this columns existed, just without a value for
RTF.

## Performance boards: RTF, peak memory, model size, per hardware tier (M2)

`arena.metrics.perf_metrics_by_tier` turns the raw per-row columns above
into board-ready aggregates, one independent set per hardware tier:

- **RTF is aggregated as the MEDIAN**, not the mean. The very first call
  against a fighter is routinely a cold model load — an order of magnitude
  slower than every call after it — and a mean lets that single outlier
  drag the whole board number away from steady-state speed. A median is
  robust to that in a way an arithmetic mean isn't; `peak_rss_mb`, by
  contrast, is aggregated as the **MAX** across a tier's rows, because the
  number a deployer actually cares about is the worst-case memory the
  process ever needed, not an average that hides the peak.
- **Hardware tiers are never blended.** Every aggregate is computed
  per-`hw["host_class"]` (e.g. `cpu-x86`, `gpu`) and a fighter benched on
  more than one tier gets one independent entry per tier — a board cell
  shows whichever tier has the most samples, badged with that tier's name,
  never a number silently averaged across a CPU box and a GPU box. Old
  rows without `hw` (pre-#90) are simply excluded from these aggregates —
  a missing tier entry, not a zero, since "no measurement" and "instant and
  free" are different claims.
- **Model download size is not a per-row aggregate at all.** It is looked
  up ONCE per fighter from the model's HuggingFace repo metadata
  (`arena.model_size.model_repo_size_mb`, summing every sibling file's
  size) during `assemble`/`export-bestiary`, using the fighter's
  `model_hf_repo` registry field, and cached for the life of that build
  process. Fighters with no `model_hf_repo` (rule-based engines, sklearn
  pipelines shipped as plugin code, …) get `model_mb = None` on their board
  entry — never a fabricated `0`, which would misleadingly read as "no
  download at all".
- RTF is additionally **ladderable**: for stt/tts/wake_word (the
  modalities that bench against a labelled audio clip with a known
  duration), `row_rtf` is a genuine per-row value, so it gets its own
  auto-only secondary BT ladder exactly like `wer_mean` or `utmos` (see
  "Per-metric ladders" above) — never blended across hardware tiers there
  either, since the ladder is fit from same-sample, same-run comparisons.

### Pareto/efficiency view

Each league+language's benchmark board carries a "Pareto frontier: best
quality per compute" table beside the ranked list: fighter A is dominated
by fighter B when B is at least as good on *both* the primary quality
metric and RTF, and strictly better on at least one — there is then no
reason to ever pick A over B. The frontier is every fighter nobody
dominates, sorted fastest-first; the dominated count is shown so a reader
knows how much of the board was excluded. This is deliberately a plain
ranked table, not a scatter-plot library — the quality/compute trade-off
only needs two numbers per row to read.

---
[← Leagues](leagues.md) · [Home](index.md) · [Operations →](operations.md)
