# OVOS Plugin Arena, Specification

**Status:** Active, maintained by TigreGotico
**Version:** 0.4

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be
interpreted as described in RFC 2119.

---

## 1. Purpose

The Plugin Arena answers one recurring question for the OVOS ecosystem:
*"which plugin should I use?"* It does so with two complementary signals:

1. **Benchmarks**, plugin predictions evaluated against labelled datasets
   (generalization accuracy/F1 for intent, WER for STT, detection metrics
   for wake word),
   published openly and reproducibly.
2. **Human preference**, chess-style ELO ratings produced by blind A/B
   "battles" where users pick the better of two plugin outputs. The initial
   ELO is derived from the benchmarks. Human votes refine it.

The arena is the *rating and voting* venue. It is **not** an execution venue.

## 2. Design principles

- **P1, GitHub-native, zero servers.** The arena is a repository: data
  artifacts are JSON files committed by scheduled Actions, the UI is a static
  site on GitHub Pages, and votes are GitHub issues. There is no backend, no
  database, and no arena account system. A FastAPI shim MAY exist as a local
  *development tool* only. It MUST NOT become a deployment target.
- **P2, Predictions are decentralized. HF is the artifact layer.** Plugins
  run *outside* the arena as offline batch jobs. Every prediction run is
  published as a HuggingFace dataset. The arena MUST NOT install, isolate, or
  execute OVOS plugins in CI.
- **P3, Everything is declarative.** Every competitor is a JSON file
  (`registry/competitors/<modality>/<id>.json`). Every dataset is a JSON file
  (`registry/datasets/<modality>/<id>.json`). Forking the repo and editing
  JSON yields a working arena.
- **P4, Every benchmark is a reproducible script.** One dedicated Python
  script per benchmark under `benchmarks/`, tracked in git. It trains each
  registry competitor, produces fresh predictions (never copied from other
  sources), records the pinned dataset revision and plugin versions in every
  row, and uploads to HF.
- **P5, Ratings are replayable.** Battle ids are content hashes. The ELO
  seed is a deterministic function of the published predictions. Human votes
  replay in issue-number order. The full standings MUST be reproducible from
  public data alone.
- **P6, All prediction data is kept**, including bad predictions. Failure
  cases guide plugin improvements and count as real benchmark content, not
  as noise to discard.

## 2.1 Leagues (modalities)

Every modality is an independent league with its own benchmarks, battle
pools and ELO standings: `stt`, `tts`, `wake_word`, and **four intent
leagues**. Intent engines are separated by what they need before they can
answer at all — a fighter handed the skill's phrasings and expected to answer
immediately is doing a different job from one that trains on them at boot,
and both differ from one shipping a model trained elsewhere. Keyword engines
consume hand-written vocabulary rules rather than phrase templates, which is
a different kind of supervision again. Engines from different leagues MUST
NOT be ranked against each other:

| League | Who competes |
|---|---|
| `intent_zero_shot` | engines that answer straight from the registered templates, with no training step |
| `intent_online` | engines that train on those templates when the device boots (Padatious, Padacioso, Nebulento, …) |
| `intent_offline` | engines carrying a pretrained artefact that never sees the device's templates |
| `intent_keyword` | keyword engines (Adapt, Palavreado, …) |

A fighter's league follows from its stages and MUST match: a fusion belongs
to its heaviest stage's league, and a fighter built only from keyword engines
is a keyword fighter. Fusions carry fun portmanteau names (Padapt =
Padatious × Adapt).

An offline fighter MUST declare the `label_set` its artefact was trained on
— the dataset_ids whose labels it can emit. It is benchmarked only on those
corpora; anywhere else its answers measure a label-space mismatch rather than
the engine. The declared claim is checked, not trusted: before scoring, the
runner intersects the loaded model's own class list with the corpus's
labels, and a fighter whose model classes reach no overlap with a corpus
in its own declared `label_set` is unranked there rather than scored on a
board of zeros. That gate is global and binary — zero overlap versus any
overlap — and identical in every league; there is no per-league floor. Each
scored row records the class overlap the fighter actually measured.

The per-league task definitions and the exact metric formulas (what each
benchmark board ranks by and what seeds ELO) are specified in
[`docs/leagues.md`](leagues.md).

## 3. System components

```
registry/*.json ──► benchmarks/<bench>.py ──► HF dataset
                                              predictions/<competitor_id>.jsonl
                                                   │ assemble.yml (daily)
                                                   ▼
                      frontend-static/public/data/*.json
                      battles · benchmark boards · elo seeds · leaderboards
                                                   │ Astro build (pages.yml)
                                                   ▼
                      GitHub Pages  ◄── votes as GitHub issues
                                                   │ tally.yml (hourly)
                                                   ▼
                      ELO replay → leaderboard JSON → commit → redeploy
```

### 3.1 Declarative registry

**Competitors** (`registry/competitors/<modality>/<id>.json`): a
*configuration you could ship*. For the intent modality, `config` MUST be a
valid `mycroft.conf` fragment, an `intents` section with an ordered
`pipeline` list of `<plugin>-<tier>` stages plus per-plugin config blocks:

```json
{
  "competitor_id": "padatious-medium",
  "modality": "intent",
  "config": {
    "intents": {
      "pipeline": ["ovos-padatious-pipeline-plugin-medium"],
      "ovos-padatious-pipeline-plugin": {}
    }
  }
}
```

A single-stage pipeline benchmarks one engine. A multi-stage pipeline is an
**ensemble** fighter in its own right (e.g. the stock OVOS cascade), first
stage whose own confidence gate fires wins, exactly like ovos-core. The same
plugin under a different config is a *different competitor*. `competitor_id`
is the stable key for predictions, battles, ELO and leaderboards. Bestiary
card fields describe the fighter for the UI: `display_name`, `species` (the
plugin class it instantiates), `types` (architecture tags: `GOFAI`,
`fuzzy-match`, `neural-net`, `template-match`, `keyword-match`, `embedding`,
`LLM`, `ensemble`), `description`, `model`, `links`.

`trained_on` lists every dataset id whose recordings were in this
competitor's training data — the deny-list mirror of `train_datasets`
below: a fighter stays registered and competes normally everywhere else,
but every scoring point (the media/wake-word bench,
`arena.predictions.group_rows`, the autorun scheduler) skips the
`(fighter, dataset)` pair instead of grading it, so a fighter is never
scored on data it was trained on. A corpus republished under a different
dataset id is a different id as far as this list is concerned — it is not
covered by an entry naming the original id, so the registrant MUST list
every id the training recordings were published under, including
republications. `validate_registry` rejects a `trained_on` id that does not
resolve to a registered dataset for the competitor's own modality.

**Datasets** (`registry/datasets/<modality>/<id>.json`): one corpus per
entry, source (HF id + revision + split or per-lang `file_pattern`),
`reference_fields` (the datashape contract), license, `lang` (or
`lang: multi` plus a `langs` list), and a `role`. Every `lang` tag MUST be a
full BCP-47 lang-REGION tag; a bare primary subtag is accepted only for a
language on `registry/schemas.py`'s allow-list, because none of the regions
it is spoken in is a dialect the corpora distinguish. That allow-list is
closed: adding a language to it is an owner ruling recorded in the schema
file itself, not a convenience a registrant can reach for by leaving off a
region. `display_name` is the
human-readable corpus name the leaderboard shows in place of the code-name
`dataset_id`, and `summary` is the plain-language paragraph that tells a
visitor where the data comes from, what one row is and what changes how to
read the score; every `role: eval` corpus MUST carry both. **Keyword-paradigm and
template-paradigm training corpora are different datasets with different
datashapes**: each gets its own `role: train` entry tagged with `paradigm`
(`template` rows carry `{slot}` phrase templates with example values. `keyword` rows carry complete Adapt-style `required_vocab`/`optional_vocab`
rules). A `role: eval` corpus links its paradigm-specific training sets via
`train_datasets`, and every stage plugin trains from the corpus matching its
paradigm.

Both `CompetitorDef` and `DatasetDef` are **closed schemas** (unknown keys
MUST be rejected, not silently ignored) and carry a `schema_version` field
(currently `1`) marking the entry's shape. `ovos-arena validate-registry`
strictly validates every `registry/**/*.json` file against these schemas
and exits non-zero on the first failure, a typo'd field name (e.g.
`"revison"`) or a value of the wrong type is a registry bug and MUST fail
CI, not degrade to a runtime warning. Runtime loaders (`list_competitors`,
`list_datasets`, …) keep their warn-and-skip behavior for resilience
against a stray bad file. `validate_registry` is the strict gate.

A `DatasetDef` MAY pin `predictions_revision`, an immutable HF commit SHA
for its `predictions_hf` predictions repo. `assemble` uses this pin (falling
back to `--revision`, typically the floating `main` branch, when unset),
resolves whichever revision it ends up with to a concrete commit SHA before
fetching, and records the resolved `{repo: sha}` mapping on the generated
`benchmark-*.json` boards (`predictions_revisions`) and at the top level of
`index.json`, so a benchmark board's provenance is always a fixed commit,
and a third party can re-fetch the exact predictions that produced any row.

### 3.2 Prediction contract

Predictions live in HF dataset repos, **one dedicated repo per benchmark
modality**, named `ovos-<modality>-bench-<dataset_id>` (league underscores
dashed), with per-competitor JSON lines at
`predictions/<lang>/<competitor_id>.jsonl` and a generated dataset card
declaring **one HF split per language**. One row per (language, sample).

Minimum columns, all modalities: `competitor_id`, `sample_id`, `dataset_id`,
`lang`, `plugin_id`, `plugin_version`, `prediction`, `runner_version`,
`created_at`. Reproducibility columns SHOULD include `dataset_revision`.
`schema_version` (default 2) marks provenance, 1 identifies a row
converted from the legacy `STTRow` column layout at load time (§4 A2). New
rows are always written directly in this shape (`competitor_id` MAY be
absent when the writer has no registry dependency, e.g. the off-repo STT
runner, `arena.predictions` re-keys via `plugin_id` and
`registry.loaders.get_competitor_by_alias`).

Per modality:

- **Intent**: `utterance`, `reference_intent` (null = out-of-scope sample),
  `reference_slots`, `predicted_slots`, `exact_match`, `confidence`,
  `bucket`, `latency_ms`. For out-of-scope samples the correct behaviour is
  *no match*: `prediction: null` scores as correct.
  Audio-input intent datasets (`registry.schemas.DatasetDef.input ==
  "audio"`, e.g. Speech-MASSIVE) are eval sets for the same leagues, not a
  new league — intent leagues stay keyed by training-data format
  (template/keyword/fusion); audio-vs-text input is a property of the eval
  dataset. The runner transcribes each clip **once per (dataset, lang)**
  with the dataset's pinned STT (`stt_plugin`/`stt_config` on the dataset
  def) and caches the transcript; every intent fighter is scored against
  that SAME cached `utterance`, isolating intent ranking from STT variance.
  Every row from such a dataset additionally carries `stt_plugin`,
  `stt_config` and `stt_revision` (the installed STT plugin version that
  produced the transcript) so the transcript's provenance travels with the
  row. v1 does not fan out combinatorial STT×intent fighters — one pinned
  default STT per language, not every STT fighter against every intent
  fighter; a later revision could add that as an opt-in matrix.
- **STT**: `reference_text`, `prediction` (transcript), `wer` (computed on
  ingest when absent), `latency_ms`.
- **Wake word**: `label`, `prediction` (decision), `latency_ms`.
- **TTS**: `input_text`, `prediction` (audio ref), `latency_ms`. No
  ground-truth reference (there is no single "correct" waveform for a
  prompt), so human votes stay the primary ranking signal. Each clip is
  also scored with an objective, reference-free naturalness metric
  (**R14**) whose per-row value and judge provenance live in `extras`:
  `utmos` (1-5, higher better), `utmos_judge`, `utmos_judge_revision`.
  Alongside it, each clip also gets an STT round-trip intelligibility score
  (**R16**): `intelligibility_wer`, `intelligibility_cer`,
  `intelligibility_judge`, `intelligibility_judge_revision`.

Performance columns (performance-metrics campaign M1), all optional, all
default to null/absent: `elapsed_ms` (wall time of the single inference
call, `time.perf_counter`-based — distinct from `latency_ms`, which some
adapters also set from the same measurement), `peak_rss_mb` (process RSS
sampled before/after the call, higher of the two — **not** a true peak and
**not** attributable to this call alone: it is process-wide, so concurrent
work in the same process folds in; see `runner.perf.measure_call`),
`audio_secs` (duration of the input clip for STT/wake word, of the produced
clip for TTS — absent for intent rows, which are text-only; lets RTF =
`elapsed_ms / 1000 / audio_secs` be computed downstream), `hw` (a per-run
hardware fingerprint captured once per process and stamped on every row of
that run: `host_class` one of `cpu-x86`/`arm64`/`gpu`, `cpu_model`,
`threads`, `accelerator`, `hostname` — see `runner.perf.hw_fingerprint`).
Rows are merged across machines into one competitor's `.jsonl` file, so `hw`
travels on the row itself rather than being assumed constant per file. The
vast majority of already-published rows predate these columns; loaders
(`arena.models.PredictionRow`) default them to `None` and MUST NOT require
them.

### 3.3 Benchmark scripts

A benchmark script (`benchmarks/<bench>.py`) MUST:

1. load its dataset definition and competitors from the registry;
2. pin the dataset revision at run start and record it in every row;
3. train/instantiate each competitor exactly as OVOS would (for intent: the
   real OPM pipeline plugin over a message bus, trained from the dataset's
   own training files, matched through the plugin's own `match_<tier>`
   confidence gate; the arena owns no threshold numbers);
4. write resumable per-competitor JSONL and upload to the HF results repo.

### 3.4 Static data artifacts

`assemble` (CI, daily) turns prediction JSONLs into:

| File | Content |
|---|---|
| `battles-<mod>-<lang>.json` | Blind A/B pool (capped, pair-interleaved) |
| `benchmark-<mod>-<lang>.json` | Auto-metric board straight from predictions |
| `elo-seed-<mod>-<lang>.json` | Benchmark-derived initial ELO ledger |
| `leaderboard-<mod>-<lang>.json` | ELO board (seed + human votes) |
| `competitors.json` | Bestiary export of the registry |
| `index.json` | Catalogue of all of the above |

### 3.5 Frontend

Static Astro site: leaderboards (benchmark boards beside ELO boards),
blind battle page, and the fighters bestiary. Plugin identities MUST NOT be
revealed before the vote is committed (post-vote reveal is encouraged).
Voting options MUST include: candidate A, candidate B, tie, both-wrong.

## 4. Matchmaking rules

- **R1, Same stimulus.** A battle pairs two predictions for the *same*
  `sample_id` from the same dataset, by two different competitors. Where the
  dataset publishes a sample-set manifest (a `sample_policy`-capped
  dataset's `sample_sets/<lang>.json`, see `docs/runner.md`), both
  predictions MUST also fall inside that manifest's id set — a sample
  either competitor answered outside the published manifest carries no
  signal about the managed comparison and MUST NOT seed a battle
  (`arena.cli._load_sample_set`, invoked during `cmd_assemble`).
- **R2, Identical outputs are never battled** (no signal for a voter).
- **R3, Prefer discriminative samples.** Within each competitor pair,
  both-wrong samples sort first, then one-wrong disagreements. Battle pools
  interleave competitor pairs so no pair dominates.
- **R4, Deterministic and blind.** `battle_id` is a content hash of
  (modality, dataset, lang, sample, sorted pair). A/B display order derives
  from that hash, not from competitor names. Re-running `assemble` keeps
  ids stable so open votes never dangle.
- **R5, ELO seeding.** Auto-battles derive an outcome from the reference
  metric (intent: exact match incl. OOD rejection. STT: lower WER) for every
  (sample, pair) with signal, replayed in deterministic order at **K/4**.
  Auto votes are never attributed to users and are reported separately.
- **R5a, Significance gate.** A competitor pair contributes auto-battles
  for a dataset only when their aggregate primary-metric confidence
  intervals (R11) do not overlap (`arena.metrics.pair_metric_significant`).
  Per-sample disagreement within a pair whose overall performance is
  statistically indistinguishable is benchmark noise and MUST NOT seed the
  rating.
- **R5b, Weight cap.** A pair's total Bradley-Terry auto-vote weight is
  capped, in human-vote-equivalent units, at a fixed ceiling applied
  symmetrically to both members of the pair; the weight below the cap is
  scaled proportionally so it preserves the pair's observed auto-battle win
  rate rather than distorting it. Dataset size MUST NOT be a lever on how
  much the auto-vote seed can move a pair's rating — a benchmark corpus with
  ten thousand samples must not outweigh a hand-voted pair the way an
  uncapped weight would let it. Replay MUST reproduce the capped weight
  deterministically from the published predictions alone, same as any other
  seed value (§2 P5). The tuned ceiling is the `MAX_AUTO_WEIGHT_PER_PAIR`
  constant in `arena/assembler.py`, worked through alongside the per-round
  scaling in `docs/methodology.md`. `ovos-arena audit-seeds` reports
  every pair's weight and whether it sits at the cap.
- **R12, Full-history replay.** The vote record (§6) is the complete history
  of votes as they were publicly cast, one committed line per `vote`-labelled
  issue, appended once and never rewritten. `tally` fetches every
  `vote`-labelled issue (open and closed) so it can detect any one not yet
  in the record, records that issue exactly once with the battle context it
  was cast on, and then rebuilds every leaderboard by replaying the FULL
  record from the start — not only the newly appended lines — so the rating
  never depends on which run first observed a given vote. An issue already
  in the record MUST NOT be re-parsed, so a later title edit changes no
  rating, and an already-closed issue MUST NOT be re-commented on or
  re-closed. `assemble`, `tally` and `verify-replay` MUST build every
  leaderboard through that same replay of the committed record, so what
  `assemble` publishes is what `verify-replay` reproduces.
- **R13, Vote fraud rules** (`arena/fraud.py`, pure functions of the vote
  log, see `docs/methodology.md` for the full rationale):
  - one vote per (voter, battle), R1's battle identity dedupe. - a per-voter, per-league, per-UTC-day cap (`DAILY_VOTE_CAP = 50`). - an account-age gate (`NEW_ACCOUNT_MIN_DAYS = 7`) using a creation-date
    cache fetched once per author and persisted (`voter-age-cache.json`),
    the replay step itself MUST NOT touch the network. An issue whose
    author's creation date cannot be fetched MUST NOT be recorded or
    actioned, so the next run retries it rather than counting it ungated. - a one-sided-voter down-weight (`ONE_SIDED_MIN_VOTES = 20`,
    `ONE_SIDED_THRESHOLD = 0.95`, weight `ONE_SIDED_WEIGHT = 0.5`), keyed on
    the literal A/B choice (blind, randomized per battle-id hash), not
    competitor identity.
  Discards and down-weights MUST be recorded (`vote-audit.json`), never
  silently dropped — including an issue discarded at ingest, a recorded
  vote whose competitor has since left the seed roster
  (`competitor_retired`), and a second record line for an issue number
  already recorded (`duplicate_record`, one issue is one vote). Every
  artifact R19 compares MUST come from that one replay, so `assemble` and
  `tally` both publish the leaderboards and `vote-audit.json` together. Weighting affects only the Bradley-Terry rating (R6-R10
  above). The legacy sequential ELO column is unaffected.
- **R14, Objective TTS scoring.** `runner/tts_bench.py` MUST score every
  synthesised clip with a reference-free naturalness metric (UTMOS) and
  record the per-row score plus judge identity/revision in `extras`
  (`utmos`, `utmos_judge`, `utmos_judge_revision`), scoring is not optional
  for a TTS run. `arena/metrics.py:score_tts` aggregates it into the `tts`
  benchmark board's primary metric (mean `utmos`, higher better,
  `n_scored` reported). `arena/assembler.py:seed_elo` treats a
  significantly-higher-UTMOS clip as an auto-battle win (§4 R5) under the
  same significance gate (R5a) and per-pair weight cap (R5b) as every other
  league. It also carries a bootstrap confidence interval (R11) like every
  other benchmark board. This objective board and its ELO seed sit
  *alongside* human votes,
  which remain the TTS league's primary ranking signal (§2.1, §3.2), there
  is still no ground-truth reference to call a synthesis definitively
  "correct".

## 5. Rating system

- **R6, Primary rating is Bradley-Terry, batch-fit.** `EloEntry.bt_rating`
  (`arena/rating.py`, minorization-maximization, Hunter 2004) is the ranking
  key for every leaderboard. It is a **batch fit over the full replayed vote
  log**, not a sequential update, re-ordering the vote log (same votes,
  different order) MUST NOT change the result, since a battle's outcome
  should not depend on which other battles happened to be voted on first.
  See `docs/methodology.md` for the full derivation and the "why not
  sequential ELO alone" / "why not TrueSkill" rationale.
- **R7, Sequential ELO is a secondary display column.** `EloEntry.elo`
  (initial 1200, K=32, K=16 after 30 battles, 400-point logistic curve) is
  kept for continuity with earlier boards but is never the ranking key.
- **R8, Bootstrap confidence intervals.** `EloEntry.ci_lower` /
  `ci_upper` are a seeded bootstrap (§P5: fixed seed ⇒ reproducible) over the
  human vote log only. The auto-vote seed is held fixed in every bootstrap
  round, it is a deterministic function of a fixed benchmark corpus, not an
  i.i.d. sample, so resampling it would not model a real source of
  uncertainty. A board with fewer than `PROVISIONAL_MIN_HUMAN_VOTES` (10)
  human votes sets `EloBoard.provisional = true`. The frontend MUST show a
  provisional badge rather than presenting the ranking as settled.
- **R9, Auto/human weighting.** Auto (benchmark-seed) votes carry weight
  `BT_AUTO_WEIGHT = 1/4` in the Bradley-Terry fit, the same §4 R5 intent as
  the legacy K/4, expressed as a pairwise weight rather than a K-factor.
- **R10, Convergence prior.** Every competitor gets one virtual weighted
  tie against a fixed-strength "field average" phantom opponent
  (`PRIOR_WEIGHT = 1.0`). This connects the comparison graph (competitors
  that never played each other, directly or transitively, still get a
  well-defined relative order) and guarantees a competitor with zero
  recorded wins or zero recorded losses converges to a finite, ordered
  rating rather than collapsing to 0 or diverging to infinity.
- **R11, Benchmark boards carry confidence intervals.** Every
  `BenchmarkEntry` (§3, `benchmark-<mod>-<lang>.json`) carries a seeded
  bootstrap 95% CI on its primary metric
  (`arena/metrics.py:primary_metric_ci`), mean-of-indicator bootstrap for
  accuracy/error-rate metrics, ratio-of-summed-counts bootstrap (resampling
  `(errors, reference_words)` pairs, never per-utterance WER values) for
  STT's `wer_mean`. `BenchmarkEntry.tied_with_leader` marks entries whose CI
  overlaps the #1 entry's CI. See `docs/methodology.md` for the full
  rationale.
- **R14, WER normalization is canonical and versioned.** Before computing
  word-level edit distance, both the reference and the hypothesis text pass
  through `arena.metrics.normalize_transcript`
  (`WER_NORMALIZER_VERSION`, currently `1`): Unicode NFKC normalization,
  casefolding, punctuation stripping, ASCII digit runs spelled out
  digit-by-digit as English words (`"7"` → `"seven"`, `"123"` → `"one two
  three"`, no magnitude words, no locale grouping), and whitespace
  collapsing. This is scoring-time only, ingested prediction rows are never
  mutated. `arena.metrics.row_wer` (used by `score_stt`) prefers
  recomputing WER from `reference_text`/`prediction` through this
  normalizer over a row's precomputed `wer`, so scores stay comparable
  across runners that may have normalized differently before publishing. It falls back to the stored `wer` only when the raw text is unavailable.
  `benchmark-stt-*.json` boards carry `wer_normalizer_version` so scores
  produced under a future normalizer revision are distinguishable from
  past ones.
- **R15, Version-blend guard.** When a competitor's rows for one benchmark
  board span more than one distinct `plugin_version`, `build_benchmark_board`
  MUST NOT silently aggregate them into one indistinguishable score. Every
  `BenchmarkEntry` carries `plugin_versions` (the distinct versions present)
  and `version_blended` (true when more than one), the frontend can flag
  blended entries, and CI/tooling can grep for them, rather than a reader
  assuming a single score reflects a single shipped version.
- **R15b, Dataset-revision guard.** Rows are scored against the dataset
  revision the registry pins. When a dataset entry's `source.revision` names a
  commit sha, `build_benchmark_board` MUST drop every prediction row whose
  `dataset_revision` differs from it before scoring, and those rows MUST NOT
  reach a battle or seed a rating either. The board records the pin as
  `dataset_revision` and each `BenchmarkEntry` records the number of rows
  dropped as `rows_other_revision`. A competitor left with no rows on the
  pinned revision is unscored — unranked, with a reason naming the pin, and
  contributing nothing to the ELO seed — rather than ranked on rows from a
  corpus it was not measured against. A revision can change the row
  population, rename buckets, or move the train/test boundary, so a score
  carried across one describes neither revision. Where `source.revision` names
  a branch there is no fixed revision to compare against and every row is
  scored. Runners apply the same rule to resume, and MUST compare against the
  DECLARED `source.revision`, never the sha a branch resolves to at run time:
  a branch tip moves with every upstream commit, so comparing against the
  resolved sha would discard every shard of every branch-pinned dataset. Where
  `source.revision` is a commit sha, a sample counts as done only when its
  existing row carries that same sha — a row with no `dataset_revision` cannot
  be shown to belong to the pin and so is not done — and rows from any other
  revision are cleared from an output shard before it is appended to. Where
  `source.revision` is a branch, a runner MUST NOT prune anything and MUST
  count every existing row as done, including rows published before
  `dataset_revision` existed as a column (§3.2: loaders MUST NOT require it).
- **R16, TTS intelligibility (STT round-trip WER/CER).** Alongside UTMOS
  (R14), `runner/tts_bench.py` MUST transcribe every rendered clip whose
  language has an ASR judge back to text with a pinned STT judge, the best
  offline onnx-asr model for the
  clip's language, resolved per `runner/asr_judges.py` from ovos-config's
  offline-STT recommends with an onnx-asr fallback table, never
  faster-whisper (model + revision recorded
  in `extras` as `intelligibility_judge`/`intelligibility_judge_revision`,
  same provenance discipline as the UTMOS judge) and score it against the
  prompt text with the canonical `normalize_transcript` WER/CER (§E),
  recorded per row as `intelligibility_wer`/`intelligibility_cer`. This
  metric is computed via the plugin's direct `get_tts` synthesis path (no
  audio-player/playback round trip). Rendered audio MUST be transcoded and
  resampled to 16 kHz mono before transcription regardless of the source
  container/rate, feeding the judge raw bytes at the wrong sample rate or
  an undecoded non-wav container produces a meaningless score, not an
  error, so this is a silent-corruption risk rather than a crash. For a
  language with a judge, a synthesis failure MUST still emit a row, scored
  `intelligibility_wer = intelligibility_cer = 1.0`, rather than being
  silently dropped by the runner's per-sample exception handling (§3.3),
  and a judge-side failure (e.g. transcribing silence/noise) forces the
  same worst-case score rather than leaving the row's intelligibility
  fields missing.
  `arena/metrics.py:score_tts` aggregates `intelligibility_wer` into the
  `tts` benchmark board as a **secondary** metric (mean + bootstrap 95% CI,
  R11), UTMOS (R14) stays primary. Low-resource languages the judge
  transcribes weakly are warn-only: the real WER is always recorded and
  never gates a board or blocks a run, since a high round-trip WER there
  reflects the STT judge's own blind spot as often as the TTS clip's actual
  intelligibility.

  Every row of one board for one language MUST be judged by the same model,
  since a WER from one ASR is not comparable with a WER from another.
  `runner/rescore_tts.py --rejudge-intelligibility` MUST therefore re-judge
  any row whose stored `intelligibility_judge` differs from the model the
  row's language resolves to, including a row marked
  `intelligibility: not_available` whose language the registry has since
  claimed. Boards MUST record the judges their rows carry and MUST NOT
  compare two rows judged by different models: such a pair contributes no
  intelligibility auto-vote, and a board holding more than one judge for
  its language carries a warning naming them.

  A language has an ASR judge when some ASR model covers it:
  `runner/asr_judges.py:judge_available` MUST decide this from an
  ovos-config offline-STT recommend, a judge this repo pins for the
  language, or an entry in the installed `ovos-stt-plugin-onnx-asr`
  registry, which it MUST read from the plugin rather than from a copy held
  here — a copy drifts, and a language the plugin covers then reads as
  unjudgeable and silently loses its scores. A language is judged if and
  only if the registry lists it, a deliberate `whisper-base` entry
  included; a language that reaches `whisper-base` only through the
  plugin's blanket `DEFAULT_CPU_MODEL` fallback is NOT judged, since with
  no model trained on it the panel transcribes noise and the round-trip error
  rate measures the ASR fleet's coverage rather than the voice. Rows for such
  a language MUST NOT carry a round-trip number at all. They MUST carry
  `intelligibility: not_available` with null `intelligibility_wer` and
  `intelligibility_cer` and `intelligibility_judge: none`, including on a
  synthesis failure, so a reader can tell an unmeasurable metric from a bad
  one. `arena/metrics.py:tts_seed_score` MUST seed those rows from the
  perceptual judges alone — UTMOS (R14), with SIGMOS/DNSMOS/NISQA recorded
  as their own board columns — and MUST NOT substitute a default for the
  absent intelligibility and agreement factors. With no
  `intelligibility_wer` aggregated, the board's judge-agreement matrix
  (R14 ext.) omits the intelligibility axis rather than reporting it
  agreeing with itself. Every writer of TTS prediction rows — the bench
  runner, the backfill tool and the rescore tool — MUST apply the same
  test, so rows for one language never disagree about whether a round trip
  exists, and a rescore MUST NOT replace a score produced by a
  language-specific judge with the marker. Availability is decided per clip
  language: the dataset-level `"multi"` tag is not a language, and a
  multilingual corpus is judged one real language at a time. TTS boards are
  keyed by language, so a seed of this shape is never ranked against an
  intelligibility-weighted one.
- **R18, Intent leagues share one battle pool.** `battle_group()`
  (`arena/models.py`) maps every intent league to the single `intent` group;
  every other modality is its own group. Matchmaking
  (`arena/assembler.py`) pairs any two fighters that answered the same
  stimulus in the same language, whatever preparation each needed: a blind
  vote judges two outputs, and the training regime is invisible in that
  judgement. There is one `battles-intent-*.json` pool, one
  `elo-seed-intent-*.json` and one `leaderboard-intent-*.json`. Benchmark
  boards stay per league (`benchmark-<league>-*.json`), where the regime
  does change what a number means. An embedding classifier is not a league
  of its own, it is a strategy: it competes in the league its own training
  regime implies (§2.1), same as any other engine.

  **Vote replay policy.** A vote counts toward the replay only if the battle
  it references is present in the currently committed pool for its group. A
  battle id is a content hash of `(battle_group, dataset, lang, sample,
  sorted(competitor_a, competitor_b))`, so an id survives a fighter changing
  league — the group it hashed does not change. A vote whose battle is no
  longer in the pool (a retired fighter, a re-swept sample) is discarded
  exactly like any other "battle not in pool" vote
  (`arena/cli.py:cmd_tally`), recorded in `vote-audit.json`, never silently
  dropped. This is a pure function of the vote log plus the committed
  battles pools (themselves a pure function of the registry and published
  predictions, P5), so replay stays deterministic and network-free, same as
  the rest of `tally`. The public vote log (the GitHub issue) itself is
  never edited or deleted by this rule.
- **R17, Streaming wake word is a separate board, not a compat shim.**
  Isolated-clip benchmarking (§3, wake_word league) structurally favors
  clip-shaped detectors: a streaming detector never gets to fire the way it
  does against a live mic, and a false-accept rate needs hours of continuous
  negative audio, not seconds-long clips. `ww_stream` (`arena/metrics.py:
  score_ww_stream`) is therefore a distinct modality/board scored from
  continuous-audio detection events (`(timestamp_s, score)` per activation)
  matched against ground-truth onsets within `EVENT_TOLERANCE_S` (1.5 s).
  `CompetitorDef.capabilities` MUST list `"stream"` for a fighter to be
  eligible (`runner.ww_bench.WakeWordStreamBench.filter_competitors`), clip-only fighters are excluded outright, never zero-scored. The primary
  metric, `error_at_2fa_per_hour`, is FRR at the lowest scanned threshold
  keeping FA/hour within `TARGET_FA_PER_HOUR` (2/hour), not raw
  threshold-0.5 FRR alone. The `wake_word` board and `score_wake_word` are
  unchanged by this rule.
- **R19, Published leaderboards must be provably reproducible from the
  public vote record.** `arena/cli.py:cmd_verify_replay` (`verify-replay`
  subcommand) re-runs `replay_boards`, the same pure path `tally` and
  `assemble` build their boards with, never a reimplementation, over the
  committed record (or a snapshot of it given with `--votes-file`), and
  diffs the result field-by-field against the committed
  `leaderboard-<league>-<lang>.json` and `vote-audit.json` (ratings,
  ranks, vote counts. `generated_at` ignored). A mismatch means the
  published data is not what the public log actually supports, and the
  command exits non-zero with the exact diff.
  `.github/workflows/replay-proof.yml` runs it on every push to `dev` and
  daily, failing the build on any divergence, strictly: a league with
  counted votes but no published board is itself a mismatch, not a
  tolerated gap. Published leaderboard/ELO-seed/battles artifacts are
  derived data, not a compatibility surface, when a change (e.g. a
  league split such as R18, a new rating field) makes old committed
  artifacts no longer reproducible from the current replay path, they are
  deleted and regenerated (`assemble` + `tally`), never grandfathered in
  to keep the proof passing.
- **R20, Vote-less leagues carry no battle/ELO artifacts.** A league in
  `arena.models.VOTELESS_MODALITIES` is scored entirely by its benchmark
  board. `cmd_assemble` MUST skip such a modality's battle-sample and
  ELO-sample accumulation outright — it MUST NOT write `battles-*.json`,
  `elo-seed-*.json`, or `leaderboard-*.json` for it, not even empty
  placeholders. `leagues()`
  marks the league `voteless: true` so the frontend renders the benchmark
  board section with no ladder section at all, rather than an always-empty
  "no standings yet" placeholder. This is a standing property of the
  league (its scoring model has no votes to replay), not a temporary gap
  that `verify-replay` (R19) needs to reconcile — R19's reproducibility
  proof only ever covers modalities that publish a vote-backed leaderboard.
- Human votes: tie and both-wrong score 0.5/0.5.
- Auto votes (seeding): K/4 (sequential ELO) / weight 1/4 (Bradley-Terry),
  outcomes from benchmark metrics only.
- Separate standings per (modality, lang).
- **Two human-vote sources, one ladder.** *Blind battles* pair two predictions
  for the same sample. *Free-form votes* are direct subjective preferences
  ("which plugin do you prefer?") cast by someone who tested the plugins out of
  band, no stimulus, identities shown. Both are pairwise human preferences and
  replay into the same (modality, lang) ELO. A free-form matchup is a
  stimulus-less battle whose `battle_id` hashes `(group, "freeform", lang, A, B)`,
  so it reuses the battle pool, the one-vote-per-(author, pair) dedupe, and the
  ELO replay unchanged.
- `tally` (CI, hourly) parses open `vote`-labelled issues
  (`vote|<battle_id>|<choice>` titles), validates against the battles pool,
  dedupes one vote per (author, battle), replays on top of the seed in
  issue-number order, commits boards, closes processed issues.

## 6. Vote flow (end to end)

```
Voter opens battle page → picks A / B / Tie / Both wrong
  → prefilled GitHub issue opens (template applies the `vote` label;
    title: vote|<battle_id>|<choice>)
  → voter submits (free GitHub account, no arena account)
Hourly tally Action:
  record every issue not yet in votes.jsonl (title, author, moment,
    battle context) → replay the record onto the ELO seed
  → write leaderboard-*.json → commit → Pages redeploys
  → close newly recorded issues with a comment (label `processed`)
```

The vote log **is** the issue history, public, auditable, replayable.
`frontend-static/public/data/votes.jsonl` is that history as it was cast:
one committed, append-only line per `vote`-labelled issue, written the
first time the issue is seen and never rewritten. Each line carries the
issue number, author, creation time, the title exactly as seen, and the
battle context resolved from the pool at that moment — modality, dataset,
lang, the competitor pair and the sample. An issue whose title does not
parse is recorded as invalid, and one naming a battle absent from the
pool is recorded as discarded; neither is examined again. The record is
replaced atomically and read strictly: a line that does not parse stops
the run rather than dropping the vote it holds.

Recording the battle context is what makes the log replayable: the
battles pool is regenerated and pruned on every assemble, and a title is
editable by its author forever. Both are inputs to ingest only. Every
published rating is a pure function of the committed record, the ELO
seeds and `voter-age-cache.json`.

## 7. Modalities

Every modality has registry fighters and a reproducible benchmark script. The
audio modalities share `runner/media_bench.py` (the intent leagues share
`runner/intent_bench.py`). Each script takes the same flags
(`--competitors`, `--langs`, `--max-samples`, `--dataset`, `--upload`).

1. **Intent**, `benchmarks/intent_intents_for_eval.py` and
   `benchmarks/intent_massive_templates.py`: the intent leagues over
   `OpenVoiceOS/intents-for-eval` (12 langs) and `OpenVoiceOS/massive-templates`
   (52 langs). Ranked by `generalization_accuracy` with an ELO seed.

   `intents-for-eval` separates train and test at the template level: the
   template-paradigm train split (`train_templates.jsonl`) holds out 20% of
   each intent's phrase templates per language, and the `template` test
   bucket is built only from expansions of those held-out templates, so a
   template engine cannot have seen the exact phrase during training. The
   `template` and `in_distribution` buckets stay excluded from
   `generalization_accuracy` regardless — matching a held-out template still
   measures a narrower kind of generalization than the free-form
   `paraphrase`, `typos`, `asr_noise` and `far_ood` buckets the ranked
   metric is scored on. Plain `accuracy` and every per-bucket column remain
   published beside it. `in_distribution` (raw bucket name in the dataset)
   holds in-domain paraphrase-adjacent utterances with a real expected
   intent, distinct from the null-intent `far_ood` bucket.
2. **STT**, `benchmarks/stt_minds14.py` (`runner/stt_bench.py`): transcribes
   each fighter over MInDS-14. Ranked by WER with an ELO seed. A separate
   off-repo prediction runner also feeds legacy `ovos-stt-bench-*` rows,
   re-keyed to competitors via the registry `alias` field.
3. **Wake word**, `benchmarks/ww_hey_mycroft.py` (`runner/ww_bench.py`):
   runs each hotword engine frame-by-frame over the held-out ww-bench manifest.
   Ranked by detection error rate (with false-accept / false-reject) and an
   ELO seed.
4. **TTS**, `benchmarks/tts_intents_prompts.py` (`runner/tts_bench.py`):
   synthesises a prompt corpus per fighter and stores the clips. Human-vote
   only, no objective metric, no benchmark board, no ELO seed. The ELO board
   accrues purely from blind A/B listening votes.

---
[Home](index.md) · [Local testing →](local-testing.md)
