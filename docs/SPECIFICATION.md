# OVOS Plugin Arena — Specification

**Status:** Active — maintained by TigreGotico
**Version:** 0.4

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be
interpreted as described in RFC 2119.

---

## 1. Purpose

The Plugin Arena answers one recurring question for the OVOS ecosystem:
*"which plugin should I use?"* It does so with two complementary signals:

1. **Benchmarks** — plugin predictions evaluated against labelled datasets
   (accuracy/F1 for intent, WER for STT, detection metrics for wake word),
   published openly and reproducibly.
2. **Human preference** — chess-style ELO ratings produced by blind A/B
   "battles" where users pick the better of two plugin outputs. The initial
   ELO is derived from the benchmarks; human votes refine it.

The arena is the *rating and voting* venue. It is **not** an execution venue.

## 2. Design principles

- **P1 — GitHub-native, zero servers.** The arena is a repository: data
  artifacts are JSON files committed by scheduled Actions, the UI is a static
  site on GitHub Pages, and votes are GitHub issues. There is no backend, no
  database, and no arena account system. A FastAPI shim MAY exist as a local
  *development tool* only; it MUST NOT become a deployment target.
- **P2 — Predictions are decentralized; HF is the artifact layer.** Plugins
  run *outside* the arena as offline batch jobs. Every prediction run is
  published as a HuggingFace dataset. The arena MUST NOT install, isolate, or
  execute OVOS plugins in CI.
- **P3 — Everything is declarative.** Every competitor is a JSON file
  (`registry/competitors/<modality>/<id>.json`); every dataset is a JSON file
  (`registry/datasets/<modality>/<id>.json`). Forking the repo and editing
  JSON yields a working arena.
- **P4 — Every benchmark is a reproducible script.** One dedicated Python
  script per benchmark under `benchmarks/`, tracked in git. It trains each
  registry competitor, produces fresh predictions (never copied from other
  sources), records the pinned dataset revision and plugin versions in every
  row, and uploads to HF.
- **P5 — Ratings are replayable.** Battle ids are content hashes; the ELO
  seed is a deterministic function of the published predictions; human votes
  replay in issue-number order. The full standings MUST be reproducible from
  public data alone.
- **P6 — All prediction data is kept**, including bad predictions — failure
  cases guide plugin improvements and are first-class benchmark content.

## 2.1 Leagues (modalities)

Every modality is an independent league with its own benchmarks, battle
pools and ELO standings: `stt`, `tts`, `wake_word`, and **three intent
leagues**. Keyword-paradigm engines (hand-written vocabulary rules) and
template-paradigm engines (phrase-template corpora) consume different
supervision, so they MUST NOT be ranked against each other:

| League | Who competes |
|---|---|
| `intent_template` | template/embedding engines (Padatious, Padacioso, Nebulento, …) |
| `intent_keyword` | keyword engines (Adapt, Palavreado, …) |
| `intent` | open league — mixed-paradigm pipeline **fusions** (ensembles) |

Paradigm leagues are pure: a fighter in `intent_template` may only carry
template-paradigm stages (enforced by the bench script). Fusions carry
fun portmanteau names (Padapt = Padatious × Adapt).

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
valid `mycroft.conf` fragment — an `intents` section with an ordered
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
**ensemble** fighter in its own right (e.g. the stock OVOS cascade) — first
stage whose own confidence gate fires wins, exactly like ovos-core. The same
plugin under a different config is a *different competitor*; `competitor_id`
is the stable key for predictions, battles, ELO and leaderboards. Bestiary
card fields describe the fighter for the UI: `display_name`, `species` (the
plugin class it instantiates), `types` (architecture tags: `GOFAI`,
`fuzzy-match`, `neural-net`, `template-match`, `keyword-match`, `embedding`,
`LLM`, `ensemble`), `description`, `model`, `links`.

**Datasets** (`registry/datasets/<modality>/<id>.json`): one corpus per
entry — source (HF id + revision + split or per-lang `file_pattern`),
`reference_fields` (the datashape contract), license, `lang` (or
`lang: multi` plus a `langs` list), and a `role`. **Keyword-paradigm and
template-paradigm training corpora are different datasets with different
datashapes**: each gets its own `role: train` entry tagged with `paradigm`
(`template` rows carry `{slot}` phrase templates with example values;
`keyword` rows carry complete Adapt-style `required_vocab`/`optional_vocab`
rules). A `role: eval` corpus links its paradigm-specific training sets via
`train_datasets`, and every stage plugin trains from the corpus matching its
paradigm.

### 3.2 Prediction contract

Predictions live in HF dataset repos — **one dedicated repo per benchmark
modality**, named `ovos-<modality>-bench-<dataset_id>` (league underscores
dashed), with per-competitor JSON lines at
`predictions/<lang>/<competitor_id>.jsonl` and a generated dataset card
declaring **one HF split per language**. One row per (language, sample).

Minimum columns, all modalities: `competitor_id`, `sample_id`, `dataset_id`,
`lang`, `plugin_id`, `plugin_version`, `prediction`, `runner_version`,
`created_at`. Reproducibility columns SHOULD include `dataset_revision`.
`schema_version` (default 2) marks provenance — 1 identifies a row
converted from the legacy `STTRow` column layout at load time (§4 A2); new
rows are always written directly in this shape (`competitor_id` MAY be
absent when the writer has no registry dependency, e.g. the off-repo STT
runner — `arena.predictions` re-keys via `plugin_id` and
`registry.loaders.get_competitor_by_alias`).

Per modality:

- **Intent**: `utterance`, `reference_intent` (null = out-of-scope sample),
  `reference_slots`, `predicted_slots`, `exact_match`, `confidence`,
  `bucket`, `latency_ms`. For out-of-scope samples the correct behaviour is
  *no match*: `prediction: null` scores as correct.
- **STT**: `reference_text`, `prediction` (transcript), `wer` (computed on
  ingest when absent), `latency_ms`.
- **Wake word**: `label`, `prediction` (decision), `latency_ms`.
- **TTS**: `input_text`, `prediction` (audio ref); no objective metric —
  TTS ranks by human votes only.

### 3.3 Benchmark scripts

A benchmark script (`benchmarks/<bench>.py`) MUST:

1. load its dataset definition and competitors from the registry;
2. pin the dataset revision at run start and record it in every row;
3. train/instantiate each competitor exactly as OVOS would (for intent: the
   real OPM pipeline plugin over a message bus, trained from the dataset's
   own training files, matched through the plugin's own `match_<tier>`
   confidence gate — the arena owns no threshold numbers);
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

- **R1 — Same stimulus.** A battle pairs two predictions for the *same*
  `sample_id` from the same dataset, by two different competitors.
- **R2 — Identical outputs are never battled** (no signal for a voter).
- **R3 — Prefer discriminative samples.** Within each competitor pair,
  both-wrong samples sort first, then one-wrong disagreements. Battle pools
  interleave competitor pairs so no pair dominates.
- **R4 — Deterministic and blind.** `battle_id` is a content hash of
  (modality, dataset, lang, sample, sorted pair); A/B display order derives
  from that hash, not from competitor names. Re-running `assemble` keeps
  ids stable so open votes never dangle.
- **R5 — ELO seeding.** Auto-battles derive an outcome from the reference
  metric (intent: exact match incl. OOD rejection; STT: lower WER) for every
  (sample, pair) with signal, replayed in deterministic order at **K/4**.
  Auto votes are never attributed to users and are reported separately.
- **R5a — Significance gate.** A competitor pair contributes auto-battles
  for a dataset only when their aggregate primary-metric confidence
  intervals (R11) do not overlap (`arena.metrics.pair_metric_significant`).
  Per-sample disagreement within a pair whose overall performance is
  statistically indistinguishable is benchmark noise and MUST NOT seed the
  rating.
- **R5b — Weight cap.** A pair's total Bradley-Terry auto-vote weight is
  capped at `MAX_AUTO_WEIGHT_PER_PAIR` (5 human-vote-equivalents),
  proportionally scaled to preserve the observed win rate — dataset size
  MUST NOT be a lever on how much the auto-vote seed can move a pair's
  rating. `ovos-arena audit-seeds` reports every pair's weight and whether
  it sits at the cap.
- **R12 — Full-history replay.** `tally` MUST fetch every `vote`-labelled
  issue (open and closed), not only issues opened since the previous run —
  the vote log is the complete issue history (§6), and every tally run
  replays it from scratch. Already-closed issues MUST NOT be re-commented
  on or re-closed.
- **R13 — Vote fraud rules** (`arena/fraud.py`, pure functions of the vote
  log — see `docs/methodology.md` for the full rationale):
  - one vote per (voter, battle) — R1's battle identity dedupe;
  - a per-voter, per-league, per-UTC-day cap (`DAILY_VOTE_CAP = 50`);
  - an account-age gate (`NEW_ACCOUNT_MIN_DAYS = 7`) using a creation-date
    cache fetched once per author and persisted (`voter-age-cache.json`) —
    the replay step itself MUST NOT touch the network;
  - a one-sided-voter down-weight (`ONE_SIDED_MIN_VOTES = 20`,
    `ONE_SIDED_THRESHOLD = 0.95`, weight `ONE_SIDED_WEIGHT = 0.5`), keyed on
    the literal A/B choice (blind, randomized per battle-id hash), not
    competitor identity.
  Discards and down-weights MUST be recorded (`vote-audit.json`), never
  silently dropped. Weighting affects only the Bradley-Terry rating (R6-R10
  above); the legacy sequential ELO column is unaffected.

## 5. Rating system

- **R6 — Primary rating is Bradley-Terry, batch-fit.** `EloEntry.bt_rating`
  (`arena/rating.py`, minorization-maximization, Hunter 2004) is the ranking
  key for every leaderboard. It is a **batch fit over the full replayed vote
  log**, not a sequential update — re-ordering the vote log (same votes,
  different order) MUST NOT change the result, since a battle's outcome
  should not depend on which other battles happened to be voted on first.
  See `docs/methodology.md` for the full derivation and the "why not
  sequential ELO alone" / "why not TrueSkill" rationale.
- **R7 — Sequential ELO is a secondary display column.** `EloEntry.elo`
  (initial 1200, K=32, K=16 after 30 battles, 400-point logistic curve) is
  kept for continuity with earlier boards but is never the ranking key.
- **R8 — Bootstrap confidence intervals.** `EloEntry.ci_lower` /
  `ci_upper` are a seeded bootstrap (§P5: fixed seed ⇒ reproducible) over the
  human vote log only. The auto-vote seed is held fixed in every bootstrap
  round — it is a deterministic function of a fixed benchmark corpus, not an
  i.i.d. sample, so resampling it would not model a real source of
  uncertainty. A board with fewer than `PROVISIONAL_MIN_HUMAN_VOTES` (10)
  human votes sets `EloBoard.provisional = true`; the frontend MUST show a
  provisional badge rather than presenting the ranking as settled.
- **R9 — Auto/human weighting.** Auto (benchmark-seed) votes carry weight
  `BT_AUTO_WEIGHT = 1/4` in the Bradley-Terry fit — the same §4 R5 intent as
  the legacy K/4, expressed as a pairwise weight rather than a K-factor.
- **R10 — Convergence prior.** Every competitor gets one virtual weighted
  tie against a fixed-strength "field average" phantom opponent
  (`PRIOR_WEIGHT = 1.0`). This connects the comparison graph (competitors
  that never played each other, directly or transitively, still get a
  well-defined relative order) and guarantees a competitor with zero
  recorded wins or zero recorded losses converges to a finite, ordered
  rating rather than collapsing to 0 or diverging to infinity.
- **R11 — Benchmark boards carry confidence intervals.** Every
  `BenchmarkEntry` (§3, `benchmark-<mod>-<lang>.json`) carries a seeded
  bootstrap 95% CI on its primary metric
  (`arena/metrics.py:primary_metric_ci`) — mean-of-indicator bootstrap for
  accuracy/error-rate metrics, ratio-of-summed-counts bootstrap (resampling
  `(errors, reference_words)` pairs, never per-utterance WER values) for
  STT's `wer_mean`. `BenchmarkEntry.tied_with_leader` marks entries whose CI
  overlaps the #1 entry's CI; see `docs/methodology.md` for the full
  rationale.
- Human votes: tie and both-wrong score 0.5/0.5.
- Auto votes (seeding): K/4 (sequential ELO) / weight 1/4 (Bradley-Terry),
  outcomes from benchmark metrics only.
- Separate standings per (modality, lang).
- **Two human-vote sources, one ladder.** *Blind battles* pair two predictions
  for the same sample. *Free-form votes* are direct subjective preferences
  ("which plugin do you prefer?") cast by someone who tested the plugins out of
  band — no stimulus, identities shown. Both are pairwise human preferences and
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
  parse + validate + dedupe → replay ELO (seed + votes, ordered)
  → write leaderboard-*.json → commit → Pages redeploys
  → close issues with a thank-you comment (label `processed`)
```

The vote log **is** the issue history — public, auditable, replayable.

## 7. Modalities

Every modality has registry fighters and a reproducible benchmark script; the
audio modalities share `runner/media_bench.py` (the intent leagues share
`runner/intent_bench.py`). Each script takes the same flags
(`--competitors`, `--langs`, `--max-samples`, `--dataset`, `--upload`).

1. **Intent** — `benchmarks/intent_intents_for_eval.py` and
   `benchmarks/intent_massive_templates.py`: the three intent leagues over
   `OpenVoiceOS/intents-for-eval` (12 langs) and `OpenVoiceOS/massive-templates`
   (52 langs). Ranked by accuracy with an ELO seed.
2. **STT** — `benchmarks/stt_minds14.py` (`runner/stt_bench.py`): transcribes
   each fighter over MInDS-14. Ranked by WER with an ELO seed. A separate
   off-repo prediction runner also feeds legacy `ovos-stt-bench-*` rows,
   re-keyed to competitors via the registry `alias` field.
3. **Wake word** — `benchmarks/ww_hey_mycroft.py` (`runner/ww_bench.py`):
   runs each hotword engine frame-by-frame over the held-out ww-bench manifest.
   Ranked by detection error rate (with false-accept / false-reject) and an
   ELO seed.
4. **TTS** — `benchmarks/tts_intents_prompts.py` (`runner/tts_bench.py`):
   synthesises a prompt corpus per fighter and stores the clips. Human-vote
   only — no objective metric, no benchmark board, no ELO seed; the ELO board
   accrues purely from blind A/B listening votes.
