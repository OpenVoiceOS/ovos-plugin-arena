# OVOS Plugin Arena — Specification

**Status:** Active — maintained by TigreGotico
**Version:** 0.3

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

Predictions live in HF dataset repos as per-competitor JSON lines:
`predictions/<competitor_id>.jsonl`, one row per (language, sample).

Minimum columns, all modalities: `competitor_id`, `sample_id`, `dataset_id`,
`lang`, `plugin_id`, `plugin_version`, `prediction`, `runner_version`,
`created_at`. Reproducibility columns SHOULD include `dataset_revision`.

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

## 5. Rating system

- Standard ELO: initial 1200, K=32 (K=16 after 30 battles), expected score
  with the 400-point logistic curve.
- Human votes: tie and both-wrong score 0.5/0.5.
- Auto votes (seeding): K/4, outcomes from benchmark metrics only.
- Separate standings per (modality, lang).
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

## 7. Modality roadmap

1. **Intent** — live: `benchmarks/intent_intents_for_eval.py`, 5 engines ×
   12 languages over `OpenVoiceOS/intents-for-eval`.
2. **STT** — prediction runner exists (`runner/`, deployed off-repo);
   arena benchmark script + registry entries pending.
3. **Wake word** — pending (same contracts).
4. **TTS** — pending; human-vote-only boards (no auto metric, no ELO seed).
