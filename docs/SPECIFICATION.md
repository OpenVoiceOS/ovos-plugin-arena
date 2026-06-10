# OVOS Plugin Arena — Specification

**Status:** Active — maintained by TigreGotico
**Version:** 0.2

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be
interpreted as described in RFC 2119.

---

## 1. Purpose

The Plugin Arena answers one recurring question for the OVOS ecosystem:
*"which plugin should I use?"* It does so with two complementary signals:

1. **Traditional benchmarks** — plugin predictions evaluated against labelled
   datasets (WER/CER for STT, detection metrics for wake word, accuracy for
   intent), published openly and reproducibly.
2. **Human preference** — chess-style ELO ratings produced by blind A/B
   "battles" where users pick the better of two plugin outputs.

The arena is the *voting and rating* venue. It is **not** an execution venue.

## 2. Design principles

- **P1 — Predictions are decentralized.** Plugins run *outside* the arena, as
  offline batch jobs (cron, laptop, CI — anywhere). The arena MUST NOT
  install, isolate, or execute OVOS plugins. This removes cloud cost,
  sandboxing, and dependency-conflict concerns from the service entirely.
- **P2 — HuggingFace is the artifact layer.** Every prediction run is
  published as a queryable HF dataset (e.g.
  `OpenVoiceOS/ovos-stt-bench-pt-PT`). The predictions are themselves a
  public benchmark, independent of the arena.
- **P3 — The arena database stores battle outcomes only.** Votes, ratings,
  users. Never raw predictions, never audio blobs (it references HF rows).
- **P4 — Evals grow organically.** New datasets and new plugins are added
  over time by publishing more prediction datasets — no arena code changes.
  Domain-specific evals (medical, ATC, accented speech, long-form audio)
  arrive the same way.
- **P5 — Ratings are replayable.** The ELO standings MUST be deterministically
  recomputable from the persisted vote log.
- **P6 — All prediction data is kept**, including bad predictions — failure
  cases guide plugin improvements and are first-class benchmark content.

## 3. System components

```
┌────────────────────┐   publish    ┌──────────────────────────┐
│ Prediction Runners │ ───────────► │ HuggingFace datasets     │
│ (offline, cron,    │              │ ovos-<mod>-bench-<ds>-<lang>
│  anywhere)         │              └───────────┬──────────────┘
└────────────────────┘                          │ sample
                                                ▼
                       ┌──────────────────────────────────────┐
                       │ Arena backend                        │
                       │  matchmaking · battles · votes · ELO │
                       └───────────┬──────────────────────────┘
                                   │ blind A/B
                                   ▼
                       ┌──────────────────────┐
                       │ Frontend (web)       │
                       │  listen/read · vote  │
                       └──────────────────────┘
```

### 3.1 Prediction runners (out of scope for this service, in scope for this spec)

- A runner takes (plugin, dataset, lang) and produces one prediction row per
  sample, plus automatic metrics where references exist.
- Runners MUST be reproducible: pinned plugin version, dataset revision, and
  runner version recorded in every row.
- Reference tooling lives in the OVOS ecosystem (`ovos-stt-bench-*` datasets
  already exist; `tts-benchmarks`, `ww-benchmarks`, `ovos-intent-benchmark`
  are the metric sources to converge on).

### 3.2 HF dataset contract

Naming: `OpenVoiceOS/ovos-<modality>-bench-<dataset>-<lang>` (one dataset per
source corpus + language; plugins are rows, not repos).

Minimum columns, all modalities:

| column | type | notes |
| --- | --- | --- |
| `sample_id` | str | stable ID within the source dataset |
| `dataset_id` | str | source corpus identifier + revision |
| `lang` | str | BCP-47 |
| `plugin_id` | str | OPM plugin name |
| `plugin_version` | str | exact installed version |
| `prediction` | str/audio | modality-specific payload (see below) |
| `runner_version` | str | reproducibility |
| `created_at` | timestamp | |

Per modality:
- **STT**: `audio` (ref to source corpus sample), `reference_text`,
  `prediction` (text), `wer`, `cer`, `rtf`.
- **TTS**: `input_text`, `prediction` (synthesized audio file), `voice`,
  `rtf`; objective metrics optional (MOS-predictors MAY be added later).
- **Wake word**: `audio`, `label` (contains-ww or not), `prediction`
  (score/decision), `latency_ms`.
- **Intent**: `utterance`, `reference_intent`, `prediction` (intent +
  entities), `exact_match`, `entity_f1`.

### 3.3 Arena backend

Responsibilities, and nothing more:
1. **Plugin registry** — admin-registered plugin identities per modality
   (mirrors what exists in the HF datasets).
2. **Sampling/matchmaking** — assemble battles from HF prediction rows.
3. **Battles & votes** — serve blind A/B pairs, record votes.
4. **Ratings** — maintain ELO standings per (modality, lang); recompute on
   demand from the vote log.
5. **Leaderboards** — query endpoints + periodic static JSON/HTML export so
   results remain public at zero infra cost even if the live service is down.

The backend builds on the existing FastAPI + auth + Postgres work in this
repo (alembic, users/roles). The committed `docs/models.sql` is the starting
point with one structural change: **battles are assembled, not executed** —
the `RUNNING` worker lifecycle is replaced by sampling from already-published
predictions (see §5).

### 3.4 Frontend

- Blind battles per modality: STT shows the source audio + two transcripts;
  TTS plays two audio renditions of the same text; intent shows utterance +
  two predicted intents; wake word plays a clip + two detector verdicts.
- Voting options MUST include: candidate A, candidate B, tie, both-wrong
  (these exist in `vote_result_enum` already).
- Plugin identities MUST NOT be revealed before the vote is committed
  (post-vote reveal is encouraged for engagement).

## 4. Matchmaking rules

- **R1 — Same stimulus.** A battle pairs two predictions for the *same*
  `sample_id` from the *same* source dataset, by two different plugins.
- **R2 — ELO proximity.** Opponents SHOULD be selected with similar current
  ELO (configurable window), so battles are informative; cold-start plugins
  (few battles) MAY be matched more broadly.
- **R3 — Prefer discriminative samples.** For modalities with references,
  sampling SHOULD prefer samples where *both* plugins erred (e.g. both
  `wer > 0`) — these are the cases where human judgement adds signal beyond
  the automatic metric. A configurable fraction of battles MAY come from
  all-samples to avoid bias.
- **R4 — No repeat exposure.** A user SHOULD NOT see the same
  (sample, plugin-pair) battle twice.
- **R5 — Auto-battles (seeding).** Before human votes exist, the system MAY
  generate auto-battles judged by the automatic metric (winner = lower WER)
  to bootstrap initial ELO. Auto-votes MUST be stored with
  `voter = system:wer` (never attributed to a user), MUST be distinguishable
  in the vote log, and SHOULD carry lower K-factor weight than human votes.

## 5. Data model (delta against docs/models.sql)

Kept: `users` (+roles), `votes` (+`vote_result_enum`), plugin registry,
ratings/snapshots.

Changed:
- `battles` no longer has worker lifecycle (`PENDING/RUNNING/...`). A battle
  row is created at *assembly time* and references two prediction rows by
  (hf_dataset, row ref) — it is READY by construction. A nightly assembler
  job MAY pre-generate battle pools per modality/lang.
- New: `prediction_sources` table — registered HF datasets (name, revision,
  modality, lang, ingested_at) so battles always pin an exact dataset
  revision.
- `ratings` keyed by (plugin_id, modality, lang); every change references the
  causal vote id (P5 replayability).

## 6. Rating system

- Standard ELO; parameters (initial=1000, K=32 fresh / 16 veteran,
  veteran-threshold) are configuration, not code constants.
- Human votes: tie → 0.5/0.5; both-wrong → 0.5/0.5 *plus* the sample is
  flagged into a `hard_samples` view (P6: failure cases are content).
- Separate standings per (modality, lang); a global aggregate MAY be shown
  but per-language boards are the primary product.
- Auto-battle votes use reduced K (suggested K/4) and are excluded from
  "human preference" leaderboard views (shown as a separate "seeded" column
  until a plugin has ≥N human votes).

## 7. Roadmap (MVP first)

1. **M1 — Spec agreed** (this document reviewed by the three of us).
2. **M2 — Prediction-runner contract + first ingests**: register existing
   `ovos-stt-bench-*` datasets; assembler builds a battle pool.
3. **M3 — STT battles MVP**: read-only sampling + voting + ELO on the
   existing auth/UI base. Human votes flowing end-to-end.
4. **M4 — Leaderboards + static export.**
5. **M5 — Auto-battle seeding.**
6. **M6 — TTS battles** (pre-synthesized audio from HF, no live inference).
7. **M7 — Wake word + intent battles.**
8. **M8 — Unlabeled mode (stretch)**: battles on unlabeled audio where votes
   *create* a new labelled dataset (consent + licensing reviewed before this
   ships).

Explicitly out of scope for MVP: live plugin inference (browser mic → backend
inference for WW/TTS was discussed and parked: HF Zero-GPU endpoints are an
option later, with cold-start caveats), user-uploaded audio, write access to
HF from the arena service.

## 8. Decisions

1. **Hosting:** HF Space (or any small box) for the live service + static
   leaderboard export to GitHub Pages — results stay public at zero infra
   cost regardless of backend uptime.
2. **Database:** SQLite for the MVP (single small service, replayable vote
   log makes migration trivial); the Postgres/alembic schema remains the
   target once multi-user load is real.
3. **Voting:** accounts-only at MVP (existing auth); anonymous voting with
   rate limits is a later experiment.
4. **`both_wrong`:** scores 0.5/0.5 *and* flags the sample into
   `hard_samples`.
5. **Plugin versions:** a new `plugin_version` enters as a new competitor
   seeded at its predecessor's rating; the predecessor's history is frozen,
   never merged.

## 9. Relationship to existing code

- `main`/`dev` (Suvan): FastAPI template, alembic auth, frontend scaffold,
  `docs/models.sql` — the base this spec builds on.
- PR #3 (`feat/arena-core`): the ELO engine, replayable vote log, blind
  matchup API, and SQLite persistence align with this spec and SHOULD be
  rebased onto it (Postgres + assembled battles). Its OPM plugin discovery
  and in-arena execution adapters are superseded by P1/P2 (predictions are
  external) and SHOULD be dropped or moved to the prediction-runner tooling.
- Roadmap issue #2 is superseded by §7 of this document.
