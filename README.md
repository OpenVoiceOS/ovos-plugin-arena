# OVOS Plugin Arena

Easy to fork, power to the community.

Blind A/B plugin evaluation for the OpenVoiceOS ecosystem — STT, TTS,
wake-word, and intent — with ELO-style leaderboards.  No live plugin execution
required: all predictions come from pre-published HuggingFace datasets.

## Three deployment modes

### Mode A — Docker (self-host)

A multi-stage Docker image bundles the FastAPI backend + the built static
frontend.  Uses SQLite — no Postgres or Kafka.

```bash
docker compose -f docker-compose.arena.yml up -d
```

The UI is at `http://localhost:8000/`; the API is at `/api/v1/`.

### Mode B — Static read-only mirror (GitHub Pages)

Push the built Astro site to GitHub Pages.  The leaderboard and battle data
are pre-rendered JSON files committed by scheduled Actions.  Voting is
disabled; the site shows current rankings only.

### Mode C — GitHub-native (preferred, zero servers)

Votes are GitHub Issues.  No server needed.  This is how the canonical
OpenVoiceOS arena runs.

**End-to-end vote flow:**

1. Voter opens a battle page and picks A / B / Tie / Both wrong.
2. The browser opens a prefilled GitHub issue URL:
   `https://github.com/OpenVoiceOS/ovos-plugin-arena/issues/new?template=vote.yml&labels=vote&title=vote|<battle_id>|<choice>`
3. Voter submits the issue (free GitHub account required — no arena account).
4. A scheduled Action (`tally.yml`, runs hourly) reads all vote issues,
   deduplicates (one vote per user per battle — later duplicates silently
   dropped), replays ELO deterministically from the ordered issue history,
   commits updated `data/leaderboard-*.json`, closes processed issues with a
   thank-you comment.
5. GitHub Pages redeploys automatically — leaderboard live within ~30 s.

The vote log **is** the issue history — fully public, fully auditable, fully
replayable at any time from scratch.

## Fork your own arena

1. Fork this repo on GitHub.
2. In Settings → Pages, set Source to "GitHub Actions".
3. In Settings → Variables, set:
   - `ASTRO_SITE` — your Pages URL (e.g. `https://yourorg.github.io`)
   - `ASTRO_BASE` — sub-path (e.g. `/ovos-plugin-arena`)
   - `HF_DATASETS` — comma-separated HuggingFace dataset names
     (e.g. `OpenVoiceOS/ovos-stt-bench-pt-PT`)
4. Run the `assemble.yml` Action manually to build the initial battles pool.
5. Voters open issues; `tally.yml` runs hourly and updates the leaderboard.

That is all.  No servers, no databases, no billing.

## Running locally

```bash
source ~/.venvs/ovos/bin/activate
cd backend
ARENA_DB_PATH=arena.sqlite3 fastapi dev app/main.py
```

```bash
cd frontend-static
npm install
npm run build   # static output in dist/
npm run dev     # dev server at http://localhost:4321/
```

## Tests

```bash
source ~/.venvs/ovos/bin/activate
python -m pytest tests/ -q
# 143 tests, <10 s
```

## Architecture

```
┌────────────────────┐  publish   ┌──────────────────────────────┐
│ Prediction Runners │ ──────────►│ HuggingFace datasets         │
│ (offline, cron)    │            │ ovos-<mod>-bench-<ds>-<lang> │
└────────────────────┘            └──────────┬───────────────────┘
                                             │ assemble.yml (daily)
                                             ▼
                      ┌──────────────────────────────────────────┐
                      │ frontend-static/public/data/*.json       │
                      │  battles-stt-pt-PT.json                  │
                      │  leaderboard-stt-pt-PT.json              │
                      │  index.json                              │
                      └──────────┬───────────────────────────────┘
                                 │ Astro build (pages.yml)
                                 ▼
                      ┌──────────────────────┐   GitHub Issues
                      │ GitHub Pages (static)│◄──────────────────
                      │  / leaderboard/      │   vote|battle|a
                      │  / battle/           │
                      └──────────────────────┘
                                 │ tally.yml (hourly)
                                 ▼
                      ELO replay → leaderboard JSON → commit → Pages
```

## Key files

| Path | Purpose |
|---|---|
| `frontend-static/` | Astro static UI (leaderboard + blind battle pages) |
| `frontend-static/public/data/` | Generated JSON data (battles pool, leaderboards) |
| `backend/app/arena/cli.py` | `assemble` + `tally` + `export-index` CLI for Actions |
| `backend/app/arena/elo.py` | Deterministic ELO engine (replayable from vote log) |
| `.github/workflows/assemble.yml` | Daily battle pool refresh from HF datasets |
| `.github/workflows/tally.yml` | Hourly vote-issue tally + leaderboard update |
| `.github/workflows/pages.yml` | Astro build + Pages deploy (disabled while private) |
| `.github/ISSUE_TEMPLATE/vote.yml` | Prefilled vote issue template |
| `Dockerfile` | Multi-stage: Astro build + FastAPI backend (Mode A) |
| `docker-compose.arena.yml` | Single-container compose for Mode A |
| `docs/SPECIFICATION.md` | Full spec including §9 deployment modes |

## Implementation status

| Component | Status |
|---|---|
| Spec §9 deployment modes | Done |
| ELO engine + replayable vote log | Done, 143 tests green |
| HF dataset ingestion | Done |
| Battle assembler | Done |
| Static UI (Astro) — leaderboard | Done |
| Static UI (Astro) — blind battle + vote | Done |
| GitHub Actions: assemble.yml | Done |
| GitHub Actions: tally.yml | Done |
| GitHub Actions: pages.yml | Done (deploy commented — enable after making repo public) |
| GitHub issue template: vote.yml | Done |
| Docker multi-stage build | Done |
| Tests: tally logic (parse, dedupe, ELO replay, JSON shape) | Done (+31 new) |

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / [NLnet](https://nlnet.nl)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429),
through the European Commission's [Next Generation Internet](https://ngi.eu) programme.
