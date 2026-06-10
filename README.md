# OVOS Plugin Arena

A structured, reproducible evaluation platform for OVOS plugins — TTS, STT,
wake-word (WW), and intent — with blind A/B comparison and ELO-style ranking.
Designed to inform default OVOS configurations by surfacing which plugins
actually perform best across real prompts.

## Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│ FastAPI backend                                              │
│                                                              │
│  /api/v1/arena/*  ←── arena core (this PR, P1+P2)           │
│       │                                                      │
│  ┌────▼─────────────────────────────────────────────────┐   │
│  │ arena/                                               │   │
│  │  models.py    — Pydantic data models                 │   │
│  │  db.py        — SQLite persistence (stdlib sqlite3)  │   │
│  │  discovery.py — OPM entry-point scanner              │   │
│  │  elo.py       — Deterministic ELO / replay engine    │   │
│  │  ranking.py   — Bridges ELO math ↔ SQLite            │   │
│  │  router.py    — FastAPI endpoints                     │   │
│  │  adapters/                                           │   │
│  │    base.py        — Adapter interface                │   │
│  │    tts.py         — TTS adapter (working)            │   │
│  │    intent.py      — Intent adapter (working)         │   │
│  │    stt.py         — STT adapter (stub)               │   │
│  │    wakeword.py    — WW adapter (stub)                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  /api/v1/*  ←── original scaffold (PostgreSQL + Kafka)      │
└──────────────────────────────────────────────────────────────┘
```

The arena core is **self-contained**: it uses SQLite via Python stdlib and
requires no PostgreSQL, Kafka, MinIO, or Docker.  The original scaffold
(users, auth, admin competitors) remains untouched alongside it.

## Data model

| Model | Purpose |
|---|---|
| `Plugin` | A registered OVOS plugin entry point with config and OPM metadata |
| `EvalRun` | A batch evaluation run for one plugin over a fixed prompt set |
| `Sample` | A single output artifact (audio file, transcript, prediction) |
| `Matchup` | A blind A/B pair of two Samples for the same input |
| `Vote` | A human or automated preference on a Matchup |
| `RatingSnapshot` | Immutable ELO snapshot after each Vote |

## ELO ranking

Ratings follow the standard ELO formula with K=32 (new) / K=16 (veteran,
≥30 battles).  The full vote log is **replayable**: calling
`/api/v1/arena/ranking/recompute` replays all votes in chronological order
and produces bit-identical ratings regardless of the current `elo_current`
state.  This is verified by the property test in `tests/test_elo.py`.

Automated benchmark results (WER, RTF, detection F1) flow through the same
ELO pipeline via `POST /api/v1/arena/votes/metric`.

## Running locally

```bash
# Activate the shared venv
source ~/.venvs/ovos/bin/activate

# From the project root — no extra installs needed beyond the venv
cd backend
ARENA_DB_PATH=arena.sqlite3 fastapi dev app/main.py
```

The arena SQLite DB is created at `ARENA_DB_PATH` on first startup.

Interactive docs: http://localhost:8000/docs

### Quick walkthrough

```bash
BASE=http://localhost:8000/api/v1/arena

# 1. Discover installed plugins
curl -sX POST $BASE/plugins/discover -H 'Content-Type: application/json' \
  -d '{"families": ["tts"]}' | python3 -m json.tool | head -30

# 2. List plugins
curl -s "$BASE/plugins?family=tts"

# 3. Get next blind matchup (requires at least one matchup seeded)
curl -s $BASE/matchups/next/tts

# 4. Submit a vote
curl -sX POST $BASE/matchups/<id>/vote \
  -H 'Content-Type: application/json' -d '{"outcome": "candidate_a"}'

# 5. Leaderboard
curl -s "$BASE/leaderboard/tts?lang=en-us"
```

## Running tests

```bash
source ~/.venvs/ovos/bin/activate
cd /path/to/ovos-plugin-arena
python -m pytest tests/ -q
```

All 59 tests run in under 30 seconds (including the OPM discovery scan).
No external services required.

## Writing a new adapter

Subclass `arena.adapters.base.BaseAdapter`:

```python
from app.arena.adapters.base import BaseAdapter
from app.arena.models import PluginFamily

class MyFamilyAdapter(BaseAdapter):
    family = PluginFamily.TTS  # or STT / WAKE_WORD / INTENT

    def _load_plugin(self) -> None:
        # load self._plugin using OPM factory
        ...

    def _unload_plugin(self) -> None:
        # release resources
        ...

    def _run_one(self, input_ref: str, output_dir) -> tuple[str | None, dict[str, float]]:
        # run the plugin, return (artifact_path_or_None, metrics_dict)
        ...
```

Use as a context manager:

```python
adapter = MyFamilyAdapter(plugin_name="ovos-tts-plugin-phoonnx", config={})
with adapter:
    samples = adapter.run_eval(run, prompts, output_dir="/tmp/arena_out")
```

## REST API reference

All endpoints are under `/api/v1/arena/`.

| Method | Path | Description |
|---|---|---|
| POST | `/plugins/discover` | Scan OPM entry points, upsert into DB |
| GET | `/plugins` | List registered plugins (`?family=tts&lang=en-us`) |
| GET | `/plugins/{id}` | Get plugin by UUID |
| POST | `/runs` | Create an EvalRun (PENDING) |
| GET | `/runs/{id}` | Get run status |
| GET | `/runs` | List runs (`?plugin_id=…`) |
| POST | `/matchups` | Create blind matchup from two sample IDs |
| GET | `/matchups/next/{family}` | Fetch oldest pending matchup (blind) |
| GET | `/matchups/{id}` | Get matchup details (admin, reveals plugin IDs) |
| POST | `/matchups/{id}/vote` | Submit a human vote |
| POST | `/votes/metric` | Ingest automated metric vote |
| GET | `/leaderboard/{family}` | ELO leaderboard (`?lang=en-us&limit=50`) |
| POST | `/ranking/recompute` | Replay vote log, rebuild ELO table |

## Implementation status

| Component | Status |
|---|---|
| Pydantic models | Done |
| SQLite persistence | Done |
| OPM discovery (TTS/STT/WW/Intent) | Done |
| TTS adapter | Working end-to-end |
| Intent adapter (padatious + adapt) | Working end-to-end |
| STT adapter | Stubbed (interface complete) |
| WW adapter | Stubbed (interface complete) |
| ELO ranking + replay | Done, property-tested |
| Automated metric vote ingestion | Done |
| REST endpoints (P1+P2) | Done |
| Tests (59 passing) | Done |
| P3 blind UI | Not started |
| P4 Docker / static export | Not started |
| P5 seeding | Not started |
