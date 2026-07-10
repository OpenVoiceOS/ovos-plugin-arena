# ovos-plugin-arena

GitHub-native benchmarking + blind-A/B voting arena for OVOS plugins. No
servers: registry JSONs declare fighters/datasets, `benchmarks/*.py` scripts
produce predictions (published to HF), GitHub Actions assemble battles +
leaderboards as committed JSON, the Astro static site renders them, and votes
are GitHub issues. See `README.md` for the data flow and
`docs/SPECIFICATION.md` for the contracts.

## Setup

```bash
uv sync --extra hf --extra test --extra lint
cd frontend-static && npm install
```

## Test

```bash
uv run pytest tests/ -q                                             # fast, no network
uv run pytest tests/ -q --cov=arena --cov=registry --cov=runner \
  --cov-report=term-missing                                         # coverage gate (see pyproject fail_under)
```

## Lint

```bash
uv run ruff check .        # style + import order (tool.ruff in pyproject.toml)
uv run mypy arena registry # strict on arena/registry; runner/ is advisory (ignore_errors)
```

Run both locally before opening/undrafting a PR — CI billing on this org can be
unavailable, so the local run is the real gate; paste its output in the PR
description.

## Layout

- `arena/` — core library. `predictions.py` (JSONL/HF loading),
  `metrics.py` (benchmark scoring), `assembler.py` (deterministic battles +
  ELO seeding), `elo.py` (rating engine), `cli.py`
  (assemble / tally / export-index / export-bestiary), `models.py`
  (pydantic contracts for every JSON artifact).
- `registry/` — declarative registry: `competitors/<modality>/<id>.json`,
  `datasets/<modality>/<id>.json`, pydantic schemas + loaders.
- `benchmarks/` — one reproducible script per benchmark; writes
  `predictions/<competitor_id>.jsonl` (§3.2 rows), uploads to HF.
- `runner/` — plugin adapters + shared bench engines. Intent:
  `intent_bench.py` + `intent_pipeline.py` (OPM intent plugins over FakeBus).
  Audio: `media_bench.py` (shared driver) + `stt_bench.py` / `ww_bench.py` /
  `tts_bench.py` adapters + `audio_io.py` (decode/stream). Legacy off-repo STT
  runner (`plugin_runner.py` + `queue.yaml`) still feeds `ovos-stt-bench-*`.
- `frontend-static/` — Astro site; generated data lives in `public/data/`
  (committed by CI, owned by the workflows).
- `.github/workflows/` — `assemble.yml` (daily), `tally.yml` (hourly),
  `pages.yml` (build+deploy), `unit_tests.yml` (gh-automations build-tests +
  lint + type-check, all reusable `@dev`).

## Gotchas

- **Battle ids are content hashes** — never make them random; open votes
  reference them across assemble runs.
- **Determinism is a contract (§P5):** seed ELO and vote replay must stay
  byte-reproducible; iterate sorted, never over set/dict order from input.
- **The API/CI layer never runs plugins.** Only `benchmarks/` scripts execute
  plugins, offline, on a maintainer machine.
- `frontend-static/public/data/*.json` is generated — regenerate via
  `arena.cli`, don't hand-edit.
- Issue forms apply the `vote` label; URL `labels=` params are ignored for
  non-collaborators — don't move labelling back into the vote URL.
- Intent leagues are independent modalities (`intent_template`,
  `intent_keyword`, open `intent` for fusions) with separate boards/ELO;
  paradigm leagues must stay pure (bench script enforces it). Fusions get
  portmanteau names (Padapt), never "default cascade" style ids.
- Pages deploy job is gated on the repo being public; the build job always
  runs.

## Conventions (org hard rules)

- Branch `dev` for work, `master` for stable — never `main`.
- Never edit `arena/version.py`; gh-automations bumps it from
  conventional-commit prefixes (`feat:` / `fix:` / `feat!:`).
- Commit identity: JarbasAi <jarbasai@mailfence.com>.
- Reference OpenVoiceOS/gh-automations reusable workflows at `@dev`.
- No meta-commentary in docs/commits/code — describe current state only.
- This repo is public; the Pages deploy job runs on every build.
- Work is planned in `ROADMAP.md` (local file, not committed — see
  `.gitignore`; durable copy in the workspace wiki at
  `knowledge/wiki/plans/roadmap-plugin-arena.md`). Pick tasks from it and
  follow each task's test/docs/acceptance requirements.
