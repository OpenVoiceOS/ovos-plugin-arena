# ovos-plugin-arena

GitHub-native benchmarking + blind-A/B voting arena for OVOS plugins. No
servers: registry JSONs declare fighters/datasets, `benchmarks/*.py` scripts
produce predictions (published to HF), GitHub Actions assemble battles +
leaderboards as committed JSON, the Astro static site renders them, and votes
are GitHub issues. See `README.md` for the data flow and
`docs/SPECIFICATION.md` for the contracts.

## Setup

```bash
pip install -e ".[hf,test]"
cd frontend-static && npm install
```

## Test

```bash
python -m pytest tests/ -q          # fast, no network
```

## Lint

No enforced linter config; match the existing style (PEP 8, ~80-90 cols).

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
  `pages.yml` (build+deploy), `unit_tests.yml` (gh-automations build-tests).

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
- This repo is private until the user flips visibility; never make it public
  unprompted.
