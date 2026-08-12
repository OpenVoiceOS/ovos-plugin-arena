# Testing the arena locally

A step-by-step guide to running every part of the arena on your own machine:
unit tests, a benchmark smoke run, the assembler, replay verification, and the
Astro frontend. Each section says what the command proves and what its output
looks like, so you know it worked without needing CI.

This page complements the top-level `README.md` ("Running a benchmark",
"Assembling the arena locally") with the *why* and the full picture, install,
tests, replay, frontend, in one place. See [`benchmarks.md`](benchmarks.md)
for per-modality benchmark detail, [`runner.md`](runner.md) for the
always-on STT prediction runner, and [`operations.md`](operations.md) for the
maintainer's day-to-day vote-loop runbook.

## 1. Environment setup

The project uses [uv](https://docs.astral.sh/uv/). Some dependencies (e.g.
`ovoscope`) are only published as prereleases, so installs need
`--prerelease=allow`.

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --prerelease=allow -e ".[test,audio,hf]"
```

**What each extra buys you** (see `pyproject.toml`):

| Extra | Needed for |
|---|---|
| `test` | running `pytest` at all (pulls in `ovoscope` for the wake-word adapter tests) |
| `audio` | STT / wake-word / TTS benchmarks and their tests (`soundfile`, `onnx-asr`, `faster-whisper`, `speechonnxmetrics`) |
| `hf` | anything that talks to HuggingFace: `assemble`, `verify-replay` against a live repo, `--upload` |

If you only need the intent leagues and the assembler, `.[test,hf]` is
enough, you can skip `audio` and its heavier downloads.

## 2. Run the unit tests

```bash
python -m pytest tests/ -q
```

**Proves:** the core library (`arena/`), the registry loaders, and the
benchmark engines (`runner/`) behave correctly, with no network or plugin
calls.

**Expected output shape:**

```
584 passed in 5.43s
```

If you installed only `.[test,hf]` (no `audio`), four `TestIntelligibility`
cases in `tests/test_tts_bench.py` fail with `No module named 'soundfile'`, that's expected. They're audio-decode tests gated on the `audio` extra, not a
real failure.

## 3. Validate the registry

```bash
python -m arena.cli validate-registry --registry registry
```

**Proves:** every fighter and dataset JSON file in `registry/` matches the
schema, the same check CI's `validate_registry` job runs on every push and
PR. See [`add-a-fighter.md`](add-a-fighter.md) for what a fighter file looks
like.

**Expected output:**

```
Registry OK — every competitor/dataset file validated cleanly.
```

## 4. Run one benchmark as a smoke test

Every benchmark script accepts `--competitors`, `--langs` and `--max-samples`
so you can run a tiny slice without downloading a full dataset or training
every fighter (see [`benchmarks.md`](benchmarks.md) for the full flag table).

```bash
python benchmarks/intent_snips.py --competitors padacioso-medium --max-samples 5
```

**Proves:** the benchmark engine can load a registry fighter, download and
cache the eval dataset from HuggingFace, train/run the plugin offline, and
write resumable §3.2 prediction rows to disk, the same path a full sweep
uses, just capped to 5 samples for one fighter.

**Expected output shape:**

```
INFO  Fighter padacioso-medium [intent_template]
INFO    training padacioso-medium for en-US (stages: ovos-padacioso-pipeline-plugin-medium)
INFO    padacioso-medium/en-US: wrote 5 rows
```

No `--upload` flag was passed, so nothing is published to HuggingFace, the
rows land locally at `predictions/<dataset>/<modality>/<lang>/<competitor>.jsonl`
(here: `predictions/snips/intent_template/en-US/padacioso-medium.jsonl`).
`predictions/` is gitignored. Re-running the same command skips any
`sample_id` already present in that file (resumable runs, per
[`benchmarks.md`](benchmarks.md)).

## 5. Assemble battles and boards locally

`assemble` also accepts a local predictions directory instead of an HF repo
id, point it at what step 4 just wrote:

```bash
python -m arena.cli assemble \
    --predictions predictions/snips \
    --modality intent_template \
    --output /tmp/arena-assemble-test
```

**Proves:** the assembler can turn raw prediction rows into a benchmark
board, a battle pool and an ELO seed without touching HuggingFace or the
committed `frontend-static/public/data/`, useful for checking a new
benchmark script's output shape before publishing anything.

**Expected output shape:**

```
INFO  Loaded 5 rows from intent_template/en-US/padacioso-medium.jsonl
INFO  Wrote /tmp/arena-assemble-test/benchmark-intent_template-snips-en-US.json
INFO  Assembled 0 battles for intent_template/snips/en-US (0 pairs, 0 reference mismatches skipped)
INFO  Wrote /tmp/arena-assemble-test/battles-intent_template-snips-en-US.json
INFO  Wrote /tmp/arena-assemble-test/elo-seed-intent_template-en-US.json
INFO  Wrote /tmp/arena-assemble-test/battles-intent_template-freeform-en-US.json
INFO  Wrote /tmp/arena-assemble-test/leaderboard-intent_template-en-US.json
```

`0 battles` here is expected, not a bug: a battle is an A/B pair between two
*different* competitors on the same sample, and this smoke run only predicted
with one fighter (`padacioso-medium`). Point `--predictions` at a directory
(or comma-separated HF repo ids) covering two or more competitors on the same
dataset to see nonzero battle counts.

## 6. Verify replay

`verify-replay` re-derives every published leaderboard from the vote log and
diffs it against what's on disk, see
[`operations.md`](operations.md#replay-proof) for the full explanation. Two
ways to run it locally:

```bash
# offline, against an empty/saved vote-issue snapshot — no GitHub calls
echo '[]' > /tmp/votes.json
python -m arena.cli verify-replay \
    --data-dir /tmp/arena-assemble-test \
    --votes-file /tmp/votes.json

# against the real committed data + a live vote log
python -m arena.cli verify-replay \
    --data-dir frontend-static/public/data \
    --repo OpenVoiceOS/ovos-plugin-arena
```

**Proves:** the committed leaderboards are exactly reproducible from the
public vote log, the same check `.github/workflows/replay-proof.yml` runs on
every push to `dev`.

**Expected output:**

```
INFO  Loaded 0 battles, 1 ELO seeds
INFO  Reading vote issues from /tmp/votes.json (offline)
INFO    → 0 vote issue(s)
INFO    → 0 deduped vote(s)
INFO    → 0 counted vote(s), 0 discarded by fraud rules
INFO  verify-replay OK — 1 published board(s) reproduced exactly by replaying the vote log
```

Any other exit code means a published board doesn't match what the vote log
supports, see [`operations.md`](operations.md#troubleshooting) for how to
dig into a mismatch.

## 7. Preview the frontend

The Astro site reads its data straight from
`frontend-static/public/data/*.json`, the same files `assemble` and `tally`
commit. There's no build step needed to point it at different data. Just
overwrite those JSON files before building.

```bash
cd frontend-static
npm ci
npm run build      # static export, writes frontend-static/dist/
npm run preview    # serves the built dist/ locally
# or, for hot-reload while editing components:
npm run dev
```

**Proves:** the site builds cleanly against whatever is currently committed
under `frontend-static/public/data/`, and serves it.

**Expected output shape:**

```
21:50:39 [build] 326 page(s) built in 1.70s
21:50:39 [build] Complete!
...
 astro  v6.4.8 ready in 12 ms
┃ Local    http://localhost:4321/ovos-plugin-arena
```

`curl -s -o /dev/null -w "%{http_code}\n" http://localhost:4321/ovos-plugin-arena/`
returns `200` once the preview server is up. Stop it with `pkill -f "astro preview"`
or Ctrl-C in the foreground terminal.

## Where the data files live

| Path | What's in it |
|---|---|
| `registry/competitors/<modality>/*.json` | fighter definitions (see [`add-a-fighter.md`](add-a-fighter.md)) |
| `registry/datasets/<modality>/*.json` | eval dataset definitions |
| `predictions/` (local, gitignored) | raw benchmark output rows before assembly |
| `frontend-static/public/data/*.json` | committed, assembled artifacts: `battles-*`, `benchmark-*`, `elo-seed-*`, `leaderboard-*`, `vote-audit.json`, `patch-notes.json`, `competitors.json` |
| `frontend-static/dist/` | Astro's static build output (gitignored, regenerated by `npm run build`) |

## Resume and local-output behavior, in one place

- **Benchmark scripts** are resumable: re-running the same command skips
  `sample_id`s already present in the output JSONL for that competitor/lang,
  and only loads a plugin/model if that language still has work left
  ([`benchmarks.md`](benchmarks.md)).
- **`assemble`** is idempotent and safe to re-run: battle ids are content
  hashes of `(modality, dataset, lang, sample, competitor pair)`, so
  re-assembling never invalidates an already-open vote
  ([`operations.md`](operations.md)).
- **`tally`** always replays the *entire* vote issue history (open and
  closed) from scratch on every run, it has no incremental state of its
  own. `vote-audit.json` is fully regenerated, not appended to
  ([`operations.md`](operations.md#auditing-discards)).

---
[← Specification](SPECIFICATION.md) · [Home](index.md) · [Add a fighter →](add-a-fighter.md)
