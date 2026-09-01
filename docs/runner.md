# Arena STT Prediction Runner

The prediction runner is an offline batch job that generates STT transcriptions
for (plugin × dataset) pairs and appends the results to the corresponding
HuggingFace benchmark dataset (`OpenVoiceOS/ovos-stt-bench-<lang>`).

It runs 24/7 on the ser9 compute box (192.168.1.111, tailnet 100.77.120.109), cycling through the job
queue and sleeping between cycles.

---

## Row schema

Each row written to the JSONL output and uploaded to HF is the canonical
arena §3.2 contract (`docs/SPECIFICATION.md`) directly, the runner has no
registry dependency by design (it can run standalone on a plugin-execution
box), so it never resolves `competitor_id`. `arena.predictions` re-keys
`plugin_id` to a `competitor_id` at load time via
`registry.loaders.get_competitor_by_alias` (§4 A2 schema convergence).

| column | type | notes |
|---|---|---|
| `sample_id` | str | stable filename within source corpus (was `dataset_entry_id`) |
| `plugin_id` | str | OPM entry-point name (was `plugin_name`) |
| `extras.model_id` | str | composite `plugin/model[/cfghash]` |
| `prediction` | str | STT output (was `prediction_transcript`) |
| `reference_text` | str | ground truth (was `transcript`) |
| `confidence` | float | 0.0–1.0 (was `prediction_confidence`) |
| `modality` | str | always `"stt"` (was `prediction_type: "STT"`) |
| `dataset_id` | str | source corpus + split path |
| `lang` | str | BCP-47 |

Already-published data in the old column layout (`dataset_entry_id` /
`plugin_name` / `prediction_transcript` / `transcript` /
`prediction_confidence` / `prediction_type`) is still readable.
`runner/schema.py:STTRow` is kept as a read-compat shim, and
`arena.predictions.parse_row` detects and converts that shape
automatically, tagging the resulting row `schema_version: 1` for
provenance. New runs never construct an `STTRow`.

---

## Sampling policy

How many rows a sweep draws per (dataset, language) is a registry decision,
not an operator one. A dataset's `sample_policy` (`max_samples`, `seed`) in
its `registry/datasets/<modality>/<id>.json` entry caps the row count and
pins a deterministic subset — the corpus rows are shuffled with that seed and
truncated to `max_samples`, so every fighter and every run draw the exact
same rows for that dataset. Large corpora that would otherwise stream
whole-corpus sweeps (VoxPopuli, MLS, mTEDx, Speech-MASSIVE, and the pooled
`ml_spoken_words` wake-word negatives) carry a policy; small curated eval
sets are left uncapped and stream in full.

The `--max-samples` CLI flag still exists for smoke runs. The effective cap
is always the SMALLER of the CLI value and the dataset's policy cap: an
operator cap smaller than the policy shrinks a run for a smoke test; an
operator cap larger than the policy (or the default of 0, meaning
unbounded) is clamped down to the policy's cap and logged, never allowed to
stream past it. Without a `sample_policy`, a dataset keeps the old
behaviour — `--max-samples`, if given, takes the first rows in corpus order.

### Publishing sample-set manifests

A `sample_policy` fixes WHICH rows a sweep draws, but a sweep run doesn't by
itself record which rows those were, and a benchmark board scores each
fighter over its own rows. Without a shared record of the selected subset,
two fighters swept at different times — one before the policy existed, one
against it, one against a smaller ad hoc `--max-samples` — end up ranked
together over different sample populations. `runner/publish_sample_set.py`
closes that gap: it recomputes a dataset's policy-selected sample ids
directly from the source corpus (no audio decoding, just the id-bearing
columns) and publishes them as `sample_sets/<lang>.json` inside the
dataset's predictions repo, right next to `predictions/<lang>/*.jsonl`.

```bash
python -m runner.publish_sample_set --upload
```

runs every dataset that declares a `sample_policy`; scope it with
`--modality` or `--dataset` for a single one, or drop `--upload` (the
default) to print row counts without touching HF. `arena.cli assemble`
downloads each dataset's manifest automatically when one exists and
restricts every fighter's rows — and battle/ELO assembly — to it before
scoring, so board rankings are always over a comparable sample set. A
fighter whose rows cover less than 90% of the manifest is marked unranked
rather than silently scored on a smaller population.

Manifests are cheap to regenerate: if a dataset's `sample_policy` changes
(a different cap or seed), republishing its manifest is enough — the next
`assemble` run picks up the new ids and re-filters/re-scores every board
automatically, no resweep required, no runner changes needed.

---

## Adding jobs

Edit `runner/queue.yaml`. Each job entry looks like:

```yaml
jobs:
  - plugin:
      plugin_name: ovos-stt-plugin-fasterwhisper
      model_name: small          # passed as config["model"] to the plugin
      lang: pt-PT
      extra_config:
        compute_type: int8
        cpu_threads: 1
        beam_size: 3

    dataset:
      hf_repo: PolyAI/minds14
      subset: pt-PT
      split: train
      ground_truth_key: transcription
      max_samples: 0             # 0 = all; positive integer for a capped run

    hf_output_dataset: OpenVoiceOS/ovos-stt-bench-pt-PT
```

The daemon reloads the queue file on the next cycle whenever its mtime changes, no restart needed.

---

## Generating a full-sweep queue

`runner/queue_tools.py` diffs the declarative registry (every competitor ×
every compatible `role: eval` dataset, per modality) against what is
actually published on HuggingFace, and prints `queue.yaml`-shaped job
entries for pairs that are missing or incomplete. It never writes
`runner/queue.yaml` itself and never runs a benchmark. Review its output
and paste in the entries you want.

"Missing or incomplete" means one of three things: no
`predictions/<competitor_id>.jsonl` file in the dataset's `predictions_hf`
repo, a 0-byte file (this happens, for example
`onnx-asr-parakeet-tdt-11b.jsonl` today), or fewer rows than `--min-rows`
(default 1). Dataset size is not tracked in the registry, so this is a
heuristic, not an exact "smaller than the corpus" check. A fighter with an
empty `langs` list is treated as compatible with every dataset language.

```bash
# Human-readable table of what's missing, no downloads beyond file listings
python -m runner.queue_tools --dry-run --modality stt

# Full sweep across all modalities, skip per-file row counting (listing only)
python -m runner.queue_tools --no-row-check --out /tmp/sweep-queue.yaml

# One modality, default row-count check (downloads only files that exist
# and are non-empty, to count rows — never a full snapshot_download)
python -m runner.queue_tools --modality wake_word > /tmp/ww-queue.yaml
```

Entries are ordered cheapest-engine-first (a static weight heuristic, e.g.
`vosk`/`webrtc` before `whisper`/cloud STT) then by competitor id, so a
sweep run burns through fast/cheap fighters before slow/expensive ones.

Emitted jobs use the registry-referenced job shape:

```yaml
jobs:
  - competitor: vosk-pt
    dataset_ref: minds14-pt-PT
    hf_output_dataset: OpenVoiceOS/ovos-stt-bench-minds14-pt-PT
    max_samples: 0
```

---

## Deploy on ser9

Stage the runner code from your laptop:

```bash
scp -r /home/miro/AgentWorkspaces/ovos/web/ovos-plugin-arena/runner \
    miro@192.168.1.111:/home/miro/arena-runner/runner
```

Run one-time setup on ser9 (already done):

```bash
python3.14 -m venv ~/venvs/arena-runner
~/venvs/arena-runner/bin/pip install \
    ovos-stt-plugin-fasterwhisper \
    ovos-stt-plugin-vosk \
    datasets huggingface_hub pyyaml
```

Start the daemon detached in tmux:

```bash
ssh miro@192.168.1.111
tmux new -s arena-runner
cd ~/arena-runner
nice -n 15 ~/venvs/arena-runner/bin/python3.14 -m runner \
    --queue runner/queue.yaml \
    --base-dir /home/miro/arena-runner \
    --max-workers 4 \
    --ort-threads 1 \
    --timeout 60 \
    --sleep 300
# Ctrl-B D to detach
```

The runner writes logs to `/home/miro/arena-runner/runner.log`.

---

## Monitor from the laptop

```bash
# Tail the log in real time
tail -f /mnt/ser9/arena-runner/runner.log

# Count rows written in current output files
wc -l /mnt/ser9/arena-runner/output/*.jsonl

# Check manifest state (how many samples done per job)
python3 -c "
import json, pathlib
for p in pathlib.Path('/mnt/ser9/arena-runner').glob('manifest_*.json'):
    d = json.loads(p.read_text())
    print(len(d['done_ids']), d['job_key'])
"
```

---

## HuggingFace publish

After each job cycle completes the runner automatically uploads the output
JSONL as new shards to `hf_output_dataset` using the `HF_TOKEN` environment
variable. The shard naming convention mirrors the existing dataset files:
`stt_<lang>_<plugin>_<n>.jsonl`.

To disable auto-publish (write JSONL only):

```bash
python3.14 -m runner --no-publish ...
```

To publish manually from the laptop:

```bash
python3 - <<'EOF'
from pathlib import Path
from runner.publish import publish_output
publish_output(
    output_file=Path("/mnt/ser9/arena-runner/output/stt_pt-PT_ovos_stt_plugin_fasterwhisper_small.jsonl"),
    hf_repo="OpenVoiceOS/ovos-stt-bench-pt-PT",
)
EOF
```

---

## Plugins deployed on ser9

| plugin | model | py3.14 status |
|---|---|---|
| `ovos-stt-plugin-fasterwhisper` | `small`, `base` | installed, working |
| `ovos-stt-plugin-vosk` | `vosk-model-small-pt-0.3` | installed, working |
| `ovos-stt-plugin-citrinet` | — | **skipped**, nemo deps do not build on py3.14 |
| `ovos-stt-plugin-whisper` | — | skipped, already covered by existing dataset |

---

## Autonomous mode (fleet-wide, registry-driven)

`runner.autorun` runs forever across the STT / wake-word / TTS / VAD boards
(the ones backed by `runner.media_bench`, not the standalone STT queue
daemon above). It does not need a curated queue file: each sweep it
re-reads the registry, works out every eligible `(fighter, dataset, lang)`
pair, and round-robins across them — 10 new samples per pair, then the next
pair, forever. Add a fighter or dataset json to the registry and it joins
the rotation on the next sweep, no restart needed.

**Not covered**: the intent leagues. Those run through the separate
`runner.intent_bench` train+eval flow (a different row shape from the
audio benches) and are not wired into this scheduler. Autonomous, fleet-wide
coverage for intent is a follow-up, not part of this tool yet.

Start one instance per host. Pick the flags for that host's class:

```bash
# GPU host — heavyweight fighters only
python -m runner.autorun --modalities stt,tts --batch 10 --host-class auto

# CPU host — everything else (the default complement of --heavy)
python -m runner.autorun --modalities stt,tts,wake_word,vad --batch 10 --host-class auto

# Explicit split instead of auto-detection
python -m runner.autorun --modalities stt --batch 10 --heavy   # GPU box
python -m runner.autorun --modalities stt --batch 10 --light   # CPU box

# Narrow a host to a subset of fighters or languages
python -m runner.autorun --modalities stt --include 'vosk-*,fasterwhisper-*' \
    --exclude '*-large-*' --langs en-US,pt-PT --batch 10
```

`--host-class auto` (the default) reads `runner.perf.hw_fingerprint()`: a
box with a CUDA/onnxruntime GPU accelerator gets the heavyweight split,
everything else gets the light split. Pass `--heavy`/`--light` explicitly
to override.

**Heavy/light classification is a heuristic, not a verified parameter
count.** A fighter is "heavy" when any of: its registry `size` field is
`large` or above; its `competitor_id`/`plugin` string contains an `<N>b`
token (e.g. `2b`, `1.1b`, `2.5b`) for a param count `>= 1B`, which is how
`cohere-transcribe-2b` and similar fighters get caught even though nothing
enumerates them by name; or the string matches a small hardcoded list of
known heavy engine families (`speech-llm`, `whisper-large`, `canary`) for
names that don't carry an explicit param count (`whisper-large-v3-turbo`).
This is pattern-matching over a string, not a lookup against real model
metadata — it can misclassify a fighter whose name doesn't follow either
convention. To make that visible instead of silent, autorun logs every
distinct fighter's classification once at startup
(`classified <id> as heavy|light (registry size=...)`) — check that log on
a new host before trusting the `--heavy`/`--light` split, and fix a wrong
classification by setting the registry `size` field or filtering the
fighter explicitly with `--include`/`--exclude`.

### Round-robin, not drain-to-completion

Every pair gets one `--batch`-sized turn per sweep, then autorun moves to
the next pair — a slow fighter against a huge dataset never blocks the rest
from making progress.

A pair is marked complete (and skipped from then on) only when a batch
returns fewer new rows than requested **and reports zero errors**. A short
batch alone is not proof of completion: `adapter.predict` can raise on
every remaining sample (a model crash, an OOM, a transient network blip)
and produce the exact same "wrote fewer rows than asked" shape as a
genuinely exhausted dataset. Conflating the two would silently and
permanently drop every sample after the failure point. A pair that keeps
returning all-error batches is NOT marked complete and NOT retried forever
either: after `--max-consecutive-error-batches` (default 5) all-error turns
in a row with no progress, autorun quarantines the fighter instead (see
below) so the pair stops burning cycles without ever being falsely counted
as done.

### Resume

Resume is layered, same as every other bench script:

- **Row-level**: each pair's local shard
  (`<output-dir>/<dataset_id>/<modality>/predictions/<lang>/<competitor_id>.jsonl`)
  is the same file `run_competitor_lang` uses everywhere else — the
  `sample_id` scheme is the existing v2 index-prefixed one, so a killed
  process resumes mid-dataset by re-reading `done_samples()` from that file,
  no recomputation.
- **Cross-host**: the first time a pair is touched, if there's no local
  shard yet, autorun downloads the already-published HF shard first
  (`predictions/<lang>/<competitor_id>.jsonl` in the dataset's
  `predictions_hf` repo) so one host never redoes samples another host in
  the fleet already published. Disable with `--no-seed`.
- **Pair/fighter state**: `<output-dir>/autorun_state.json` records which
  pairs are already complete and which fighters are quarantined (with their
  retry timers — see below), so a restart doesn't have to re-stream an
  already-finished dataset just to rediscover there's nothing left to do.

### Quarantine

A fighter whose plugin fails to load outright (missing deps, bad config,
…), or whose samples fail with `--max-consecutive-error-batches` all-error
batches in a row, is quarantined — but not forever. Each quarantine carries
an exponential backoff retry timer: 30 minutes after the first failure,
doubling on each subsequent failure (60min, 2h, 4h, …), capped at 24h. When
the timer expires the fighter is put back in rotation for one retry attempt
automatically; if it fails again the backoff doubles and the cycle repeats.
This is what lets a weeks-long daemon recover from a transient blip
(network hiccup fetching a model, a momentarily locked file) on its own
instead of a fighter being quarantined for the rest of the process's
lifetime after one bad attempt. Each quarantine event (initial failure or a
failed retry) is logged once, at the moment it happens — never replayed
every sweep while a fighter is still within its backoff window.

### Uploads

Batched, not per-10-row: a pair's shard is pushed to HF immediately when
that pair completes, and everything else still pending (dirty shards from
in-progress pairs) flushes on a timer, `--flush-every` minutes (default 15).
`--no-upload` writes local shards only.

### ctrl-C / SIGTERM

Both are caught: the current pair finishes, every dirty shard is flushed,
and `autorun_state.json` is saved before exit.

## Hourly CI predictions (`--one-shot`)

The forever-mode daemon above runs on the fleet, not in CI. For a scheduled
GitHub Actions job — no long-lived process, one bounded run per invocation —
`runner.autorun` also has a `--one-shot` mode, wired up as
`.github/workflows/hourly-predictions.yml`:

```bash
python -m runner.autorun --one-shot --light \
    --max-samples 1000 --time-budget-secs 1500 --max-attempts 3
```

Each run:

1. enumerates every eligible `(fighter, dataset, lang)` pair (same
   eligibility rules as forever-mode) and **shuffles** the candidate list
   with a seeded RNG — `--seed` makes the shuffle (and so the pick)
   reproducible/debuggable; unset (the default, and what the workflow
   uses) draws from entropy;
2. walks the shuffled list ONE candidate at a time and only THEN checks
   whether that one candidate is already complete (seeds its local shard
   from the published HF shard, compares sample ids already written
   against the eval set's sample ids — no inference). This is a **lazy**
   draw: discovery cost is proportional to how many candidates were looked
   at before landing on a usable one, never to the full size of the
   registry. An earlier version of this mode ran the completeness check
   against *every* eligible pair before picking one, which made discovery
   alone take longer than the entire time budget on a registry with
   hundreds of pairs — fixed by this walk-and-stop design;
3. scores new samples for the pair it landed on, until either
   `--max-samples` is written or `--time-budget-secs` of wall-clock
   elapses. **The budget covers the WHOLE call**, not just inference: it's
   counted from the very start, so it also pays for the discovery walk in
   step 2 (each completeness check is itself deadline-bounded) and the
   fighter's model load, which happens inside this step — a slow-loading
   model, or a registry that takes a while to walk before landing on a
   pair, doesn't get that time "for free";
4. uploads the shard, then — if more than 5 minutes of the time budget
   remain — draws another random pair and repeats, so a CI job's time slot
   isn't left idle after one small/fast pair finishes. With no
   `--time-budget-secs`, exactly one pair runs;
5. prints a `one-shot: pair=... written=... errored=... elapsed=...s
   upload=...` summary line per pair run, plus a totals line, and always
   exits 0. Two other outcomes get their own DISTINCT summary line, never
   conflated with each other or with a normal run:
   - `one-shot: nothing to do — ...` — every eligible candidate was
     checked and is genuinely complete;
   - `one-shot: discovery-bound, 0 pairs drawn, N candidate(s) probed —
     ran out of time-budget before finding a usable pair` — the deadline
     elapsed while still walking candidates (a slow completeness check on
     a huge dataset, or a very large registry with a short budget); this
     is NOT the same as "nothing to do" — there might well be work left,
     the run simply never got to look at all of it.

**Resilience**: if the chosen fighter's plugin fails to load outright (a
missing system dependency on the runner, e.g. `espeak-ng`), the pair's
fighter is quarantined exactly as in forever-mode and the shuffled walk
continues onto the next candidate instead, up to `--max-attempts` (default
3) real draws before that round gives up — the job still exits 0 rather
than failing outright on one broken fighter.

**Filters**: `--include`/`--exclude` (competitor id globs) and the new
`--datasets` (dataset id globs) narrow the random pick to a subset instead
of the full pool — e.g. `--include 'vosk-*' --datasets common-voice-*`. The
workflow's `workflow_dispatch` exposes these as the `competitor` and
`dataset` inputs.

### Triggering a run

- **Scheduled**: cron `17 * * * *` (offset from the hour to avoid
  colliding with `tally.yml`/`assemble.yml`, which also run on cron).
- **Manual** (Actions tab → "Hourly random-pair predictions" → *Run
  workflow*): set `competitor`/`dataset` globs to target a specific
  fighter or dataset instead of the full random pool — useful for
  populating a newly-added fighter/dataset's coverage or validating that a
  fighter loads cleanly on the runner without waiting for the cron to pick
  it. `time_budget_secs`/`max_samples` are also dispatch inputs (defaults
  1500s / 1000 samples).

### Runner environment

- **System deps**: the workflow installs `espeak-ng`, `libsndfile1`,
  `ffmpeg` via `apt-get` — needed by several TTS/audio-decode fighters.
- **Python deps**: `pip install ".[hf,audio]"` plus a small, best-effort
  list of CPU-only OVOS plugin packages (see the workflow file for the
  current list). **Not every registry fighter's plugin is pre-installed**
  — a fighter whose plugin is missing just fails to load, gets quarantined
  for that run, and a different pair is drawn (point 3 above); it is not a
  job failure. Extend the installed-plugin list over time as more
  CPU-friendly plugins are confirmed to install cleanly on a GitHub-hosted
  runner.
- **HF upload token**: the `HF_TOKEN` repo secret (same environment
  variable name `runner.__main__`/the fleet daemon already read via
  `os.environ.get("HF_TOKEN")` — see "HuggingFace publish" above). No
  *existing* workflow in this repo needed a write token before this one:
  `tally.yml`/`assemble.yml` only read already-published public HF data.
  `HF_TOKEN` must be added as a repository secret before this workflow can
  publish; without it, `--one-shot` still runs and writes the local shard,
  it just can't push to HF.
- **Caching**: `~/.cache/huggingface` is cached across runs on a coarse
  key (registry file hashes, with an OS-level restore-key fallback) so a
  model already downloaded by a previous run doesn't get re-fetched cold
  every hour; `actions/setup-python`'s `cache: pip` covers the Python
  dependency install.
- **Concurrency**: `concurrency: group: hourly-predictions,
  cancel-in-progress: false` — an overlapping run (a slow scheduled run
  plus a manual dispatch) queues instead of overlapping or being
  cancelled mid-write. `timeout-minutes: 55` keeps a stuck run from
  bumping into the next hour's scheduled trigger.

---
[← Benchmarks](benchmarks.md) · [Home](index.md) · [Reproduce a row →](reproduce-a-row.md)
