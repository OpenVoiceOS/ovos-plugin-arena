# Reproduce a leaderboard row

Every number on the arena's leaderboards comes from a plugin run offline
against a public dataset, published as raw prediction rows on HuggingFace. You
do not have to trust the board: install the plugin, run the same benchmark
script against the same dataset with a small sample cap, and check that your
number falls where the published row's confidence interval says it should.
Five minutes and a laptop are enough for a CPU fighter; a GPU-free onnx-asr
STT model takes a couple of minutes more the first time, mostly downloading
the model.

This page walks the two cheapest cases end to end: an intent fighter
(`padacioso-medium`, template matching, pure CPU) and an STT fighter
(`onnx-asr-parakeet-tdt-ctc-110m`, a 110M-parameter ONNX model). Every
command below was actually run to produce the numbers quoted.

## 1. Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --prerelease=allow -e ".[hf,audio]"
uv pip install --python .venv/bin/python --prerelease=allow ovos-adapt-parser ovos-stt-plugin-onnx-asr
```

`padacioso` (installed transitively with the arena package) registers the
`ovos-padacioso-pipeline-plugin` entry point used by the template and
freeform leagues, and `ovos-adapt-parser` registers `ovos-adapt-pipeline-plugin`
the same way for the keyword leagues — `ovos-core` itself only ships the
converse/fallback/stop pipeline plugins. Either way, there is no separate
`ovos-*-pipeline-plugin` package to install: the entry point comes with the
library. The `.[hf,audio]` extra pulls in `onnx-asr`, `soundfile` and
`huggingface_hub`, needed for the STT case.

## 2. Run the intent fighter

```bash
python benchmarks/intent_snips.py --competitors padacioso-medium --max-samples 5
```

This trains and runs `padacioso-medium` against 5 English SNIPS utterances
and writes rows to `predictions/snips/intent_template/en-US/padacioso-medium.jsonl`.
It took under 10 seconds on a 12-thread box. Expected output:

```
INFO    training padacioso-medium for en-US (stages: ovos-padacioso-pipeline-plugin-medium)
INFO    padacioso-medium/en-US: wrote 5 rows
```

## 3. Run the STT fighter

```bash
python benchmarks/stt_minds14.py --dataset minds14-en-US \
    --competitors onnx-asr-parakeet-tdt-ctc-110m --max-samples 20
```

This downloads the 110M ONNX model once, transcribes 20 MInDS-14 clips, and
writes rows to `predictions/minds14-en-US/stt/predictions/en-US/onnx-asr-parakeet-tdt-ctc-110m.jsonl`.
It took about 17 seconds after the model was cached. Expected output:

```
INFO    loading onnx-asr-parakeet-tdt-ctc-110m for en-US
INFO    onnx-asr-parakeet-tdt-ctc-110m/en-US: wrote 20 rows (0 errored)
```

## 4. Turn the rows into a board

`assemble` needs the directory that directly contains the `<lang>/` shards,
not the top-level `predictions/<dataset>/` directory a benchmark script logs
you into — see the caveat at the bottom of this page.

```bash
python -m arena.cli assemble \
    --predictions predictions/snips/intent_template \
    --modality intent_template --output /tmp/arena-assemble-intent

python -m arena.cli assemble \
    --predictions predictions/minds14-en-US/stt/predictions \
    --modality stt --output /tmp/arena-assemble-stt
```

Each writes a `benchmark-<modality>-<dataset>-<lang>.json` with one entry per
competitor, in the same shape as the committed files under
`frontend-static/public/data/`. Read the `metrics` block for the competitor
you ran — for the intent case, `metrics.accuracy`; for STT,
`metrics.wer_mean` — along with `primary_metric_ci_lower` /
`primary_metric_ci_upper`, the confidence interval computed from your sample
count.

## 5. Compare against the published row

Open the matching file under `frontend-static/public/data/` and find the same
`competitor_id`:

```bash
python3 -c "
import json
d = json.load(open('frontend-static/public/data/benchmark-intent_template-snips-en-US.json'))
row = next(e for e in d['entries'] if e['competitor_id'] == 'padacioso-medium')
print(row['metrics']['accuracy'], row['samples'])
"
```

The published `padacioso-medium` row on `snips`/`en-US` scores 0.0164
accuracy over 1400 samples (95% CI 0.0107–0.0236). A 5-sample local run
scored 0/5 = 0.0, well inside that interval — at a 1.6% true positive rate,
0 hits in 5 draws is the expected outcome most of the time, not a
contradiction. The published `onnx-asr-parakeet-tdt-ctc-110m` row on
`minds14-en-US` scores 0.3451 mean WER over 563 samples; a 20-sample local
run scored 0.2355, inside the local run's own CI (0.121–0.407) and
consistent with the published value given the sample size.

To go one step further and check the *exact* published prediction rows
(not just the aggregate metric), fetch the one file for your competitor from
the dataset's HuggingFace `predictions_hf` repo (see `dataset_info` in the
board JSON) and diff row counts:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='OpenVoiceOS/ovos-intent-template-bench-snips', repo_type='dataset',
    revision='<predictions_revisions value from the board JSON>',
    filename='predictions/en-US/padacioso-medium.jsonl')
print(p)
"
```

This is a read of one file, not a full dataset snapshot download. The
`predictions_revisions` field in the board JSON pins the exact commit the
published numbers were computed from, so the file you fetch is the file the
board was built from, not whatever happens to be on `main` today.

## Caveats

- **Sampling cap.** `--max-samples` trades statistical power for speed. A
  5- or 20-sample run's confidence interval is wide; treat agreement as
  "consistent with the published row," not as an independent re-measurement
  of the true score. Widen the cap if you want a tighter interval.
- **Nondeterministic plugins.** Fighters with any learned component (fuzzy
  matchers, ONNX beam search with more than one worker) can return slightly
  different rows between runs. The two fighters used here are deterministic
  (template regex match, greedy ONNX decode), but not every fighter in the
  registry is.
- **Model downloads.** The first STT run for a given model downloads and
  caches it (over a gigabyte for `onnx-asr-parakeet-tdt-ctc-110m`); later runs
  reuse the cache. Budget for that on a slow connection.
- **`assemble --predictions` path depth.** Every benchmark script writes to
  `predictions/<dataset>/<modality>/<lang>/<competitor>.jsonl`, but `assemble`
  expects to be pointed at the directory that directly contains the `<lang>/`
  subdirectories. For the STT adapter that directory is one level deeper
  still (`predictions/<dataset>/<modality>/predictions/<lang>/...`) than for
  the intent engine. Passing `assemble` the dataset-level directory the
  benchmark script's log line implies (as `docs/local-testing.md` step 5
  shows for a flat single-dataset case) silently assembles nothing for either
  layout used here — no error, just `nothing to assemble for modality ...`.
  Use the paths in the commands above, or list the actual directory tree
  under `predictions/` before assembling.

---
[← Runner](runner.md) · [Home](index.md) · [Leagues →](leagues.md)
