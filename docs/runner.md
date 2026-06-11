# Arena STT Prediction Runner

The prediction runner is an offline batch job that generates STT transcriptions
for (plugin × dataset) pairs and appends the results to the corresponding
HuggingFace benchmark dataset (`OpenVoiceOS/ovos-stt-bench-<lang>`).

It runs 24/7 on the ser9 compute box (192.168.1.116), cycling through the job
queue and sleeping between cycles.

---

## Row schema

Each row written to the JSONL output and uploaded to HF matches the legacy
`ovos-stt-bench-*` column layout; the arena §3.2 contract (docs/SPECIFICATION.md) is the convergence target for a future STT bench script:

| column | type | notes |
|---|---|---|
| `dataset_entry_id` | str | stable filename within source corpus |
| `plugin_name` | str | OPM entry-point name |
| `model_id` | str | composite `plugin/model[/cfghash]` |
| `prediction_transcript` | str | STT output |
| `transcript` | str | ground truth |
| `prediction_confidence` | float | 0.0–1.0 |
| `prediction_type` | str | always `"STT"` |
| `dataset_id` | str | source corpus + split path |
| `lang` | str | BCP-47 |

---

## Adding jobs

Edit `runner/queue.yaml`.  Each job entry looks like:

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

The daemon reloads the queue file on the next cycle whenever its mtime changes
— no restart needed.

---

## Deploy on ser9

```bash
# Stage from laptop
scp -r /home/miro/AgentWorkspaces/ovos/web/ovos-plugin-arena/runner \
    miro@192.168.1.116:/home/miro/arena-runner/runner

# On ser9 — one-time setup (already done):
python3.14 -m venv ~/venvs/arena-runner
~/venvs/arena-runner/bin/pip install \
    ovos-stt-plugin-fasterwhisper \
    ovos-stt-plugin-vosk \
    datasets huggingface_hub pyyaml

# Start the daemon detached in tmux:
ssh miro@192.168.1.116
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
variable.  The shard naming convention mirrors the existing dataset files:
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
| `ovos-stt-plugin-citrinet` | — | **skipped** — nemo deps do not build on py3.14 |
| `ovos-stt-plugin-whisper` | — | skipped — already covered by existing dataset |
