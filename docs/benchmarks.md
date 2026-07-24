# Benchmarks

Every league has one reproducible benchmark script under `benchmarks/` (spec
§P4). A script loads its eval dataset and the registry fighters, runs the real
OVOS plugins **offline** (never in CI — spec §P2), writes §3.2 prediction rows
and, with `--upload`, publishes them to one HuggingFace dataset repo per
modality (`OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`). The arena's
`assemble` workflow turns those rows into benchmark boards, blind A/B battle
pools and a benchmark-seeded ELO ladder.

## Shared engines

| Engine | Modalities | What it does |
|---|---|---|
| `runner/intent_bench.py` | `intent*` | Trains each pipeline fighter from the dataset's paradigm-specific corpora, runs the OPM cascade over a FakeBus. |
| `runner/media_bench.py` | `stt`, `wake_word`, `tts` | Drives a `MediaBenchAdapter`: load dataset samples → instantiate the plugin → predict → write resumable §3.2 rows → upload. |

The audio adapters (`runner/stt_bench.py`, `runner/ww_bench.py`,
`runner/tts_bench.py`) supply the per-modality sample loading and plugin call;
audio decoding lives in `runner/audio_io.py`. All plugin/audio imports are
lazy, so the engine, row contract and publishing logic import (and unit-test)
without any plugin or audio stack installed.

## Common flags

Every benchmark script accepts the same arguments:

| Flag | Meaning |
|---|---|
| `--dataset <id>` | Registry dataset id to run (defaults to the script's canonical set) |
| `--competitors a,b` | Restrict to these competitor ids (default: all registered for the modality) |
| `--langs a,b` | Restrict to these languages (default: the dataset's languages) |
| `--max-samples N` | Cap samples per language (smoke runs) |
| `--output-dir DIR` | Local root for prediction JSONLs / synthesised audio |
| `--upload` | Publish predictions to the per-modality HF repo |

Runs are resumable: re-running skips `sample_id`s already present in the output
JSONL, and the plugin/model is only loaded when a language still has work left.

## Modalities

### Intent

```bash
python benchmarks/intent_intents_for_eval.py            # 3 leagues, 12 langs
python benchmarks/intent_massive_templates.py           # template league, 52 langs
```

Fighters are mycroft.conf pipeline fragments. Paradigm leagues are pure
(`check_league`); a fighter is skipped where the benchmark lacks the training
corpus its engine needs (e.g. keyword engines on a template-only corpus).
Scored by accuracy / macro-F1 / OOD-FPR / slot-EM.

> Audio benchmarks need the `audio` extra (`pip install .[hf,audio]` —
> numpy/soundfile/pyarrow) plus the relevant STT/TTS/wake-word OVOS plugins.

### STT

```bash
python benchmarks/stt_minds14.py                        # minds14-pt-PT
python benchmarks/stt_minds14.py --dataset minds14-en-US --max-samples 50
```

Each fighter's `stt` config block names an `ovos-stt-plugin-*` module; the
adapter streams the audio corpus from parquet, transcribes each clip and stores
the hypothesis next to the reference. WER is computed by the arena on ingest
(`arena.metrics.row_wer`) so the dataset stays the single source of truth for
the formula. Scored by mean/median WER; lower WER seeds higher ELO.

### Wake word

```bash
python benchmarks/ww_hey_mycroft.py
python benchmarks/ww_hey_mycroft.py --competitors openwakeword-hey-mycroft --max-samples 50
```

The eval set is the held-out [ww-bench](https://github.com/TigreGotico/ww-benchmarks)
manifest for one wake phrase (eval-only donor voices). Each clip is fed to the
hotword engine frame by frame (80 ms chunks) exactly as the listening loop
does; the binary decision is recorded against the ground-truth label
(role `positive` = wake word present, `negative`/`adversarial` = absent).
Scored by detection `error_rate` (primary), `false_accept_rate` and
`false_reject_rate`. The plugin owns its threshold — the arena owns none.

### TTS

```bash
python benchmarks/tts_intents_prompts.py --langs en-US --max-samples 30
python benchmarks/tts_intents_prompts.py --langs en-US --upload
```

TTS has **no ground-truth reference** (spec §2.1, §3.2) — there is no single
"correct" waveform for a prompt — so human votes stay the league's primary
ranking signal. Each fighter synthesises every prompt, the clip is stored
under `audio/<lang>/<competitor>/<hash>.wav`, its HF resolve URL becomes the
§3.2 `prediction`, and the clip is scored with UTMOS (reference-free
naturalness MOS, §4 R14) — the score and judge provenance
(`utmos`/`utmos_judge`/`utmos_judge_revision`) land in the row's `extras`.
The arena assembles the clips into blind A/B listening battles for human
voting *and* builds an objective `tts` benchmark board from the UTMOS mean,
feeding a benchmark-seeded ELO contribution alongside the human votes.

## Adding a benchmark

1. Add the dataset JSON under `registry/datasets/<modality>/`.
2. Add the fighter JSONs under `registry/competitors/<modality>/` (a shippable
   plugin config + bestiary card fields).
3. For a new audio dataset, reuse the existing adapter; for a new corpus shape,
   point the dataset `source` at it (parquet split, per-lang `file_pattern`, or
   a `manifest.jsonl` sidecar for wake word).
4. Add a thin `benchmarks/<modality>_<corpus>.py` that calls
   `run_benchmark(<Adapter>(), "<dataset_id>", __doc__...)`.
5. Run with `--max-samples` first, then `--upload`.
