"""
STT prediction runner — single-job execution logic.

Each job runs in its own worker process so model state is fully isolated and
ORT/BLAS thread counts can be applied before any import.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

from runner.queue_config import DatasetSpec, JobSpec, PluginSpec
from runner.schema import JobManifest, STTRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-ID helper (mirrors STTPluginDefinition.model_id in ovos_plugin_bench)
# ---------------------------------------------------------------------------


def _model_id(plugin: PluginSpec) -> str:
    from ovos_plugin_manager.utils.tts_cache import hash_sentence
    pid = f"{plugin.plugin_name}/{plugin.model_name}"
    if plugin.extra_config:
        cfg_hash = hash_sentence(json.dumps(plugin.extra_config, sort_keys=True))
        pid += f"/{cfg_hash}"
    return pid


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------


def _entry_id(sample: dict, audio_key: str, entry_id_key: Optional[str]) -> str:
    if entry_id_key and entry_id_key in sample:
        return sample[entry_id_key]
    audio = sample.get(audio_key, {})
    if isinstance(audio, dict):
        path = audio.get("path", "")
        # Strip snapshot prefix added by datasets lib
        name = path.split("/snapshots/")[-1].split("/", 1)[-1]
        return name or path
    return str(audio)


def _stream_dataset(spec: DatasetSpec) -> Iterator[Tuple[str, str, object, int]]:
    """
    Yield (entry_id, ground_truth, audio_array, sample_rate) per sample.
    """
    from datasets import load_dataset

    kwargs: dict = dict(split=spec.split, streaming=True)
    if spec.subset:
        kwargs["name"] = spec.subset
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True

    ds = load_dataset(spec.hf_repo, **kwargs)
    sample_rate: Optional[int] = None

    count = 0
    for sample in ds:
        if spec.max_samples and count >= spec.max_samples:
            break

        ground_truth = sample.get(spec.ground_truth_key, "")
        if not ground_truth:
            continue

        audio = sample.get(spec.audio_key, {})
        if not isinstance(audio, dict):
            continue
        array = audio.get("array")
        if array is None:
            continue

        if sample_rate is None:
            sample_rate = audio.get("sampling_rate", 16000)

        entry_id = _entry_id(sample, spec.audio_key, spec.entry_id_key)
        yield entry_id, ground_truth, array, sample_rate
        count += 1


# ---------------------------------------------------------------------------
# Plugin wrapper
# ---------------------------------------------------------------------------


def _load_plugin(plugin: PluginSpec):
    from ovos_plugin_manager.stt import load_stt_plugin

    clazz = load_stt_plugin(plugin.plugin_name)
    if clazz is None:
        raise RuntimeError(f"Plugin not found: {plugin.plugin_name}")
    cfg = {"lang": plugin.lang, "model": plugin.model_name, **plugin.extra_config}
    return clazz(cfg)


def _transcribe(stt_instance, array, sample_rate: int, lang: str) -> Tuple[str, float]:
    from ovos_plugin_manager.utils.audio import AudioData

    audio_data = AudioData.from_array(array, sample_rate=sample_rate, sample_width=2)
    result = stt_instance.transcribe(audio_data, lang=lang)
    if isinstance(result, list) and result:
        text, conf = result[0] if isinstance(result[0], tuple) else (result[0], 1.0)
    elif isinstance(result, tuple):
        text, conf = result
    else:
        text, conf = str(result or ""), 1.0
    return text, float(conf)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def run_job(
    job: JobSpec,
    base_dir: Path,
    per_sample_timeout: int = 60,
    flush_every: int = 100,
) -> Path:
    """
    Run a single (plugin × dataset) job.

    Returns the path to the output JSONL file.
    Writes rows incrementally; safe to resume after interruption.
    """
    import signal

    plugin = job.plugin
    dataset = job.dataset

    mid = _model_id(plugin)
    job_key = f"{plugin.plugin_name}|{plugin.model_name}|{dataset.dataset_id}"

    manifest = JobManifest.load(base_dir, job_key)

    # Determine output file
    safe_model = plugin.model_name.replace("/", "__")
    safe_plugin = plugin.plugin_name.replace("-", "_")
    out_name = f"stt_{plugin.lang}_{safe_plugin}_{safe_model}.jsonl"
    output_path = base_dir / "output" / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.output_file = str(output_path)

    logger.info("job_key=%s  output=%s  already_done=%d",
                job_key, output_path, len(manifest.done_ids))

    # Load plugin once for the whole job
    stt = _load_plugin(plugin)

    written = 0

    # Per-sample alarm timeout handler
    def _timeout_handler(signum, frame):
        raise TimeoutError("per-sample timeout")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)

    with output_path.open("a", encoding="utf-8") as fh:
        for entry_id, ground_truth, array, sample_rate in _stream_dataset(dataset):
            if manifest.is_done(entry_id):
                continue

            try:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(per_sample_timeout)

                t0 = time.perf_counter()
                text, conf = _transcribe(stt, array, sample_rate, plugin.lang)
                elapsed = time.perf_counter() - t0

                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

            except TimeoutError:
                logger.warning("timeout on sample %s — skipping", entry_id)
                manifest.mark_done(entry_id, base_dir)
                continue
            except Exception as exc:
                logger.warning("error on sample %s: %s — skipping", entry_id, exc)
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                continue

            row = STTRow(
                dataset_entry_id=entry_id,
                plugin_name=plugin.plugin_name,
                model_id=mid,
                prediction_transcript=text,
                transcript=ground_truth,
                prediction_confidence=conf,
                prediction_type="STT",
                dataset_id=dataset.dataset_id,
                lang=plugin.lang,
            )
            fh.write(row.to_jsonl() + "\n")
            written += 1

            if written % flush_every == 0:
                fh.flush()
                manifest.save(base_dir)
                logger.info("flushed %d rows (job %s)", written, job_key)

            manifest.mark_done(entry_id, base_dir)

    manifest.save(base_dir)
    logger.info("job done: %s  rows_written=%d", job_key, written)
    return output_path
