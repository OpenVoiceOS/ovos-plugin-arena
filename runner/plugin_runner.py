"""
STT prediction runner — single-job execution logic.

Each job runs in its own worker process so model state is fully isolated and
ORT/BLAS thread counts can be applied before any import.

§4 A2 schema convergence: rows are written directly in the canonical §3.2
``PredictionRow`` shape (see ``arena.models.PredictionRow`` /
``docs/SPECIFICATION.md``) — never the legacy ``STTRow`` layout. This runner
does not resolve ``competitor_id`` (it has no registry dependency by
design, so it can run standalone on a plugin-execution box); ``plugin_id``
is written instead, and ``arena.predictions`` re-keys it to a
``competitor_id`` at load time via ``registry.loaders.get_competitor_by_alias``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from runner.queue_config import DatasetSpec, JobSpec, PluginSpec
from runner.schema import JobManifest

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


def _entry_id(sample: dict, audio_key: str, entry_id_key: str | None) -> str:
    if entry_id_key and entry_id_key in sample:
        return sample[entry_id_key]
    audio = sample.get(audio_key, {})
    if isinstance(audio, dict):
        path = audio.get("path", "")
        # Strip snapshot prefix added by datasets lib
        name = path.split("/snapshots/")[-1].split("/", 1)[-1]
        return name or path
    return str(audio)


def _decode_audio_bytes(raw_bytes: bytes, target_sr: int = 16000):
    """
    Decode raw audio bytes to a float32 numpy array resampled to *target_sr*.
    Handles WAV, MP3, and other formats via soundfile (falling back to av/pydub).
    Returns (array, sample_rate).
    """
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO(raw_bytes)
    try:
        array, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        # Try with av (installed as part of fasterwhisper deps)
        try:
            import av
            buf.seek(0)
            container = av.open(buf)
            stream = next(s for s in container.streams if s.type == "audio")
            sr = stream.rate
            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray())
            array = np.concatenate(frames, axis=-1).astype(np.float32)
            if array.ndim > 1:
                array = array.mean(axis=0)
            array /= max(np.abs(array).max(), 1e-6)
        except Exception as e:
            raise RuntimeError(f"Cannot decode audio: {e}") from e

    # Resample if needed
    if sr != target_sr:
        try:
            from faster_whisper.audio import decode_audio
            buf.seek(0)
            array = decode_audio(buf, sampling_rate=target_sr)
            sr = target_sr
        except Exception:
            pass  # Keep original sr; plugin will handle it

    return array, sr


def _list_parquet_files(hf_repo: str, subset: str | None, split: str) -> list:
    """Return list of parquet file paths for a HF dataset split."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        files = list(api.list_repo_files(hf_repo, repo_type="dataset"))
    except Exception as e:
        raise RuntimeError(f"Cannot list files in {hf_repo}: {e}") from e

    # Try prefix patterns: subset/split-*.parquet or data/split-*.parquet
    candidates = []
    for pattern_prefix in [
        f"{subset}/{split}-" if subset else f"{split}-",
        f"data/{split}-",
        f"{split}-",
    ]:
        candidates = [f for f in files if f.startswith(pattern_prefix) and f.endswith(".parquet")]
        if candidates:
            break

    if not candidates:
        # Fallback: any parquet file
        candidates = [f for f in files if f.endswith(".parquet")]

    return sorted(candidates)


def _stream_dataset(spec: DatasetSpec) -> Iterator[tuple[str, str, object, int]]:
    """
    Yield (entry_id, ground_truth, audio_array, sample_rate) per sample.

    Uses direct parquet + huggingface_hub download to avoid dill/datasets
    incompatibility on Python 3.14.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet_files = _list_parquet_files(spec.hf_repo, spec.subset, spec.split)
    if not parquet_files:
        raise RuntimeError(
            f"No parquet files found for {spec.hf_repo} subset={spec.subset} split={spec.split}"
        )

    count = 0
    for pfile in parquet_files:
        local_path = hf_hub_download(spec.hf_repo, pfile, repo_type="dataset")
        table = pq.read_table(local_path)

        for batch in table.to_batches(max_chunksize=64):
            rows = batch.to_pydict()
            n = len(rows.get(spec.ground_truth_key, []))
            for i in range(n):
                if spec.max_samples and count >= spec.max_samples:
                    return

                ground_truth = (rows.get(spec.ground_truth_key) or [None])[i]
                if not ground_truth:
                    continue

                audio_cell = (rows.get(spec.audio_key) or [None])[i]
                if audio_cell is None:
                    continue

                # audio_cell may be dict(bytes=..., path=...) or dict(array=..., sampling_rate=...)
                if isinstance(audio_cell, dict):
                    raw_bytes = audio_cell.get("bytes")
                    path_hint = audio_cell.get("path", f"sample_{count}.wav")
                    array = audio_cell.get("array")
                    sr = audio_cell.get("sampling_rate", 16000)
                    if array is None and raw_bytes:
                        try:
                            array, sr = _decode_audio_bytes(raw_bytes, target_sr=16000)
                        except Exception as e:
                            logger.warning("audio decode error on %s: %s", path_hint, e)
                            continue
                    elif array is None:
                        continue
                else:
                    continue

                if spec.entry_id_key and spec.entry_id_key in rows:
                    entry_id = str(rows[spec.entry_id_key][i])
                else:
                    entry_id = path_hint or f"sample_{count}.wav"

                import numpy as np
                if not isinstance(array, np.ndarray):
                    try:
                        array = np.array(array, dtype=np.float32)
                    except Exception:
                        continue

                yield entry_id, ground_truth, array, sr
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


def _transcribe(stt_instance, array, sample_rate: int, lang: str) -> tuple[str, float]:
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

                text, conf = _transcribe(stt, array, sample_rate, plugin.lang)

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

            row = {
                "sample_id": entry_id,
                "dataset_id": dataset.dataset_id,
                "lang": plugin.lang,
                "plugin_id": plugin.plugin_name,
                "modality": "stt",
                "prediction": text,
                "reference_text": ground_truth,
                "confidence": conf,
                "extras": {"model_id": mid},
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()  # flush every row so output is visible immediately
            written += 1

            if written % flush_every == 0:
                manifest.save(base_dir)
                logger.info("flushed %d rows (job %s)", written, job_key)

            manifest.mark_done(entry_id, base_dir)

    manifest.save(base_dir)
    logger.info("job done: %s  rows_written=%d", job_key, written)
    return output_path
