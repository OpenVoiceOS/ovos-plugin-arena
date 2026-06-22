"""
Audio dataset streaming for the STT and wake-word benchmarks.

Reads HuggingFace audio corpora straight from their parquet files via
``huggingface_hub`` + ``pyarrow`` (no ``datasets`` dependency), decoding each
clip to a mono float32 array resampled to 16 kHz.  All imports are lazy so the
arena core and tests do not pull in audio stacks.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def decode_audio_bytes(raw_bytes: bytes, target_sr: int = TARGET_SR):
    """Decode encoded audio bytes → (mono float32 array, sample_rate)."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO(raw_bytes)
    try:
        array, sr = sf.read(buf, dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
    except Exception as exc:  # fall back to libav for mp3/opus/…
        try:
            import av

            buf.seek(0)
            container = av.open(buf)
            stream = next(s for s in container.streams if s.type == "audio")
            sr = stream.rate
            frames = [f.to_ndarray() for f in container.decode(stream)]
            array = np.concatenate(frames, axis=-1).astype(np.float32)
            if array.ndim > 1:
                array = array.mean(axis=0)
            array /= max(np.abs(array).max(), 1e-6)
        except Exception as exc2:  # noqa: F841
            raise RuntimeError(f"cannot decode audio: {exc}") from exc

    if sr != target_sr:
        try:
            from faster_whisper.audio import decode_audio
            array = decode_audio(io.BytesIO(raw_bytes), sampling_rate=target_sr)
            sr = target_sr
        except Exception:
            pass  # plugin handles the native rate
    return array, sr


def _parquet_files(hf_repo: str, subset: Optional[str], split: str,
                   revision: str) -> List[str]:
    from huggingface_hub import HfApi

    files = list(HfApi().list_repo_files(hf_repo, repo_type="dataset",
                                         revision=revision))
    for prefix in (f"{subset}/{split}-" if subset else f"{split}-",
                   f"data/{subset}/" if subset else "data/",
                   f"data/{split}-", f"{split}-"):
        hits = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
        if hits:
            return sorted(hits)
    return sorted(f for f in files if f.endswith(".parquet"))


def _sample_id(sample: dict, audio_cell, audio_key: str,
               id_key: Optional[str], index: int) -> str:
    if id_key and sample.get(id_key) is not None:
        return str(sample[id_key])
    if isinstance(audio_cell, dict):
        path = audio_cell.get("path") or ""
        name = path.split("/snapshots/")[-1].split("/", 1)[-1]
        if name:
            return name
    return f"sample_{index:06d}"


_AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def _even(paths: List[str], cap: int) -> List[str]:
    """Pick *cap* paths evenly spaced across the sorted list (deterministic).

    Striding spans the full range — across TTS voices within a folder and
    across folders in a concatenated negative list — instead of taking a
    same-voice run from the front.
    """
    if not cap or len(paths) <= cap:
        return paths
    step = len(paths) / cap
    return [paths[int(i * step)] for i in range(cap)]


def _emit_ww_clips(hf_id, pos, neg, revision, max_per_class):
    """Download + decode positive/negative clip lists → labelled WW samples."""
    from urllib.parse import quote

    from huggingface_hub import hf_hub_download

    if max_per_class:
        pos = _even(sorted(pos), max_per_class)
        neg = _even(sorted(neg), max_per_class)
    for label, paths in (("positive", pos), ("negative", neg)):
        for rel in paths:
            try:
                local = hf_hub_download(hf_id, rel, repo_type="dataset",
                                        revision=revision)
                with open(local, "rb") as fh:
                    array, sr = decode_audio_bytes(fh.read())
            except Exception as exc:
                logger.warning("ww clip %s failed: %s", rel, exc)
                continue
            url = (f"https://huggingface.co/datasets/{hf_id}"
                   f"/resolve/{revision}/{quote(rel)}")
            yield rel, {"array": array, "sr": sr, "label": label,
                        "audio_url": url}


def stream_audiofolder_ww(
    source,
    wakeword: str,
    negative_dirs: Optional[List[str]],
    revision: str,
    max_per_class: int = 0,
) -> Iterator[Tuple[str, dict]]:
    """Yield labelled WW clips from an audiofolder corpus (one folder per phrase).

    Positives are clips under ``<wakeword>/``; negatives are clips under the
    other top-level folders (``negative_dirs``, or every other folder when
    None) — other wake phrases make strong adversarial hard negatives.
    """
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(source.hf_id, repo_type="dataset",
                                                revision=revision)
             if f.lower().endswith(_AUDIO_EXT) and "/" in f]
    pos = [f for f in files if f.split("/")[0] == wakeword]
    negset = set(negative_dirs) if negative_dirs else None
    neg = [f for f in files if f.split("/")[0] != wakeword
           and (negset is None or f.split("/")[0] in negset)]
    yield from _emit_ww_clips(source.hf_id, pos, neg, revision, max_per_class)


def stream_metadata_csv_ww(
    source,
    wakeword: str,
    negative_labels: Optional[List[str]],
    revision: str,
    max_per_class: int = 0,
    audio_col: str = "file_name",
    label_col: str = "label",
) -> Iterator[Tuple[str, dict]]:
    """Yield labelled WW clips from an audiofolder corpus with a ``metadata.csv``.

    Some HF audio corpora list every clip in a CSV (``file_name``, ``label``)
    rather than relying on folder names — and their file tree may be too large
    to enumerate.  Positives are rows whose label is *wakeword*; negatives are
    rows in *negative_labels* (or every other label).
    """
    import csv

    from huggingface_hub import hf_hub_download

    meta = hf_hub_download(source.hf_id, source.file_pattern or "metadata.csv",
                           repo_type="dataset", revision=revision)
    rows = list(csv.DictReader(open(meta, encoding="utf-8")))
    pos = [r[audio_col] for r in rows if r.get(label_col) == wakeword]
    negset = set(negative_labels) if negative_labels else None
    neg = [r[audio_col] for r in rows if r.get(label_col) != wakeword
           and (negset is None or r.get(label_col) in negset)]
    yield from _emit_ww_clips(source.hf_id, pos, neg, revision, max_per_class)


def stream_manifest_audio(
    source,
    audio_key: str,
    extra_keys: Dict[str, str],
    revision: str,
    max_samples: int = 0,
) -> Iterator[Tuple[str, dict]]:
    """Yield ``(sample_id, {"array", "sr", **extras})`` from a JSONL manifest.

    For datasets stored as a per-sample ``manifest.jsonl`` next to audio files
    (the ww-bench layout: one JSON record per clip with a relative ``path`` and
    a ``role``).  ``source.file_pattern`` names the manifest file in the repo;
    each record's *audio_key* field is the repo-relative audio path.  Lines
    without the audio field (e.g. the ``_manifest_header``) are skipped.
    """
    import json

    from huggingface_hub import hf_hub_download

    manifest_path = source.file_pattern or "manifest.jsonl"
    manifest = hf_hub_download(source.hf_id, manifest_path, repo_type="dataset",
                              revision=revision)
    count = 0
    with open(manifest, encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping bad manifest line: %s", exc)
                continue
            rel = record.get(audio_key)
            if not rel:
                continue  # header or rows without an audio path
            if max_samples and count >= max_samples:
                return
            try:
                local = hf_hub_download(source.hf_id, rel, repo_type="dataset",
                                        revision=revision)
                with open(local, "rb") as fh:
                    array, sr = decode_audio_bytes(fh.read())
            except Exception as exc:
                logger.warning("manifest sample %s failed: %s", rel, exc)
                continue
            # the source clip is a real file in the repo → a playable URL the
            # battle UI can offer the voter
            sample = {
                "array": array, "sr": sr,
                "audio_url": (f"https://huggingface.co/datasets/{source.hf_id}"
                              f"/resolve/{revision}/{rel}"),
            }
            for out_field, col in extra_keys.items():
                sample[out_field] = record.get(col)
            yield rel, sample
            count += 1


def stream_audio_dataset(
    source,
    audio_key: str,
    extra_keys: Dict[str, str],
    revision: str,
    max_samples: int = 0,
    id_key: Optional[str] = None,
) -> Iterator[Tuple[str, dict]]:
    """Yield ``(sample_id, {"array", "sr", **extras})`` per audio sample.

    *extra_keys* maps an output field name to its source column name (e.g.
    ``{"ground_truth": "transcription"}`` for STT, ``{"label": "label"}`` for
    wake word); the named columns are copied through verbatim.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = _parquet_files(source.hf_id, source.subset, source.split, revision)
    if not files:
        raise RuntimeError(
            f"no parquet files for {source.hf_id} "
            f"subset={source.subset} split={source.split}"
        )

    count = 0
    for pfile in files:
        local = hf_hub_download(source.hf_id, pfile, repo_type="dataset",
                                revision=revision)
        table = pq.read_table(local)
        for batch in table.to_batches(max_chunksize=64):
            rows = batch.to_pydict()
            n = len(rows.get(audio_key, []))
            for i in range(n):
                if max_samples and count >= max_samples:
                    return
                audio_cell = (rows.get(audio_key) or [None])[i]
                if audio_cell is None:
                    continue
                array = sr = None
                if isinstance(audio_cell, dict):
                    if audio_cell.get("array") is not None:
                        array = np.asarray(audio_cell["array"], dtype=np.float32)
                        sr = audio_cell.get("sampling_rate", TARGET_SR)
                    elif audio_cell.get("bytes"):
                        try:
                            array, sr = decode_audio_bytes(audio_cell["bytes"])
                        except Exception as exc:
                            logger.warning("decode error: %s", exc)
                            continue
                if array is None:
                    continue
                sample = {"array": array, "sr": sr}
                for out_field, col in extra_keys.items():
                    sample[out_field] = (rows.get(col) or [None])[i]
                sid = _sample_id(
                    {k: (rows.get(k) or [None])[i] for k in (rows or {})},
                    audio_cell, audio_key, id_key, count,
                )
                yield sid, sample
                count += 1
