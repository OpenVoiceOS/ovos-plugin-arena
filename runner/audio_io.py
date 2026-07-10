"""
Audio dataset streaming for the STT and wake-word benchmarks.

Reads HuggingFace audio corpora straight from their parquet files via
``huggingface_hub`` + ``pyarrow`` (no ``datasets`` dependency), decoding each
clip to a mono float32 array resampled to 16 kHz.  All imports are lazy so the
arena core and tests do not pull in audio stacks.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

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


def _parquet_files(hf_repo: str, subset: str | None, split: str,
                   revision: str) -> list[str]:
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
               id_key: str | None, index: int) -> str:
    if id_key and sample.get(id_key) is not None:
        return str(sample[id_key])
    if isinstance(audio_cell, dict):
        path = audio_cell.get("path") or ""
        name = path.split("/snapshots/")[-1].split("/", 1)[-1]
        if name:
            return name
    return f"sample_{index:06d}"


_AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def _even(paths: list[str], cap: int) -> list[str]:
    """Pick *cap* paths evenly spaced across the sorted list (deterministic).

    Striding spans the full range — across TTS voices within a folder and
    across folders in a concatenated negative list — instead of taking a
    same-voice run from the front.
    """
    if not cap or len(paths) <= cap:
        return paths
    step = len(paths) / cap
    return [paths[int(i * step)] for i in range(cap)]


def _repo_audio(hf_id: str, revision: str) -> list[str]:
    from huggingface_hub import HfApi

    return [f for f in HfApi().list_repo_files(hf_id, repo_type="dataset",
                                               revision=revision)
            if f.lower().endswith(_AUDIO_EXT) and "/" in f]


def _all_audio(hf_id: str, revision: str, subdir: str | None = None) -> list[str]:
    """Every audio file in a repo (root-level too), optionally under *subdir*."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(hf_id, repo_type="dataset",
                                                revision=revision)
             if f.lower().endswith(_AUDIO_EXT)]
    if subdir:
        files = [f for f in files
                 if f == subdir or f.startswith(subdir + "/")]
    return files


def _csv_rows(hf_id: str, csv_path: str, revision: str) -> list:
    import csv

    from huggingface_hub import hf_hub_download

    meta = hf_hub_download(hf_id, csv_path, repo_type="dataset", revision=revision)
    return list(csv.DictReader(open(meta, encoding="utf-8")))


def _emit_labelled(pos, neg, max_per_class, pos_label="positive",
                   neg_label="negative"):
    """Download + decode clip tuples ``(hf_id, rel, rev)`` → labelled samples.

    Shared by the binary-detection benchmarks (wake word: positive/negative;
    VAD: speech/non_speech).
    """
    from urllib.parse import quote

    from huggingface_hub import hf_hub_download

    if max_per_class:
        pos = _even(sorted(pos), max_per_class)
        neg = _even(sorted(neg), max_per_class)
    for label, clips in ((pos_label, pos), (neg_label, neg)):
        for hf_id, rel, rev in clips:
            try:
                local = hf_hub_download(hf_id, rel, repo_type="dataset", revision=rev)
                with open(local, "rb") as fh:
                    array, sr = decode_audio_bytes(fh.read())
            except Exception as exc:
                logger.warning("clip %s/%s failed: %s", hf_id, rel, exc)
                continue
            url = (f"https://huggingface.co/datasets/{hf_id}"
                   f"/resolve/{rev}/{quote(rel)}")
            yield rel, {"array": array, "sr": sr, "label": label, "audio_url": url}


def _emit_ww(pos, neg, max_per_class):
    """Wake-word labels (positive = wake word present)."""
    yield from _emit_labelled(pos, neg, max_per_class, "positive", "negative")


def _pool_negatives(sources, max_per_class):
    """Even share of audio across several corpora → ``(hf_id, rel, rev)`` tuples.

    An entry is ``org/name`` optionally followed by a subdir
    (``org/name/subdir``); a big corpus never dominates the pool.
    """
    per = max(1, -(-max_per_class // len(sources))) if max_per_class else 0
    out = []
    for spec in sources:
        parts = spec.split("/")
        nhf = "/".join(parts[:2])
        sub = "/".join(parts[2:]) or None
        files = sorted(_all_audio(nhf, "main", sub))
        for rel in (_even(files, per) if per else files):
            out.append((nhf, rel, "main"))
    return out


def stream_vad(dataset_def, revision: str, max_per_class: int = 0
               ) -> Iterator[tuple[str, dict]]:
    """Yield labelled clips for the VAD league: speech vs non-speech.

    Positives are speech recordings from ``dataset_def.source`` (the whole
    repo, or a ``file_pattern`` subdir); negatives are non-speech audio (music,
    environmental sound, noise, silence) pooled across
    ``dataset_def.negatives_sources`` so the false-accept rate — firing speech
    on non-speech — spans many scenarios. ``max_per_class`` caps each class so
    the battle pool stays balanced.
    """
    src = dataset_def.source
    fields = dataset_def.reference_fields or {}
    neg = _pool_negatives(dataset_def.negatives_sources or [], max_per_class)

    if getattr(src, "split", None):
        # Parquet / HF-dataset speech positives (e.g. MInDS-14 per language) —
        # any speech corpus works as VAD positives. Negatives stay the shared
        # non-speech pool. Lets VAD run across many languages.
        audio_key = fields.get("audio", "audio")
        for sid, sample in stream_audio_dataset(
                src, audio_key=audio_key, extra_keys={}, revision=revision,
                max_samples=max_per_class):
            sample["label"] = "speech"
            sample.setdefault("audio_url", None)
            yield f"speech/{sid}", sample
        yield from _emit_labelled([], neg, max_per_class, "speech", "non_speech")
        return

    # Audiofolder speech positives (whole repo, or a subset subdir).
    subdir = (src.subset or "").rstrip("/") or None
    pos = [(src.hf_id, f, revision)
           for f in _all_audio(src.hf_id, revision, subdir)]
    yield from _emit_labelled(pos, neg, max_per_class, "speech", "non_speech")


def stream_ww(dataset_def, revision: str, max_per_class: int = 0
              ) -> Iterator[tuple[str, dict]]:
    """Yield labelled wake-word clips for a benchmark.

    Positives are the wake-word clips of ``dataset_def`` (a ``<wakeword>/``
    folder, or rows of a ``metadata.csv`` whose label is the wakeword).
    Negatives come from a separate not-wake-word corpus when ``negatives_hf``
    is set (general speech/noise that must never fire — the proper false-accept
    test), otherwise from the same corpus's other phrases.
    """
    src = dataset_def.source
    hf, ww = src.hf_id, dataset_def.wakeword
    fields = dataset_def.reference_fields or {}
    negset = set(dataset_def.negative_dirs) if dataset_def.negative_dirs else None

    if (src.file_pattern or "").endswith(".csv"):
        ac = fields.get("audio", "file_name")
        lc = fields.get("label", "label")
        rows = _csv_rows(hf, src.file_pattern, revision)
        pos = [(hf, r[ac], revision) for r in rows if r.get(lc) == ww]
        same_neg = [(hf, r[ac], revision) for r in rows if r.get(lc) != ww
                    and (negset is None or r.get(lc) in negset)]
    else:
        files = _repo_audio(hf, revision)
        # ``subset`` optionally names a wrapping folder (e.g. Picovoice's
        # ``data/<wakeword>/``); the wake phrase is the first component below it.
        prefix = (src.subset.rstrip("/") + "/") if src.subset else ""

        def _phrase(f: str) -> str | None:
            if prefix:
                if not f.startswith(prefix):
                    return None
                f = f[len(prefix):]
            return f.split("/")[0]

        pos = [(hf, f, revision) for f in files if _phrase(f) == ww]
        same_neg = [(hf, f, revision) for f in files
                    if _phrase(f) not in (ww, None)
                    and (negset is None or _phrase(f) in negset)]

    if dataset_def.negatives_sources:
        # pool negatives across several not-wake-word corpora (speech, music,
        # noise, household sounds) so false-accept rate spans many scenarios;
        # take an even share from each so a big corpus does not dominate.
        srcs = dataset_def.negatives_sources
        per = max(1, -(-max_per_class // len(srcs))) if max_per_class else 0
        neg = []
        for spec in srcs:
            # an hf id is ``org/name``; anything after the second / is a subdir
            parts = spec.split("/")
            nhf = "/".join(parts[:2])
            sub = "/".join(parts[2:]) or None
            files = sorted(_all_audio(nhf, "main", sub or None))
            for rel in (_even(files, per) if per else files):
                neg.append((nhf, rel, "main"))
    elif dataset_def.negatives_hf:
        nhf, ndir = dataset_def.negatives_hf, dataset_def.negatives_dir
        nfiles = _all_audio(nhf, "main", ndir)
        neg = [(nhf, f, "main") for f in nfiles]
    else:
        neg = same_neg

    yield from _emit_ww(pos, neg, max_per_class)


def stream_manifest_audio(
    source,
    audio_key: str,
    extra_keys: dict[str, str],
    revision: str,
    max_samples: int = 0,
) -> Iterator[tuple[str, dict]]:
    """Yield ``(sample_id, {"array", "sr", **extras})`` from a manifest file.

    For datasets stored as a per-sample manifest beside audio files: a
    ``manifest.jsonl`` (one JSON record per clip — the ww-bench layout) or a
    ``metadata.csv`` (one row per clip — the audiofolder layout, e.g.
    ``speech_MASSIVE_pt-PT``). ``source.file_pattern`` names the manifest;
    each record's *audio_key* field is the repo-relative audio path. Records
    without the audio field (e.g. a header line) are skipped.
    """
    import json

    from huggingface_hub import hf_hub_download

    manifest_path = source.file_pattern or "manifest.jsonl"
    if manifest_path.endswith((".csv", ".tsv")):
        records: Iterable[dict] = _csv_rows(source.hf_id, manifest_path, revision)
    else:
        manifest = hf_hub_download(source.hf_id, manifest_path,
                                   repo_type="dataset", revision=revision)
        def _jsonl():
            with open(manifest, encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("skipping bad manifest line: %s", exc)
        records = _jsonl()

    count = 0
    for record in records:
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
    extra_keys: dict[str, str],
    revision: str,
    max_samples: int = 0,
    id_key: str | None = None,
) -> Iterator[tuple[str, dict]]:
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
