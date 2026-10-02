#!/usr/bin/env python3
"""Build a ``ww_stream`` benchmark corpus from isolated clips.

Inputs are a directory of positive clips for one wake word (wav/flac files
plus an audiofolder-style ``metadata.csv`` whose ``file_name`` column lists
them) and one or more local directories of negative audio (wav/flac, any
sample rate). Output is a directory of long 16 kHz mono recordings plus a
``manifest.jsonl`` with one row per recording:

    audio        recording path relative to the output directory
    onsets       wake-word onset times in seconds (sample-exact: the start
                 of the inserted clip)
    duration_s   recording length in seconds
    sources      the positive clip name behind each onset (parallel to onsets)
    onset_samples, end_samples, ends, spoken_s
                 each inserted clip's first sample, end sample and end time,
                 and its length in seconds (leading and trailing silence is
                 trimmed, so the onset is where the phrase starts)
    snr_db, level_dbfs
                 each inserted clip's RMS relative to the recording's
                 negative audio, and its absolute RMS in dBFS

Positives are trimmed of leading and trailing silence, so the onset is where
the phrase starts and ``ends`` where it ends. The scorer counts a detection
for an onset when it falls in ``[onset, end + tolerance]``, because detectors
fire at the end of the phrase. A positive longer than 4 s is kept but
logged as suspicious. Positives are inserted between
negative audio, never mixed over it, at
random positions that keep consecutive onsets at least ``--min-gap-s`` of
negative audio apart. ``--file-seconds`` is the negative audio per recording;
the recording is longer by the inserted clips. The total negative audio is
exactly ``--negative-hours`` unless the negative pool is smaller than that,
in which case it is replayed in a fresh shuffle (logged as a warning).

The output is a pure function of the inputs and ``--seed``. No network.

Usage::

    python3 scripts/build_ww_stream_corpus.py \\
        --positives ./hey_mycroft_clips --negatives ./speech ./noise \\
        --out ./ww-stream-hey_mycroft --negative-hours 10 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("build_ww_stream_corpus")

SAMPLE_RATE = 16000
AUDIO_SUFFIXES = (".wav", ".flac")
FADE_S = 0.01
PEAK_LIMIT = 0.99
RMS_FLOOR = 1e-4
TRIM_REL_DB = -40.0
MIN_LEVEL_DBFS = -35.0
WARN_CLIP_S = 4.0


def trim_silence(clip: np.ndarray, rel_db: float = TRIM_REL_DB) -> np.ndarray:
    """Drop leading and trailing samples below *rel_db* of the clip's peak."""
    if not len(clip):
        return clip
    peak = float(np.max(np.abs(clip)))
    if peak <= 0:
        return clip[:0]
    loud = np.nonzero(np.abs(clip) >= peak * 10 ** (rel_db / 20))[0]
    return clip[loud[0]: loud[-1] + 1]


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode *path* to mono float32 at *sample_rate* (linear-interpolation
    resampling, with a box low-pass first when downsampling)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate and len(data):
        if sr > sample_rate:
            width = max(1, int(round(sr / sample_rate)))
            kernel = np.ones(width, dtype="float32") / width
            data = np.convolve(data, kernel, mode="same")
        n_out = int(round(len(data) * sample_rate / sr))
        x_old = np.arange(len(data), dtype="float64") / sr
        x_new = np.arange(n_out, dtype="float64") / sample_rate
        data = np.interp(x_new, x_old, data).astype("float32")
    return np.ascontiguousarray(data, dtype="float32")


def list_audio(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def read_positive_names(positives_dir: Path, file_column: str = "file_name") -> list[str]:
    """Clip names listed in ``metadata.csv``, sorted and deduplicated."""
    meta = positives_dir / "metadata.csv"
    with meta.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if file_column not in (reader.fieldnames or []):
            raise ValueError(f"{meta}: no {file_column!r} column")
        names = [row[file_column] for row in reader if row.get(file_column)]
    return sorted(dict.fromkeys(names))


class NegativeStream:
    """Endless supply of negative audio: files in a seeded shuffle, replayed
    in a fresh shuffle once the pool is spent."""

    def __init__(self, files: list[Path], rng: np.random.Generator, sample_rate: int):
        if not files:
            raise ValueError("no negative audio files found")
        self._files = files
        self._rng = rng
        self._sr = sample_rate
        self._queue: list[Path] = []
        self._buf = np.zeros(0, dtype="float32")
        self._passes = 0

    @property
    def replays(self) -> int:
        return max(0, self._passes - 1)

    def _refill(self) -> None:
        if not self._queue:
            order = self._rng.permutation(len(self._files))
            self._queue = [self._files[i] for i in order]
            self._passes += 1
        chunk = load_audio(self._queue.pop(0), self._sr)
        self._buf = np.concatenate([self._buf, chunk])

    def take(self, n: int) -> np.ndarray:
        while len(self._buf) < n:
            self._refill()
        out, self._buf = self._buf[:n], self._buf[n:]
        return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype="float64")))) if len(x) else 0.0


def _insertion_points(
    rng: np.random.Generator, n_samples: int, count: int, min_gap: int
) -> list[int]:
    """*count* sorted cut points in ``[min_gap, n_samples - min_gap]``, each at
    least *min_gap* samples after the previous one (uniform over the
    feasible arrangements)."""
    span = n_samples - 2 * min_gap - (count - 1) * min_gap
    if span < 0:
        raise ValueError(
            f"{count} positives with a {min_gap}-sample gap do not fit in "
            f"{n_samples} negative samples"
        )
    base = np.sort(rng.integers(0, span + 1, size=count))
    return [int(b) + min_gap + i * min_gap for i, b in enumerate(base)]


def build_corpus(
    positives_dir: Path,
    negative_dirs: list[Path],
    out_dir: Path,
    negative_hours: float,
    seed: int,
    file_seconds: float = 600.0,
    positives_per_file: int = 5,
    min_gap_s: float = 3.0,
    snr_db: tuple[float, float] = (5.0, 15.0),
    sample_rate: int = SAMPLE_RATE,
    file_column: str = "file_name",
    min_level_dbfs: float = MIN_LEVEL_DBFS,
) -> list[dict]:
    """Write the recordings and ``manifest.jsonl`` under *out_dir*; returns
    the manifest rows."""
    rng = np.random.default_rng(seed)
    all_names = read_positive_names(positives_dir, file_column)
    if not all_names:
        raise ValueError(f"{positives_dir}/metadata.csv lists no clips")
    clips: dict[str, np.ndarray] = {}
    for name in all_names:
        trimmed = trim_silence(load_audio(positives_dir / name, sample_rate))
        if not len(trimmed):
            continue
        if len(trimmed) > WARN_CLIP_S * sample_rate:
            log.warning("%s is %.1f s long after trimming", name, len(trimmed) / sample_rate)
        clips[name] = trimmed
    pos_names = sorted(clips)
    if not pos_names:
        raise ValueError("no usable (non-silent) positive clips")
    neg_files = [f for d in negative_dirs for f in list_audio(d)]
    stream = NegativeStream(neg_files, rng, sample_rate)

    total_neg = int(round(negative_hours * 3600 * sample_rate))
    per_file = int(round(file_seconds * sample_rate))
    min_gap = int(round(min_gap_s * sample_rate))
    fade = int(FADE_S * sample_rate)
    floor_rms = 10 ** (min_level_dbfs / 20)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pos_order: list[str] = []
    rows: list[dict] = []
    remaining = total_neg
    index = 0
    while remaining > 0:
        n_neg = min(per_file, remaining)
        remaining -= n_neg
        neg = stream.take(n_neg)
        fits = (n_neg - 2 * min_gap) // max(min_gap, 1) + 1
        k = max(0, min(positives_per_file, fits))
        points = _insertion_points(rng, n_neg, k, min_gap) if k else []

        pieces: list[np.ndarray] = []
        onsets: list[int] = []
        ends: list[int] = []
        sources: list[str] = []
        snrs: list[float] = []
        levels: list[float] = []
        ref = max(_rms(neg), RMS_FLOOR)
        cursor = 0
        out_len = 0
        for point in points:
            if not pos_order:
                pos_order = [pos_names[i] for i in rng.permutation(len(pos_names))]
            name = pos_order.pop()
            clip = clips[name].copy()
            target_db = float(rng.uniform(snr_db[0], snr_db[1]))
            clip_rms = _rms(clip)
            if clip_rms > 0:
                clip *= max(ref * 10 ** (target_db / 20), floor_rms) / clip_rms
            peak = float(np.max(np.abs(clip)))
            if peak > PEAK_LIMIT:
                clip *= PEAK_LIMIT / peak
            level = _rms(clip)
            f = min(fade, len(clip) // 2)
            if f:
                ramp = np.linspace(0.0, 1.0, f, dtype="float32")
                clip[:f] *= ramp
                clip[-f:] *= ramp[::-1]
            pieces.append(neg[cursor:point])
            out_len += point - cursor
            onsets.append(out_len)
            pieces.append(clip.astype("float32"))
            out_len += len(clip)
            ends.append(out_len)
            sources.append(name)
            snrs.append(round(float(20 * np.log10(level / ref)), 2))
            levels.append(round(float(20 * np.log10(level)), 2))
            cursor = point
        pieces.append(neg[cursor:])
        recording = np.clip(np.concatenate(pieces), -1.0, 1.0)

        rel = f"audio/ww_stream_{index:05d}.wav"
        sf.write(str(out_dir / rel), recording, sample_rate, subtype="PCM_16")
        rows.append({
            "audio": rel,
            "onsets": [o / sample_rate for o in onsets],
            "onset_samples": onsets,
            "end_samples": ends,
            "ends": [e / sample_rate for e in ends],
            "spoken_s": [(e - o) / sample_rate for o, e in zip(onsets, ends, strict=True)],
            "duration_s": len(recording) / sample_rate,
            "sources": sources,
            "snr_db": snrs,
            "level_dbfs": levels,
        })
        index += 1

    if stream.replays:
        log.warning(
            "negative pool replayed %d time(s) to reach %.2f h; the corpus "
            "repeats negative audio", stream.replays, negative_hours,
        )
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--positives", type=Path, required=True,
                    help="directory of positive clips with a metadata.csv")
    ap.add_argument("--negatives", type=Path, nargs="+", required=True,
                    help="directories of negative wav/flac audio")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--negative-hours", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--file-seconds", type=float, default=600.0,
                    help="negative audio per recording (default 600)")
    ap.add_argument("--positives-per-file", type=int, default=5)
    ap.add_argument("--min-gap-s", type=float, default=3.0,
                    help="minimum negative audio between consecutive onsets")
    ap.add_argument("--snr-db", type=float, nargs=2, default=(5.0, 15.0),
                    metavar=("MIN", "MAX"),
                    help="inserted clip RMS relative to the "
                         "negative audio, drawn uniformly per clip")
    ap.add_argument("--min-level-dbfs", type=float, default=MIN_LEVEL_DBFS,
                    help="absolute RMS floor for an inserted clip (default -35)")
    ap.add_argument("--file-column", default="file_name")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = build_corpus(
        args.positives, args.negatives, args.out, args.negative_hours, args.seed,
        file_seconds=args.file_seconds, positives_per_file=args.positives_per_file,
        min_gap_s=args.min_gap_s, snr_db=tuple(args.snr_db),
        file_column=args.file_column,
        min_level_dbfs=args.min_level_dbfs,
    )
    n_pos = sum(len(r["onsets"]) for r in rows)
    log.info("wrote %d recordings, %d onsets to %s", len(rows), n_pos, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
