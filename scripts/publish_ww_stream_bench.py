#!/usr/bin/env python3
"""Build the ``ww_stream`` corpus of every wake-word test set and its registry entries.

The corpus is one Hugging Face dataset repo:

    negatives/audio/*.wav     hours of audio with no wake word, shared by every test set
    <set>/audio/*.wav         the test set's clips with the phrase, between seconds of speech
    manifest-<set>.jsonl      the test set's recordings, then every negative recording

Every recording comes from ``scripts/build_ww_stream_corpus.py``. The sources
are pinned below; the clip sets are read at the commit their ``main`` points
at when the script runs, and the card and the registry entries record it.

Usage::

    python3 scripts/publish_ww_stream_bench.py --work ~/tmp/ww-stream --seed 0
    python3 scripts/publish_ww_stream_bench.py --work ~/tmp/ww-stream --upload
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_ww_stream_corpus import build_corpus, load_audio, trim_silence  # noqa: E402

from registry.loaders import list_competitors, list_datasets, results_repo_for  # noqa: E402
from runner.audio_io import _even, decode_audio_bytes, ww_positive_files  # noqa: E402
from runner.dataset_cards import FUNDING_BLOCK, check_funding_block  # noqa: E402
from runner.intent_bench import resolve_revision  # noqa: E402

log = logging.getLogger("publish_ww_stream_bench")

REPO = "OpenVoiceOS/ww-stream-bench"
LIBRISPEECH = ("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1")
NEGATIVE_SPEECH = ("all/test.clean/0000.parquet", "all/test.other/0000.parquet")
FILLER_SPEECH = ("all/validation.clean/0000.parquet", "all/validation.other/0000.parquet")
SOUNDS = (
    "TigreGotico/public_domain_sounds_3secs",
    "f47136f9dd92b3b4ae1c4a6b9d545f35a1a0666d",
    "pd_3sec",
)
SOUND_FILES = 2000
NEGATIVE_HOURS = 12.0
MAX_POSITIVES = 400
FILLER_S_PER_POSITIVE = 6.0
MIN_GAP_S = 3.0


def wake_phrases() -> list[str]:
    """Every phrase a wake-word test set or fighter answers to, lower case with spaces."""
    phrases = {d.wakeword for d in list_datasets("wake_word") if d.wakeword}
    for competitor in list_competitors("wake_word"):
        phrases |= set(competitor.config.get("hotwords", {}))
    return sorted({p.replace("_", " ").lower() for p in phrases})


def phrase_pattern(phrases: list[str]) -> re.Pattern[str]:
    """Match any of *phrases* as whole words, ignoring case."""
    return re.compile(r"\b(?:" + "|".join(map(re.escape, phrases)) + r")\b", re.IGNORECASE)


def export_parquet(
    repo: str, revision: str, files: tuple[str, ...], out: Path, drop: re.Pattern[str]
) -> tuple[int, list[str]]:
    """Write each row of the parquet *files* as ``<id>.wav`` under *out*.

    A row whose transcript matches *drop* is left out. Returns the count written
    and the ids left out.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    count = 0
    dropped = []
    for name in files:
        table = pq.read_table(
            hf_hub_download(repo, name, repo_type="dataset", revision=revision),
            columns=["id", "text", "audio"],
        )
        for row_id, text, audio in zip(
            table.column("id").to_pylist(),
            table.column("text").to_pylist(),
            table.column("audio").to_pylist(),
            strict=True,
        ):
            if drop.search(text):
                dropped.append(row_id)
                continue
            target = out / f"{row_id}.wav"
            if not target.exists():
                array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                sf.write(target, array, sr, subtype="PCM_16")
            count += 1
    return count, dropped


def export_files(entries: list[tuple[str, str, str]], out: Path) -> list[str]:
    """Download *entries* (``hf_id, path, revision``) as 16 kHz wav files.

    Writes ``metadata.csv`` beside them.
    """
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    names = []
    for k, (hf_id, rel, revision) in enumerate(entries):
        name = f"{k:05d}.wav"
        target = out / name
        if not target.exists():
            local = hf_hub_download(hf_id, rel, repo_type="dataset", revision=revision)
            array, sr = decode_audio_bytes(Path(local).read_bytes())
            sf.write(target, np.asarray(array, dtype="float32"), sr, subtype="PCM_16")
        names.append((name, rel))
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name", "source"])
        writer.writerows(names)
    return [n for n, _ in names]


def clip_sets() -> list:
    """Every wake-word test set that has clips with a phrase (not the negative-only pools)."""
    return [d for d in list_datasets("wake_word") if d.wakeword and d.role == "eval"]


def prefixed(rows: list[dict], prefix: str) -> list[dict]:
    return [{**row, "audio": f"{prefix}/{row['audio']}"} for row in rows]


def build(work: Path, seed: int, only: list[str]) -> dict:
    """Write the repo tree under ``work/repo``; returns the facts the card and the registry need."""
    src = work / "sources"
    repo = work / "repo"
    facts_path = work / "facts.json"
    if only and facts_path.is_file():
        # add the named test sets to an earlier build; its negatives stay as they are
        facts = json.loads(facts_path.read_text())
        negatives = [
            {**row, "audio": row["audio"].removeprefix("negatives/")}
            for row in map(json.loads, (repo / "manifest-negatives.jsonl").read_text().splitlines())
        ]
        build_sets(src, repo, seed, only, negatives, facts)
        facts_path.write_text(json.dumps(facts, indent=1, sort_keys=True))
        return facts
    phrases = wake_phrases()
    drop = phrase_pattern(phrases)
    n_speech, dropped_speech = export_parquet(
        *LIBRISPEECH, NEGATIVE_SPEECH, src / "negative-speech", drop
    )
    n_filler, dropped_filler = export_parquet(
        *LIBRISPEECH, FILLER_SPEECH, src / "filler-speech", drop
    )
    from huggingface_hub import HfApi

    sounds = sorted(
        f
        for f in HfApi().list_repo_files(SOUNDS[0], repo_type="dataset", revision=SOUNDS[1])
        if f.startswith(SOUNDS[2] + "/") and f.endswith(".wav")
    )
    sound_entries = [(SOUNDS[0], f, SOUNDS[1]) for f in _even(sounds, SOUND_FILES)]
    export_files(sound_entries, src / "negative-sounds")
    log.info(
        "sources: %d negative utterances (%d with a phrase left out), "
        "%d filler utterances (%d left out), %d sounds",
        n_speech,
        len(dropped_speech),
        n_filler,
        len(dropped_filler),
        len(sound_entries),
    )

    negatives = build_corpus(
        None,
        [src / "negative-speech", src / "negative-sounds"],
        repo / "negatives",
        NEGATIVE_HOURS,
        seed,
        file_seconds=1800.0,
    )
    facts = {
        "seed": seed,
        "negative_hours": sum(r["duration_s"] for r in negatives) / 3600,
        "negative_recordings": len(negatives),
        "phrases": phrases,
        "dropped_utterances": sorted(dropped_speech + dropped_filler),
        "sets": {},
    }
    (repo / "dropped-utterances.txt").write_text(
        "".join(f"{u}\n" for u in facts["dropped_utterances"])
    )
    (repo / "manifest-negatives.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in prefixed(negatives, "negatives"))
    )

    build_sets(src, repo, seed, only, negatives, facts)
    facts_path.write_text(json.dumps(facts, indent=1, sort_keys=True))
    return facts


def build_sets(
    src: Path, repo: Path, seed: int, only: list[str], negatives: list[dict], facts: dict
) -> None:
    """Write one recording and one manifest per clip test set; adds each set to *facts*."""
    for dataset in clip_sets():
        if only and dataset.dataset_id not in only:
            continue
        revision = resolve_revision(dataset.source.hf_id, dataset.source.revision)
        positives = ww_positive_files(dataset, revision)
        chosen = _even(sorted(positives, key=lambda e: e[1]), MAX_POSITIVES)
        if not chosen:
            log.warning("%s: no clips with the phrase, skipped", dataset.dataset_id)
            continue
        clips_dir = src / "positives" / dataset.dataset_id
        names = export_files(chosen, clips_dir)
        # build_corpus drops a clip that is silent after trimming; each other clip plays once
        silent = [
            rel
            for name, (_, rel, _) in zip(names, chosen, strict=True)
            if not len(trim_silence(load_audio(clips_dir / name)))
        ]
        for rel in silent:
            log.warning("%s: %s is silent, left out", dataset.dataset_id, rel)
        usable = len(chosen) - len(silent)
        rows = build_corpus(
            clips_dir,
            [src / "filler-speech"],
            repo / dataset.dataset_id,
            usable * FILLER_S_PER_POSITIVE / 3600,
            seed,
            file_seconds=usable * FILLER_S_PER_POSITIVE,
            positives_per_file=usable,
            min_gap_s=MIN_GAP_S,
        )
        sources = [name for r in rows for name in r["sources"]]
        if len(sources) != usable or len(set(sources)) != usable:
            raise RuntimeError(
                f"{dataset.dataset_id}: {usable} usable clips in, {len(sources)} onsets "
                f"from {len(set(sources))} clips out"
            )
        onsets = usable
        manifest = prefixed(rows, dataset.dataset_id) + prefixed(negatives, "negatives")
        (repo / f"manifest-{dataset.dataset_id}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest)
        )
        facts["sets"][dataset.dataset_id] = {
            "source": dataset.source.hf_id,
            "revision": revision,
            "license": dataset.license,
            "wakeword": dataset.wakeword,
            "lang": dataset.lang,
            "available": len(positives),
            "silent": silent,
            "onsets": onsets,
            "hours": sum(r["duration_s"] for r in rows) / 3600,
        }
        log.info("%s: %d of %d clips with the phrase", dataset.dataset_id, onsets, len(positives))


def card(facts: dict) -> str:
    rows = "\n".join(
        f"| `{sid}` | `{s['wakeword']}` "
        f"| [{s['source']}](https://huggingface.co/datasets/{s['source']}) `{s['revision'][:8]}` "
        f"| {s['license']} | {s['onsets']} of {s['available']} | {s['hours']:.2f} |"
        for sid, s in sorted(facts["sets"].items())
    )
    configs = "\n".join(
        f"- config_name: {sid}\n  data_files: manifest-{sid}.jsonl"
        for sid in ["negatives", *sorted(facts["sets"])]
    )
    negatives = (
        f"{facts['negative_recordings']} recordings, {facts['negative_hours']:.2f} hours, "
        "with no wake word"
    )
    left_out = [
        f"`{rel}` of `{sid}`"
        for sid, s in sorted(facts["sets"].items())
        for rel in s.get("silent", [])
    ]
    silent = (
        "\n\nThese clips hold no sound and are left out: " + ", ".join(left_out) + "."
        if left_out
        else ""
    )
    return f"""---
license: cc-by-4.0
task_categories:
- audio-classification
tags:
- openvoiceos
- wake-word
- keyword-spotting
- benchmark
pretty_name: OVOS streaming wake-word benchmark
configs:
{configs}
---

# OVOS streaming wake-word benchmark

Continuous recordings for testing a wake-word detector the way it runs on a
live microphone. A detector streams each recording in 80 ms blocks. The
manifest gives the time of every real wake word, so a test counts the wake
words it missed and the times it woke with no wake word. The
[OVOS plugin arena](https://github.com/OpenVoiceOS/ovos-plugin-arena) scores
its `ww_stream` boards from this repo.

## Layout

| Path | Contents |
|---|---|
| `negatives/audio/` | {negatives} |
| `<set>/audio/` | One recording per test set: its clips with the phrase, with speech between them |
| `manifest-<set>.jsonl` | The test set's recording, then every negative recording |
| `manifest-negatives.jsonl` | The negative recordings only |
| `dropped-utterances.txt` | The LibriSpeech ids left out because their transcript holds a phrase |

A manifest row has `audio`, `duration_s`, `onsets` and `ends` (seconds),
`sources` (the clip behind each onset), `snr_db` and `level_dbfs`. A
negative recording has empty `onsets`.

## How it is made

- **Audio with no wake word**: LibriSpeech `test-clean` and `test-other`
  (read English speech, CC BY 4.0), and {SOUND_FILES} three-second clips of the
  public-domain sounds of pdsounds.org, shuffled together and cut into
  30-minute recordings.
- **Speech between the wake words**: LibriSpeech `dev-clean` and `dev-other`,
  so it never repeats the audio with no wake word.
- **No phrase in the speech**: an utterance whose transcript holds any of the
  {len(facts["phrases"])} phrases of the arena's wake-word test sets and fighters, as whole words,
  is left out of both. That removes {len(facts["dropped_utterances"])} utterances, named in
  `dropped-utterances.txt`. The sounds have no transcript and are not filtered.
- **Wake words**: up to {MAX_POSITIVES} clips with the phrase from each test
  set, picked evenly across the set. Silence before and after each clip is cut,
  so its onset is where the phrase starts. Each clip is set 5 to 15 dB above the
  level of the speech in its recording. Consecutive clips are at least
  {MIN_GAP_S:.0f} seconds of speech apart, and {FILLER_S_PER_POSITIVE:.0f} seconds on average.
- The build is a pure function of the sources, the phrases above and seed `{facts["seed"]}`:
  `python3 scripts/publish_ww_stream_bench.py --work <folder> --seed {facts["seed"]}` in
  the arena.

| Test set | Phrase | Source | Licence | Clips | Hours |
|---|---|---|---|---|---|
{rows}

LibriSpeech is pinned at `{LIBRISPEECH[1][:8]}` of
[{LIBRISPEECH[0]}](https://huggingface.co/datasets/{LIBRISPEECH[0]}); the sounds at
`{SOUNDS[1][:8]}` of [{SOUNDS[0]}](https://huggingface.co/datasets/{SOUNDS[0]}).
Each recording keeps the licence of its sources.{silent}

{FUNDING_BLOCK}
"""


def registry_entry(sid: str, s: dict, revision: str) -> dict:
    phrase = s["wakeword"].replace("_", " ")
    return {
        "dataset_id": f"ww_stream_{sid}",
        "display_name": f"Streaming wake word — {sid}",
        "summary": (
            f"The clips of “{phrase}” from the `{sid}` test set, placed between seconds of read "
            f"speech, followed by hours of audio with no wake word. A detector streams every "
            f"recording and is scored on the wake words it caught and on how often per hour "
            f"it woke with no wake word."
        ),
        "modality": "ww_stream",
        "source": {
            "type": "huggingface",
            "hf_id": REPO,
            "revision": revision,
            "file_pattern": f"manifest-{sid}.jsonl",
        },
        "reference_fields": {"audio": "audio", "onsets": "onsets", "ends": "ends"},
        "wakeword": s["wakeword"],
        "lang": s["lang"],
        "license": "CC-BY-4.0",
        "role": "eval",
        "sample_rate_hz": 16000,
        "event_manifest_fields": {"audio": "audio", "onsets": "onsets", "duration_s": "duration_s"},
        "min_negative_hours": 10.0,
        "predictions_hf": results_repo_for("ww_stream", f"ww_stream_{sid}"),
        "event_tolerance_s": 1.5,
        "test_set_kind": "synthetic" if sid.startswith("synthetic-") else "real",
        "notes": (
            f"Built by scripts/publish_ww_stream_bench.py from `{sid}` at "
            f"{s['source']}@{s['revision'][:8]} ({s['onsets']} of {s['available']} clips with the "
            f"phrase) and the shared negative recordings."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--work", type=Path, required=True, help="folder for the sources and the repo tree"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[], help="test set ids to build (default: all)")
    ap.add_argument(
        "--upload",
        action="store_true",
        help="upload work/repo to the Hub and write the registry entries at the new commit",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    facts_path = args.work / "facts.json"
    facts = (
        json.loads(facts_path.read_text())
        if args.upload
        else build(args.work, args.seed, args.only)
    )
    text = check_funding_block(card(facts), REPO)
    (args.work / "repo" / "README.md").write_text(text)
    if not args.upload:
        log.info("built %d test sets under %s", len(facts["sets"]), args.work / "repo")
        return 0
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(args.work / "repo"),
        commit_message=f"Build with seed {facts['seed']}",
    )
    revision = api.dataset_info(REPO).sha
    out = ROOT / "registry" / "datasets" / "ww_stream"
    out.mkdir(parents=True, exist_ok=True)
    for sid, s in sorted(facts["sets"].items()):
        (out / f"ww_stream_{sid}.json").write_text(
            json.dumps(registry_entry(sid, s, revision), indent=2, ensure_ascii=False) + "\n"
        )
    log.info("uploaded %s at %s; wrote %d registry entries", REPO, revision, len(facts["sets"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
