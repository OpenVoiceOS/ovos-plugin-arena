#!/usr/bin/env python3
"""Build the ``ww_stream`` corpus and the registry entries of its false wake-up sets.

The corpus is one Hugging Face dataset repo:

    negatives/<group>/<category>/audio/*.flac   audio with no wake word, by language and kind
    <set>/audio/*.flac                          the test set's clips with the phrase, between
                                                seconds of speech in its language
    manifest-negatives-<lang>.jsonl             every recording with no wake word of a language
    manifest-<set>.jsonl                        the test set's recording, then the negatives
                                                of its language

The negatives of a language are its own speech plus a shared part: music,
household and outdoor sounds, and speech in other languages. Each negative row
names its ``category``, so a board can say which kind of audio woke a
detector. Every source is pinned in SOURCES, and every recording comes from
``scripts/build_ww_stream_corpus.py``. The clip sets are read at the commit
their ``main`` points at when the script runs, and the card records it. The
registry gets one ``ww-negatives-<lang>`` entry per language with a featured
phrase (``arena.ww_listing``); the recordings of the test sets get none.

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
import math
import re
import sys
import tarfile
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_ww_stream_corpus import (  # noqa: E402
    SAMPLE_RATE,
    build_corpus,
    load_audio,
    resample,
    trim_silence,
)

from arena.ww_listing import board_positives, registry_featured_phrases  # noqa: E402
from registry.loaders import (  # noqa: E402
    list_competitors,
    list_datasets,
    results_repo_for,
    wakeword_key,
)
from runner.audio_io import _even, decode_audio_bytes, ww_positive_files  # noqa: E402
from runner.dataset_cards import FUNDING_BLOCK, check_funding_block  # noqa: E402
from runner.intent_bench import resolve_revision  # noqa: E402

log = logging.getLogger("publish_ww_stream_bench")

REPO = "OpenVoiceOS/ww-stream-bench"
BOARDS = ROOT / "frontend-static" / "public" / "data"
SUFFIX = ".flac"
RECORDING_S = 1800.0
MAX_POSITIVES = 400
FILLER_S_PER_POSITIVE = 6.0
MIN_GAP_S = 3.0
# speech held back from the negatives of a language, for the speech between its wake words
FILLER_HOURS = 1.0
# the shared recordings each language other than English gets, per category
SHARED_RECORDINGS = 2
# FSD50K labels that mean a human voice; a clip with any of them is not a noise clip
VOICE_LABELS = frozenset(
    {
        "Human_voice",
        "Speech",
        "Male_speech_and_man_speaking",
        "Female_speech_and_woman_speaking",
        "Child_speech_and_kid_speaking",
        "Conversation",
        "Chatter",
        "Crowd",
        "Cheering",
        "Speech_synthesizer",
        "Singing",
        "Male_singing",
        "Female_singing",
        "Chant",
        "Yell",
        "Shout",
        "Screaming",
        "Whispering",
        "Laughter",
        "Giggle",
        "Chuckle_and_chortle",
        "Crying_and_sobbing",
        "Child_crying",
    }
)


@dataclass(frozen=True)
class Source:
    """One pinned audio source and how to read it.

    ``reader`` is ``parquet`` (rows with an audio column), ``tar`` (an archive
    with an index file of transcripts) or ``files`` (one audio file per row of
    an index, or every file under the folders in ``files``).
    """

    name: str
    hf_id: str
    revision: str
    reader: str
    files: tuple[str, ...]
    title: str
    split: str
    license: str
    nc: bool = False
    text: str = ""
    key: str = ""
    audio: str = "audio"
    index: str = ""
    index_columns: tuple[str, ...] = ()
    where: tuple[tuple[str, str], ...] = ()
    skip_labels: frozenset[str] = frozenset()
    path_format: str = "{}"
    seconds: str = ""
    seconds_scale: float = 1.0
    bytes_per_s: float = 0.0

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.hf_id}"


LIBRISPEECH = ("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1")
FLEURS = ("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")
COMMON_VOICE = ("fsicoli/common_voice_17_0", "8262c16bf297c87a9cd88c51997c4758ed7a8ba2")
FLEURS_COLUMNS = (
    "id",
    "file_name",
    "raw_transcription",
    "transcription",
    "phonemes",
    "num_samples",
    "gender",
)


def _fleurs(lang: str) -> Source:
    return Source(
        f"fleurs-{lang}",
        *FLEURS,
        "tar",
        (f"data/{lang}/audio/test.tar.gz",),
        f"FLEURS {lang}",
        "test",
        "CC BY 4.0",
        text="raw_transcription",
        key="file_name",
        index=f"data/{lang}/test.tsv",
        index_columns=FLEURS_COLUMNS,
        seconds="num_samples",
        seconds_scale=1 / 16000,
    )


def _common_voice(lang: str) -> Source:
    return Source(
        f"cv-{lang}",
        *COMMON_VOICE,
        "tar",
        (f"audio/{lang}/test/{lang}_test_0.tar",),
        f"Common Voice 17 {lang}",
        "test",
        "CC0",
        text="sentence",
        key="path",
        index=f"transcript/{lang}/test.tsv",
        bytes_per_s=8000,
    )


SOURCES = {
    s.name: s
    for s in [
        Source(
            "librispeech-test",
            *LIBRISPEECH,
            "parquet",
            ("all/test.clean/0000.parquet", "all/test.other/0000.parquet"),
            "LibriSpeech",
            "test-clean, test-other",
            "CC BY 4.0",
            text="text",
            key="id",
        ),
        Source(
            "librispeech-dev",
            *LIBRISPEECH,
            "parquet",
            ("all/validation.clean/0000.parquet", "all/validation.other/0000.parquet"),
            "LibriSpeech",
            "dev-clean, dev-other",
            "CC BY 4.0",
            text="text",
            key="id",
        ),
        Source(
            "ami-sdm-test",
            "edinburghcstr/ami",
            "46f28f2503e2ec48f8867a84eef356c70476beab",
            "parquet",
            tuple(f"sdm/test-0000{i}-of-00004.parquet" for i in range(4)),
            "AMI meetings, distant microphone",
            "sdm test",
            "CC BY 4.0",
            text="text",
            key="audio_id",
        ),
        Source(
            "peoples-speech-test",
            "MLCommons/peoples_speech",
            "f10597c5d3d3a63f8b6827701297c3afdf178272",
            "parquet",
            ("test/test-00000-of-00008.parquet", "test/test-00001-of-00008.parquet"),
            "People's Speech",
            "test",
            "CC BY and CC BY-SA",
            text="text",
            key="id",
        ),
        Source(
            "voxpopuli-en-test",
            "facebook/voxpopuli",
            "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
            "parquet",
            ("en/test-00000-of-00001.parquet",),
            "VoxPopuli en",
            "test",
            "CC0",
            text="normalized_text",
            key="audio_id",
        ),
        Source(
            "mls-test",
            "facebook/multilingual_librispeech",
            "2e83e61823b4c47dcbcb1980bb88601274127609",
            "parquet",
            tuple(
                f"{lang}/test-00000-of-00001.parquet"
                for lang in ("german", "french", "spanish", "italian", "dutch", "portuguese")
            ),
            "Multilingual LibriSpeech",
            "test, de/fr/es/it/nl/pt",
            "CC BY 4.0",
            text="transcript",
            key="id",
        ),
        Source(
            "musan-music",
            "corypaik/musan",
            "76f9882cfa4475efe11508ac9aa32722f84ca5b7",
            "parquet",
            ("music/train-00000-of-00010.parquet", "music/train-00005-of-00010.parquet"),
            "MUSAN music",
            "music",
            "CC BY 4.0, per-track terms",
            key="path",
        ),
        Source(
            "musan-noise",
            "corypaik/musan",
            "76f9882cfa4475efe11508ac9aa32722f84ca5b7",
            "parquet",
            ("noise/train-00000-of-00002.parquet", "noise/train-00001-of-00002.parquet"),
            "MUSAN noise",
            "noise",
            "CC BY 4.0, per-clip terms",
            key="path",
        ),
        Source(
            "demand-16k",
            "philgzl/demand",
            "29f46e613920cc05db198fd4ad7992e874e7f60f",
            "parquet",
            tuple(f"data/16k-00{i}-000000.parquet" for i in range(4)),
            "DEMAND",
            "16k",
            "CC BY 4.0",
            key="name",
        ),
        Source(
            "fsd50k-eval",
            "Fhrozen/FSD50k",
            "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b",
            "files",
            (),
            "FSD50K, clips with no human voice",
            "eval",
            "CC0, CC BY or CC BY-NC per clip",
            nc=True,
            key="fname",
            index="labels/eval.csv",
            skip_labels=VOICE_LABELS,
            path_format="clips/eval/{}.wav",
            bytes_per_s=88200,
        ),
        Source(
            "massive-pt-test",
            "TigreGotico/speech_MASSIVE_pt-PT",
            "288e7b5ffe102e303c71abd68ebcacba16f82cd9",
            "files",
            (),
            "Speech-MASSIVE pt-PT",
            "test",
            "CC BY-NC-SA 4.0",
            nc=True,
            text="text",
            key="file_name",
            index="metadata.csv",
            where=(("split", "test"),),
            bytes_per_s=32000,
        ),
        Source(
            "basque-parliament-eu",
            "gttsehu/basque_parliament_1",
            "46208b3001c500aa53ddb5f4a9da81493bbb8067",
            "tar",
            ("audio/test_0.tar",),
            "Basque Parliament, Basque segments",
            "test",
            "CC0",
            text="sentence",
            key="path",
            index="metadata/test.tsv",
            where=(("language", "eu"),),
            seconds="length",
        ),
        *(_fleurs(lang) for lang in ("de_de", "es_419", "it_it", "ca_es", "da_dk", "fr_fr")),
        *(_common_voice(lang) for lang in ("pt", "de", "es", "it", "nl", "ca", "da", "eu", "fr")),
    ]
}

# group -> category -> [(source, hours)]; "shared" is part of every language's negatives
NEGATIVES = {
    "shared": {
        "music": [("musan-music", 2.0)],
        "noise": [
            ("musan-noise", 0.8),
            ("demand-16k", 0.6),
            ("fsd50k-eval", 0.6),
        ],
    },
    "en-US": {
        "read_speech": [("librispeech-test", 6.0)],
        "conversation": [("ami-sdm-test", 4.0)],
        "broadcast": [("peoples-speech-test", 2.0), ("voxpopuli-en-test", 2.0)],
        "foreign_speech": [("mls-test", 2.0)],
    },
    "pt-PT": {"read_speech": [("massive-pt-test", 2.0), ("cv-pt", 2.0)]},
    "de-DE": {"read_speech": [("fleurs-de_de", 2.0), ("cv-de", 2.0)]},
    "es-ES": {"read_speech": [("fleurs-es_419", 1.8), ("cv-es", 2.2)]},
    "it-IT": {"read_speech": [("fleurs-it_it", 2.0), ("cv-it", 2.0)]},
    "nl-NL": {"read_speech": [("cv-nl", 4.0)]},
    "ca-ES": {"read_speech": [("fleurs-ca_es", 1.8), ("cv-ca", 2.2)]},
    "da-DK": {"read_speech": [("fleurs-da_dk", 2.2), ("cv-da", 1.8)]},
    "fr-FR": {"read_speech": [("fleurs-fr_fr", 1.2), ("cv-fr", 2.8)]},
    "eu-ES": {"read_speech": [("cv-eu", 3.0)], "broadcast": [("basque-parliament-eu", 1.5)]},
}
# a language with no open speech set of its own takes a near language's speech,
# counted as speech in another language
BORROWED = {"an-ES": "es-ES"}
LANGUAGES = {
    "en-US": "English",
    "pt-PT": "Portuguese",
    "de-DE": "German",
    "es-ES": "Spanish",
    "it-IT": "Italian",
    "nl-NL": "Dutch",
    "ca-ES": "Catalan",
    "da-DK": "Danish",
    "fr-FR": "French",
    "eu-ES": "Basque",
    "an-ES": "Aragonese",
}
CATEGORY_TEXT = {
    "read_speech": "read speech",
    "conversation": "meeting conversation",
    "broadcast": "radio, parliament and talk recordings",
    "music": "music",
    "noise": "household and outdoor sounds",
    "foreign_speech": "speech in other languages",
}


def group_of(lang: str) -> str | None:
    """The negatives group of a test set language, or None when there is none."""
    if lang.startswith("en-"):
        return "en-US"
    return lang if lang in NEGATIVES or lang in BORROWED else None


def normalise(text: str) -> str:
    """Lower case, accents removed, every run of other characters one space."""
    text = unicodedata.normalize("NFKD", text.lower())
    return " ".join(re.findall(r"\w+", "".join(c for c in text if not unicodedata.combining(c))))


def wake_phrases() -> list[str]:
    """Every phrase a wake-word test set or fighter answers to, normalised."""
    phrases = {d.wakeword for d in list_datasets("wake_word") if d.wakeword}
    for competitor in list_competitors("wake_word"):
        for key, hotword in competitor.config.get("hotwords", {}).items():
            phrases.add(key)
            phrases.update(s for s in hotword.get("samples") or [] if isinstance(s, str))
    return sorted({p for p in (normalise(p.replace("_", " ")) for p in phrases) if p})


def phrase_pattern(phrases: list[str]) -> re.Pattern[str]:
    """Match any of *phrases* as whole words of a normalised text."""
    return re.compile(r"\b(?:" + "|".join(map(re.escape, phrases)) + r")\b")


@dataclass
class Candidate:
    key: str
    text: str
    seconds: float | None
    data: bytes | None = None
    # reads the audio bytes of a row whose bytes are not loaded into ``data``
    fetch: Callable[[], bytes] | None = None


def _download(source: Source, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source.hf_id, name, repo_type="dataset", revision=source.revision))


def _index_rows(source: Source) -> list[dict]:
    text = _download(source, source.index).read_text(encoding="utf-8")
    tsv = source.index.endswith(".tsv")
    reader = csv.DictReader(
        io.StringIO(text),
        fieldnames=source.index_columns or None,
        delimiter="\t" if tsv else ",",
        quoting=csv.QUOTE_NONE if tsv else csv.QUOTE_MINIMAL,
    )
    return [
        r
        for r in reader
        if all(r.get(k) == v for k, v in source.where)
        and not source.skip_labels & set(r.get("labels", "").split(","))
    ]


def candidates(source: Source) -> tuple[list[Candidate], Callable[[list[Candidate]], None]]:
    """The rows of *source* and a function that loads the audio bytes of chosen rows."""
    if source.reader == "parquet":
        import pyarrow.parquet as pq

        def cell(audio, i: int) -> bytes:
            value = audio[i].as_py()
            return value["bytes"] if isinstance(value, dict) else value

        rows = []
        columns = [source.key, source.audio] + ([source.text] if source.text else [])
        for name in source.files:
            # the audio stays in Arrow memory; a row's bytes are copied out only when written
            table = pq.read_table(_download(source, name), columns=columns)
            audio = table.column(source.audio)
            texts = table.column(source.text).to_pylist() if source.text else [""] * len(table)
            for i, (key, text) in enumerate(
                zip(table.column(source.key).to_pylist(), texts, strict=True)
            ):
                rows.append(Candidate(str(key), text or "", None, fetch=partial(cell, audio, i)))
        return rows, lambda chosen: None
    if source.reader == "tar":
        archive = _download(source, source.files[0])
        sizes = {}
        if source.bytes_per_s:
            with tarfile.open(archive) as tf:
                sizes = {Path(m.name).name: m.size for m in tf if m.isfile()}
        rows = [
            Candidate(
                Path(r[source.key]).name,
                r.get(source.text) or "",
                float(r[source.seconds]) * source.seconds_scale
                if source.seconds
                else sizes.get(Path(r[source.key]).name, 0) / source.bytes_per_s,
            )
            for r in _index_rows(source)
        ]

        def load(chosen: list[Candidate]) -> None:
            wanted = {c.key: c for c in chosen}
            with tarfile.open(archive) as tf:
                for member in tf:
                    c = wanted.get(Path(member.name).name)
                    if c is not None and member.isfile():
                        c.data = tf.extractfile(member).read()

        return rows, load
    from huggingface_hub import HfApi, RepoFile

    tree = {
        f.path: f.size
        for f in HfApi().list_repo_tree(
            source.hf_id, repo_type="dataset", revision=source.revision, recursive=True
        )
        if isinstance(f, RepoFile) and f.path.endswith(".wav")
    }
    if source.index:
        paths = [
            (source.path_format.format(r[source.key]), r.get(source.text) or "")
            for r in _index_rows(source)
        ]
    else:
        paths = [(p, "") for p in sorted(tree) if p.startswith(source.files)]
    rows = [Candidate(p, text, tree[p] / source.bytes_per_s) for p, text in paths if p in tree]

    def load(chosen: list[Candidate]) -> None:
        with ThreadPoolExecutor(8) as pool:
            paths = pool.map(lambda c: _download(source, c.key), chosen)
            for c, path in zip(chosen, paths, strict=True):
                c.data = path.read_bytes()

    return rows, load


def decode(data: bytes) -> np.ndarray:
    """Encoded audio bytes as mono float32 at the corpus rate."""
    try:
        array, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        array, sr = decode_audio_bytes(data)
    return resample(np.asarray(array, dtype="float32"), sr, SAMPLE_RATE)


def export(
    source: Source, budgets: list[tuple[Path, float]], seed: int, drop: re.Pattern[str]
) -> dict:
    """Write rows of *source* in a seeded order as 16 kHz wav files, filling each
    ``(folder, seconds)`` of *budgets* in turn.

    A row whose transcript holds a phrase is left out. Raises when the source
    runs out before the last budget is full. Returns what was written and left out.
    """
    rows, load = candidates(source)
    order = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
    dropped = [c.key for c in order if c.text and drop.search(normalise(c.text))]
    kept = [c for c in order if not (c.text and drop.search(normalise(c.text)))]
    loaded: list[Candidate] = []

    def load_more(need: float) -> None:
        # load only the rows the budgets can use, with room for estimates that run short
        nonlocal kept
        if kept and kept[0].seconds is not None:
            reach = np.cumsum([c.seconds for c in kept])
            cut = int(np.searchsorted(reach, need * 1.25 + 60)) + 1
        else:
            cut = len(kept)
        batch, kept = kept[:cut], kept[cut:]
        load(batch)
        loaded.extend(batch)

    written = []
    left = sum(s for _, s in budgets)
    for folder, seconds in budgets:
        folder.mkdir(parents=True, exist_ok=True)
        got, count = 0.0, 0
        while got < seconds and (loaded or kept):
            if not loaded:
                load_more(left)
            c = loaded.pop(0)
            data = c.data if c.data is not None else c.fetch() if c.fetch else None
            if data is None:
                raise RuntimeError(f"{source.name}: no audio for {c.key}")
            audio = decode(data)
            c.data = None
            if not len(audio):
                continue
            sf.write(folder / f"{count:06d}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
            got += len(audio) / SAMPLE_RATE
            left -= len(audio) / SAMPLE_RATE
            count += 1
        if got < seconds:
            raise RuntimeError(f"{source.name}: {got / 3600:.2f} h of {seconds / 3600:.2f} h")
        written.append({"folder": folder.name, "clips": count, "hours": got / 3600})
    log.info("%s: %s, %d rows with a phrase left out", source.name, written, len(dropped))
    return {"written": written, "dropped": dropped}


def export_files(entries: list[tuple[str, str, str]], out: Path) -> list[str]:
    """Download *entries* (``hf_id, path, revision``) as 16 kHz wav files.

    Writes ``metadata.csv`` beside them.
    """
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    names = [(f"{k:05d}.wav", rel) for k, (_, rel, _) in enumerate(entries)]
    missing = [
        (out / name, e)
        for (name, _), e in zip(names, entries, strict=True)
        if not (out / name).exists()
    ]

    def fetch(job: tuple[Path, tuple[str, str, str]]) -> tuple[Path, str]:
        target, (hf_id, rel, revision) = job
        return target, hf_hub_download(hf_id, rel, repo_type="dataset", revision=revision)

    with ThreadPoolExecutor(8) as pool:
        for target, local in pool.map(fetch, missing):
            array, sr = decode_audio_bytes(Path(local).read_bytes())
            sf.write(target, np.asarray(array, dtype="float32"), sr, subtype="PCM_16")
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name", "source"])
        writer.writerows(names)
    return [n for n, _ in names]


def clip_sets() -> list:
    """Every wake-word test set that has clips with a phrase (not the negative-only pools)."""
    return [d for d in list_datasets("wake_word") if d.wakeword and d.role == "eval"]


def prefixed(rows: list[dict], prefix: str, **extra) -> list[dict]:
    return [{**row, **extra, "audio": f"{prefix}/{row['audio']}"} for row in rows]


def build_negatives(src: Path, repo: Path, seed: int, drop: re.Pattern[str], facts: dict) -> dict:
    """Write the negative recordings; returns ``{group: manifest rows}`` with paths in the repo."""
    built: dict[str, dict[str, list[dict]]] = {}
    filler: dict[str, list[str]] = {}
    for group, categories in NEGATIVES.items():
        built[group] = {}
        for category, parts in categories.items():
            for name, hours in parts:
                budgets = [(src / name / "negatives", hours * 3600)]
                if group != "en-US" and category == "read_speech":
                    budgets.append((src / name / "filler", FILLER_HOURS * 3600 / len(parts)))
                    filler.setdefault(group, []).append(f"{name}/filler")
                facts["sources"][name] = export(SOURCES[name], budgets, seed, drop)
            prefix = f"negatives/{group}/{category}"
            rows = build_corpus(
                None,
                [src / name / "negatives" for name, _ in parts],
                repo / prefix,
                sum(h for _, h in parts),
                seed,
                file_seconds=RECORDING_S,
                suffix=SUFFIX,
            )
            built[group][category] = prefixed(rows, prefix, category=category)
    shared = built.pop("shared")
    english = [
        {**r, "category": "foreign_speech"}
        for r in built["en-US"]["read_speech"][:SHARED_RECORDINGS]
    ]
    manifests = {
        "en-US": [r for rows in [*built["en-US"].values(), *shared.values()] for r in rows]
    }
    shared_part = [r for rows in shared.values() for r in rows[:SHARED_RECORDINGS]]
    for group in [g for g in built if g != "en-US"]:
        manifests[group] = [r for rows in built[group].values() for r in rows]
    for group, near in BORROWED.items():
        manifests[group] = [{**r, "category": "foreign_speech"} for r in manifests[near]]
        filler[group] = filler[near]
    for group in manifests:
        if group != "en-US":
            manifests[group] += shared_part + english
    facts["filler"] = filler
    facts["negatives"] = {
        group: {
            category: sum(r["duration_s"] for r in rows if r["category"] == category) / 3600
            for category in CATEGORY_TEXT
            if any(r["category"] == category for r in rows)
        }
        for group, rows in manifests.items()
    }
    return manifests


def build(work: Path, seed: int, only: list[str]) -> dict:
    """Write the repo tree under ``work/repo``; returns the facts the card and the registry need."""
    src = work / "sources"
    repo = work / "repo"
    facts_path = work / "facts.json"
    if only and facts_path.is_file():
        # add the named test sets to an earlier build; its negatives stay as they are
        facts = json.loads(facts_path.read_text())
        manifests = {
            group: [
                json.loads(line)
                for line in (repo / f"manifest-negatives-{group}.jsonl").read_text().splitlines()
            ]
            for group in facts["negatives"]
        }
        build_sets(src, repo, seed, only, manifests, facts)
        facts_path.write_text(json.dumps(facts, indent=1, sort_keys=True))
        return facts
    phrases = wake_phrases()
    drop = phrase_pattern(phrases)
    facts = {"seed": seed, "phrases": phrases, "sources": {}, "sets": {}}
    manifests = build_negatives(src, repo, seed, drop, facts)
    facts["sources"]["librispeech-dev"] = export(
        SOURCES["librispeech-dev"], [(src / "librispeech-dev" / "filler", 8.0 * 3600)], seed, drop
    )
    facts["filler"]["en-US"] = ["librispeech-dev/filler"]
    for group, rows in manifests.items():
        (repo / f"manifest-negatives-{group}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        )
    (repo / "dropped-utterances.txt").write_text(
        "".join(
            f"{name}\t{key}\n"
            for name, s in sorted(facts["sources"].items())
            for key in sorted(s["dropped"])
        )
    )
    build_sets(src, repo, seed, only, manifests, facts)
    facts_path.write_text(json.dumps(facts, indent=1, sort_keys=True))
    return facts


def build_sets(
    src: Path, repo: Path, seed: int, only: list[str], manifests: dict, facts: dict
) -> None:
    """Write one recording and one manifest per clip test set; adds each set to *facts*."""
    for dataset in clip_sets():
        if only and dataset.dataset_id not in only:
            continue
        group = group_of(dataset.lang)
        if group is None:
            log.warning("%s: no negatives for %s, skipped", dataset.dataset_id, dataset.lang)
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
            [src / p for p in facts["filler"][group]],
            repo / dataset.dataset_id,
            usable * FILLER_S_PER_POSITIVE / 3600,
            seed,
            file_seconds=usable * FILLER_S_PER_POSITIVE,
            positives_per_file=usable,
            min_gap_s=MIN_GAP_S,
            suffix=SUFFIX,
        )
        sources = [name for r in rows for name in r["sources"]]
        if len(sources) != usable or len(set(sources)) != usable:
            raise RuntimeError(
                f"{dataset.dataset_id}: {usable} usable clips in, {len(sources)} onsets "
                f"from {len(set(sources))} clips out"
            )
        manifest = prefixed(rows, dataset.dataset_id) + manifests[group]
        (repo / f"manifest-{dataset.dataset_id}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest)
        )
        facts["sets"][dataset.dataset_id] = {
            "source": dataset.source.hf_id,
            "revision": revision,
            "license": dataset.license,
            "wakeword": dataset.wakeword,
            "lang": dataset.lang,
            "group": group,
            "available": len(positives),
            "silent": silent,
            "onsets": usable,
            "hours": sum(r["duration_s"] for r in rows) / 3600,
        }
        log.info("%s: %d of %d clips with the phrase", dataset.dataset_id, usable, len(positives))


def card(facts: dict) -> str:
    rows = "\n".join(
        f"| `{sid}` | `{s['wakeword']}` | {s['lang']} "
        f"| [{s['source']}](https://huggingface.co/datasets/{s['source']}) `{s['revision'][:8]}` "
        f"| {s['license']} | {s['onsets']} of {s['available']} | {s['hours']:.2f} |"
        for sid, s in sorted(facts["sets"].items())
    )
    groups = sorted(facts["negatives"])
    configs = "\n".join(
        [
            f"- config_name: negatives-{g}\n  data_files: manifest-negatives-{g}.jsonl"
            for g in groups
        ]
        + [
            f"- config_name: {sid}\n  data_files: manifest-{sid}.jsonl"
            for sid in sorted(facts["sets"])
        ]
    )
    used = {
        name: category
        for g in NEGATIVES.values()
        for category, parts in g.items()
        for name, _ in parts
    }
    sources = "\n".join(
        f"| {s.title} | [{s.hf_id}]({s.url}) `{s.revision[:8]}` | {s.split} "
        f"| {s.license}{' (non-commercial)' if s.nc else ''} "
        f"| {CATEGORY_TEXT[used[name]] if name in used else 'speech between the wake words'} "
        f"| {sum(w['hours'] for w in facts['sources'][name]['written']):.2f} "
        f"| {len(facts['sources'][name]['dropped'])} |"
        for name, s in SOURCES.items()
        if name in facts["sources"]
    )
    by_language = "\n".join(
        f"| {LANGUAGES[g]} (`{g}`) | "
        + " | ".join(f"{facts['negatives'][g].get(c, 0):.1f}" for c in CATEGORY_TEXT)
        + f" | {sum(facts['negatives'][g].values()):.1f} |"
        for g in groups
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
    dropped = sum(len(s["dropped"]) for s in facts["sources"].values())
    return f"""---
license: other
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
its `ww_stream` boards from this repo. It also ranks every detector of a
language on how often per hour it wakes on the negatives of that language.

## Layout

| Path | Contents |
|---|---|
| `negatives/<language>/<kind>/audio/` | Recordings with no wake word, 30 minutes each |
| `negatives/shared/<kind>/audio/` | Music and sounds, part of the negatives of every language |
| `<set>/audio/` | One recording per test set: its clips with the phrase, with speech between them |
| `manifest-negatives-<language>.jsonl` | Every recording with no wake word of a language |
| `manifest-<set>.jsonl` | The test set's recording, then the negatives of its language |
| `dropped-utterances.txt` | The source rows left out because their transcript holds a phrase |

A manifest row has `audio`, `duration_s`, `onsets` and `ends` (seconds),
`sources` (the clip behind each onset), `snr_db` and `level_dbfs`. A
negative recording has empty `onsets` and a `category`: the kind of audio it holds.

## The audio with no wake word

Hours of each kind in the negatives of each language:

| Language | {" | ".join(CATEGORY_TEXT.values())} | Total |
|---|{"---|" * len(CATEGORY_TEXT)}---|
{by_language}

English has its own conversation, broadcast and other-language parts. Every
other language gets the first hour of the shared music and sounds, and the
first hour of the English read speech as speech in another language. No open
Aragonese speech set was found, so the Aragonese negatives hold the Spanish
speech, counted as speech in another language.

| Source | Repo and revision | Split | Licence | Used as | Hours | Rows with a phrase left out |
|---|---|---|---|---|---|---|
{sources}

Each recording keeps the licence of its sources. A source marked
non-commercial holds clips that must not be used commercially. The other
sources are open for any use under their licence.

These sources stay out:

- **Sound-alike and near-miss sets** (`not-wake-words-soundalikes-*`,
  `near-miss-wakewords`, `OpenVoiceOS/wake_word_noise`). They are made to be
  hard, so a rate on them does not say how often a detector wakes in a home.
- **Sources a fighter trained on**: `TigreGotico/not-wake-words-speech-en`,
  AudioSet, LibriSpeech train, and the train splits of Common Voice and FLEURS.
  Only the test splits of Common Voice and FLEURS are used. `TigreGotico/ESC-50`,
  `TigreGotico/public_domain_sounds_3secs` and `TigreGotico/FMA_3secs` are in the
  default negative and background-noise pools of the wakeforge trainer, so they stay
  out too.
- **Sources with no known licence**: `TigreGotico/ambient_noises` and
  `TigreGotico/building_106_kitchen_3secs`.
- **`TigreGotico/NAR`**: it is for research use only, and it holds the word "stop".

## How it is made

- **No phrase in the speech**: a row whose transcript holds any of the
  {len(facts["phrases"])} phrases of the arena's wake-word test sets and fighters, as whole
  words, with case and accents ignored, is left out. That removes {dropped} rows, named in
  `dropped-utterances.txt`. Music and sounds have no transcript and are not filtered.
  FSD50K clips with a human-voice label are left out.
- **Speech between the wake words**: LibriSpeech `dev-clean` and `dev-other` for
  English. For another language, {FILLER_HOURS:.0f} hour of its own speech that its
  negatives do not use.
- **Wake words**: up to {MAX_POSITIVES} clips with the phrase from each test
  set, picked evenly across the set. Silence before and after each clip is cut,
  so its onset is where the phrase starts. Each clip is set 5 to 15 dB above the
  level of the speech in its recording. Consecutive clips are at least
  {MIN_GAP_S:.0f} seconds of speech apart, and {FILLER_S_PER_POSITIVE:.0f} seconds on average.
- The build is a pure function of the sources, the phrases above and seed `{facts["seed"]}`:
  `python3 scripts/publish_ww_stream_bench.py --work <folder> --seed {facts["seed"]}` in
  the arena.

| Test set | Phrase | Language | Source | Licence | Clips | Hours |
|---|---|---|---|---|---|---|
{rows}{silent}

{FUNDING_BLOCK}
"""


def _negatives_text(hours: dict) -> str:
    kinds = [CATEGORY_TEXT[c] for c in CATEGORY_TEXT if c in hours]
    listed = f"{', '.join(kinds[:-1])} and {kinds[-1]}" if len(kinds) > 1 else kinds[0]
    return f"{sum(hours.values()):g} hours of {listed}"


def _hours_note(hours: dict) -> str:
    parts = [f"{hours[c]:g} h of {CATEGORY_TEXT[c]}" for c in CATEGORY_TEXT if c in hours]
    return f"The negatives hold {', '.join(parts)}: {sum(hours.values()):g} hours in all."


def _stream_entry(dataset_id: str, file_pattern: str, revision: str, hours: dict) -> dict:
    return {
        "modality": "ww_stream",
        "source": {
            "type": "huggingface",
            "hf_id": REPO,
            "revision": revision,
            "file_pattern": file_pattern,
        },
        "reference_fields": {"audio": "audio", "onsets": "onsets", "ends": "ends"},
        "license": "other",
        "role": "eval",
        "sample_rate_hz": 16000,
        "event_manifest_fields": {"audio": "audio", "onsets": "onsets", "duration_s": "duration_s"},
        "min_negative_hours": float(math.floor(sum(hours.values()))),
        "predictions_hf": results_repo_for("ww_stream", dataset_id),
        "event_tolerance_s": 1.5,
    }


def negatives_entry(group: str, revision: str, hours: dict) -> dict:
    language = LANGUAGES[group]
    dataset_id = f"ww-negatives-{group}"
    return {
        "dataset_id": dataset_id,
        "display_name": f"False wake-ups — {language}",
        "summary": (
            f"{_negatives_text(hours)}, with no wake word, for every {language} wake-word "
            "detector. A detector streams every recording, and the board ranks it on how often "
            "per hour it woke, with a column for each kind of audio."
        ),
        **_stream_entry(dataset_id, f"manifest-negatives-{group}.jsonl", revision, hours),
        "lang": group,
        "primary_metric": "fa_per_hour",
        "notes": (
            f"Built by scripts/publish_ww_stream_bench.py: the `negatives-{group}` part of "
            f"{REPO}. {_hours_note(hours)} Its sources and their licences are in the dataset "
            "card."
        ),
    }


def featured_groups() -> set[str]:
    """The negatives groups of the languages with a featured phrase."""
    featured = registry_featured_phrases(
        list_datasets("wake_word"), list_competitors("wake_word"), board_positives(BOARDS))
    return {
        group for d in clip_sets()
        if wakeword_key(d.wakeword) in featured and (group := group_of(d.lang))
    }


def write_registry(facts: dict, revision: str) -> None:
    """Write the ``ww-negatives-<lang>`` registry entries of a build published at
    *revision*, for the languages with a featured phrase. The recordings of the
    test sets stay in the corpus but get no registry entry: the arena runs only
    the false wake-up boards."""
    out = ROOT / "registry" / "datasets" / "ww_stream"
    out.mkdir(parents=True, exist_ok=True)
    groups = featured_groups()
    entries = [
        negatives_entry(group, revision, hours)
        for group, hours in sorted(facts["negatives"].items())
        if group in groups
    ]
    for entry in entries:
        (out / f"{entry['dataset_id']}.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n"
        )
    log.info("wrote %d registry entries at %s", len(entries), revision)


def upload(api, folder: Path) -> str:
    """Make *REPO* hold exactly the files under *folder*; returns the new commit."""
    from huggingface_hub import CommitOperationDelete

    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(folder))
    local = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    stale = [
        f
        for f in api.list_repo_files(REPO, repo_type="dataset")
        if f not in local and f != ".gitattributes"
    ]
    if stale:
        api.create_commit(
            REPO,
            repo_type="dataset",
            operations=[CommitOperationDelete(f) for f in stale],
            commit_message=f"Remove {len(stale)} files of the earlier build",
        )
    log.info("%d files uploaded, %d stale files removed", len(local), len(stale))
    return api.dataset_info(REPO).sha


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

    revision = upload(HfApi(), args.work / "repo")
    log.info("uploaded %s at %s", REPO, revision)
    write_registry(facts, revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
