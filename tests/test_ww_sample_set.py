"""Wake-word clip boards score every fighter on one published sample set.

A wake-word sweep draws a per-class cap of clips from the dataset's own
positives, its negatives corpora and its pooled registry negatives. The
sample-set manifest must hold exactly the ids that draw yields, worked out
from the file listings and parquet rows alone, so that board assembly can
drop the rows of an older sweep that drew a different set.

The fake hub below answers only at the Hugging Face boundary
(``HfApi.list_repo_files`` and ``hf_hub_download``); the listings, the csv
manifest, the parquet shards and the wav bytes are real files read by the
real selection, decode and row-walk code.
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from arena.metrics import build_benchmark_board
from arena.models import PredictionRow
from registry.loaders import list_datasets, load_dataset
from registry.schemas import SamplePolicy
from runner.publish_sample_set import compute_sample_set
from runner.ww_bench import WakeWordBench

MLSW = "MLCommons/ml_spoken_words"
# rows 3, 11 and 30 carry no audio: both the sweep and the manifest skip them
MLSW_NULL_ROWS = {3, 11, 30}
MLSW_SHARD_ROWS = (25, 15)

LISTINGS = {
    "Picovoice/wake-word-benchmark": (
        [f"data/jarvis/{i:02d}.wav" for i in range(23)]
        + [f"data/alexa/{i:02d}.wav" for i in range(5)]
        + ["README.md", "data/LICENSE"]),
    "OpenVoiceOS/ovos-community-wakewords-dataset": ["metadata.csv"],
    "quo-labs/sam-wake-word": (
        [f"positive/p{i:02d}.wav" for i in range(19)]
        + [f"negative/n{i:02d}.wav" for i in range(31)]
        + ["other/x.wav", "README.md"]),
    "TigreGotico/not-wake-words-speech-en": (
        [f"not_wake_word_subset/a{i}.wav" for i in range(10)] + ["other/b.wav"]),
    "TigreGotico/public_domain_sounds_3secs": [f"pd_3sec/s{i}.wav" for i in range(5)],
    "TigreGotico/ESC-50": [f"1-{i}-A-0.wav" for i in range(12)] + ["meta.csv"],
    "TigreGotico/FMA_3secs": [f"rock/f{i}.mp3" for i in range(8)],
    "TigreGotico/ambient_noises": [f"amb/a{i}.flac" for i in range(6)],
    MLSW: ["en_wav/partial-test/0000.parquet", "en_wav/partial-test/0001.parquet"],
}
COMMUNITY_ROWS = ([("hey_k9", f"hey_k9/c{i:02d}.wav") for i in range(15)]
                  + [("hey_floyd", f"hey_floyd/c{i:02d}.wav") for i in range(10)])


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve LISTINGS, a real csv, real parquet shards and a real wav."""
    import huggingface_hub

    wav = tmp_path / "clip.wav"
    wav.write_bytes(_wav_bytes())
    meta = tmp_path / "metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name", "label"])
        for label, rel in COMMUNITY_ROWS:
            writer.writerow([rel, label])
    shards = {}
    position = 0
    for k, n in enumerate(MLSW_SHARD_ROWS):
        audio = []
        for _ in range(n):
            audio.append(None if position in MLSW_NULL_ROWS
                         else {"bytes": wav.read_bytes(), "path": f"en_{position:03d}.wav"})
            position += 1
        path = tmp_path / f"{k:04d}.parquet"
        pq.write_table(pa.table({"audio": audio}), path)
        shards[f"en_wav/partial-test/{k:04d}.parquet"] = str(path)

    downloads = []

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            return list(LISTINGS[repo_id])

    def fake_download(repo_id, filename, repo_type=None, revision=None, **kwargs):
        downloads.append((repo_id, filename))
        if filename in shards:
            return shards[filename]
        if filename == "metadata.csv":
            return str(meta)
        assert filename in LISTINGS[repo_id] or repo_id in (
            "OpenVoiceOS/ovos-community-wakewords-dataset",), (repo_id, filename)
        return str(wav)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return downloads


def _capped(dataset_id: str, cap: int = 7):
    return load_dataset("wake_word", dataset_id).model_copy(
        update={"sample_policy": SamplePolicy(max_samples=cap, seed=1337)})


def _sweep(dataset_def) -> list[tuple[str, str]]:
    return [(sid, sample["label"]) for sid, sample in
            WakeWordBench().iter_samples(dataset_def, dataset_def.lang, "rev", 0)]


# The ids and labels the sweep yielded at a per-class cap of 7, recorded
# from the sampler before the manifest shared its selection.
SWEEP_AT_7 = {
    "picovoice-jarvis": [
        ("data/jarvis/00.wav", "positive"),
        ("1-0-A-0.wav", "negative"),
        ("data/jarvis/03.wav", "positive"),
        ("mlsw-negatives-en-US/en_004.wav", "negative"),
        ("data/jarvis/06.wav", "positive"),
        ("1-4-A-0.wav", "negative"),
        ("data/jarvis/09.wav", "positive"),
        ("mlsw-negatives-en-US/en_008.wav", "negative"),
        ("data/jarvis/13.wav", "positive"),
        ("rock/f0.mp3", "negative"),
        ("data/jarvis/16.wav", "positive"),
        ("mlsw-negatives-en-US/en_014.wav", "negative"),
        ("data/jarvis/19.wav", "positive"),
        ("amb/a0.flac", "negative"),
        ("mlsw-negatives-en-US/en_018.wav", "negative"),
        ("amb/a3.flac", "negative"),
        ("mlsw-negatives-en-US/en_028.wav", "negative"),
        ("not_wake_word_subset/a5.wav", "negative"),
        ("mlsw-negatives-en-US/en_031.wav", "negative"),
        ("pd_3sec/s0.wav", "negative"),
        ("mlsw-negatives-en-US/en_032.wav", "negative"),
    ],
    "community-hey-k9": [
        ("hey_k9/c00.wav", "positive"),
        ("1-0-A-0.wav", "negative"),
        ("hey_k9/c02.wav", "positive"),
        ("mlsw-negatives-en-US/en_004.wav", "negative"),
        ("hey_k9/c04.wav", "positive"),
        ("1-4-A-0.wav", "negative"),
        ("hey_k9/c06.wav", "positive"),
        ("mlsw-negatives-en-US/en_008.wav", "negative"),
        ("hey_k9/c08.wav", "positive"),
        ("rock/f0.mp3", "negative"),
        ("hey_k9/c10.wav", "positive"),
        ("mlsw-negatives-en-US/en_014.wav", "negative"),
        ("hey_k9/c12.wav", "positive"),
        ("amb/a0.flac", "negative"),
        ("mlsw-negatives-en-US/en_018.wav", "negative"),
        ("amb/a3.flac", "negative"),
        ("mlsw-negatives-en-US/en_028.wav", "negative"),
        ("not_wake_word_subset/a5.wav", "negative"),
        ("mlsw-negatives-en-US/en_031.wav", "negative"),
        ("pd_3sec/s0.wav", "negative"),
        ("mlsw-negatives-en-US/en_032.wav", "negative"),
    ],
    "sam-wake-word": [
        ("positive/p00.wav", "positive"),
        ("negative/n00.wav", "negative"),
        ("positive/p02.wav", "positive"),
        ("mlsw-negatives-en-US/en_004.wav", "negative"),
        ("positive/p05.wav", "positive"),
        ("negative/n04.wav", "negative"),
        ("positive/p08.wav", "positive"),
        ("mlsw-negatives-en-US/en_008.wav", "negative"),
        ("positive/p10.wav", "positive"),
        ("negative/n08.wav", "negative"),
        ("positive/p13.wav", "positive"),
        ("mlsw-negatives-en-US/en_014.wav", "negative"),
        ("positive/p16.wav", "positive"),
        ("negative/n13.wav", "negative"),
        ("mlsw-negatives-en-US/en_018.wav", "negative"),
        ("negative/n17.wav", "negative"),
        ("mlsw-negatives-en-US/en_028.wav", "negative"),
        ("negative/n22.wav", "negative"),
        ("mlsw-negatives-en-US/en_031.wav", "negative"),
        ("negative/n26.wav", "negative"),
        ("mlsw-negatives-en-US/en_032.wav", "negative"),
    ],
}


class TestManifestIsTheSweepDraw:
    @pytest.mark.parametrize("dataset_id", [
        "picovoice-jarvis",        # audiofolder positives, negatives_sources
        "community-hey-k9",        # metadata.csv positives
        "sam-wake-word",           # positives_dir / negative_dirs, same-corpus negatives
    ])
    def test_manifest_ids_equal_the_ids_the_sweep_yields(self, fake_hub, dataset_id):
        dataset_def = _capped(dataset_id)
        swept = [sid for sid, _ in _sweep(dataset_def)]
        manifest = compute_sample_set(dataset_def, "rev")
        assert manifest["sample_ids"] == swept
        assert len(set(swept)) == len(swept)
        assert manifest["max_samples"] == 7
        assert manifest["seed"] == 1337

    def test_manifest_downloads_no_clip(self, fake_hub):
        compute_sample_set(_capped("picovoice-jarvis"), "rev")
        clips = [f for _, f in fake_hub if not f.endswith((".parquet", ".csv"))]
        assert clips == []

    def test_each_class_and_the_pooled_negatives_hold_the_cap(self, fake_hub):
        """7 positives, 7 negatives from the negatives corpora and 7 pooled
        spoken-word negatives: the pooled channel is why a cap of 100 gives
        200 negatives per set."""
        ids = compute_sample_set(_capped("picovoice-jarvis"), "rev")["sample_ids"]
        pooled = [i for i in ids if i.startswith("mlsw-negatives-en-US/")]
        positives = [i for i in ids if i.startswith("data/jarvis/")]
        assert len(positives) == 7
        assert len(pooled) == 7
        assert len(ids) == 21
        # positions drawn by the seeded shuffle over all 40 rows; the null
        # rows are skipped, so fewer than 7 would mean a null was counted
        assert not {f"mlsw-negatives-en-US/en_{n:03d}.wav" for n in MLSW_NULL_ROWS} & set(ids)

    @pytest.mark.parametrize("dataset_id", sorted(SWEEP_AT_7))
    def test_sweep_draw_is_unchanged(self, fake_hub, dataset_id):
        assert _sweep(_capped(dataset_id)) == SWEEP_AT_7[dataset_id]


class TestBoardScoresOnlyTheManifest:
    def test_off_manifest_rows_are_dropped_before_scoring(self, fake_hub):
        dataset_def = _capped("picovoice-jarvis")
        manifest = compute_sample_set(dataset_def, "rev")
        ids = set(manifest["sample_ids"])
        labels = dict(_sweep(dataset_def))

        def row(sample_id, label, prediction):
            return PredictionRow(
                competitor_id="old", sample_id=sample_id, dataset_id="picovoice-jarvis",
                lang="en-US", plugin_id="p", modality="wake_word",
                label=label, prediction=prediction)

        # every on-manifest row right except two missed positives and one
        # false accept; the older sweep's extra clips are all wrong
        missed = {"data/jarvis/00.wav", "data/jarvis/03.wav"}
        false_accept = {"1-0-A-0.wav"}
        rows = [row(sid, labels[sid],
                    ("not_detected" if labels[sid] == "positive" else "detected")
                    if sid in missed | false_accept else
                    ("detected" if labels[sid] == "positive" else "not_detected"))
                for sid in manifest["sample_ids"]]
        rows += [row(f"data/jarvis/{i:02d}.wav", "positive", "not_detected")
                 for i in (1, 2, 4, 5)]
        rows += [row(f"1-{i}-A-0.wav", "negative", "detected") for i in (1, 2, 3)]
        assert not {r.sample_id for r in rows[21:]} & ids

        board = build_benchmark_board(
            "wake_word", "picovoice-jarvis", "en-US", {"old": rows},
            "2026-01-01T00:00:00Z", sample_set_ids=ids, min_samples=1)
        (entry,) = board.entries
        assert entry.samples == 21
        assert entry.sample_set_coverage == 1.0
        # 7 positives, 2 missed; 14 negatives, 1 accepted; 3 errors in 21
        assert entry.metrics["false_reject_rate"] == pytest.approx(2 / 7, abs=5e-5)
        assert entry.metrics["false_accept_rate"] == pytest.approx(1 / 14, abs=5e-5)
        assert entry.metrics["error_rate"] == pytest.approx(3 / 21, abs=5e-5)


CLIP_SETS = [d for d in list_datasets(modality="wake_word")
             if d.role == "eval" and d.wakeword]


def test_every_eval_clip_set_carries_the_per_class_policy():
    assert len(CLIP_SETS) == 34
    off = {d.dataset_id: d.sample_policy for d in CLIP_SETS
           if d.sample_policy != SamplePolicy(max_samples=100, seed=1337)}
    assert off == {}


def test_pooled_negatives_sources_keep_their_own_policy():
    pools = [d for d in list_datasets(modality="wake_word")
             if d.dataset_id.startswith("mlsw-negatives-")]
    assert len(pools) == 10
    assert {json.dumps(d.sample_policy.model_dump()) for d in pools} == {
        json.dumps({"max_samples": 5000, "seed": 1337})}
