"""The false-wake-up board: a ww_stream corpus with no wake word, ranked on false wake-ups
per hour."""

from __future__ import annotations

import pytest

from arena.metrics import build_benchmark_board, score_ww_stream
from arena.models import PredictionRow
from registry.schemas import DatasetDef
from runner import audio_io


def _negative(
    cid: str, sample_id: str, scores: list[float], duration_s: float, category: str | None = None
) -> PredictionRow:
    extras = {
        "events": [[60.0 * (i + 1), s] for i, s in enumerate(scores)],
        "truth_onsets": [],
        "duration_s": duration_s,
    }
    if category:
        extras["category"] = category
    return PredictionRow(
        competitor_id=cid,
        sample_id=sample_id,
        dataset_id="ww-negatives-en-US",
        lang="en-US",
        plugin_id="p",
        modality="ww_stream",
        prediction="WW_STREAM",
        extras=extras,
    )


def _hours(cid: str, hours: int, events_per_hour: list[int]) -> list[PredictionRow]:
    return [_negative(cid, f"neg{h}", [0.9] * events_per_hour[h], 3600.0) for h in range(hours)]


class TestNegativesBoard:
    def _board(self, by_competitor):
        return build_benchmark_board(
            "ww_stream",
            "ww-negatives-en-US",
            "en-US",
            by_competitor,
            generated_at="2026-01-01T00:00:00+00:00",
            primary_metric="fa_per_hour",
        )

    def test_ranks_every_fighter_on_false_wake_ups_per_hour(self):
        # 5 h each: 1 wake-up in 5 h is 0.2/h, 5 in 5 h is 1.0/h, 10 in 5 h is 2.0/h
        board = self._board(
            {
                "quiet": _hours("quiet", 5, [1, 0, 0, 0, 0]),
                "one-per-hour": _hours("one-per-hour", 5, [1, 1, 1, 1, 1]),
                "noisy": _hours("noisy", 5, [2, 2, 2, 2, 2]),
            }
        )
        by_id = {e.competitor_id: e for e in board.entries}
        assert board.primary_metric == "fa_per_hour"
        assert [by_id[c].metrics["fa_per_hour"] for c in ("quiet", "one-per-hour", "noisy")] == [
            0.2,
            1.0,
            2.0,
        ]
        # 1.0 is a rate here, not the "caught no wake word" sentinel of error_at_1fa_per_hour
        assert [(by_id[c].rank, by_id[c].unranked) for c in ("quiet", "one-per-hour", "noisy")] == [
            (1, False),
            (2, False),
            (3, False),
        ]

    def test_a_fighter_with_too_few_hours_is_not_ranked(self):
        board = self._board(
            {"full": _hours("full", 5, [0] * 5), "short": _hours("short", 3, [0] * 3)}
        )
        by_id = {e.competitor_id: e for e in board.entries}
        assert by_id["full"].rank == 1
        assert by_id["short"].unranked

    def test_zero_wake_ups_keep_their_count_and_hours(self):
        board = self._board({"silent": _hours("silent", 5, [0] * 5)})
        metrics = board.entries[0].metrics
        assert (metrics["fa_per_hour"], metrics["n_false_accepts"], metrics["negative_hours"]) == (
            0.0,
            0.0,
            5.0,
        )


def test_false_wake_ups_per_source_category():
    rows = [
        _negative("c", "a", [0.9, 0.9, 0.2], 3600.0, "read_speech"),
        _negative("c", "b", [0.9], 1800.0, "music"),
        _negative("c", "d", [], 1800.0, "music"),
    ]
    metrics = score_ww_stream(rows)
    # read speech: 2 events at or above 0.5 in 1 h; music: 1 event in 1 h
    assert metrics["fa_per_hour@read_speech"] == 2.0
    assert metrics["fa_per_hour@music"] == 1.0
    assert metrics["negative_hours@music"] == 1.0
    assert metrics["n_false_accepts"] == 3.0
    assert metrics["fa_per_hour"] == 1.5


def test_primary_metric_is_only_fa_per_hour_on_a_stream_set():
    source = {"type": "huggingface", "hf_id": "org/x", "revision": "r"}
    DatasetDef(
        dataset_id="n",
        modality="ww_stream",
        lang="en-US",
        source=source,
        primary_metric="fa_per_hour",
    )
    with pytest.raises(ValueError, match="primary_metric"):
        DatasetDef(
            dataset_id="n",
            modality="wake_word",
            lang="en-US",
            source=source,
            primary_metric="fa_per_hour",
        )
    with pytest.raises(ValueError, match="primary_metric"):
        DatasetDef(
            dataset_id="n", modality="ww_stream", lang="en-US", source=source, primary_metric="frr"
        )


def test_excluded_folder_never_reaches_the_clip_lists(monkeypatch):
    keep = [f"train/edge-tts/{i}.wav" for i in range(3)] + [
        f"train/piper/{i}.wav" for i in range(2)
    ]
    dropped = [f"train/openvoiceos-synthetic-wakewords/{i}.wav" for i in range(4)]
    negatives = [f"test/edge-tts/{i}.wav" for i in range(2)]
    dropped_negatives = ["test/openvoiceos-synthetic-wakewords-extra/0.wav"]
    monkeypatch.setattr(
        audio_io, "_repo_audio", lambda hf, rev: keep + dropped + negatives + dropped_negatives
    )
    ds = DatasetDef(
        dataset_id="tg-alexa",
        modality="wake_word",
        wakeword="alexa",
        positives_dir="train",
        exclude_dirs=[
            "train/openvoiceos-synthetic-wakewords",
            "test/openvoiceos-synthetic-wakewords-extra/",
        ],
        lang="en-US",
        source={"type": "huggingface", "hf_id": "org/alexa", "revision": "r"},
    )
    pos, neg = audio_io._ww_split(ds, "r")
    assert pos == [("org/alexa", f, "r") for f in keep]
    assert neg == [("org/alexa", f, "r") for f in negatives]


def test_positives_filter_keeps_the_clips_whose_metadata_matches(monkeypatch):
    files = [
        f"test/omnivoice-single/wake_up_{lang}_{i:04d}.wav"
        for lang in ("ca", "gl")
        for i in range(3)
    ]
    metadata = [{"file_name": f.removeprefix("test/"), "lang": f.split("_")[2]} for f in files]
    monkeypatch.setattr(audio_io, "_repo_audio", lambda hf, rev: files)
    monkeypatch.setattr(
        audio_io,
        "_jsonl_rows",
        lambda hf, path, rev: metadata if path == "test/metadata.jsonl" else [],
    )
    ds = DatasetDef(
        dataset_id="tg-desperta-gl",
        modality="wake_word",
        wakeword="desperta",
        positives_dir="test",
        positives_filter={"lang": "gl"},
        lang="gl-ES",
        source={"type": "huggingface", "hf_id": "org/desperta", "revision": "r"},
    )
    assert audio_io.ww_positive_files(ds, "r") == [("org/desperta", f, "r") for f in files[3:]]



def test_a_list_in_positives_filter_allows_any_of_its_values(monkeypatch):
    voices = ("Amy", "Brian", "Cristiano", "Ines")
    files = [f"train/polly/Hey_Jarbas-{v}-{i}.wav" for v in voices for i in range(2)]
    metadata = [{"file_name": f.removeprefix("train/"), "voice": f.split("-")[1]} for f in files]
    monkeypatch.setattr(audio_io, "_repo_audio", lambda hf, rev: files)
    monkeypatch.setattr(
        audio_io,
        "_jsonl_rows",
        lambda hf, path, rev: metadata if path == "train/metadata.jsonl" else [],
    )
    ds = DatasetDef(
        dataset_id="tg-hey_jarbas",
        modality="wake_word",
        wakeword="hey_jarbas",
        positives_dir="train",
        positives_filter={"voice": ["Amy", "Brian"]},
        lang="en-US",
        source={"type": "huggingface", "hf_id": "org/hey_jarbas", "revision": "r"},
    )
    assert audio_io.ww_positive_files(ds, "r") == [("org/hey_jarbas", f, "r") for f in files[:4]]


class TestHoursFloor:
    def _entry(self, hours: float):
        rows = [_negative("c", "only", [], hours * 3600.0)]
        board = build_benchmark_board(
            "ww_stream",
            "ww-negatives-en-US",
            "en-US",
            {"c": rows},
            generated_at="2026-01-01T00:00:00+00:00",
            primary_metric="fa_per_hour",
        )
        return board.entries[0]

    def test_four_hours_rank(self):
        assert self._entry(4.0).rank == 1

    def test_just_under_four_hours_does_not_rank_and_says_why_in_hours(self):
        entry = self._entry(3.9)
        assert entry.unranked
        assert entry.unranked_reason.startswith("too_few_samples — 3.9 h of audio, below the 4 h")


def test_stream_adapter_copies_the_manifest_category(monkeypatch):
    import numpy as np

    from runner import ww_bench

    seen = {}

    def manifest(source, audio_key, extra_keys, revision, max_samples):
        seen.update(extra_keys)
        yield (
            "neg/0.flac",
            {
                "array": np.zeros(16000, dtype=np.float32),
                "sr": 16000,
                "audio_url": "u",
                "truth_onsets": [],
                "truth_ends": [],
                "category": "music",
            },
        )

    monkeypatch.setattr(ww_bench, "stream_manifest_audio", manifest)
    monkeypatch.setattr(ww_bench, "_cached_events", lambda stack, sample, ctx: [])
    bench = ww_bench.WakeWordStreamBench()
    ds = DatasetDef(
        dataset_id="n",
        modality="ww_stream",
        lang="en-US",
        source={"type": "huggingface", "hf_id": "org/x", "revision": "r"},
    )
    _, sample = next(bench.iter_samples(ds, "en-US", "r", 0))
    assert seen["category"] == "category"
    assert bench.predict(None, sample, None)["extras"]["category"] == "music"


def test_assemble_ranks_a_registry_negatives_set_on_fa_per_hour(tmp_path, monkeypatch):
    import json

    from arena import cli as arena_cli
    from registry.loaders import load_dataset

    dataset_id = "ww_stream_community-amelia"
    revision = load_dataset("ww_stream", dataset_id).source.revision
    preds = tmp_path / "predictions"
    preds.mkdir()
    for cid, per_hour in (("vosk-ww-amelia", 0), ("wakeforge-zeroshot-amelia", 2)):
        rows = [
            _negative(cid, f"neg{h}", [0.9] * per_hour, 3600.0).model_copy(
                update={"dataset_id": dataset_id, "dataset_revision": revision}
            )
            for h in range(5)
        ]
        (preds / f"{cid}.jsonl").write_text("\n".join(r.model_dump_json() for r in rows) + "\n")
    monkeypatch.setattr(arena_cli, "_load_sample_set", lambda *args: None)
    out = tmp_path / "data"
    board_file = out / f"benchmark-ww_stream-{dataset_id}-en-US.json"

    def assemble():
        try:
            arena_cli.main(["assemble", "--predictions", str(preds), "--output", str(out)])
        except SystemExit as exc:
            assert exc.code == 0
        return json.loads(board_file.read_text())

    monkeypatch.setattr(arena_cli, "_registry_primary_metrics", lambda: {dataset_id: "fa_per_hour"})
    board = assemble()
    assert board["primary_metric"] == "fa_per_hour"
    assert [(e["competitor_id"], e["rank"]) for e in board["entries"]] == [
        ("vosk-ww-amelia", 1),
        ("wakeforge-zeroshot-amelia", 2),
    ]
    # the same rows under the modality's own metric: the override is part of the cache key
    monkeypatch.setattr(arena_cli, "_registry_primary_metrics", lambda: {})
    assert assemble()["primary_metric"] == "error_at_1fa_per_hour"


def test_positives_filter_reads_the_metadata_below_the_subset(monkeypatch):
    files = [f"data/test/wake_up_{lang}_0.wav" for lang in ("ca", "gl")] + ["data/other/x.wav"]
    calls = []

    def rows(hf, path, rev):
        calls.append(path)
        return [
            {"file_name": "wake_up_gl_0.wav", "lang": "gl"},
            {"file_name": "wake_up_ca_0.wav", "lang": "ca"},
        ]

    monkeypatch.setattr(audio_io, "_repo_audio", lambda hf, rev: files)
    monkeypatch.setattr(audio_io, "_jsonl_rows", rows)
    ds = DatasetDef(
        dataset_id="x",
        modality="wake_word",
        wakeword="desperta",
        positives_dir="test",
        positives_filter={"lang": "gl"},
        lang="gl-ES",
        source={"type": "huggingface", "hf_id": "org/x", "revision": "r", "subset": "data"},
    )
    assert audio_io.ww_positive_files(ds, "r") == [("org/x", "data/test/wake_up_gl_0.wav", "r")]
    assert calls == ["data/test/metadata.jsonl"]


def test_positives_filter_is_refused_on_a_csv_manifest():
    with pytest.raises(ValueError, match="positives_filter"):
        DatasetDef(
            dataset_id="x",
            modality="wake_word",
            wakeword="w",
            positives_filter={"lang": "gl"},
            lang="gl-ES",
            source={
                "type": "huggingface",
                "hf_id": "org/x",
                "revision": "r",
                "file_pattern": "labels.csv",
            },
        )
