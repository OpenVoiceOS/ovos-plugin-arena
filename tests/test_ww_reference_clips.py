"""A few-shot fighter's `samples_from` becomes a folder of the dataset's positive clips."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import registry.loaders
import runner.intent_bench
from runner import ww_bench


def _stream(dataset, revision, max_per_class):
    for k in range(max_per_class):
        yield (
            f"pos/{k}",
            {"array": np.full(1600, k / 10, dtype=np.float32), "sr": 16000, "label": "positive"},
        )
        yield (
            f"neg/{k}",
            {"array": np.zeros(1600, dtype=np.float32), "sr": 16000, "label": "negative"},
        )


def _setup(monkeypatch, tmp_path, calls):
    def stream(dataset, revision, max_per_class):
        calls.append(revision)
        return _stream(dataset, revision, max_per_class)

    monkeypatch.setattr(ww_bench, "REFERENCE_CACHE", tmp_path)
    monkeypatch.setattr(ww_bench, "stream_ww", stream)
    monkeypatch.setattr(
        registry.loaders,
        "load_dataset",
        lambda m, did: SimpleNamespace(source=SimpleNamespace(hf_id="org/corpus", revision="main")),
    )
    monkeypatch.setattr(runner.intent_bench, "resolve_revision", lambda hf_id, rev: "abc123")


def test_the_folder_holds_only_positive_clips(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, calls := [])
    folder = ww_bench.reference_clips("community-fake", 3)
    clips = sorted(p.name for p in (tmp_path / "community-fake@abc123-3").glob("*.wav"))
    assert folder == str(tmp_path / "community-fake@abc123-3") and clips == [
        "ref_0.wav",
        "ref_1.wav",
        "ref_2.wav",
    ]
    levels = [
        round(float(sf.read(tmp_path / "community-fake@abc123-3" / c)[0].max()), 1) for c in clips
    ]
    assert levels == [0.0, 0.1, 0.2] and calls == ["abc123"]


def test_a_filled_folder_is_reused(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, calls := [])
    ww_bench.reference_clips("community-fake", 2)
    ww_bench.reference_clips("community-fake", 2)
    assert len(calls) == 1


def test_load_engine_hands_the_plugin_a_folder_not_samples_from(monkeypatch, tmp_path):
    seen = {}

    class Engine:
        def __init__(self, key_phrase, config):
            seen.update(config)

    monkeypatch.setattr(ww_bench, "reference_clips", lambda did, n: f"/refs/{did}/{n}")
    monkeypatch.setattr(ww_bench, "load_plugin_class", lambda loader, module: Engine)
    competitor = SimpleNamespace(
        plugin="ovos-ww-plugin-local-wake",
        config={
            "hotwords": {
                "computer": {
                    "module": "ovos-ww-plugin-local-wake",
                    "samples_from": {"dataset_id": "community-computer", "count": 4},
                }
            }
        },
    )
    ww_bench.WakeWordBench().load_engine(competitor, "en-US")
    assert seen["samples"] == "/refs/community-computer/4" and "samples_from" not in seen
    assert "samples_from" in competitor.config["hotwords"]["computer"]


def test_a_load_killed_while_writing_leaves_no_folder_that_looks_complete(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [])

    def dies_after_one(dataset, revision, max_per_class):
        yield "pos/0", {"array": np.zeros(1600, dtype=np.float32), "sr": 16000, "label": "positive"}
        raise KeyboardInterrupt

    monkeypatch.setattr(ww_bench, "stream_ww", dies_after_one)
    try:
        ww_bench.reference_clips("community-fake", 3)
    except KeyboardInterrupt:
        pass
    assert not (tmp_path / "community-fake@abc123-3").exists()
    monkeypatch.setattr(ww_bench, "stream_ww", _stream)
    folder = ww_bench.reference_clips("community-fake", 3)
    assert len(list((tmp_path / "community-fake@abc123-3").glob("*.wav"))) == 3
    assert folder == str(tmp_path / "community-fake@abc123-3")


def test_a_folder_built_by_another_load_meanwhile_is_kept(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [])
    winner = tmp_path / "community-fake@abc123-2"

    def other_load_wins(dataset, revision, max_per_class):
        winner.mkdir()
        (winner / "ref_0.wav").write_bytes(b"theirs")
        yield from _stream(dataset, revision, max_per_class)

    monkeypatch.setattr(ww_bench, "stream_ww", other_load_wins)
    assert ww_bench.reference_clips("community-fake", 2) == str(winner)
    assert [p.name for p in winner.iterdir()] == ["ref_0.wav"]
    assert [p.name for p in tmp_path.iterdir()] == [winner.name]


def test_a_short_set_is_not_cached_and_the_next_load_retries(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [])

    def one_download_failed(dataset, revision, max_per_class):
        for k, item in enumerate(_stream(dataset, revision, max_per_class)):
            if k != 0:
                yield item

    monkeypatch.setattr(ww_bench, "stream_ww", one_download_failed)
    with pytest.raises(RuntimeError, match="gave 2 of 3"):
        ww_bench.reference_clips("community-fake", 3)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(ww_bench, "stream_ww", _stream)
    ww_bench.reference_clips("community-fake", 3)
    assert len(list((tmp_path / "community-fake@abc123-3").glob("*.wav"))) == 3
