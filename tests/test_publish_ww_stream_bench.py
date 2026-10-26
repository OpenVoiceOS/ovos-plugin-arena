"""Tests for ``scripts/publish_ww_stream_bench.py``: no wake phrase reaches the speech, and
every language gets its own negatives plus the shared part."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from registry.schemas import DatasetDef

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "publish_ww_stream_bench", ROOT / "scripts" / "publish_ww_stream_bench.py"
)
pwb = importlib.util.module_from_spec(_spec)
sys.modules["publish_ww_stream_bench"] = pwb
_spec.loader.exec_module(pwb)


def _wav_bytes(seconds: float = 0.1) -> bytes:
    buf = io.BytesIO()
    noise = np.random.default_rng(0).normal(0, 0.05, int(16000 * seconds)).astype("float32")
    sf.write(buf, noise, 16000, format="WAV")
    return buf.getvalue()


def _parquet(path: Path, rows: dict[str, str], seconds: float = 0.1) -> Path:
    audio = [{"bytes": _wav_bytes(seconds), "path": None} for _ in rows]
    pq.write_table(pa.table({"id": list(rows), "text": list(rows.values()), "audio": audio}), path)
    return path


def _source(name: str) -> pwb.Source:
    return pwb.Source(
        name,
        f"org/{name}",
        "rev",
        "parquet",
        (f"{name}.parquet",),
        name,
        "test",
        "CC BY 4.0",
        text="text",
        key="id",
    )


@pytest.fixture
def parquet_sources(tmp_path, monkeypatch):
    """Serve ``<name>.parquet`` under *tmp_path* for every source download."""
    monkeypatch.setattr(pwb, "_download", lambda source, name: tmp_path / name)
    return tmp_path


def test_every_set_and_fighter_phrase_is_filtered():
    phrases = pwb.wake_phrases()
    for phrase in (
        "stop",
        "sam",
        "wake up",
        "weather",
        "hey mycroft",
        "acorda",
        "liga a luz",
        "wakker worden",
        "joao",
    ):
        assert phrase in phrases
    assert all(p == pwb.normalise(p) for p in phrases)


def test_case_and_accents_do_not_hide_a_phrase():
    drop = pwb.phrase_pattern(["joao", "liga a luz", "stop"])
    assert drop.search(pwb.normalise("Olá, JOÃO!"))
    assert drop.search(pwb.normalise("Liga  a LUZ, por favor."))
    assert not drop.search(pwb.normalise("O STOPPER estava solto"))
    assert not drop.search(pwb.normalise("Joãozinho chegou"))


def test_an_utterance_that_says_a_phrase_is_left_out(parquet_sources):
    rows = {
        "1-1-0001": "AND THEN HE SAID STOP AT ONCE",
        "1-1-0002": "SAM WENT HOME",
        "1-1-0003": "WHEN WILL YOU WAKE UP",
        "1-1-0004": "THE STOPPER WAS LOOSE",
        "1-1-0005": "SAMUEL READ THE PAPER",
        "1-1-0006": "IT WAS A COLD DAY",
    }
    _parquet(parquet_sources / "speech.parquet", rows)
    out = parquet_sources / "speech"
    result = pwb.export(
        _source("speech"), [(out, 0.3)], 0, pwb.phrase_pattern(["stop", "sam", "wake up"])
    )
    assert sorted(result["dropped"]) == ["1-1-0001", "1-1-0002", "1-1-0003"]
    assert result["written"] == [
        {"folder": "speech", "clips": 3, "hours": pytest.approx(0.3 / 3600)}
    ]
    assert len(list(out.glob("*.wav"))) == 3


def test_budgets_fill_in_turn_and_a_short_source_raises(parquet_sources):
    _parquet(parquet_sources / "speech.parquet", {f"u{i}": "a cold day" for i in range(10)})
    drop = pwb.phrase_pattern(["stop"])
    a, b = parquet_sources / "a", parquet_sources / "b"
    result = pwb.export(_source("speech"), [(a, 0.6), (b, 0.3)], 0, drop)
    assert [w["clips"] for w in result["written"]] == [6, 3]
    assert len(list(a.glob("*.wav"))) == 6 and len(list(b.glob("*.wav"))) == 3
    with pytest.raises(RuntimeError, match="speech"):
        pwb.export(
            _source("speech"), [(parquet_sources / "c", 0.6), (parquet_sources / "d", 0.6)], 0, drop
        )


def test_each_language_gets_its_own_speech_and_the_shared_part(parquet_sources, monkeypatch):
    for name in ("music", "english", "portuguese"):
        _parquet(
            parquet_sources / f"{name}.parquet",
            {f"{name}{i}": "a cold day" for i in range(80)},
            seconds=1.0,
        )
    monkeypatch.setattr(pwb, "SOURCES", {n: _source(n) for n in ("music", "english", "portuguese")})
    monkeypatch.setattr(
        pwb,
        "NEGATIVES",
        {
            "shared": {"music": [("music", 20 / 3600)]},
            "en-US": {"read_speech": [("english", 20 / 3600)]},
            "pt-PT": {"read_speech": [("portuguese", 20 / 3600)]},
        },
    )
    monkeypatch.setattr(pwb, "BORROWED", {"an-ES": "pt-PT"})
    monkeypatch.setattr(pwb, "RECORDING_S", 10.0)
    monkeypatch.setattr(pwb, "SHARED_RECORDINGS", 1)
    monkeypatch.setattr(pwb, "FILLER_HOURS", 10 / 3600)
    facts = {"sources": {}}
    manifests = pwb.build_negatives(
        parquet_sources / "src", parquet_sources / "repo", 0, pwb.phrase_pattern(["stop"]), facts
    )

    def kinds(group):
        return [(r["audio"].rsplit("/", 2)[0], r["category"]) for r in manifests[group]]

    assert (
        kinds("en-US")
        == [("negatives/en-US/read_speech", "read_speech")] * 2
        + [("negatives/shared/music", "music")] * 2
    )
    # one shared music recording and one English recording, counted as another language
    assert kinds("pt-PT") == [("negatives/pt-PT/read_speech", "read_speech")] * 2 + [
        ("negatives/shared/music", "music"),
        ("negatives/en-US/read_speech", "foreign_speech"),
    ]
    assert kinds("an-ES") == [("negatives/pt-PT/read_speech", "foreign_speech")] * 2 + [
        ("negatives/shared/music", "music"),
        ("negatives/en-US/read_speech", "foreign_speech"),
    ]
    assert facts["filler"] == {"pt-PT": ["portuguese/filler"], "an-ES": ["portuguese/filler"]}
    assert len(list((parquet_sources / "src" / "portuguese" / "filler").glob("*.wav"))) == 10
    assert facts["negatives"]["pt-PT"] == {
        "read_speech": pytest.approx(20 / 3600),
        "music": pytest.approx(10 / 3600),
        "foreign_speech": pytest.approx(10 / 3600),
    }
    for group in manifests:
        for row in manifests[group]:
            assert (parquet_sources / "repo" / row["audio"]).is_file()
            assert row["onsets"] == []
    # a negative recording never repeats inside one language's manifest
    for rows in manifests.values():
        assert len({r["audio"] for r in rows}) == len(rows)


def test_set_languages_map_to_their_negatives():
    assert pwb.group_of("en-GB") == "en-US"
    assert pwb.group_of("pt-PT") == "pt-PT"
    assert pwb.group_of("an-ES") == "an-ES"
    assert pwb.group_of("kab-DZ") is None


def test_registry_entries_are_valid_datasets():
    hours = {"read_speech": 4.0, "music": 1.0, "noise": 1.0, "foreign_speech": 1.0}
    negatives = DatasetDef(**pwb.negatives_entry("pt-PT", "a" * 40, hours))
    assert negatives.dataset_id == "ww-negatives-pt-PT"
    assert negatives.primary_metric == "fa_per_hour"
    assert negatives.source.file_pattern == "manifest-negatives-pt-PT.jsonl"
    assert negatives.min_negative_hours == 7.0
    assert negatives.wakeword is None
    s = {
        "wakeword": "acorda",
        "group": "pt-PT",
        "lang": "pt-PT",
        "source": "TigreGotico/x",
        "revision": "b" * 40,
        "onsets": 400,
        "available": 900,
    }
    stream = DatasetDef(**pwb.registry_entry("tigregotico-acorda-pt", s, "a" * 40, hours))
    assert stream.dataset_id == "ww_stream_tigregotico-acorda-pt"
    assert stream.wakeword == "acorda" and stream.primary_metric is None
    assert stream.test_set_kind == "synthetic"
    assert "Portuguese speech" in stream.summary
    assert json.dumps(pwb.registry_entry("tigregotico-acorda-pt", s, "a" * 40, hours))


def test_estimates_that_run_long_load_more_rows(tmp_path, monkeypatch):
    # every row claims 1 s but holds 0.25 s, so the first batch the estimates pick is too small
    rows = [pwb.Candidate(f"u{i}", "a cold day", 1.0) for i in range(600)]

    def load(chosen):
        for c in chosen:
            c.data = _wav_bytes(0.25)

    monkeypatch.setattr(pwb, "candidates", lambda source: (rows, load))
    result = pwb.export(
        _source("speech"),
        [(tmp_path / "a", 100.0), (tmp_path / "b", 20.0)],
        0,
        pwb.phrase_pattern(["stop"]),
    )
    assert [w["clips"] for w in result["written"]] == [400, 80]


def test_export_files_keeps_the_entry_order_and_skips_files_on_disk(tmp_path, monkeypatch):
    import huggingface_hub

    sources = tmp_path / "hub"
    sources.mkdir()
    for k in range(5):
        sf.write(sources / f"c{k}.wav", np.full(1600, k / 10, dtype="float32"), 16000)
    fetched = []

    def download(hf_id, rel, repo_type, revision):
        fetched.append(rel)
        return str(sources / rel)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    out = tmp_path / "out"
    out.mkdir()
    sf.write(out / "00001.wav", np.zeros(1600, dtype="float32"), 16000)
    entries = [("repo", f"c{k}.wav", "sha") for k in range(5)]
    names = pwb.export_files(entries, out)
    assert names == [f"{k:05d}.wav" for k in range(5)]
    assert sorted(fetched) == ["c0.wav", "c2.wav", "c3.wav", "c4.wav"]
    for k in (0, 2, 3, 4):
        audio, _ = sf.read(out / f"{k:05d}.wav", dtype="float32")
        assert audio[0] == pytest.approx(k / 10, abs=1e-3)


def test_upload_calls_only_what_the_installed_hub_has_and_removes_stale_files(tmp_path):
    from unittest.mock import create_autospec

    from huggingface_hub import CommitOperationDelete, HfApi

    (tmp_path / "negatives").mkdir()
    (tmp_path / "README.md").write_text("card")
    (tmp_path / "negatives" / "en-US.flac").write_bytes(b"x")
    api = create_autospec(HfApi, instance=True)
    api.list_repo_files.return_value = [
        ".gitattributes",
        "README.md",
        "negatives/en-US.flac",
        "old/set.flac",
    ]
    api.dataset_info.return_value.sha = "abc123"

    assert pwb.upload(api, tmp_path) == "abc123"
    api.upload_folder.assert_called_once_with(
        repo_id=pwb.REPO, repo_type="dataset", folder_path=str(tmp_path)
    )
    (deleted,) = [c.kwargs["operations"] for c in api.create_commit.call_args_list]
    assert deleted == [CommitOperationDelete("old/set.flac")]


def test_entry_texts_name_the_speech_used_and_the_exact_hours():
    eu = {"read_speech": 3.0, "broadcast": 1.5, "music": 1.0, "noise": 1.0, "foreign_speech": 1.0}
    an = {"music": 1.0, "noise": 1.0, "foreign_speech": 5.0}
    aragonese = {
        "wakeword": "despierta",
        "group": "an-ES",
        "lang": "an-ES",
        "source": "TigreGotico/synthetic-wakeword-despierta",
        "revision": "0" * 40,
        "onsets": 267,
        "available": 267,
    }

    entry = pwb.registry_entry("tigregotico-despierta-an", aragonese, "abc", an)
    negatives = pwb.negatives_entry("eu-ES", "abc", eu)

    assert "between seconds of Spanish speech" in entry["summary"]
    assert "Aragonese speech" not in entry["summary"]
    assert negatives["summary"].startswith(f"{sum(eu.values())} hours of ")
    assert f"{sum(eu.values())} hours in all" in negatives["notes"]
    assert pwb._negatives_text({"music": 2.0}) == "2 hours of music"


# the default negative, background-noise and music pools of the wakeforge trainer
# (TigreGotico/wakeforge ww_trainer/datagen.py NEGATIVE_DATASETS)
WAKEFORGE_TRAINING_POOLS = {
    "TigreGotico/ESC-50",
    "TigreGotico/NAR",
    "agkphysics/AudioSet",
    "TigreGotico/not-wake-words-speech-en",
    "TigreGotico/ambient_noises",
    "TigreGotico/building_106_kitchen_3secs",
    "TigreGotico/public_domain_sounds_3secs",
    "TigreGotico/FMA_3secs",
}


def test_no_source_is_a_training_pool_of_a_fighter():
    assert not {s.hf_id for s in pwb.SOURCES.values()} & WAKEFORGE_TRAINING_POOLS


def test_fsd50k_clips_with_crowd_or_chatter_are_left_out(monkeypatch, tmp_path):
    index = tmp_path / "eval.csv"
    index.write_text(
        "fname,labels,mids\n"
        '1,"Dishes_and_pots_and_pans,Domestic_sounds_and_home_sounds",x\n'
        '2,"Chatter,Human_group_actions",x\n'
        '3,"Crowd,Human_group_actions",x\n'
        '4,"Cheering,Crowd",x\n'
        '5,"Rain,Water",x\n'
    )
    monkeypatch.setattr(pwb, "_download", lambda source, path: index)

    rows = pwb._index_rows(pwb.SOURCES["fsd50k-eval"])

    assert [r["fname"] for r in rows] == ["1", "5"]
