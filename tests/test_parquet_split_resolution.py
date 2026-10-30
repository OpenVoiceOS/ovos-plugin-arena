"""runner.audio_io._parquet_files reads exactly the named subset and split.

The listings below copy the shape of the real Hugging Face repos (file names
trimmed to a few per directory), so no test touches the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from runner.audio_io import _parquet_files

REGISTRY_ROOT = Path(__file__).resolve().parent.parent / "registry"

FLEURS = [
    "README.md",
    "data/af_za/audio/test.tar.gz",
    "data/af_za/test.tsv",
    "data/hi_in/audio/test.tar.gz",
    "data/hi_in/test.tsv",
    "data/hi_in/train.tsv",
    "parquet-data/af_za/test-00000-of-00001.parquet",
    "parquet-data/af_za/train-00000-of-00001.parquet",
    "parquet-data/all/test-00000-of-00004.parquet",
    "parquet-data/hi_in/test-00000-of-00001.parquet",
    "parquet-data/hi_in/train-00000-of-00001.parquet",
    "parquet-data/hi_in/validation-00000-of-00001.parquet",
]

EDACC = [
    "README.md",
    "data/test-00000-of-00002-f0aceb1c.parquet",
    "data/test-00001-of-00002-856b016d.parquet",
    "data/validation-00000-of-00001-17ec18f4.parquet",
]

MLSW = [
    "ar_opus/test/0000.parquet",
    "ca_wav/partial-test/0000.parquet",
    "ca_wav/partial-train/0000.parquet",
    "ca_wav/partial-validation/0000.parquet",
]


def _listing(monkeypatch, files):
    class FakeApi:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            return files

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())


def test_fleurs_subset_reads_only_its_language_test_split(monkeypatch):
    _listing(monkeypatch, FLEURS)
    assert _parquet_files("google/fleurs", "parquet-data/hi_in", "test", "main") == [
        "parquet-data/hi_in/test-00000-of-00001.parquet",
    ]


def test_subset_that_matches_no_parquet_raises(monkeypatch):
    # The bare FLEURS code only names tar and tsv files. Falling back to
    # every parquet file would score Afrikaans audio on a Hindi board.
    _listing(monkeypatch, FLEURS)
    with pytest.raises(RuntimeError, match="subset=hi_in split=test"):
        _parquet_files("google/fleurs", "hi_in", "test", "main")


def test_subset_never_falls_back_to_another_subset(monkeypatch):
    _listing(monkeypatch, ["data/test-00000-of-00001.parquet", "en/test-00000-of-00001.parquet"])
    with pytest.raises(RuntimeError):
        _parquet_files("some/repo", "hr", "test", "main")


def test_null_subset_reads_only_the_named_split(monkeypatch):
    _listing(monkeypatch, EDACC)
    assert _parquet_files("edinburghcstr/edacc", None, "test", "main") == [
        "data/test-00000-of-00002-f0aceb1c.parquet",
        "data/test-00001-of-00002-856b016d.parquet",
    ]


def test_split_that_matches_no_parquet_raises(monkeypatch):
    # The ml_spoken_words parquet mirror names its splits partial-test etc.
    _listing(monkeypatch, MLSW)
    with pytest.raises(RuntimeError):
        _parquet_files("MLCommons/ml_spoken_words", "ca_wav", "test", "main")
    assert _parquet_files("MLCommons/ml_spoken_words", "ca_wav", "partial-test", "main") == [
        "ca_wav/partial-test/0000.parquet"
    ]


def test_every_fleurs_dataset_names_a_parquet_language_directory():
    entries = [json.loads(p.read_text()) for p in sorted(REGISTRY_ROOT.glob("datasets/*/*.json"))]
    fleurs = {
        d["dataset_id"]: d["source"]["subset"]
        for d in entries
        if d.get("source", {}).get("hf_id") == "google/fleurs"
    }
    assert len(fleurs) >= 12
    bad = {
        k: v for k, v in fleurs.items() if not re.fullmatch(r"parquet-data/[a-z]+_[a-z]+", v or "")
    }
    assert bad == {}
