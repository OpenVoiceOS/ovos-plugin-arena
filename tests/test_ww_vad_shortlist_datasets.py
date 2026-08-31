"""Unit tests for the ml_spoken_words / AVA-Speech shortlist integration.

No network: exercises the pure label/window-boundary math and the parquet
path-resolution fallback with stubbed rows/files, proving the two new
adapters do what their registry entries claim before any live smoke test.
"""
from __future__ import annotations

from runner.audio_io import _ava_speech_windows, _parquet_files


class TestAvaSpeechWindows:
    """runner.audio_io._ava_speech_windows — pure onset/offset -> window math
    backing the ava-speech-conditions vad dataset (registry/datasets/vad/
    ava-speech-conditions.json, reference_fields.format=ava-onset-offset)."""

    def test_single_speech_segment_and_surrounding_gaps(self):
        # one 10s speech segment in the middle of a 30s clip
        speech, non_speech = _ava_speech_windows(
            onsets=[10.0], offsets=[20.0], duration_s=30.0, seg_s=3.0,
        )
        assert speech == [(10.0, 13.0)]
        # gap before [0, 10) and gap after [20, 30) are both >= seg_s
        assert non_speech == [(0.0, 3.0), (20.0, 23.0)]

    def test_short_segment_and_short_gap_are_dropped(self):
        # speech segment shorter than seg_s is dropped; likewise a short gap
        speech, non_speech = _ava_speech_windows(
            onsets=[5.0], offsets=[6.5], duration_s=10.0, seg_s=3.0,
        )
        assert speech == []  # 1.5s segment < 3s
        # gap before: [0, 5) = 5s >= 3s -> kept; gap after: [6.5, 10) = 3.5s -> kept
        assert non_speech == [(0.0, 3.0), (6.5, 9.5)]

    def test_multiple_segments(self):
        speech, non_speech = _ava_speech_windows(
            onsets=[0.0, 20.0], offsets=[10.0, 40.0], duration_s=40.0, seg_s=5.0,
        )
        assert speech == [(0.0, 5.0), (20.0, 25.0)]
        # gaps: [0,0) dropped (zero-length before first onset==0),
        # [10, 20) = 10s -> kept, [40, 40) dropped (clip ends exactly at last offset)
        assert non_speech == [(10.0, 15.0)]

    def test_no_speech_at_all_is_all_non_speech(self):
        speech, non_speech = _ava_speech_windows(
            onsets=[], offsets=[], duration_s=12.0, seg_s=3.0,
        )
        assert speech == []
        assert non_speech == [(0.0, 3.0)]


class TestParquetFilesDirectoryLayout:
    """runner.audio_io._parquet_files must resolve both the default flat
    '<subset>/<split>-NNNN.parquet' naming AND a directory-per-split layout
    ('<subset>/<split>/NNNN.parquet') — the shape HF's own
    refs/convert/parquet auto-conversion uses for a loading-script dataset
    such as MLCommons/ml_spoken_words (registry/datasets/wake_word/
    mlsw-negatives-*.json)."""

    def test_directory_per_split_layout(self, monkeypatch):
        listed = [
            "README.md",
            "de_wav/partial-test/0000.parquet",
            "de_wav/partial-test/0001.parquet",
            "de_wav/partial-train/0000.parquet",
            "en_wav/partial-test/0000.parquet",
        ]

        class FakeApi:
            def list_repo_files(self, repo, repo_type=None, revision=None):
                return listed

        monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
        files = _parquet_files(
            "MLCommons/ml_spoken_words", "de_wav", "partial-test",
            "refs/convert/parquet",
        )
        assert files == [
            "de_wav/partial-test/0000.parquet",
            "de_wav/partial-test/0001.parquet",
        ]

    def test_flat_split_layout_still_resolves(self, monkeypatch):
        listed = [
            "en-US/train-00000-of-00001.parquet",
            "en-US/test-00000-of-00001.parquet",
        ]

        class FakeApi:
            def list_repo_files(self, repo, repo_type=None, revision=None):
                return listed

        monkeypatch.setattr("huggingface_hub.HfApi", lambda: FakeApi())
        files = _parquet_files("some/repo", "en-US", "test", "main")
        assert files == ["en-US/test-00000-of-00001.parquet"]
