"""Unit tests for runner.rescore_tts — backfilling SIGMOS/DNSMOS/NISQA onto
existing TTS prediction rows from their already-stored wavs, no re-synthesis
and no network (the real judges are monkeypatched, per §4 test policy).
"""
from __future__ import annotations

import json

import pytest

from runner import rescore_tts


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class TestNeedsRescoring:
    def test_missing_both_needs_rescoring(self):
        assert rescore_tts._needs_rescoring({"extras": {"utmos": 4.0}})

    def test_missing_one_needs_rescoring(self):
        assert rescore_tts._needs_rescoring(
            {"extras": {"utmos": 4.0, "sigmos.ovrl": 4.0, "dnsmos.ovrl": 3.0}})

    def test_all_three_present_skips(self):
        assert not rescore_tts._needs_rescoring(
            {"extras": {"sigmos.ovrl": 4.0, "dnsmos.ovrl": 3.0, "nisqa.mos": 4.5}})

    def test_no_extras_needs_rescoring(self):
        assert rescore_tts._needs_rescoring({})


class TestResolveWavPath:
    def test_resolves_relative_path_under_audio_dir(self, tmp_path):
        wav = tmp_path / "audio" / "en-US" / "voice_a" / "abc123.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF....WAVEfmt ")
        row = {"audio_url": (
            "https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
            "/resolve/main/audio/en-US/voice_a/abc123.wav")}
        resolved = rescore_tts._resolve_wav_path(tmp_path, row)
        assert resolved == wav

    def test_missing_file_returns_none(self, tmp_path):
        row = {"audio_url": (
            "https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
            "/resolve/main/audio/en-US/voice_a/nope.wav")}
        assert rescore_tts._resolve_wav_path(tmp_path, row) is None

    def test_no_audio_url_returns_none(self, tmp_path):
        assert rescore_tts._resolve_wav_path(tmp_path, {"audio_url": None}) is None
        assert rescore_tts._resolve_wav_path(tmp_path, {}) is None


class TestRescoreFile:
    def test_rescores_rows_missing_the_metric(self, tmp_path, monkeypatch):
        wav = tmp_path / "audio" / "en-US" / "voice_a" / "abc.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF....WAVEfmt ")
        url = ("https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
               "/resolve/main/audio/en-US/voice_a/abc.wav")
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        _write_jsonl(jsonl_path, [
            {"sample_id": "s1", "competitor_id": "voice_a", "audio_url": url,
             "extras": {"utmos": 4.0}},
        ])

        monkeypatch.setattr(
            rescore_tts, "_score_quality_dimensions",
            lambda p: {"sigmos.ovrl": 4.5, "dnsmos.ovrl": 3.2, "nisqa.mos": 4.6})

        rescored, skipped = rescore_tts.rescore_file(jsonl_path, tmp_path)
        assert (rescored, skipped) == (1, 0)

        updated = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        assert updated[0]["extras"]["sigmos.ovrl"] == pytest.approx(4.5)
        assert updated[0]["extras"]["dnsmos.ovrl"] == pytest.approx(3.2)
        assert updated[0]["extras"]["nisqa.mos"] == pytest.approx(4.6)
        # existing utmos survives untouched
        assert updated[0]["extras"]["utmos"] == pytest.approx(4.0)

    def test_already_scored_rows_are_skipped(self, tmp_path, monkeypatch):
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        _write_jsonl(jsonl_path, [
            {"sample_id": "s1", "competitor_id": "voice_a",
             "extras": {"sigmos.ovrl": 4.0, "dnsmos.ovrl": 3.0, "nisqa.mos": 4.2}},
        ])
        called = []
        monkeypatch.setattr(
            rescore_tts, "_score_quality_dimensions",
            lambda p: called.append(p) or {})

        rescored, skipped = rescore_tts.rescore_file(jsonl_path, tmp_path)
        assert (rescored, skipped) == (0, 1)
        assert called == []

    def test_row_with_no_audio_url_is_skipped_not_fatal(self, tmp_path):
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        _write_jsonl(jsonl_path, [
            {"sample_id": "s1", "competitor_id": "voice_a", "audio_url": None,
             "extras": {"synthesis_error": "boom"}},
        ])
        rescored, skipped = rescore_tts.rescore_file(jsonl_path, tmp_path)
        assert (rescored, skipped) == (0, 1)

    def test_memory_stays_bounded_regardless_of_file_size(self, tmp_path, monkeypatch):
        """``rescore_file`` must stream rows through a temp file rather than
        parsing the whole predictions file into a list held until the final
        rewrite. Each row here carries an inflated payload (standing in for
        the per-row state a real bench run accumulates); if the old
        "parse everything, rewrite at the end" implementation is in effect,
        peak traced memory grows with the number of rows. A streaming
        implementation's peak stays close to the cost of a small, constant
        number of rows no matter how many are in the file.
        """
        import tracemalloc

        n_rows = 400
        row_payload_bytes = 200_000  # ~200 KB "decoded wav"-sized per-row footprint

        audio_dir = tmp_path / "audio" / "en-US" / "voice_a"
        audio_dir.mkdir(parents=True)
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        rows = []
        for i in range(n_rows):
            wav = audio_dir / f"clip{i}.wav"
            wav.write_bytes(b"RIFF....WAVEfmt ")
            url = (f"https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
                   f"/resolve/main/audio/en-US/voice_a/clip{i}.wav")
            rows.append({
                "sample_id": f"s{i}", "competitor_id": "voice_a", "audio_url": url,
                "extras": {"utmos": 4.0, "_padding": "x" * row_payload_bytes},
            })
        _write_jsonl(jsonl_path, rows)

        monkeypatch.setattr(
            rescore_tts, "_score_quality_dimensions",
            lambda p: {"sigmos.ovrl": 4.5, "dnsmos.ovrl": 3.2, "nisqa.mos": 4.6})

        tracemalloc.start()
        try:
            rescored, skipped = rescore_tts.rescore_file(jsonl_path, tmp_path)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert (rescored, skipped) == (n_rows, 0)
        # Bound peak to a handful of rows' worth, not the whole file's worth —
        # the old list-everything-then-rewrite implementation blows past this.
        bound = row_payload_bytes * 10
        assert peak < bound, (
            f"peak traced memory {peak} bytes exceeded {bound} bytes for "
            f"{n_rows} rows of ~{row_payload_bytes} bytes each — rows are "
            f"apparently being held in memory for the whole file")

    def test_judge_failure_during_rescore_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        wav = tmp_path / "audio" / "en-US" / "voice_a" / "abc.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF....WAVEfmt ")
        url = ("https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
               "/resolve/main/audio/en-US/voice_a/abc.wav")
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        _write_jsonl(jsonl_path, [
            {"sample_id": "s1", "competitor_id": "voice_a", "audio_url": url,
             "extras": {}},
        ])

        def boom(p):
            raise RuntimeError("onnx session exploded")

        monkeypatch.setattr(rescore_tts, "_score_quality_dimensions", boom)
        rescored, skipped = rescore_tts.rescore_file(jsonl_path, tmp_path)
        assert (rescored, skipped) == (0, 1)
