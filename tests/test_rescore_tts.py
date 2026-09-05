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


class TestNeedsIntelligibilityRejudge:
    def test_legacy_row_without_marker_needs_rejudge(self):
        assert rescore_tts._needs_intelligibility_rejudge(
            {"extras": {"intelligibility_wer": 0.2, "intelligibility_cer": 0.1}})

    def test_row_with_rover_marker_does_not_need_rejudge(self):
        assert not rescore_tts._needs_intelligibility_rejudge(
            {"extras": {"intelligibility_rover": True}})

    def test_no_extras_needs_rejudge(self):
        assert rescore_tts._needs_intelligibility_rejudge({})


def _stub_panel_result(wer=0.05, cer=0.02):
    return {
        "wer": wer,
        "cer": cer,
        "judge_model_id": "stub-model-a",
        "judge_revision": "main",
        "judges": [{"model": "stub-model-a", "revision": "main", "transcript": "hello world"}],
        "consensus": "hello world",
        "agreement": 1.0,
    }


class TestRejudgeIntelligibility:
    """``--rejudge-intelligibility`` upgrades legacy single-judge rows to a
    full #143 ROVER panel result, from the stored wav, without touching the
    quality-dims backfill's default (flag-off) behaviour."""

    def _write_row(self, tmp_path, extras):
        wav = tmp_path / "audio" / "en-US" / "voice_a" / "abc.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF....WAVEfmt ")
        url = ("https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
               "/resolve/main/audio/en-US/voice_a/abc.wav")
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        _write_jsonl(jsonl_path, [
            {"sample_id": "en-US/00000", "competitor_id": "voice_a",
             "audio_url": url, "lang": "en-US", "input_text": "hello world",
             "extras": {
                 "sigmos.ovrl": 4.0, "dnsmos.ovrl": 3.0, "nisqa.mos": 4.2,
                 **extras}},
        ])
        return jsonl_path

    def test_legacy_row_fully_upgraded(self, tmp_path, monkeypatch):
        jsonl_path = self._write_row(tmp_path, {
            "intelligibility_wer": 0.9, "intelligibility_cer": 0.8,
            "intelligibility_judge": "old-model", "intelligibility_judge_revision": "main",
        })
        calls = []

        def fake_score(wav_path, text, lang):
            calls.append((wav_path, text, lang))
            return _stub_panel_result()

        monkeypatch.setattr(rescore_tts, "_score_intelligibility", fake_score)

        rescored, skipped = rescore_tts.rescore_file(
            jsonl_path, tmp_path, rejudge_intelligibility=True)
        assert (rescored, skipped) == (1, 0)
        assert calls == [(tmp_path / "audio" / "en-US" / "voice_a" / "abc.wav",
                           "hello world", "en-US")]

        row = json.loads(jsonl_path.read_text().splitlines()[0])
        extras = row["extras"]
        assert extras["intelligibility_wer"] == pytest.approx(0.05)
        assert extras["intelligibility_cer"] == pytest.approx(0.02)
        assert extras["intelligibility_judge"] == "stub-model-a"
        assert extras["intelligibility_judge_revision"] == "main"
        assert extras["intelligibility_judges"] == [
            {"model": "stub-model-a", "revision": "main", "transcript": "hello world"}]
        assert extras["intelligibility_consensus"] == "hello world"
        assert extras["intelligibility_agreement"] == pytest.approx(1.0)
        assert extras["intelligibility_rover"] is True
        # quality dims untouched (already present, not re-scored)
        assert extras["sigmos.ovrl"] == pytest.approx(4.0)

    def test_row_already_rover_scored_is_untouched(self, tmp_path, monkeypatch):
        jsonl_path = self._write_row(tmp_path, {
            "intelligibility_wer": 0.05, "intelligibility_cer": 0.02,
            "intelligibility_judge": "stub-model-a", "intelligibility_judge_revision": "main",
            "intelligibility_judges": [{"model": "stub-model-a", "revision": "main",
                                         "transcript": "hello world"}],
            "intelligibility_consensus": "hello world", "intelligibility_agreement": 1.0,
            "intelligibility_rover": True,
        })
        called = []
        monkeypatch.setattr(
            rescore_tts, "_score_intelligibility",
            lambda w, t, l: called.append((w, t, l)) or _stub_panel_result())

        rescored, skipped = rescore_tts.rescore_file(
            jsonl_path, tmp_path, rejudge_intelligibility=True)
        assert (rescored, skipped) == (0, 1)
        assert called == []

    def test_missing_wav_is_skipped_and_counted(self, tmp_path, monkeypatch):
        jsonl_path = tmp_path / "predictions" / "en-US" / "voice_a.jsonl"
        url = ("https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
               "/resolve/main/audio/en-US/voice_a/nope.wav")
        _write_jsonl(jsonl_path, [
            {"sample_id": "en-US/00000", "competitor_id": "voice_a",
             "audio_url": url, "lang": "en-US", "input_text": "hello world",
             "extras": {
                 "sigmos.ovrl": 4.0, "dnsmos.ovrl": 3.0, "nisqa.mos": 4.2,
                 "intelligibility_wer": 0.9, "intelligibility_cer": 0.8}},
        ])
        called = []
        monkeypatch.setattr(
            rescore_tts, "_score_intelligibility",
            lambda w, t, l: called.append((w, t, l)) or _stub_panel_result())

        rescored, skipped = rescore_tts.rescore_file(
            jsonl_path, tmp_path, rejudge_intelligibility=True)
        assert (rescored, skipped) == (0, 1)
        assert called == []
        row = json.loads(jsonl_path.read_text().splitlines()[0])
        # legacy fields untouched — nothing to score against
        assert row["extras"]["intelligibility_wer"] == pytest.approx(0.9)
        assert "intelligibility_rover" not in row["extras"]

    def test_without_flag_legacy_intelligibility_untouched(self, tmp_path, monkeypatch):
        jsonl_path = self._write_row(tmp_path, {
            "intelligibility_wer": 0.9, "intelligibility_cer": 0.8,
            "intelligibility_judge": "old-model", "intelligibility_judge_revision": "main",
        })
        called = []
        monkeypatch.setattr(
            rescore_tts, "_score_intelligibility",
            lambda w, t, l: called.append((w, t, l)) or _stub_panel_result())

        # flag defaults to False; quality dims already present so nothing
        # to rescore at all — file must be left byte-identical.
        original_bytes = jsonl_path.read_bytes()
        rescored, skipped = rescore_tts.rescore_file(jsonl_path, tmp_path)
        assert (rescored, skipped) == (0, 1)
        assert called == []
        assert jsonl_path.read_bytes() == original_bytes


class TestUnjudgeableLanguageMigration:
    """``--rejudge-intelligibility`` marks rows whose language has no ASR
    judge, and must not touch a row that carries a real measurement."""

    def _row(self, lang, extras):
        url = (f"https://huggingface.co/datasets/OpenVoiceOS/ovos-tts-bench-d"
               f"/resolve/main/audio/{lang}/voice_a/abc.wav")
        return {"sample_id": "s1", "competitor_id": "voice_a", "lang": lang,
                "audio_url": url, "extras": extras}

    def _run(self, tmp_path, lang, row, monkeypatch, judge=None):
        wav = tmp_path / "audio" / lang / "voice_a" / "abc.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF....WAVEfmt ")
        jsonl_path = tmp_path / "predictions" / lang / "voice_a.jsonl"
        _write_jsonl(jsonl_path, [row])
        monkeypatch.setattr(
            rescore_tts, "_score_quality_dimensions",
            lambda p: {"sigmos.ovrl": 4.5, "dnsmos.ovrl": 3.2, "nisqa.mos": 4.6})

        def must_not_run(*a, **kw):
            raise AssertionError("no judge exists for this language")

        monkeypatch.setattr(rescore_tts, "_score_intelligibility",
                            judge or must_not_run)
        rescore_tts.rescore_file(jsonl_path, tmp_path, rejudge_intelligibility=True)
        return json.loads(jsonl_path.read_text().splitlines()[0])

    def test_an_es_whisper_scored_row_is_replaced_by_the_marker(
            self, tmp_path, monkeypatch):
        row = self._row("an-ES", {
            "utmos": 3.3061, "intelligibility_wer": 1.2996,
            "intelligibility_cer": 0.9, "intelligibility_judge": "whisper-base"})
        updated = self._run(tmp_path, "an-ES", row, monkeypatch)

        assert updated["extras"]["intelligibility"] == "not_available"
        assert updated["extras"]["intelligibility_wer"] is None
        assert updated["extras"]["intelligibility_judge"] == "none"
        assert updated["extras"]["utmos"] == pytest.approx(3.3061)

    def test_row_scored_by_a_dedicated_model_is_never_overwritten(
            self, tmp_path, monkeypatch):
        # Russian resolves to gigaam-v2-rnnt in the plugin registry. Even if
        # judge_available were to regress, a real measurement must survive.
        row = self._row("ru-RU", {
            "utmos": 3.9, "intelligibility_wer": 0.21,
            "intelligibility_cer": 0.08, "intelligibility_judge": "gigaam-v2-rnnt"})
        monkeypatch.setattr(rescore_tts, "judge_available", lambda lang: False)
        panel = lambda *a, **kw: {
            "wer": 0.21, "cer": 0.08, "judge_model_id": "gigaam-v2-rnnt",
            "judge_revision": "abc", "judges": [], "consensus": "x",
            "agreement": 1.0,
        }
        updated = self._run(tmp_path, "ru-RU", row, monkeypatch, judge=panel)

        assert updated["extras"]["intelligibility_wer"] == pytest.approx(0.21)
        assert updated["extras"]["intelligibility_cer"] == pytest.approx(0.08)
        assert updated["extras"]["intelligibility_judge"] == "gigaam-v2-rnnt"
        assert "intelligibility" not in updated["extras"]

    def test_marked_row_is_not_rejudged_again(self):
        row = {"extras": {"intelligibility": "not_available",
                          "intelligibility_wer": None}}
        assert rescore_tts._needs_intelligibility_rejudge(row) is False
