"""Tests for runner.publish — spec-path publishing (§3.2).

The daemon's legacy root-level shards are invisible to both the assembler
(reads ``predictions/``) and the sweep diff (checks
``predictions/<competitor_id>.jsonl``): publishing anywhere else makes the
runner's work silently unusable, so the remote path shape is pinned here.
"""
from pathlib import Path

from runner.publish import publish_competitor_output


class FakeApi:
    def __init__(self):
        self.uploads = []

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                    repo_type, commit_message):
        self.uploads.append((path_in_repo, repo_id, repo_type))


class TestPublishCompetitorOutput:
    def _patch_api(self, monkeypatch):
        import huggingface_hub
        fake = FakeApi()
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake)
        return fake

    def test_uploads_full_file_to_spec_path(self, tmp_path, monkeypatch):
        fake = self._patch_api(monkeypatch)
        out = tmp_path / "stt_en-AU_vosk-big-en.jsonl"
        out.write_text('{"sample_id": "s1"}\n')
        uploaded = publish_competitor_output(
            out, "OpenVoiceOS/ovos-stt-bench-minds14-en-AU",
            lang="en-AU", competitor_id="vosk-big-en")
        assert uploaded == ["predictions/en-AU/vosk-big-en.jsonl"]
        assert fake.uploads == [(
            "predictions/en-AU/vosk-big-en.jsonl",
            "OpenVoiceOS/ovos-stt-bench-minds14-en-AU", "dataset")]

    def test_empty_or_missing_file_skips(self, tmp_path, monkeypatch):
        fake = self._patch_api(monkeypatch)
        empty = tmp_path / "x.jsonl"
        empty.write_text("")
        assert publish_competitor_output(
            empty, "r", lang="en", competitor_id="c") == []
        assert publish_competitor_output(
            tmp_path / "nope.jsonl", "r", lang="en", competitor_id="c") == []
        assert fake.uploads == []
