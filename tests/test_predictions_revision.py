"""Tests for pinned predictions revisions (§C reproducibility).

All HuggingFace Hub calls are mocked — these tests never touch the network.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arena.cli import _predictions_revision_for, cmd_assemble
from arena.predictions import reset_revision_cache, resolve_predictions_revision


@pytest.fixture(autouse=True)
def _reset_revision_cache():
    reset_revision_cache()
    yield
    reset_revision_cache()


def _response(status: int):
    import httpx

    return httpx.Response(status, request=httpx.Request("GET", "https://hf.co/x"))


class _StubCompetitor:
    def __init__(self, competitor_id):
        self.competitor_id = competitor_id


@pytest.fixture(autouse=True)
def _permissive_registry(monkeypatch):
    # "comp-a" predates the board-truth registry filter added to
    # arena.predictions.group_rows — see tests/test_predictions.py for
    # the filter's own coverage.
    import registry.loaders as loaders_mod

    def fake_list_competitors(modality=None):
        ids = {"stt": {"comp-a"}}.get(modality, set())
        return [_StubCompetitor(cid) for cid in ids]

    monkeypatch.setattr(loaders_mod, "list_competitors", fake_list_competitors)


class TestResolvePredictionsRevision:
    def test_resolves_to_commit_sha(self):
        fake_info = MagicMock(sha="abc123deadbeef")
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.dataset_info.return_value = fake_info
            sha = resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="main")
        assert sha == "abc123deadbeef"
        MockApi.return_value.dataset_info.assert_called_once_with(
            "OpenVoiceOS/ovos-stt-bench-x", revision="main"
        )

    def test_memoized_at_most_once_per_repo_per_run(self):
        """§assemble scalability — a board that shares its predictions repo
        across two (dataset, lang) boards must resolve that repo's revision
        ONCE for the whole run, not once per board. Fails against the
        unmemoized implementation (HfApi().dataset_info called twice)."""
        fake_info = MagicMock(sha="abc123deadbeef")
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.dataset_info.return_value = fake_info
            sha1 = resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="main")
            sha2 = resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="main")
        assert sha1 == sha2 == "abc123deadbeef"
        MockApi.return_value.dataset_info.assert_called_once_with(
            "OpenVoiceOS/ovos-stt-bench-x", revision="main"
        )

    def test_retries_a_rate_limited_lookup_then_resolves(self, monkeypatch):
        """A 429 on the revision lookup is the same rate limiter
        ``fetch_hf_predictions`` retries around (§ daily unauthenticated
        assemble walks ~120 repos). Failing to retry here degraded a
        board's provenance from a resolved commit to the floating ref
        with only a warning to show for it."""
        from huggingface_hub.utils import HfHubHTTPError

        attempts = []
        fake_info = MagicMock(sha="abc123deadbeef")

        def flaky_dataset_info(repo_id, revision=None):
            attempts.append(repo_id)
            if len(attempts) < 3:
                raise HfHubHTTPError("429 Client Error: Too Many Requests",
                                      response=_response(429))
            return fake_info

        sleeps = []
        import arena.predictions as predictions_mod
        monkeypatch.setattr(predictions_mod.time, "sleep", sleeps.append)
        monkeypatch.setattr(predictions_mod, "HF_FETCH_BACKOFF_SECONDS",
                            (5.0, 15.0, None))

        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.dataset_info.side_effect = flaky_dataset_info
            sha = resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="main")

        assert sha == "abc123deadbeef"
        assert len(attempts) == 3
        assert sleeps == [5.0, 15.0]

    def test_gives_up_after_exhausting_backoff_on_persistent_429(self, monkeypatch):
        from huggingface_hub.utils import HfHubHTTPError

        attempts = []

        def always_429(repo_id, revision=None):
            attempts.append(repo_id)
            raise HfHubHTTPError("429 Client Error: Too Many Requests",
                                  response=_response(429))

        import arena.predictions as predictions_mod
        monkeypatch.setattr(predictions_mod.time, "sleep", lambda *_: None)
        monkeypatch.setattr(predictions_mod, "HF_FETCH_BACKOFF_SECONDS",
                            (0.0, 0.0, None))

        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.dataset_info.side_effect = always_429
            with pytest.raises(HfHubHTTPError):
                resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="main")

        assert len(attempts) == 3

    def test_pinned_revision_not_found_is_never_retried(self, monkeypatch):
        """#162: a stale pin is fatal, not transient — it will not resolve
        no matter how many times it is retried, so it must fail on the
        first attempt."""
        from huggingface_hub.utils import RevisionNotFoundError

        attempts = []

        def vanished(repo_id, revision=None):
            attempts.append(repo_id)
            raise RevisionNotFoundError("404 Client Error: Revision Not Found",
                                         response=_response(404))

        import arena.predictions as predictions_mod
        sleeps = []
        monkeypatch.setattr(predictions_mod.time, "sleep", sleeps.append)

        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.dataset_info.side_effect = vanished
            with pytest.raises(RevisionNotFoundError):
                resolve_predictions_revision("OpenVoiceOS/pinned-bench", revision="v-gone")

        assert attempts == ["OpenVoiceOS/pinned-bench"]
        assert sleeps == []

    def test_different_revision_is_not_cached_together(self):
        fake_info = MagicMock(sha="shaA")
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.dataset_info.return_value = fake_info
            resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="main")
            resolve_predictions_revision("OpenVoiceOS/ovos-stt-bench-x", revision="v2")
        assert MockApi.return_value.dataset_info.call_count == 2


class TestPredictionsRevisionFor:
    def test_local_dir_skips_resolution(self, tmp_path):
        revision, meta = _predictions_revision_for(str(tmp_path), "main")
        assert revision == "main"
        assert meta == {}

    def test_registry_pin_used_when_present(self, tmp_path, monkeypatch):
        ds = MagicMock(predictions_hf="OpenVoiceOS/ovos-stt-bench-x",
                        predictions_revision="pinned-sha-123")
        with patch("registry.loaders.list_datasets", return_value=[ds]), \
             patch("arena.predictions.resolve_predictions_revision",
                   return_value="pinned-sha-123") as mock_resolve:
            revision, meta = _predictions_revision_for(
                "OpenVoiceOS/ovos-stt-bench-x", "main"
            )
        assert revision == "pinned-sha-123"
        assert meta == {"resolved_sha": "pinned-sha-123"}
        mock_resolve.assert_called_once_with(
            "OpenVoiceOS/ovos-stt-bench-x", revision="pinned-sha-123"
        )

    def test_falls_back_to_default_revision_when_no_pin(self):
        ds = MagicMock(predictions_hf="OpenVoiceOS/ovos-stt-bench-x",
                        predictions_revision=None)
        with patch("registry.loaders.list_datasets", return_value=[ds]), \
             patch("arena.predictions.resolve_predictions_revision",
                   return_value="resolved-main-sha") as mock_resolve:
            revision, meta = _predictions_revision_for(
                "OpenVoiceOS/ovos-stt-bench-x", "main"
            )
        assert revision == "resolved-main-sha"
        mock_resolve.assert_called_once_with(
            "OpenVoiceOS/ovos-stt-bench-x", revision="main"
        )

    def test_resolution_failure_propagates_instead_of_floating_the_ref(self):
        """A resolution failure that isn't the unpinned-``RevisionNotFoundError``
        case (see the test above) has no known non-transient, non-fatal
        cause — silently falling back here would ship a board whose
        provenance claims a resolved commit but is actually the floating
        ref. It must propagate so the caller records the source as
        failed instead."""
        with patch("registry.loaders.list_datasets", return_value=[]), \
             patch("arena.predictions.resolve_predictions_revision",
                   side_effect=RuntimeError("network down")):
            with pytest.raises(RuntimeError, match="network down"):
                _predictions_revision_for("OpenVoiceOS/ovos-stt-bench-x", "main")


class TestAssembleEmbedsRevisions:
    def test_board_carries_resolved_revisions(self, tmp_path, monkeypatch):
        import argparse

        from arena.models import PredictionRow

        predictions_dir = tmp_path / "preds"
        predictions_dir.mkdir()
        (predictions_dir / "comp-a.jsonl").write_text(
            PredictionRow(
                competitor_id="comp-a", sample_id="s1", dataset_id="ds1",
                lang="en-US", plugin_id="p", modality="stt",
                reference_text="hello", prediction="hello",
            ).model_dump_json() + "\n"
        )

        out_dir = tmp_path / "out"
        args = argparse.Namespace(
            predictions=str(predictions_dir),
            revision="main",
            output=str(out_dir),
            modality="",
            max_battles=200,
        )
        # local dir source — no HF resolution should happen, and no
        # predictions_revisions should be recorded for it.
        rc = cmd_assemble(args)
        assert rc == 0
        board_path = out_dir / "benchmark-stt-ds1-en-US.json"
        assert board_path.exists()
        import json
        payload = json.loads(board_path.read_text())
        assert payload.get("predictions_revisions") in (None, {})
