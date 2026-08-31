"""Tests for the full-sweep queue generator (``runner.queue_tools``).

Uses a fixture mini-registry written under ``tmp_path`` and a fake
``HFLister`` — no network calls, no real HuggingFace access.
"""
from __future__ import annotations

import json

import pytest

from runner.queue_tools import (
    HFLister,
    dataset_langs,
    engine_weight,
    enumerate_pairs,
    find_missing_pairs,
    is_compatible,
    render_dry_run_table,
    render_queue_yaml,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class FakeLister:
    """In-memory HFLister — files: {repo_id: {path: size}}, rows: {(repo,path): n}."""

    def __init__(self, files: dict[str, dict[str, int]], rows: dict[tuple, int] | None = None):
        self.files = files
        self.rows = rows or {}
        self.list_calls: list[str] = []
        self.count_calls: list[tuple] = []

    def list_files(self, repo_id: str) -> dict[str, int]:
        self.list_calls.append(repo_id)
        return dict(self.files.get(repo_id, {}))

    def count_rows(self, repo_id: str, path_in_repo: str) -> int:
        self.count_calls.append((repo_id, path_in_repo))
        return self.rows.get((repo_id, path_in_repo), 0)


@pytest.fixture()
def mini_registry(tmp_path):
    """A tiny stt registry: 3 competitors, 2 datasets.

    - vosk-en: langs=[en-US]        -> compatible with minds14-en-US only
    - fasterwhisper-multi: langs=[] -> compatible with everything
    - vosk-pt: langs=[pt-PT]        -> compatible with minds14-pt-PT only
    """
    root = tmp_path

    _write(
        root / "competitors" / "stt" / "vosk-en.json",
        {
            "competitor_id": "vosk-en",
            "modality": "stt",
            "plugin": "ovos-stt-plugin-vosk",
            "langs": ["en-US"],
        },
    )
    _write(
        root / "competitors" / "stt" / "fasterwhisper-multi.json",
        {
            "competitor_id": "fasterwhisper-multi",
            "modality": "stt",
            "plugin": "ovos-stt-plugin-fasterwhisper",
            "langs": [],
        },
    )
    _write(
        root / "competitors" / "stt" / "vosk-pt.json",
        {
            "competitor_id": "vosk-pt",
            "modality": "stt",
            "plugin": "ovos-stt-plugin-vosk",
            "langs": ["pt-PT"],
        },
    )
    # non-eval competitor of a different modality — must never leak into stt pairs
    _write(
        root / "competitors" / "vad" / "webrtc.json",
        {
            "competitor_id": "webrtc-vad",
            "modality": "vad",
            "plugin": "ovos-vad-plugin-webrtcvad",
            "langs": [],
        },
    )

    _write(
        root / "datasets" / "stt" / "minds14-en-US.json",
        {
            "dataset_id": "minds14-en-US",
            "modality": "stt",
            "source": {"type": "huggingface", "hf_id": "PolyAI/minds14", "subset": "en-US"},
            "reference_fields": {"audio": "audio", "ground_truth": "transcription"},
            "lang": "en-US",
            "role": "eval",
            "predictions_hf": "OpenVoiceOS/ovos-stt-bench-minds14-en-US",
        },
    )
    _write(
        root / "datasets" / "stt" / "minds14-pt-PT.json",
        {
            "dataset_id": "minds14-pt-PT",
            "modality": "stt",
            "source": {"type": "huggingface", "hf_id": "PolyAI/minds14", "subset": "pt-PT"},
            "reference_fields": {"audio": "audio", "ground_truth": "transcription"},
            "lang": "pt-PT",
            "role": "eval",
            "predictions_hf": "OpenVoiceOS/ovos-stt-bench-minds14-pt-PT",
        },
    )
    # a train-role dataset must never enter the sweep
    _write(
        root / "datasets" / "stt" / "minds14-pt-PT-train.json",
        {
            "dataset_id": "minds14-pt-PT-train",
            "modality": "stt",
            "source": {"type": "huggingface", "hf_id": "PolyAI/minds14", "subset": "pt-PT"},
            "reference_fields": {"audio": "audio", "ground_truth": "transcription"},
            "lang": "pt-PT",
            "role": "train",
            "predictions_hf": "OpenVoiceOS/ovos-stt-bench-minds14-pt-PT-train",
        },
    )
    return root


# ---------------------------------------------------------------------------
# Compatibility / enumeration
# ---------------------------------------------------------------------------


class TestCompatibility:
    def test_empty_langs_is_universal(self, mini_registry):
        pairs = enumerate_pairs("stt", registry_root=mini_registry)
        fw_datasets = {ds.dataset_id for c, ds in pairs if c.competitor_id == "fasterwhisper-multi"}
        assert fw_datasets == {"minds14-en-US", "minds14-pt-PT"}

    def test_lang_mismatch_excluded(self, mini_registry):
        pairs = enumerate_pairs("stt", registry_root=mini_registry)
        vosk_en_datasets = {ds.dataset_id for c, ds in pairs if c.competitor_id == "vosk-en"}
        assert vosk_en_datasets == {"minds14-en-US"}
        vosk_pt_datasets = {ds.dataset_id for c, ds in pairs if c.competitor_id == "vosk-pt"}
        assert vosk_pt_datasets == {"minds14-pt-PT"}

    def test_train_role_dataset_excluded(self, mini_registry):
        pairs = enumerate_pairs("stt", registry_root=mini_registry)
        dataset_ids = {ds.dataset_id for _, ds in pairs}
        assert "minds14-pt-PT-train" not in dataset_ids

    def test_other_modality_excluded(self, mini_registry):
        pairs = enumerate_pairs("stt", registry_root=mini_registry)
        assert all(c.modality == "stt" for c, _ in pairs)

    def test_multi_lang_dataset_matches_any_listed_lang(self):
        from registry.schemas import CompetitorDef, DatasetDef

        comp = CompetitorDef(
            competitor_id="c", modality="stt", plugin="p", langs=["fr-FR"]
        )
        ds = DatasetDef(
            dataset_id="d",
            modality="stt",
            source={"type": "huggingface", "hf_id": "x"},
            lang="multi",
            langs=["en-US", "fr-FR"],
        )
        assert is_compatible(comp, ds)
        assert dataset_langs(ds) == ["en-US", "fr-FR"]

    def test_bare_primary_subtag_matches_full_bcp47(self):
        """Fighters and datasets both carry full BCP-47 tags, but under
        different regions of the same primary subtag (``de-AT`` vs
        ``de-DE``) — exact-string overlap silently dropped ~90% of real
        pairs. Matching must go through the primary-subtag rule the
        benches use."""
        from registry.schemas import CompetitorDef, DatasetDef

        comp = CompetitorDef(
            competitor_id="c", modality="stt", plugin="p", langs=["de-AT"]
        )
        ds = DatasetDef(
            dataset_id="d",
            modality="stt",
            source={"type": "huggingface", "hf_id": "x"},
            lang="de-DE",
        )
        assert is_compatible(comp, ds)
        # and the reverse orientation
        comp2 = CompetitorDef(
            competitor_id="c2", modality="stt", plugin="p", langs=["ca-ES"]
        )
        ds2 = DatasetDef(
            dataset_id="d2",
            modality="stt",
            source={"type": "huggingface", "hf_id": "x"},
            lang="ca-AD",
        )
        assert is_compatible(comp2, ds2)

    def test_different_primary_subtags_still_excluded(self):
        from registry.schemas import CompetitorDef, DatasetDef

        comp = CompetitorDef(
            competitor_id="c", modality="stt", plugin="p", langs=["pt-BR"]
        )
        ds = DatasetDef(
            dataset_id="d",
            modality="stt",
            source={"type": "huggingface", "hf_id": "x"},
            lang="pl-PL",
        )
        assert not is_compatible(comp, ds)


# ---------------------------------------------------------------------------
# find_missing_pairs — the diff logic
# ---------------------------------------------------------------------------


class TestFindMissingPairs:
    def test_no_file_is_missing(self, mini_registry):
        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        reasons = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp.reason for mp in missing}
        assert reasons[("vosk-en", "minds14-en-US")] == "no_file"

    def test_zero_byte_file_is_missing(self, mini_registry):
        """Adversarial case mirroring the real onnx-asr-parakeet-tdt-11b.jsonl 0-byte file."""
        lister = FakeLister(
            files={
                "OpenVoiceOS/ovos-stt-bench-minds14-en-US": {
                    "predictions/vosk-en.jsonl": 0,
                    "predictions/fasterwhisper-multi.jsonl": 500,
                },
                "OpenVoiceOS/ovos-stt-bench-minds14-pt-PT": {
                    "predictions/vosk-pt.jsonl": 500,
                    "predictions/fasterwhisper-multi.jsonl": 500,
                },
            },
            rows={
                ("OpenVoiceOS/ovos-stt-bench-minds14-en-US", "predictions/fasterwhisper-multi.jsonl"): 200,
                ("OpenVoiceOS/ovos-stt-bench-minds14-pt-PT", "predictions/vosk-pt.jsonl"): 200,
                ("OpenVoiceOS/ovos-stt-bench-minds14-pt-PT", "predictions/fasterwhisper-multi.jsonl"): 200,
            },
        )
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp for mp in missing}
        assert by_key[("vosk-en", "minds14-en-US")].reason == "empty_file"
        assert ("fasterwhisper-multi", "minds14-en-US") not in by_key
        assert ("vosk-pt", "minds14-pt-PT") not in by_key

    def test_partial_rows_below_min_rows_is_missing(self, mini_registry):
        lister = FakeLister(
            files={
                "OpenVoiceOS/ovos-stt-bench-minds14-en-US": {
                    "predictions/vosk-en.jsonl": 50,
                    "predictions/fasterwhisper-multi.jsonl": 5000,
                },
                "OpenVoiceOS/ovos-stt-bench-minds14-pt-PT": {
                    "predictions/vosk-pt.jsonl": 5000,
                    "predictions/fasterwhisper-multi.jsonl": 5000,
                },
            },
            rows={
                ("OpenVoiceOS/ovos-stt-bench-minds14-en-US", "predictions/vosk-en.jsonl"): 3,
                ("OpenVoiceOS/ovos-stt-bench-minds14-en-US", "predictions/fasterwhisper-multi.jsonl"): 200,
                ("OpenVoiceOS/ovos-stt-bench-minds14-pt-PT", "predictions/vosk-pt.jsonl"): 200,
                ("OpenVoiceOS/ovos-stt-bench-minds14-pt-PT", "predictions/fasterwhisper-multi.jsonl"): 200,
            },
        )
        missing = find_missing_pairs(
            "stt", registry_root=mini_registry, lister=lister, min_rows=100
        )
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp for mp in missing}
        assert by_key[("vosk-en", "minds14-en-US")].reason == "low_rows"
        assert by_key[("vosk-en", "minds14-en-US")].rows == 3
        assert ("fasterwhisper-multi", "minds14-en-US") not in by_key

    def test_no_row_check_skips_download(self, mini_registry):
        lister = FakeLister(
            files={
                "OpenVoiceOS/ovos-stt-bench-minds14-en-US": {
                    "predictions/vosk-en.jsonl": 50,
                    "predictions/fasterwhisper-multi.jsonl": 5000,
                },
                "OpenVoiceOS/ovos-stt-bench-minds14-pt-PT": {
                    "predictions/vosk-pt.jsonl": 5000,
                    "predictions/fasterwhisper-multi.jsonl": 5000,
                },
            }
        )
        missing = find_missing_pairs(
            "stt", registry_root=mini_registry, lister=lister, check_rows=False
        )
        assert missing == []
        assert lister.count_calls == []

    def test_missing_predictions_hf_repo(self, tmp_path):
        _write(
            tmp_path / "competitors" / "stt" / "c.json",
            {"competitor_id": "c", "modality": "stt", "plugin": "p", "langs": []},
        )
        _write(
            tmp_path / "datasets" / "stt" / "d.json",
            {
                "dataset_id": "d",
                "modality": "stt",
                "source": {"type": "huggingface", "hf_id": "x"},
                "lang": "en-US",
                "role": "eval",
            },
        )
        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=tmp_path, lister=lister)
        assert len(missing) == 1
        assert missing[0].reason == "no_repo"
        assert lister.list_calls == []

    def test_results_sorted_deterministically(self, mini_registry):
        lister = FakeLister(files={})
        r1 = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        r2 = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        key = lambda mps: [(mp.competitor.competitor_id, mp.dataset.dataset_id) for mp in mps]
        assert key(r1) == key(r2)

    def test_empty_modality_returns_empty(self, mini_registry):
        missing = find_missing_pairs("tts", registry_root=mini_registry, lister=FakeLister(files={}))
        assert missing == []

    def test_transient_hf_failure_aborts_instead_of_emitting_jobs(self, mini_registry):
        """A network blip/rate-limit/auth error must not be swallowed into
        '{} files' → every fighter looks 'no_file' → a queue that re-runs
        already-done work. It must propagate and abort generation."""

        class RaisingLister:
            def list_files(self, repo_id: str) -> dict[str, int]:
                raise ConnectionError("simulated network failure")

            def count_rows(self, repo_id: str, path_in_repo: str) -> int:
                raise AssertionError("must not be reached")

        with pytest.raises(ConnectionError):
            find_missing_pairs("stt", registry_root=mini_registry, lister=RaisingLister())


# ---------------------------------------------------------------------------
# Engine weight ordering
# ---------------------------------------------------------------------------


class TestEngineWeight:
    def test_vosk_cheaper_than_whisper(self):
        assert engine_weight("vosk-pt", "ovos-stt-plugin-vosk") < engine_weight(
            "fasterwhisper-small-pt", "ovos-stt-plugin-fasterwhisper"
        )

    def test_unknown_engine_gets_default_weight(self):
        from runner.queue_tools import _DEFAULT_WEIGHT

        assert engine_weight("mystery-engine", "ovos-stt-plugin-mystery") == _DEFAULT_WEIGHT

    def test_sort_places_cheap_engines_first(self, mini_registry):
        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        weights = [engine_weight(mp.competitor.competitor_id, mp.competitor.plugin) for mp in missing]
        assert weights == sorted(weights)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_dry_run_table_empty(self):
        assert "No missing" in render_dry_run_table([])

    def test_dry_run_table_lists_reason_and_rows(self, mini_registry):
        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        table = render_dry_run_table(missing)
        assert "no_file" in table
        assert "vosk-en" in table
        assert f"{len(missing)} missing" in table

    def test_queue_yaml_is_parseable_and_matches_schema(self, mini_registry):
        import yaml

        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        rendered = render_queue_yaml(missing)
        parsed = yaml.safe_load(rendered)
        assert "jobs" in parsed
        assert len(parsed["jobs"]) == len(missing)
        for job in parsed["jobs"]:
            assert set(job) == {
                "competitor", "dataset_ref", "lang", "hf_output_dataset", "max_samples",
            }
            assert job["max_samples"] == 0
            # every mini_registry dataset has a concrete lang (en-US/pt-PT)
            assert job["lang"] in {"en-US", "pt-PT"}

    def test_queue_yaml_loads_via_queue_config(self, mini_registry, tmp_path, monkeypatch):
        """Rendered YAML must actually resolve through runner.queue_config.load_queue
        against the same registry it was generated from."""
        from runner.queue_config import load_queue

        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        rendered = render_queue_yaml(missing)
        out = tmp_path / "generated_queue.yaml"
        out.write_text(rendered)

        jobs = load_queue(out, registry_root=mini_registry)
        assert len(jobs) == len(missing)
        assert {j.competitor_id for j in jobs} == {
            mp.competitor.competitor_id for mp in missing
        }


# ---------------------------------------------------------------------------
# HFLister protocol sanity
# ---------------------------------------------------------------------------


def test_fake_lister_satisfies_protocol():
    lister: HFLister = FakeLister(files={})
    assert hasattr(lister, "list_files")
    assert hasattr(lister, "count_rows")


class TestHubListerErrorHandling:
    """Unit-tests the real HubLister's exception split directly (mocking
    HfApi.list_repo_tree), since the fake-lister-based tests above bypass
    HubLister entirely and can't catch a regression there."""

    def test_repository_not_found_yields_empty_files(self, monkeypatch):
        import runner.queue_tools as qt
        from huggingface_hub.utils import RepositoryNotFoundError

        class FakeApi:
            def list_repo_tree(self, *a, **kw):
                # huggingface_hub >=1.0 makes `response` required — build a
                # real (empty) Response so this constructs on every version.
                import requests
                resp = requests.Response()
                resp.status_code = 404
                raise RepositoryNotFoundError("no such repo", response=resp)

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

        lister = qt.HubLister()
        assert lister.list_files("OpenVoiceOS/does-not-exist") == {}

    def test_connection_error_propagates_not_swallowed(self, monkeypatch):
        """The defect this guards: a transient failure must NOT be caught
        into {} (which would make every competitor look 'no_file')."""
        import runner.queue_tools as qt

        class FakeApi:
            def list_repo_tree(self, *a, **kw):
                raise ConnectionError("simulated network failure")

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

        lister = qt.HubLister()
        with pytest.raises(ConnectionError):
            lister.list_files("OpenVoiceOS/ovos-stt-bench-minds14-pt-PT")


class TestCLIAbortsOnHFFailure:
    def test_main_exits_nonzero_and_prints_error_on_hf_failure(self, mini_registry, monkeypatch, capsys):
        import runner.queue_tools as qt

        class RaisingLister:
            def list_files(self, repo_id: str) -> dict[str, int]:
                raise ConnectionError("simulated network failure")

            def count_rows(self, repo_id: str, path_in_repo: str) -> int:
                raise AssertionError("must not be reached")

        monkeypatch.setattr(qt, "HubLister", lambda: RaisingLister())

        with pytest.raises(SystemExit) as exc_info:
            qt.main(["--modality", "stt", "--registry-root", str(mini_registry)])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "simulated network failure" in captured.err
        # No partial/garbage queue output on stdout
        assert captured.out == ""


class TestFindMissingPairsLangNestedLayout:
    """Regression: the published layout nests per lang
    (predictions/<lang>/<id>.jsonl — what media_bench and the daemon write).
    A diff that checks only the flat predictions/<id>.jsonl path reports
    "no_file" forever for every published pair and re-queues finished work."""

    def test_lang_nested_file_counts_as_published(self, mini_registry):
        lister = FakeLister(
            files={
                "OpenVoiceOS/ovos-stt-bench-minds14-en-US": {
                    "predictions/en-US/vosk-en.jsonl": 500,
                },
            },
            rows={
                ("OpenVoiceOS/ovos-stt-bench-minds14-en-US",
                 "predictions/en-US/vosk-en.jsonl"): 200,
            },
        )
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp.reason
                  for mp in missing}
        assert ("vosk-en", "minds14-en-US") not in by_key

    def test_rows_aggregate_across_lang_dirs_for_multi_lang_dataset(self, tmp_path):
        """Any-lang aggregation is only correct for a genuinely multi/unknown
        -lang dataset (lang='multi') — a single-language dataset must NOT
        aggregate rows across unrelated lang dirs (see
        TestConcreteLangMatching below for why)."""
        _write(
            tmp_path / "competitors" / "stt" / "fasterwhisper-multi.json",
            {
                "competitor_id": "fasterwhisper-multi",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-fasterwhisper",
                "langs": [],
            },
        )
        _write(
            tmp_path / "datasets" / "stt" / "fleurs-multi.json",
            {
                "dataset_id": "fleurs-multi",
                "modality": "stt",
                "source": {"type": "huggingface", "hf_id": "google/fleurs"},
                "reference_fields": {"audio": "audio", "ground_truth": "transcription"},
                "lang": "multi",
                "langs": ["en-US", "en-GB", "en-AU"],
                "role": "eval",
                "predictions_hf": "OpenVoiceOS/ovos-stt-bench-fleurs-multi",
            },
        )
        repo = "OpenVoiceOS/ovos-stt-bench-fleurs-multi"
        lister = FakeLister(
            files={repo: {
                "predictions/en-US/fasterwhisper-multi.jsonl": 300,
                "predictions/en-GB/fasterwhisper-multi.jsonl": 300,
                "predictions/en-AU/fasterwhisper-multi.jsonl": 0,
            }},
            rows={
                (repo, "predictions/en-US/fasterwhisper-multi.jsonl"): 3,
                (repo, "predictions/en-GB/fasterwhisper-multi.jsonl"): 4,
            },
        )
        missing = find_missing_pairs(
            "stt", registry_root=tmp_path, lister=lister, min_rows=5)
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp
                  for mp in missing}
        assert ("fasterwhisper-multi", "fleurs-multi") not in by_key
        # 0-byte files are never row-counted
        assert (repo, "predictions/en-AU/fasterwhisper-multi.jsonl") \
            not in lister.count_calls

    def test_similar_id_in_lang_dir_does_not_leak(self, mini_registry):
        """predictions/<lang>/xx-vosk-en.jsonl must NOT satisfy vosk-en."""
        repo = "OpenVoiceOS/ovos-stt-bench-minds14-en-US"
        lister = FakeLister(files={repo: {
            "predictions/en-US/other-vosk-en-variant.jsonl": 500,
        }})
        missing = find_missing_pairs(
            "stt", registry_root=mini_registry, lister=lister, check_rows=False)
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp.reason
                  for mp in missing}
        assert by_key[("vosk-en", "minds14-en-US")] == "no_file"


class TestConcreteLangMatching:
    """Regression for the production bug: a multilingual fighter (e.g.
    onnx-asr-canary, langs=[en, de, fr, es]) queued against a single-lang
    dataset (speech-massive-de-DE) with no explicit ``lang`` resolved
    lang=en (fighter default) in ``queue_config``, ran mis-conditioned, and
    published to predictions/en/<id>.jsonl inside the de-DE dataset's repo.
    ``find_missing_pairs``' old any-lang suffix match then treated that
    wrong-lang shard as a completed pair and it was never re-queued."""

    def test_generator_emits_lang_for_lang_suffixed_dataset(self, mini_registry):
        lister = FakeLister(files={})
        missing = find_missing_pairs("stt", registry_root=mini_registry, lister=lister)
        rendered = render_queue_yaml(missing)
        import yaml
        parsed = yaml.safe_load(rendered)
        by_key = {(j["competitor"], j["dataset_ref"]): j for j in parsed["jobs"]}
        assert by_key[("vosk-en", "minds14-en-US")]["lang"] == "en-US"
        assert by_key[("vosk-pt", "minds14-pt-PT")]["lang"] == "pt-PT"

    def test_wrong_lang_published_shard_does_not_mark_pair_complete(self, tmp_path):
        """A fighter's shard published under the wrong lang dir (e.g. the
        fighter's own default "en" instead of the dataset's "de-DE") must
        still count as missing/unpublished for THIS dataset."""
        _write(
            tmp_path / "competitors" / "stt" / "canary-multi.json",
            {
                "competitor_id": "canary-multi",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-onnx-asr",
                "langs": ["en-US", "de-DE", "fr-FR", "es-ES"],
            },
        )
        _write(
            tmp_path / "datasets" / "stt" / "speech-massive-de-DE.json",
            {
                "dataset_id": "speech-massive-de-DE",
                "modality": "stt",
                "source": {"type": "huggingface", "hf_id": "FBK-MT/Speech-MASSIVE-test"},
                "reference_fields": {"audio": "audio", "ground_truth": "utt"},
                "lang": "de-DE",
                "role": "eval",
                "predictions_hf": "OpenVoiceOS/ovos-stt-bench-speech-massive-de-DE",
            },
        )
        repo = "OpenVoiceOS/ovos-stt-bench-speech-massive-de-DE"
        # Shard exists, but under the WRONG lang dir (fighter's own default
        # "en", not the dataset's "de-DE").
        lister = FakeLister(
            files={repo: {"predictions/en/canary-multi.jsonl": 500}},
            rows={(repo, "predictions/en/canary-multi.jsonl"): 200},
        )
        missing = find_missing_pairs("stt", registry_root=tmp_path, lister=lister)
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp.reason
                  for mp in missing}
        assert by_key[("canary-multi", "speech-massive-de-DE")] == "no_file"

    def test_correct_lang_published_shard_marks_pair_complete(self, tmp_path):
        """Sanity companion to the wrong-lang test: the SAME shard published
        under the dataset's own lang dir must satisfy the pair."""
        _write(
            tmp_path / "competitors" / "stt" / "canary-multi.json",
            {
                "competitor_id": "canary-multi",
                "modality": "stt",
                "plugin": "ovos-stt-plugin-onnx-asr",
                "langs": ["en-US", "de-DE", "fr-FR", "es-ES"],
            },
        )
        _write(
            tmp_path / "datasets" / "stt" / "speech-massive-de-DE.json",
            {
                "dataset_id": "speech-massive-de-DE",
                "modality": "stt",
                "source": {"type": "huggingface", "hf_id": "FBK-MT/Speech-MASSIVE-test"},
                "reference_fields": {"audio": "audio", "ground_truth": "utt"},
                "lang": "de-DE",
                "role": "eval",
                "predictions_hf": "OpenVoiceOS/ovos-stt-bench-speech-massive-de-DE",
            },
        )
        repo = "OpenVoiceOS/ovos-stt-bench-speech-massive-de-DE"
        lister = FakeLister(
            files={repo: {"predictions/de-DE/canary-multi.jsonl": 500}},
            rows={(repo, "predictions/de-DE/canary-multi.jsonl"): 200},
        )
        missing = find_missing_pairs("stt", registry_root=tmp_path, lister=lister)
        by_key = {(mp.competitor.competitor_id, mp.dataset.dataset_id): mp.reason
                  for mp in missing}
        assert ("canary-multi", "speech-massive-de-DE") not in by_key

    def test_resolved_dataset_lang_falls_back_to_id_suffix(self):
        from registry.schemas import DatasetDef
        from runner.queue_tools import resolved_dataset_lang

        ds = DatasetDef(
            dataset_id="speech-massive-de-DE",
            modality="stt",
            source={"type": "huggingface", "hf_id": "x"},
            lang="",
        )
        assert resolved_dataset_lang(ds) == "de-DE"

    def test_resolved_dataset_lang_multi_returns_none(self):
        from registry.schemas import DatasetDef
        from runner.queue_tools import resolved_dataset_lang

        ds = DatasetDef(
            dataset_id="fleurs-multi",
            modality="stt",
            source={"type": "huggingface", "hf_id": "x"},
            lang="multi",
            langs=["en-US", "de-DE"],
        )
        assert resolved_dataset_lang(ds) is None


class TestHubListerEmptyRepo:
    def test_entry_not_found_yields_empty_files(self, monkeypatch):
        """Regression: a freshly created prediction repo has no predictions/
        folder; list_repo_tree raises EntryNotFoundError, which must mean
        'nothing published yet', not a crash (live failure: queue regen died
        on ovos-stt-bench-fleurs-gl, 2026-08-11)."""
        import runner.queue_tools as qt
        from huggingface_hub.utils import EntryNotFoundError

        class FakeApi:
            def list_repo_tree(self, *a, **kw):
                raise EntryNotFoundError("predictions does not exist on main")

        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
        lister = qt.HubLister()
        assert lister.list_files("OpenVoiceOS/ovos-stt-bench-fleurs-gl") == {}
