"""The funding block is defined once and cannot be dropped on the way out."""
import pytest

from runner.dataset_cards import (
    FUNDING_BLOCK,
    MissingFundingBlock,
    check_funding_block,
)


class _Source:
    hf_id = "OpenVoiceOS/intents-for-eval"


class _EvalDef:
    source = _Source


class _Adapter:
    modality = "stt"
    card_tags = ("stt",)
    card_task = "Speech-to-text"


def test_the_block_names_the_fund_the_grant_and_the_programme():
    assert "NGI0 Commons Fund" in FUNDING_BLOCK
    assert "https://nlnet.nl" in FUNDING_BLOCK
    assert "101135429" in FUNDING_BLOCK
    assert "Next Generation Internet" in FUNDING_BLOCK


def test_a_card_without_the_block_is_refused():
    card = "---\nlicense: apache-2.0\n---\n\nAn OVOS Plugin Arena benchmark repository.\n"
    with pytest.raises(MissingFundingBlock) as excinfo:
        check_funding_block(card, "OpenVoiceOS/ovos-intent-bench-meteocat")
    assert "ovos-intent-bench-meteocat" in str(excinfo.value)


def test_a_card_carrying_the_block_passes_through_unchanged():
    card = f"# a card\n\n{FUNDING_BLOCK}\n"
    assert check_funding_block(card, "repo") is card


def test_both_publishers_produce_a_card_the_guard_accepts():
    from runner.intent_bench import _dataset_card
    from runner.media_bench import dataset_card

    intent = _dataset_card("intent", "intents-for-eval", _EvalDef, ["en-US"])
    media = dataset_card(_Adapter, "minds14", _EvalDef, ["en-US"])
    assert check_funding_block(intent, "repo") is intent
    assert check_funding_block(media, "repo") is media


class _RecordingApi:
    """Stands in for ``HfApi``, remembering what a publisher tried to upload."""

    def __init__(self, repo_files=()):
        self.uploaded_files = []
        self.uploaded_folders = []
        self._repo_files = list(repo_files)

    def create_repo(self, *a, **kw):
        pass

    def list_repo_files(self, repo, repo_type=None):
        return list(self._repo_files)

    def upload_file(self, **kw):
        self.uploaded_files.append(kw)

    def upload_folder(self, **kw):
        self.uploaded_folders.append(kw)


def test_a_publisher_refuses_to_upload_a_card_missing_the_block(monkeypatch, tmp_path):
    """The guard has to stop the upload, not merely describe the rule.

    A card pass has already replaced a correct card with one naming no funder.
    The publisher must fail where the card is written.
    """
    import huggingface_hub

    from runner import intent_bench

    api = _RecordingApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **kw: api)
    monkeypatch.setattr(
        intent_bench, "_dataset_card",
        lambda *a, **kw: "---\nlicense: apache-2.0\n---\n\nA benchmark repo.\n")

    bench_dir = tmp_path / "bench"
    (bench_dir / "intent" / "en-US").mkdir(parents=True)
    (bench_dir / "intent" / "en-US" / "fighter.jsonl").write_text("{}\n")

    with pytest.raises(MissingFundingBlock):
        intent_bench.upload_predictions(bench_dir, "intents-for-eval", _EvalDef)
    assert api.uploaded_files == []
    assert api.uploaded_folders == []


class TestSampleSetCard:
    """The card the manifest publisher writes."""

    def test_a_manifest_repo_is_not_described_as_holding_predictions(self):
        from runner.dataset_cards import sample_set_card

        card = sample_set_card("minds14", "en-US", "OpenVoiceOS/minds14")
        assert "holds manifests rather than predictions" in card
        assert "predictions/<lang>/" not in card


class TestSampleSetPublisherWritesACard:
    """The path that creates manifest repos is the path that must card them."""

    class _Source:
        hf_id = "OpenVoiceOS/minds14"

    class _Modality:
        value = "stt"

    class _DatasetDef:
        dataset_id = "minds14"
        lang = "en-US"
        source = None
        modality = None

    def _dataset_def(self):
        d = self._DatasetDef()
        d.source = self._Source()
        d.modality = self._Modality()
        return d

    class _Api:
        def __init__(self, existing=None):
            self.existing = existing
            self.uploaded = []

        def hf_hub_download(self, repo, path, repo_type=None):
            if self.existing is None:
                raise FileNotFoundError(path)
            import tempfile
            f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
            f.write(self.existing)
            f.close()
            return f.name

        def upload_file(self, **kw):
            self.uploaded.append(kw)

    def test_a_repo_with_no_card_gets_one(self):
        from runner.publish_sample_set import _write_card

        api = self._Api(existing=None)
        _write_card(api, "OpenVoiceOS/ovos-stt-bench-minds14", self._dataset_def())
        assert len(api.uploaded) == 1
        assert api.uploaded[0]["path_in_repo"] == "README.md"
        assert FUNDING_BLOCK in api.uploaded[0]["path_or_fileobj"].decode()

    def test_a_prediction_publishers_card_is_left_alone(self):
        """This job runs weekly; without the check it would overwrite the
        richer prediction card every time."""
        from runner.media_bench import dataset_card
        from runner.publish_sample_set import _write_card

        prediction_card = dataset_card(_Adapter, "minds14", _EvalDef, ["en-US"])
        api = self._Api(existing=prediction_card)
        _write_card(api, "OpenVoiceOS/ovos-stt-bench-minds14", self._dataset_def())
        assert api.uploaded == []

    def test_an_unchanged_card_is_not_rewritten(self):
        from runner.dataset_cards import sample_set_card
        from runner.publish_sample_set import _write_card

        current = sample_set_card("minds14", "en-US", "OpenVoiceOS/minds14")
        api = self._Api(existing=current)
        _write_card(api, "OpenVoiceOS/ovos-stt-bench-minds14", self._dataset_def())
        assert api.uploaded == []

    def test_publishing_a_manifest_cards_the_repo(self, monkeypatch):
        """The call site, not just the helper: a publish writes both files.

        The other cells here drive ``_write_card`` directly, so they stay
        green if the call is dropped from :func:`publish_sample_set`.
        """
        import huggingface_hub

        from runner import publish_sample_set as mod

        api = self._Api(existing=None)
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)
        monkeypatch.setattr(api, "create_repo", lambda *a, **k: None,
                            raising=False)
        monkeypatch.setattr(mod, "compute_sample_set", lambda *a, **k: {
            "sample_ids": ["1"], "total_rows": 1, "seed": 0, "max_samples": 1})

        mod.publish_sample_set(self._dataset_def(), "rev", "OpenVoiceOS",
                               dry_run=False)

        written = {u["path_in_repo"] for u in api.uploaded}
        assert "README.md" in written
