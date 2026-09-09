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
