"""A prediction card carries the licence of the test set its rows repeat, and
rows that repeat text from a no-derivatives set never reach the Hub."""
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from runner import dataset_cards, intent_bench
from runner import media_bench as mb
from runner.dataset_cards import (
    FUNDING_BLOCK,
    CardFormatError,
    RestrictedLicence,
    card_licence,
    check_redistributable,
    forbids_derivatives,
    is_noncommercial,
    is_share_alike,
    relicensed_card,
)


def _eval_def(license, hf_id="tasksource/mtop", display_name="MTOP (English)"):
    return SimpleNamespace(
        dataset_id="mtop-en-US", display_name=display_name, license=license,
        source=SimpleNamespace(hf_id=hf_id, revision="main"))


class _Adapter:
    modality = "stt"
    card_tags = ("automatic-speech-recognition",)
    card_task = "Per-clip transcripts"


def _cards(eval_def):
    return [
        intent_bench._dataset_card("intent", "mtop-en-US", eval_def, ["en-US"]),
        mb.dataset_card(_Adapter, "mtop-en-US", eval_def, ["en-US"]),
    ]


@pytest.mark.parametrize(("registry", "card_id"), [
    ("cc-by-4.0", "cc-by-4.0"),
    ("CC-BY-4.0", "cc-by-4.0"),
    ("cc-by-sa-4.0", "cc-by-sa-4.0"),
    ("cc-by-sa-3.0", "cc-by-sa-3.0"),
    ("apache-2.0", "apache-2.0"),
    ("cc-by-nc-sa-4.0", "cc-by-nc-sa-4.0"),
])
def test_the_card_carries_the_licence_of_the_source_set(registry, card_id):
    for card in _cards(_eval_def(registry)):
        assert card_licence(card) == card_id
        assert f"The rows carry the licence of that set, `{card_id}`." in card
        assert "[MTOP (English)](https://huggingface.co/datasets/tasksource/mtop)" in card
        assert "Credit the authors of `tasksource/mtop`" in card
        assert FUNDING_BLOCK in card


@pytest.mark.parametrize(("registry", "noncommercial", "share_alike"), [
    ("cc-by-4.0", False, False),
    ("cc-by-sa-4.0", False, True),
    ("cc-by-nc-4.0", True, False),
    ("cc-by-nc-sa-4.0", True, True),
])
def test_the_card_states_the_noncommercial_and_share_alike_terms(
        registry, noncommercial, share_alike):
    for card in _cards(_eval_def(registry)):
        assert ("reuse the rows for non-commercial purposes only" in card) is noncommercial
        assert ("Share any adaptation of the rows under the same licence" in card) is share_alike


@pytest.mark.parametrize("registry", ["Custom research licence v2", "other"])
def test_an_unknown_licence_becomes_other_and_the_card_quotes_it(registry):
    for card in _cards(_eval_def(registry)):
        assert card_licence(card) == "other"
        assert "The card gives `other` as the Hugging Face licence id." in card
        assert f"The source licence is: `{registry}`." in card
        assert "apache-2.0" not in card.split("\n---", 1)[0]


def test_a_set_with_no_licence_becomes_other():
    for card in _cards(_eval_def(None)):
        assert card_licence(card) == "other"
        assert "The registry records no licence for this set." in card


class _RecordingApi:
    def __init__(self):
        self.uploaded_files = []
        self.uploaded_folders = []
        self.created = []

    def create_repo(self, repo, **kw):
        self.created.append(repo)

    def upload_file(self, **kw):
        self.uploaded_files.append(kw)

    def upload_folder(self, **kw):
        self.uploaded_folders.append(kw)


@pytest.fixture
def api(monkeypatch):
    api = _RecordingApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **kw: api)
    return api


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("registry", ["cc-by-nd-4.0", "cc-by-nc-nd-4.0"])
def test_intent_rows_from_an_nd_set_are_refused(api, tmp_path, registry):
    _write_rows(tmp_path / "intent" / "en-US" / "fighter.jsonl",
                [{"utterance": "wake me at six", "prediction": "alarm:set"}])
    with pytest.raises(RestrictedLicence) as excinfo:
        intent_bench.upload_predictions(tmp_path, "mtop-en-US", _eval_def(registry))
    assert "'utterance'" in str(excinfo.value)
    assert (api.created, api.uploaded_files, api.uploaded_folders) == ([], [], [])


@pytest.mark.parametrize("registry", ["cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-sa-4.0"])
def test_intent_rows_from_an_nc_or_sa_set_publish_under_its_licence(api, tmp_path, registry):
    _write_rows(tmp_path / "intent" / "en-US" / "fighter.jsonl",
                [{"utterance": "wake me at six", "prediction": "alarm:set"}])
    intent_bench.upload_predictions(tmp_path, "mtop-en-US", _eval_def(registry))
    [card] = api.uploaded_files
    assert card_licence(card["path_or_fileobj"].decode()) == registry
    assert len(api.uploaded_folders) == 1


@pytest.mark.parametrize("registry", ["cc-by-nc-4.0", "cc-by-nc-sa-4.0"])
def test_stt_rows_from_an_nc_set_publish_under_its_licence(api, tmp_path, registry):
    _write_rows(tmp_path / "stt" / "predictions" / "ca-ES" / "fighter.jsonl",
                [{"reference_text": "bon dia", "prediction": "bon dia"}])
    mb.upload_predictions(_Adapter, tmp_path, "lafrescat-ca-ES", _eval_def(registry))
    [card] = api.uploaded_files
    text = card["path_or_fileobj"].decode()
    assert card_licence(text) == registry
    assert "reuse the rows for non-commercial purposes only" in text
    assert len(api.uploaded_folders) == 1


@pytest.mark.parametrize("registry", ["cc-by-nd-4.0", "CC BY-NC-ND 4.0"])
def test_stt_rows_from_an_nd_set_are_refused(api, tmp_path, registry):
    _write_rows(tmp_path / "stt" / "predictions" / "en-US" / "fighter.jsonl",
                [{"reference_text": "", "prediction": "a"},
                 {"reference_text": "the cat sat", "prediction": "the cat"}])
    with pytest.raises(RestrictedLicence) as excinfo:
        mb.upload_predictions(_Adapter, tmp_path, "mtedx-fr-FR", _eval_def(registry))
    assert "'reference_text'" in str(excinfo.value)
    assert (api.created, api.uploaded_files, api.uploaded_folders) == ([], [], [])


def test_rows_with_no_text_from_an_nd_set_are_published(api, tmp_path):
    """Voice-activity rows hold a label and a link to the source clip only."""

    class _Vad(_Adapter):
        modality = "vad"

    _write_rows(tmp_path / "vad" / "predictions" / "ar-SA" / "fighter.jsonl",
                [{"label": "speech", "prediction": "speech",
                  "audio_url": "https://huggingface.co/x.wav"}])
    mb.upload_predictions(_Vad, tmp_path, "speech-vs-nonspeech-ar-SA",
                          _eval_def("cc-by-nc-nd-4.0", hf_id="FBK-MT/Speech-MASSIVE-test"))
    [card] = api.uploaded_files
    assert card_licence(card["path_or_fileobj"].decode()) == "cc-by-nc-nd-4.0"
    assert len(api.uploaded_folders) == 1


def test_rows_from_an_open_set_are_published_with_its_licence(api, tmp_path):
    _write_rows(tmp_path / "intent" / "en-US" / "fighter.jsonl",
                [{"utterance": "wake me at six", "prediction": "alarm:set"}])
    intent_bench.upload_predictions(tmp_path, "mtop-en-US", _eval_def("cc-by-sa-4.0"))
    [card] = api.uploaded_files
    assert card_licence(card["path_or_fileobj"].decode()) == "cc-by-sa-4.0"


GENERIC_CARD = """---
license: apache-2.0
tags:
  - openvoiceos
---

An OVOS Plugin Arena benchmark repository.
"""


def test_relicensing_a_generic_card_sets_the_licence_and_restores_the_funder():
    card = relicensed_card(GENERIC_CARD, _eval_def("cc-by-sa-4.0"), "repo")
    assert card_licence(card) == "cc-by-sa-4.0"
    assert "  - openvoiceos\n" in card
    assert "An OVOS Plugin Arena benchmark repository." in card
    assert card.index("## Licence") < card.index(FUNDING_BLOCK)


def test_relicensing_a_generated_card_keeps_its_body_and_is_idempotent():
    old = _cards(_eval_def("apache-2.0"))[0]
    new = relicensed_card(old, _eval_def("cc-by-sa-4.0"), "repo")
    assert card_licence(new) == "cc-by-sa-4.0"
    assert new.count("## Licence") == 1
    assert "`apache-2.0`" not in new
    assert new == _cards(_eval_def("cc-by-sa-4.0"))[0]
    assert relicensed_card(new, _eval_def("cc-by-sa-4.0"), "repo") == new


def test_relicensing_a_card_with_no_front_matter_adds_one():
    card = relicensed_card("# rows\n", _eval_def("cc-by-4.0"), "repo")
    assert card.startswith("---\nlicense: cc-by-4.0\n---\n")


class _HubApi(_RecordingApi):
    def __init__(self, repos, files):
        super().__init__()
        self._repos = repos
        self._files = files

    def list_datasets(self, author):
        return [SimpleNamespace(id=r) for r in self._repos]

    def list_repo_files(self, repo, repo_type=None):
        return [f for (r, f) in self._files if r == repo]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    from registry import loaders

    real = loaders.load_all_datasets()
    nd = next(d for d in real if d.dataset_id == "edacc-en-GB").model_copy(
        update={"license": "cc-by-nc-nd-4.0"})
    monkeypatch.setattr(loaders, "load_all_datasets",
                        lambda: [nd if d.dataset_id == "edacc-en-GB" else d for d in real])
    files = {
        ("OpenVoiceOS/ovos-stt-bench-edacc-en-GB", "README.md"): GENERIC_CARD,
        ("OpenVoiceOS/ovos-stt-bench-edacc-en-GB", "predictions/en-GB/f.jsonl"):
            json.dumps({"reference_text": "hello"}) + "\n",
        ("OpenVoiceOS/ovos-intent-bench-mtop-en-US", "README.md"): GENERIC_CARD,
        ("OpenVoiceOS/ovos-stt-bench-lafrescat-ca-ES", "README.md"): GENERIC_CARD,
        ("OpenVoiceOS/ovos-stt-bench-lafrescat-ca-ES", "predictions/ca-ES/f.jsonl"):
            json.dumps({"reference_text": "bon dia"}) + "\n",
        ("OpenVoiceOS/ovos-intent-bench-meteocat", "README.md"): GENERIC_CARD,
    }
    api = _HubApi(sorted({r for (r, _) in files} | {"OpenVoiceOS/ww-stream-bench"}),
                  list(files))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **kw: api)

    def download(repo, filename, repo_type=None):
        path = tmp_path / "hub" / repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[(repo, filename)])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return api


def test_the_rewrite_command_writes_nothing_by_default(hub, capsys):
    assert dataset_cards.main([]) == 0
    out = capsys.readouterr().out
    assert hub.uploaded_files == []
    assert ("ovos-intent-bench-mtop-en-US: apache-2.0 -> cc-by-sa-4.0 "
            "[source mtop-en-US, registry 'cc-by-sa-4.0'] rewrite") in out
    assert ("ovos-stt-bench-lafrescat-ca-ES: apache-2.0 -> cc-by-nc-4.0 "
            "[source lafrescat-ca-ES, registry 'cc-by-nc-4.0'] rewrite\n") in out
    assert ("ovos-stt-bench-edacc-en-GB: apache-2.0 -> cc-by-nc-nd-4.0 "
            "[source edacc-en-GB, registry 'cc-by-nc-nd-4.0'] rewrite NO-DERIVATIVES: "
            "predictions/en-GB/f.jsonl republishes text in 'reference_text'; "
            "withdraw the rows") in out
    assert out.count("NO-DERIVATIVES") == 1
    assert "4 in, 3 matched and done, 0 failed, 1 unmatched" in out
    assert "  OpenVoiceOS/ovos-intent-bench-meteocat" in out


def test_the_rewrite_command_uploads_the_cards_with_apply(hub):
    dataset_cards.main(["--apply"])
    written = {f["repo_id"]: f["path_or_fileobj"].decode() for f in hub.uploaded_files}
    assert sorted(written) == ["OpenVoiceOS/ovos-intent-bench-mtop-en-US",
                               "OpenVoiceOS/ovos-stt-bench-edacc-en-GB",
                               "OpenVoiceOS/ovos-stt-bench-lafrescat-ca-ES"]
    assert card_licence(written["OpenVoiceOS/ovos-intent-bench-mtop-en-US"]) == "cc-by-sa-4.0"
    assert card_licence(written["OpenVoiceOS/ovos-stt-bench-lafrescat-ca-ES"]) == "cc-by-nc-4.0"


@pytest.mark.parametrize(("raw", "nd", "nc", "sa"), [
    ("No-Derivatives", True, False, False),
    ("NoDerivatives", True, False, False),
    ("no derivs", True, False, False),
    ("Attribution-NonCommercial-NoDerivatives 4.0", True, True, False),
    ("Non-Commercial", False, True, False),
    ("NonCommercial", False, True, False),
    ("non commercial use only", False, True, False),
    ("ShareAlike", False, False, True),
    ("Share-Alike", False, False, True),
    ("share alike 4.0", False, False, True),
    ("CC BY-NC-SA 4.0", False, True, True),
    ("cc-by-4.0", False, False, False),
    ("NCSA", False, False, False),
    ("Standard licence", False, False, False),
    (None, False, False, False),
])
def test_licence_terms_match_codes_and_full_names_in_any_spelling(raw, nd, nc, sa):
    assert (forbids_derivatives(raw), is_noncommercial(raw), is_share_alike(raw)) == (nd, nc, sa)


def test_a_spelled_out_nd_licence_is_refused(tmp_path):
    _write_rows(tmp_path / "f.jsonl", [{"reference_text": "the cat sat"}])
    with pytest.raises(RestrictedLicence):
        check_redistributable(_eval_def("Creative Commons Attribution-No-Derivatives"),
                              tmp_path, "repo")


def test_tts_prompts_from_an_nd_set_are_refused_before_the_repo_exists(api, tmp_path):
    class _Tts(_Adapter):
        modality = "tts"

    _write_rows(tmp_path / "tts" / "predictions" / "en-US" / "voice.jsonl",
                [{"input_text": "turn on the lights",
                  "prediction": "https://huggingface.co/x.wav"}])
    with pytest.raises(RestrictedLicence) as excinfo:
        mb.upload_predictions(_Tts, tmp_path, "prompts-en-US", _eval_def("cc-by-nd-4.0"))
    assert "'input_text'" in str(excinfo.value)
    assert (api.created, api.uploaded_files, api.uploaded_folders) == ([], [], [])


def test_a_list_form_licence_is_replaced_whole():
    card = "---\nlicense:\n- apache-2.0\n- mit\ntags:\n  - x\n---\n\nbody\n"
    new = relicensed_card(card, _eval_def("cc-by-sa-4.0"), "repo")
    front = new.split("\n---", 1)[0]
    assert front == "---\nlicense: cc-by-sa-4.0\ntags:\n  - x"


def test_a_crlf_card_is_rewritten_with_one_front_matter_block():
    card = "---\r\nlicense: apache-2.0\r\n---\r\n\r\nbody\r\n"
    new = relicensed_card(card, _eval_def("cc-by-4.0"), "repo")
    assert new.count("---\n") == 2
    assert card_licence(new) == "cc-by-4.0"
    assert "\r" not in new


def test_an_unterminated_front_matter_is_refused():
    with pytest.raises(CardFormatError):
        relicensed_card("---\nlicense: apache-2.0\nbody\n", _eval_def("cc-by-4.0"), "repo")


def test_the_rewrite_command_reports_a_bad_card_and_carries_on(hub, capsys, monkeypatch):
    import huggingface_hub as hh

    download = hh.hf_hub_download

    def broken_for_mtop(repo, filename, repo_type=None):
        path = download(repo, filename, repo_type=repo_type)
        if repo.endswith("mtop-en-US") and filename == "README.md":
            Path(path).write_text("---\nlicense: apache-2.0\nno end\n")
        return path

    monkeypatch.setattr(hh, "hf_hub_download", broken_for_mtop)
    assert dataset_cards.main(["--apply"]) == 1
    out = capsys.readouterr().out
    assert "ovos-intent-bench-mtop-en-US: ERROR, card left as it is" in out
    assert sorted(f["repo_id"] for f in hub.uploaded_files) == [
        "OpenVoiceOS/ovos-stt-bench-edacc-en-GB",
        "OpenVoiceOS/ovos-stt-bench-lafrescat-ca-ES"]
    assert "4 in, 2 matched and done, 1 failed, 1 unmatched" in out


def test_autorun_skips_text_leagues_of_an_nd_set(monkeypatch):
    from runner import autorun

    def ds(did, modality, license):
        return SimpleNamespace(dataset_id=did, modality=modality, role="eval", license=license)

    sets = [ds("nd-stt", "stt", "cc-by-nc-nd-4.0"), ds("by-stt", "stt", "cc-by-4.0"),
            ds("nd-vad", "vad", "cc-by-nd-4.0")]
    fighters = [SimpleNamespace(competitor_id=f"{m}-f", modality=m, langs=[], trained_on=[])
                for m in ("stt", "vad")]
    monkeypatch.setattr(autorun, "load_all_competitors", lambda registry_root=None: fighters)
    monkeypatch.setattr(autorun, "load_all_datasets", lambda registry_root=None: sets)
    monkeypatch.setattr(autorun, "is_compatible", lambda c, d: True)
    monkeypatch.setattr(autorun, "dataset_langs", lambda d: ["en-US"])
    adapter = SimpleNamespace(filter_competitors=lambda comps: comps,
                              competitor_langs=lambda comp, langs: langs)
    monkeypatch.setattr(autorun, "adapter_factories",
                        lambda: {"stt": lambda: adapter, "vad": lambda: adapter})
    pairs = autorun.enumerate_all_pairs(["stt", "vad"])
    assert sorted(d.dataset_id for _, _, d, _ in pairs) == ["by-stt", "nd-vad"]


def test_autorun_drops_a_refused_pair_and_never_uploads_it_again(monkeypatch, tmp_path):
    from runner import autorun

    calls = []

    def refuse(*a, **kw):
        calls.append(a)
        raise RestrictedLicence("refused")

    monkeypatch.setattr(autorun.mb, "upload_predictions", refuse)
    monkeypatch.setattr(autorun.mb, "plugin_is_installed", lambda modality, plugin: True)
    monkeypatch.setattr(autorun, "adapter_factories", lambda: {"stt": lambda: _Adapter})
    runner = autorun.AutoRunner(
        autorun.AutoRunConfig(output_dir=tmp_path, batch=10, upload=True, seed_from_hf=False),
        process_fn=lambda *a, **kw: mb.BatchResult(written=3, errored=0))
    fighter = SimpleNamespace(competitor_id="c", plugin="c", size=None)
    nd = _eval_def("cc-by-nd-4.0")
    for _ in range(2):
        runner.process_pair("stt", fighter, nd, "en-US")
        runner.flush_all(force=True)
    assert len(calls) == 1
    assert ("stt", nd.dataset_id) not in runner._dirty


def test_backfill_refuses_nd_tts_rows(monkeypatch, tmp_path):
    from runner import backfill_tts_predictions as backfill

    uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi",
                        lambda *a, **kw: SimpleNamespace(upload_file=uploads.append))
    _write_rows(tmp_path / "voice.jsonl", [{"input_text": "turn on the lights"}])
    with pytest.raises(RestrictedLicence):
        backfill.upload_pair(_eval_def("cc-by-nd-4.0"), "repo", tmp_path / "voice.jsonl",
                             "en-US", "voice")
    assert uploads == []


def test_rescore_refuses_nd_tts_rows(monkeypatch, tmp_path):
    from runner import rescore_tts

    uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi",
                        lambda *a, **kw: SimpleNamespace(upload_folder=uploads.append))
    _write_rows(tmp_path / "predictions" / "en-US" / "voice.jsonl",
                [{"input_text": "turn on the lights"}])
    with pytest.raises(RestrictedLicence):
        rescore_tts.upload_rescored(_eval_def("cc-by-nd-4.0"), "repo", tmp_path)
    assert uploads == []


@pytest.mark.parametrize(("repo", "refused"), [
    ("OpenVoiceOS/ovos-stt-bench-mtedx-fr-FR", True),
    ("OpenVoiceOS/ovos-stt-bench-edacc-en-GB", False),
])
def test_the_stt_migration_refuses_rows_of_a_repo_with_no_known_licence(
        monkeypatch, tmp_path, repo, refused):
    from runner.migrate import stt_rows_to_prediction_rows as migrate

    uploads = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **kw: SimpleNamespace(
        upload_file=lambda **kw: uploads.append(kw),
        dataset_info=lambda repo, revision: SimpleNamespace(sha="abc")))
    _write_rows(tmp_path / "f.jsonl", [{"reference_text": "bonjour"}])
    if refused:
        with pytest.raises(RestrictedLicence):
            migrate.push_migrated(repo, {"f.jsonl": tmp_path / "f.jsonl"})
        assert uploads == []
    else:
        assert migrate.push_migrated(repo, {"f.jsonl": tmp_path / "f.jsonl"}) == "abc"
        assert len(uploads) == 1
