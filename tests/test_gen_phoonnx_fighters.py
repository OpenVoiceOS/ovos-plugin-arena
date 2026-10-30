"""The phoonnx fighter generator: one fighter per distinct voice, each with a prompt set."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from registry.loaders import REGISTRY_ROOT, list_datasets
from runner.media_bench import _lang_matches

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gen_phoonnx_fighters.py"
_spec = importlib.util.spec_from_file_location("gen_phoonnx_fighters", MODULE_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

PIPER = {"engine": "piper", "phoneme_type": "espeak", "alphabet": "ipa"}


def _voice(voice_id: str, url: str, lang: str, **extra) -> tuple[str, dict, list[str]]:
    meta = {"voice_id": voice_id, "model_url": url, "lang": lang, **extra}
    return voice_id, meta, gen.normalize_langs(lang)


class TestDedupeVoices:
    def test_one_upload_under_four_ids_keeps_the_curated_entry(self):
        voices = [
            _voice("hf_community/agentvibes/piper-custom-voices/kristin", "u1", "en-US", **PIPER),
            _voice("piper/en_US-kristin-medium", "u2", "en-US", **PIPER),
            _voice("piper_community/brycebeattie/en-US_Kristin", "u3", "en-US", **PIPER),
            _voice("piper_community/agentvibe/en-US_kristin", "u4", "en-US", **PIPER),
        ]
        kept, duplicates, unreachable = gen.dedupe_voices(
            voices, {"u1": "a", "u2": "a", "u3": "a", "u4": "a"}.get
        )
        assert kept == ["piper_community/agentvibe/en-US_kristin"]
        assert duplicates == {v[0]: "piper_community/agentvibe/en-US_kristin" for v in voices[:3]}
        assert unreachable == []

    def test_a_reupload_of_a_stock_piper_voice_keeps_the_piper_entry(self):
        voices = [
            _voice("piper_community/brycebeattie/en-GB_cori-high", "u1", "en-GB", **PIPER),
            _voice("piper/en_GB-cori-high", "u2", "en-GB", **PIPER),
        ]
        kept, duplicates, _ = gen.dedupe_voices(voices, {"u1": "a", "u2": "a"}.get)
        assert kept == ["piper/en_GB-cori-high"]
        assert duplicates == {
            "piper_community/brycebeattie/en-GB_cori-high": "piper/en_GB-cori-high"
        }

    @pytest.mark.parametrize(
        ("scrape_lang", "curated_lang"),
        [("en-US", "en-IE"), ("es-419", "es-MX"), ("cmn-CN", "zh-TW")],
    )
    def test_a_piper_model_is_one_voice_whatever_its_region_tag(self, scrape_lang, curated_lang):
        voices = [
            _voice("hf_community/someone/voice", "u1", scrape_lang, **PIPER),
            _voice("piper_community/someone/voice", "u2", curated_lang, **PIPER),
        ]
        kept, duplicates, _ = gen.dedupe_voices(voices, {"u1": "a", "u2": "a"}.get)
        assert kept == ["piper_community/someone/voice"]
        assert list(duplicates) == ["hf_community/someone/voice"]

    def test_a_piper_model_in_another_language_is_another_voice(self):
        voices = [
            _voice("piper/en_US-reza_ibrahim-medium", "u1", "en-US", **PIPER),
            _voice("piper/fa_IR-reza_ibrahim-medium", "u2", "fa-IR", **PIPER),
        ]
        kept, duplicates, _ = gen.dedupe_voices(voices, {"u1": "a", "u2": "a"}.get)
        assert len(kept) == 2
        assert duplicates == {}

    def test_dialects_of_one_chatterbox_model_stay_separate(self):
        chatterbox = {"engine": "chatterbox", "speech_encoder_url": "enc"}
        voices = [
            _voice("chatterbox/lahgtna/eg", "u", "ar-EG", **chatterbox),
            _voice("chatterbox/lahgtna/sa", "u", "ar-SA", **chatterbox),
        ]
        kept, duplicates, _ = gen.dedupe_voices(voices, {"u": "a"}.get)
        assert kept == ["chatterbox/lahgtna/eg", "chatterbox/lahgtna/sa"]
        assert duplicates == {}

    def test_style_packs_on_one_model_stay_separate(self):
        voices = [
            _voice("styletts2/kokoro/af_bella", "u", "en-US", engine="styletts2", style_url="s1"),
            _voice("styletts2/kokoro/af_sky", "u", "en-US", engine="styletts2", style_url="s2"),
        ]
        kept, duplicates, _ = gen.dedupe_voices(voices, {"u": "a"}.get)
        assert len(kept) == 2
        assert duplicates == {}

    def test_an_unreachable_model_gets_no_fighter(self):
        voices = [
            _voice("chatterbox/base/en", "private", "en-US", engine="chatterbox"),
            _voice("piper/en_US-amy-low", "public", "en-US", **PIPER),
        ]
        kept, duplicates, unreachable = gen.dedupe_voices(
            voices, {"public": "a", "private": None}.get
        )
        assert kept == ["piper/en_US-amy-low"]
        assert duplicates == {}
        assert unreachable == ["chatterbox/base/en"]


class TestLangTags:
    @pytest.mark.parametrize(
        ("index_lang", "tag"),
        [
            ("no-NO", "nb-NO"),
            ("azj-Latn", "az-AZ"),
            ("swh", "sw-TZ"),
            ("tl", "tl-PH"),
            ("pt-BR", "pt-BR"),
            ("fa", "fa-IR"),
        ],
    )
    def test_index_tags_normalise_to_the_language_read(self, index_lang, tag):
        assert gen.normalize_langs(index_lang) == [tag]

    def test_the_mul_tagged_f5tts_base_reads_english_and_chinese(self):
        assert gen.VOICE_LANGS["f5tts/v1-base"] == ["en-US", "zh-CN"]


TTS_DIR = REGISTRY_ROOT / "competitors" / "tts"
PROMPT_LANGS = sorted({lang for d in list_datasets("tts") for lang in d.langs})


@pytest.mark.parametrize(
    "competitor_id",
    [
        "phoonnx-chatterbox-chatterbox-multilingual-no",
        "phoonnx-f5tts-f5tts-v1-base",
        "phoonnx-mms-facebook-mms-tts-azj-script-latin-azerbaijani-north",
        "phoonnx-mms-facebook-mms-tts-swh-swahili",
        "phoonnx-mms-facebook-mms-tts-tgl-tagalog",
        "phoonnx-piper-piper-no-no-talesyntese-medium",
    ],
)
def test_retagged_voices_reach_a_prompt_set(competitor_id):
    langs = json.loads((TTS_DIR / f"{competitor_id}.json").read_text())["langs"]
    for lang in langs:
        assert any(_lang_matches(lang, p) for p in PROMPT_LANGS), lang


def test_every_tts_fighter_names_its_languages():
    """An empty ``langs`` runs a fighter on every prompt language."""
    empty = sorted(
        path.stem for path in TTS_DIR.glob("*.json") if not json.loads(path.read_text())["langs"]
    )
    assert empty == []


def test_every_tts_fighter_reaches_a_prompt_set():
    """A fighter no prompt set covers never produces a clip."""
    unreachable = sorted(
        path.stem
        for path in TTS_DIR.glob("*.json")
        if not any(
            _lang_matches(lang, p)
            for lang in json.loads(path.read_text())["langs"]
            for p in PROMPT_LANGS
        )
    )
    assert unreachable == []


HAND_AUTHORED = {
    "competitor_id": "phoonnx-hand-authored",
    "plugin": "ovos-tts-plugin-phoonnx",
    "model": "OpenVoiceOS/hand",
    "notes": "Written by hand.",
}

FAKE_INDEX = {
    "piper_community/alice/en-US_amy": {
        "model_url": "https://huggingface.co/org/voices/resolve/main/alice/model.onnx",
        "lang": "en-US",
        **PIPER,
    },
    "hf_community/alice/amy-scrape": {
        "model_url": "https://huggingface.co/org/voices/resolve/main/scrape/model.onnx",
        "lang": "en-GB",
        **PIPER,
    },
    "piper_community/bob/kbd-voice": {
        "model_url": "https://huggingface.co/org/voices/resolve/main/bob/model.onnx",
        "lang": "kbd-RU",
        **PIPER,
    },
    "piper_community/carol/pt-PT_ana": {
        "model_url": "https://huggingface.co/org/voices/resolve/main/carol/model.onnx",
        "lang": "pt-PT",
        **PIPER,
    },
}
FAKE_SHA = {
    "alice/model.onnx": "sha-amy",
    "scrape/model.onnx": "sha-amy",
    "bob/model.onnx": "sha-bob",
    "carol/model.onnx": "sha-carol",
}


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """A four-voice index, plus a fighter dir with one stale and one hand-authored file."""
    index = tmp_path / "voice_index"
    index.mkdir()
    (index / "piper_community.json").write_text(json.dumps(FAKE_INDEX))
    tts = tmp_path / "tts"
    tts.mkdir()
    (tts / "phoonnx-hand-authored.json").write_text(json.dumps(HAND_AUTHORED))
    stale = {"competitor_id": "phoonnx-piper-community-gone", "notes": gen.GENERATED_NOTES_MARKER}
    (tts / "phoonnx-piper-community-gone.json").write_text(json.dumps(stale))
    monkeypatch.setattr(gen, "TTS_DIR", str(tts))
    return index, tts


def test_main_writes_one_fighter_per_voice_with_prompts(fake_registry, monkeypatch):
    """The scrape duplicate, the kbd voice (no prompt set) and the stale file all go."""
    index, tts = fake_registry

    def fake_fetch(urls):
        return {url: FAKE_SHA[url.split("/main/")[1]] for url in urls}

    monkeypatch.setattr(gen, "fetch_model_sha256", fake_fetch)
    gen.main(["--voice-index", str(index)])
    assert sorted(p.stem for p in tts.glob("*.json")) == [
        "phoonnx-hand-authored",
        "phoonnx-piper-community-piper-community-alice-en-us-amy",
        "phoonnx-piper-community-piper-community-carol-pt-pt-ana",
    ]
    assert json.loads((tts / "phoonnx-hand-authored.json").read_text()) == HAND_AUTHORED

    before = {p.name: p.read_text() for p in tts.glob("*.json")}
    gen.main(["--voice-index", str(index)])
    assert {p.name: p.read_text() for p in tts.glob("*.json")} == before


def test_a_rate_limited_hub_stops_the_run_before_any_file_changes(fake_registry, tmp_path):
    index, tts = fake_registry
    before = {p.name: p.read_text() for p in tts.glob("*.json")}
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
import importlib.util, httpx, huggingface_hub
from huggingface_hub.errors import HfHubHTTPError

def rate_limited(self, repo_id, paths, **kwargs):
    request = httpx.Request("POST", f"https://huggingface.co/api/models/{{repo_id}}/paths-info")
    raise HfHubHTTPError("rate limited", response=httpx.Response(429, request=request))

huggingface_hub.HfApi.get_paths_info = rate_limited
spec = importlib.util.spec_from_file_location("gen", {str(MODULE_PATH)!r})
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
gen.TTS_DIR = {str(tts)!r}
gen.main(["--voice-index", {str(index)!r}])
"""
    )
    result = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "429" in result.stderr
    assert {p.name: p.read_text() for p in tts.glob("*.json")} == before
