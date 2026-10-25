"""The TigreGotico wake-word test sets and the WakeHuBERT fighters trained on them."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from registry.loaders import list_competitors, list_datasets

_spec = importlib.util.spec_from_file_location(
    "gen_ww_model_fighters", Path(__file__).parent.parent / "scripts" / "gen_ww_model_fighters.py")
gen_ww_model_fighters = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_ww_model_fighters)
wakehubert_trained_on = gen_ww_model_fighters.wakehubert_trained_on

SETS = {d.dataset_id: d for d in list_datasets("wake_word")
        if d.dataset_id.startswith("tigregotico-")}
WAKEHUBERT = [c for c in list_competitors("wake_word") if c.plugin == "ovos-ww-plugin-wakeforge"]
# models.json phrase -> TigreGotico repo word
SPELLING = {"okay_nabu": "ok_nabu"}


def test_the_sets_loaded():
    assert len(SETS) == 83


@pytest.mark.parametrize("fighter", WAKEHUBERT, ids=lambda c: c.competitor_id)
def test_an_english_model_skips_the_tigregotico_set_of_its_phrase(fighter):
    word = next(iter(fighter.config["hotwords"]))
    own = f"tigregotico-{SPELLING.get(word, word)}"
    if "en-US" in fighter.langs and own in SETS:
        assert own in (fighter.trained_on or []), fighter.notes


def test_no_fighter_is_kept_off_a_held_out_test_split():
    held_out = {i for i, d in SETS.items() if d.positives_dir == "test"}
    assert len(held_out) == 10
    skipped = {t for c in list_competitors("wake_word") for t in c.trained_on or []}
    assert held_out & skipped == set()


def test_english_train_sets_leave_out_the_folder_the_synthetic_wakewords_sets_hold():
    # the folders were read off each repo at generation time; these hold one
    with_folder = {"alexa", "amelia", "android", "athena", "hey_chatterbox", "hey_computer",
                   "hey_eggbert", "hey_floyd", "hey_jarvis", "hey_marvin", "hey_mycroft",
                   "hey_neon", "hey_robin", "hey_ziggy", "no", "stop", "wake_up", "yes"}
    for word in with_folder:
        assert "train/openvoiceos-synthetic-wakewords" in SETS[f"tigregotico-{word}"].exclude_dirs


@pytest.mark.parametrize("fighter_id", [
    "wakeforge-alexa", "wakeforge-alexa-streaming",
    "wakeforge-hey-mycroft", "wakeforge-hey-mycroft-streaming",
])
def test_a_model_trained_on_the_whole_repo_skips_the_synthetic_wakewords_set(fighter_id):
    fighter = next(c for c in list_competitors("wake_word") if c.competitor_id == fighter_id)
    word = next(iter(fighter.config["hotwords"]))
    assert f"synthetic-wakewords-{word}" in fighter.trained_on


EVAL_SETS = {"tigregotico-alexa", "synthetic-wakewords-alexa", "tigregotico-ok_nabu",
             "synthetic-wakewords-okay_nabu"}


def test_training_data_naming_the_whole_repo_skips_both_sets():
    # wording of wakehubert_alexa in models.json
    model = {"language": "en", "training_data": (
        "synthetic only: text-to-speech clips from TigreGotico/synthetic-wakeword-alexa "
        "(multi-engine) and OmniVoice clips")}
    assert wakehubert_trained_on(model, "alexa", EVAL_SETS) == [
        "tigregotico-alexa", "synthetic-wakewords-alexa"]


def test_training_data_naming_only_clips_of_the_repo_skips_the_tigregotico_set():
    # the wording the localised models use for the repo's train split
    model = {"language": "en", "training_data": (
        "OmniVoice clips and the training clips of TigreGotico/synthetic-wakeword-alexa")}
    assert wakehubert_trained_on(model, "alexa", EVAL_SETS) == ["tigregotico-alexa"]


def test_a_streaming_model_trained_on_its_heads_sources_skips_both_sets():
    model = {"language": "en", "training_data": (
        "synthetic TTS wake words (the shipped heads' positive sources, one voice in ten "
        "held out) in 30 s synthetic streams")}
    assert wakehubert_trained_on(model, "alexa", EVAL_SETS) == [
        "tigregotico-alexa", "synthetic-wakewords-alexa"]


def test_the_repo_spelling_of_a_phrase_maps_to_its_set():
    model = {"language": "en", "training_data": "TigreGotico/synthetic-wakeword-ok_nabu"}
    assert wakehubert_trained_on(model, "okay_nabu", EVAL_SETS) == ["tigregotico-ok_nabu"]


def test_a_non_english_model_skips_no_set():
    model = {"language": "pt", "training_data": "TigreGotico/synthetic-wakeword-alexa"}
    assert wakehubert_trained_on(model, "alexa", EVAL_SETS) == []


def test_a_streaming_model_skips_what_its_head_skips(monkeypatch):
    head = {"name": "wakehubert_hey_jarvis", "word": "hey jarvis", "language": "en",
            "default_threshold": 0.5, "training_data": "an edge-tts voice grid"}
    streaming = {**head, "name": "wakehubert_hey_jarvis_streaming", "training_data": (
        "synthetic TTS wake words (the shipped heads' positive sources)")}
    monkeypatch.setattr(gen_ww_model_fighters, "wakehubert_models", lambda: [head, streaming])
    sets = {"tigregotico-hey_jarvis", "synthetic-wakewords-hey_jarvis"}
    fighters = {f["competitor_id"]: f for f in gen_ww_model_fighters.wakeforge_fighters(sets)}
    assert fighters["wakeforge-hey-jarvis-streaming"]["trained_on"] == ["tigregotico-hey_jarvis"]
