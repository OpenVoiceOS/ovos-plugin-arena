#!/usr/bin/env python3
"""Generate a wake-word fighter for every published model of the model-based plugins.

- ``wakeforge-<word>[-streaming]``: every WakeHuBERT model in
  ``OpenVoiceOS/wakehubert-wakewords`` ``models.json``, at ``MODELS_REVISION``.
  Its ``trained_on`` follows the model's ``training_data``: every English model
  trained on clips from the TigreGotico repo of its phrase, so it skips
  ``tigregotico-<word>``. A model whose training data names that whole repo
  (``TigreGotico/synthetic-wakeword-<word>`` and not only its training split)
  also skips ``synthetic-wakewords-<word>``, the source of one of the repo's
  folders. A streaming model trained on "the shipped heads' positive sources"
  skips what its head skips. The held-out ``test/`` sets of the localised
  repos are no model's training data. This field is rewritten in files that
  already exist.
- ``precise-onnx-<model>``: every wake-word model in
  ``OpenVoiceOS/precise-lite-models``.
- ``microwakeword-<word>[-v2]``: every model of the ESPHome
  ``micro-wake-word-models`` repository.
- ``openwakeword-<word>``: every pretrained openWakeWord model.

A fighter file that already exists is left as it is, so the hand-written
fighters and their variants keep their settings. Then every wake-word fighter
gets the ``stream`` capability, because every OVOS wake-word plugin runs on the
live microphone stream, and every ``trained_on`` test set brings its
``ww_stream_<set>`` corpus with it.

Output is deterministic and idempotent. Run from the repository root.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from registry.loaders import list_datasets

OUT = Path("registry/competitors/wake_word")
STREAM_SETS = Path("registry/datasets/ww_stream")

WAKEHUBERT_REPO = "OpenVoiceOS/wakehubert-wakewords"
MODELS_REVISION = "2ac92d3bcf55f177a47049c3f810a78984a31ff9"
PRECISE_BASE = "https://github.com/OpenVoiceOS/precise-lite-models/raw/master/wakewords"
# (file stem under wakewords/<lang>/, phrase key, language)
PRECISE_MODELS = [
    ("android", "android", "en"),
    ("athena-en-0.3.0-20190801-eltocino", "athena", "en"),
    ("christopher", "christopher", "en"),
    ("computer", "computer", "en"),
    ("hey_chatterbox", "hey_chatterbox", "en"),
    ("hey_firefox", "hey_firefox", "en"),
    ("hey_k9", "hey_k9", "en"),
    ("hey_kit", "hey_kit", "en"),
    ("hey_moxie", "hey_moxie", "en"),
    ("hey_mycroft", "hey_mycroft", "en"),
    ("hey_mycroft_mk1", "hey_mycroft", "en"),
    ("hey_robin", "hey_robin", "en"),
    ("hey_scout", "hey_scout", "en"),
    ("jarvis", "jarvis", "en"),
    ("marvin", "hey_marvin", "en"),
    ("sheila", "sheila", "en"),
    ("eyordenador-es-0.2.0-20190815-eltocino", "ey_ordenador", "es"),
    ("eyordenador-es-0.3.0-20190815-eltocino", "ey_ordenador", "es"),
    ("o_sauro", "o_sauro", "pt"),
]
# precise-lite-models file stem -> the fighter id an earlier entry already uses
PRECISE_IDS = {
    "athena-en-0.3.0-20190801-eltocino": "precise-onnx-athena",
    "eyordenador-es-0.3.0-20190815-eltocino": "precise-onnx-ey-ordenador",
    "eyordenador-es-0.2.0-20190815-eltocino": "precise-onnx-ey-ordenador-0-2",
    "hey_mycroft_mk1": "precise-onnx-hey-mycroft-mk1",
}
MICROWAKEWORD_BASE = "https://github.com/esphome/micro-wake-word-models/raw/main/models"
# (path under models/, phrase key, model version)
MICROWAKEWORD_MODELS = [
    ("alexa", "alexa", 1),
    ("hey_jarvis", "hey_jarvis", 1),
    ("okay_nabu", "okay_nabu", 1),
    ("v2/alexa", "alexa", 2),
    ("v2/hey_jarvis", "hey_jarvis", 2),
    ("v2/hey_mycroft", "hey_mycroft", 2),
    ("v2/okay_nabu", "okay_nabu", 2),
    ("v2/experiments/choo_choo_homie", "choo_choo_homie", 2),
    ("v2/experiments/hey_home_assistant", "hey_home_assistant", 2),
    ("v2/experiments/hey_peppa_pig", "hey_peppa_pig", 2),
    ("v2/experiments/okay_computer", "okay_computer", 2),
]
OPENWAKEWORD_MODELS = ["alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "timer", "weather"]
LANGS = {
    "en": ["en-US", "en-GB", "en-AU"],
    "es": ["es-ES"],
    "de": ["de-DE"],
    "nl": ["nl-NL"],
    "it": ["it-IT"],
    "pt": ["pt-PT"],
}


def slug(key: str) -> str:
    return key.replace("_", "-").replace(" ", "-")


def fighter(
    competitor_id,
    plugin,
    key,
    hotword,
    lang,
    display_name,
    species,
    model,
    description,
    notes,
    links,
    trained_on=None,
):
    entry = {
        "competitor_id": competitor_id,
        "modality": "wake_word",
        "plugin": plugin,
        "config": {
            "listener": {"wake_word": key},
            "hotwords": {key: {"module": plugin, "listen": True, **hotword}},
        },
        "langs": LANGS[lang],
        "display_name": display_name,
        "species": species,
        "types": ["neural-net"],
        "description": description,
        "model": model,
        "links": links,
        "notes": notes,
        "capabilities": ["clip", "stream"],
    }
    if trained_on:
        entry["trained_on"] = trained_on
    return entry


def wakehubert_models() -> list[dict]:
    url = f"https://huggingface.co/{WAKEHUBERT_REPO}/resolve/{MODELS_REVISION}/models.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["models"]


# models.json phrase -> the word in the TigreGotico repo name, where the two differ
TIGREGOTICO_SPELLING = {"okay_nabu": "ok_nabu"}


def wakehubert_trained_on(model: dict, word: str, eval_sets: set[str]) -> list[str]:
    """The registry test sets whose clips the model's ``training_data`` names."""
    if model["language"] != "en":
        return []
    text = model.get("training_data", "")
    word = TIGREGOTICO_SPELLING.get(word, word)
    repo = f"TigreGotico/synthetic-wakeword-{word}"
    whole_repo = repo in text and f"clips of {repo}" not in text
    candidates = [f"tigregotico-{word}"]
    if whole_repo or "the shipped heads' positive sources" in text:
        candidates.append(f"synthetic-wakewords-{word}")
    return [s for s in candidates if s in eval_sets]


def wakeforge_fighters(eval_sets: set[str]) -> list[dict]:
    entries = []
    models = wakehubert_models()
    heads = {m["name"]: m for m in models}
    for m in models:
        streaming = m["name"].endswith("_streaming")
        # models.json spells some phrases with a space; hotword keys are snake case
        word = m["word"].replace(" ", "_")
        phrase = word.replace("_", " ")
        kind = "streaming " if streaming else ""
        head = heads.get(m["name"].removesuffix("_streaming"), m)
        trained = (wakehubert_trained_on(head, word, eval_sets)
                   if "the shipped heads' positive sources" in m.get("training_data", "")
                   else wakehubert_trained_on(m, word, eval_sets))
        entries.append(
            fighter(
                f"wakeforge-{slug(word)}{'-streaming' if streaming else ''}",
                "ovos-ww-plugin-wakeforge",
                word,
                {"model": m["name"], "models_revision": MODELS_REVISION},
                m["language"],
                f"wakeforge ({phrase}{', streaming' if streaming else ''})",
                "WakeForgeHotwordPlugin",
                f"{m['name']} ({WAKEHUBERT_REPO})",
                f"The ready WakeHuBERT {kind}'{phrase}' model at its own calibrated default "
                f"threshold ({m['default_threshold']}).",
                f"models.json at {MODELS_REVISION[:8]}: featurizer {m.get('featurizer')}. "
                "Threshold, "
                f"smoothing and debounce come from the model's own metadata. Training data per "
                f"models.json: {m.get('training_data', 'not stated')}.",
                {
                    "source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-wakeforge",
                    "model": f"https://huggingface.co/{WAKEHUBERT_REPO}",
                },
                trained_on=trained or None,
            )
        )
    return entries


def precise_fighters() -> list[dict]:
    entries = []
    for stem, key, lang in PRECISE_MODELS:
        phrase = key.replace("_", " ")
        entries.append(
            fighter(
                PRECISE_IDS.get(stem, f"precise-onnx-{slug(stem)}"),
                "ovos-ww-plugin-precise-onnx",
                key,
                {
                    "model": f"{PRECISE_BASE}/{lang}/{stem}.onnx",
                    "sensitivity": 0.5,
                    "trigger_level": 3,
                },
                lang,
                f"Precise ONNX ({stem.replace('_', ' ')})",
                "PreciseOnnxHotwordPlugin",
                f"{stem}.onnx (precise-lite-models)",
                f"Mycroft Precise recurrent wake-word network for '{phrase}', exported to ONNX.",
                "README reference configuration (sensitivity 0.5, trigger level 3).",
                {
                    "source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-precise-onnx",
                    "model": "https://github.com/OpenVoiceOS/precise-lite-models",
                },
            )
        )
    return entries


def microwakeword_manifest(path: str) -> dict:
    """The ``micro`` settings of an ESPHome model manifest."""
    with urllib.request.urlopen(f"{MICROWAKEWORD_BASE}/{path}.json", timeout=60) as response:
        return json.load(response)["micro"]


def microwakeword_fighters() -> list[dict]:
    entries = []
    for path, key, version in MICROWAKEWORD_MODELS:
        phrase = key.replace("_", " ")
        name = path.rsplit("/", 1)[-1]
        experimental = "/experiments/" in path
        if version == 2:
            micro = microwakeword_manifest(path)
            settings = {
                "probability_cutoff": micro["probability_cutoff"],
                "sliding_window_size": micro["sliding_window_size"],
            }
            notes = (
                "The model manifest's cutoff and sliding window; the plugin's refractory "
                "frames (40)."
            )
        else:
            settings = {"probability_cutoff": 0.5, "sliding_window_size": 10}
            notes = (
                "The plugin's reference cutoff (0.5), sliding window (10) and refractory "
                "frames (40)."
            )
        entries.append(
            fighter(
                f"microwakeword-{slug(key)}{'-v2' if version == 2 else ''}",
                "ovos-ww-plugin-microwakeword",
                key,
                {
                    "model": f"{MICROWAKEWORD_BASE}/{path}.tflite",
                    "model_name": name,
                    "model_version": version,
                    **settings,
                    "refractory_frames": 40,
                },
                "en",
                f"microWakeWord ({phrase}{', v2' if version == 2 else ''})",
                "MicroWakeWordPlugin",
                f"{path}.tflite (ESPHome micro-wake-word-models)",
                f"microWakeWord v{version} streaming TFLite model for '{phrase}' from the ESPHome "
                "model "
                f"repository{', experiments folder' if experimental else ''}.",
                notes,
                {
                    "source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-microwakeword",
                    "model": "https://github.com/esphome/micro-wake-word-models",
                },
            )
        )
    return entries


def openwakeword_fighters() -> list[dict]:
    return [
        fighter(
            f"openwakeword-{slug(key)}",
            "ovos-ww-plugin-openwakeword",
            key,
            {"threshold": 0.5, "custom_verifier_models": {}, "custom_verifier_threshold": 0.1},
            "en",
            f"openWakeWord ({key.replace('_', ' ')})",
            "OwwHotwordPlugin",
            f"{key} (openWakeWord pretrained)",
            f"The pretrained openWakeWord '{key.replace('_', ' ')}' model.",
            f"Pretrained openWakeWord '{key}' model at threshold 0.5.",
            {"source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-openWakeWord"},
        )
        for key in OPENWAKEWORD_MODELS
    ]


def stream_capable(entry: dict, stream_sets: set[str]) -> dict:
    """Every fighter streams, and its training sets exclude their stream corpora too."""
    caps = entry.get("capabilities") or ["clip"]
    entry["capabilities"] = sorted(set(caps) | {"stream"}, key=["clip", "stream"].index)
    trained = entry.get("trained_on") or []
    extra = [f"ww_stream_{t}" for t in trained if f"ww_stream_{t}" in stream_sets]
    if extra:
        entry["trained_on"] = list(dict.fromkeys([*trained, *extra]))
    return entry


def main() -> None:
    eval_sets = {d.dataset_id for d in list_datasets("wake_word") if d.role == "eval"}
    stream_sets = {p.stem for p in STREAM_SETS.glob("*.json")}
    wakeforge = wakeforge_fighters(eval_sets)
    for entry in wakeforge:
        path = OUT / f"{entry['competitor_id']}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("trained_on"):
                existing["trained_on"] = entry["trained_on"]
            else:
                existing.pop("trained_on", None)
            path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    generated = [
        *wakeforge,
        *precise_fighters(),
        *microwakeword_fighters(),
        *openwakeword_fighters(),
    ]
    written = 0
    for entry in generated:
        path = OUT / f"{entry['competitor_id']}.json"
        if not path.exists():
            path.write_text(
                json.dumps(stream_capable(entry, stream_sets), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written += 1
    paths = sorted(OUT.glob("*.json"))
    for path in paths:
        entry = json.loads(path.read_text(encoding="utf-8"))
        updated = stream_capable(dict(entry), stream_sets)
        if updated != entry:
            path.write_text(
                json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    print(
        f"{len(generated)} models in, {written} new fighters out, "
        f"{len(paths)} fighters in the registry"
    )


if __name__ == "__main__":
    main()
