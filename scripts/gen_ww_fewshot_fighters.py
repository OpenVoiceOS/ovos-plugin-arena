#!/usr/bin/env python3
"""Generate the wake-word fighters that need no trained model for their phrase.

- ``wakeforge-zeroshot-<phrase>``: ovos-ww-plugin-wakeforge-zeroshot, given the
  phrase as phones. The phones come from ``espeak-ng -q -v en-us --ipa`` with
  the stress marks removed, or from PHONE_OVERRIDES where espeak-ng misreads
  a name. One fighter per en-US phrase that has a test set. A phrase of
  another language uses that language's espeak-ng voice (ESPEAK_VOICES) and the
  spoken text in its test set's display name, and its id ends in the language.
- ``efficientword-net-<phrase>``: ovos-ww-plugin-efficientword-net with its
  bundled references, one per bundled phrase.
- ``local-wake-<phrase>``: ovos-ww-plugin-local-wake, given four clips of the
  phrase from one registry dataset (``samples_from``). That dataset is in
  ``trained_on``, so the fighter is scored only on the other sets.
- ``vosk-ww-<phrase>``: ovos-ww-plugin-vosk matching the transcript against the
  phrase, for each test-set phrase in every language with a small Vosk model
  (VOSK_MODELS). VOSK_SAMPLES lists the phrases that need other samples.

Output is deterministic and idempotent. Run from the repository root.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from registry.loaders import list_datasets

OUT = Path("registry/competitors/wake_word")
EFFICIENTWORD_BUNDLED = [
    "alexa",
    "balloon",
    "computer",
    "lights_off",
    "lights_on",
    "mobile",
    "mycroft",
]
VOSK_MODELS = {
    "en-US": "http://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "pt-PT": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
    "de-DE": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "es-ES": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.3.zip",
    "it-IT": "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
    "nl-NL": "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
    "ca-ES": "https://alphacephei.com/vosk/models/vosk-model-small-ca-0.4.zip",
    "fr-FR": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
}
# arena lang -> espeak-ng voice; a language with no voice gets no zero-shot fighter
ESPEAK_VOICES = {
    "en-US": "en-us", "pt-PT": "pt", "de-DE": "de", "es-ES": "es", "an-ES": "an",
    "ca-ES": "ca", "eu-ES": "eu", "it-IT": "it", "da-DK": "da", "nl-NL": "nl", "fr-FR": "fr-fr",
}
# A phrase with a word the small Vosk model lacks cannot be produced by its grammar,
# so it lists sound-alike samples here and matches them with the fuzzy rule. Every
# other phrase matches itself with the equals rule. The samples are picked without
# looking at any test clip: from the shipped OVOS default, or from model vocabulary
# words (looked up with the model's own word table) chosen by spelling or sound.
VOSK_SAMPLES = {
    ("hey_mycroft", "en-US"): (
        ["hey mycroft", "hey microsoft", "hey mike roft", "hey minecraft"],
        "fuzzy",
        "the hey_mycroft_vosk fallback that ovos-config ships (ovos_config/mycroft.conf)",
    ),
    ("hey_eggbert", "en-US"): (
        ["hey egbert", "hey egg bert"],
        "fuzzy",
        "vocabulary words closest in spelling to 'eggbert', which the model lacks",
    ),
    ("hey_k9", "en-US"): (
        ["hey k nine", "hey kay nine", "hey canine"],
        "fuzzy",
        "the spoken form of 'k9', which the model lacks; 'canine' is the vocabulary "
        "word with the same sounds (espeak-ng reads 'k nine' and 'canine' alike)",
    ),
    ("snowboy", "en-US"): (
        ["snow boy"],
        "fuzzy",
        "the two vocabulary words that spell 'snowboy', which the model lacks",
    ),
    ("ey_ordenador", "es-ES"): (
        ["ey ordenador", "hey ordenador"],
        "equals",
        "the wake word name and the text spoken in its test set, both made of vocabulary words",
    ),
}
# fighters whose id keeps no language suffix
VOSK_IDS = {("ey_ordenador", "es-ES"): "vosk-ww-ey-ordenador"}
# the small Vosk model of the language lacks a word of the phrase, so the grammar
# can never produce it and the fighter would never wake
VOSK_OUT_OF_VOCABULARY = {
    ("afonsinho", "pt-PT"),
    ("gervasio", "pt-PT"),
    ("o_flor", "pt-PT"),
    ("hello_nabu", "en-US"),
    ("okay_nabu", "en-US"),
    ("hey_jarbas", "en-US"),
    ("hey_rhasspy", "en-US"),
}
# phrase -> the dataset its local-wake references come from. A reference must
# hold only the phrase: the community clips carry seconds of room sound, and
# local-wake then matches every clip, so the references are synthetic.
LOCAL_WAKE_REFERENCES = {
    key: f"synthetic-wakewords-{key}"
    for key in ("alexa", "amelia", "athena", "hey_chatterbox", "hey_floyd", "hey_ziggy")
}
REFERENCE_COUNT = 4
# espeak-ng reads the name Mycroft with the vowel of "mick"; it is said "MY-croft"
PHONE_OVERRIDES = {"hey_mycroft": "h eɪ m aɪ k ɹ ɔ f t"}
LANGS = ["en-US", "en-GB", "en-AU"]


def phones(phrase: str, voice: str = "en-us") -> str:
    out = subprocess.run(
        ["espeak-ng", "-q", "-v", voice, "--ipa", "--sep= ", phrase],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return re.sub(r"\s+", " ", out.replace("ˈ", "").replace("ˌ", "")).strip()


def fighter(
    competitor_id,
    plugin,
    key,
    hotword,
    display_name,
    species,
    types,
    description,
    notes,
    links,
    trained_on=None,
    langs=None,
):
    entry = {
        "competitor_id": competitor_id,
        "modality": "wake_word",
        "plugin": plugin,
        "config": {
            "listener": {"wake_word": key},
            "hotwords": {key: {"module": plugin, "listen": True, **hotword}},
        },
        "langs": langs or LANGS,
        "display_name": display_name,
        "species": species,
        "types": types,
        "description": description,
        "links": links,
        "notes": notes,
        "capabilities": ["clip", "stream"],
    }
    if trained_on:
        entry["trained_on"] = trained_on
    return entry


def main() -> None:
    spoken = {}
    for d in list_datasets("wake_word"):
        if d.wakeword and d.role == "eval":
            key = d.wakeword.replace(" ", "_")
            shown = re.search(r"“(.+?)”", d.display_name or "")
            text = key.replace("_", " ") if d.lang == "en-US" or not shown else shown.group(1)
            spoken.setdefault((key, d.lang), text)
    pairs = sorted(spoken)
    entries = []
    for key, lang in pairs:
        phrase = spoken[(key, lang)]
        english = lang == "en-US"
        slug = key.replace("_", "-").replace(" ", "-")
        if not english:
            slug += f"-{lang.split('-')[0]}"
        langs = None if english else [lang]
        voice = ESPEAK_VOICES.get(lang)
        if voice is None:
            continue
        ipa = (english and PHONE_OVERRIDES.get(key)) or phones(phrase, voice)
        source = (
            "written by hand, because espeak-ng reads the name with the vowel of 'mick'"
            if english and key in PHONE_OVERRIDES
            else f"from espeak-ng ({voice}), stress marks removed"
        )
        entries.append(
            fighter(
                f"wakeforge-zeroshot-{slug}",
                "ovos-ww-plugin-wakeforge-zeroshot",
                key,
                {"ipa": ipa},
                f"wakeforge zero-shot ({phrase})",
                "WakePhoneHuBERTZeroShotPlugin",
                ["neural-net", "zero-shot"],
                f"WakePhoneHuBERT phone posteriors matched against the phones of '{phrase}'. "
                "No recordings and no training for the phrase.",
                f"Phones '{ipa}' {source}. Threshold, confirm_blocks and debounce are the "
                "plugin defaults.",
                {
                    "source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-wakeforge",
                    "model": "https://huggingface.co/TigreGotico/wakephonehubert",
                },
                langs=langs,
            )
        )
        if (key, lang) not in VOSK_OUT_OF_VOCABULARY and lang in VOSK_MODELS:
            if (key, lang) in VOSK_SAMPLES:
                samples, rule, origin = VOSK_SAMPLES[(key, lang)]
                note = f"Samples {samples} with the {rule} rule: {origin}."
            else:
                samples, rule = [phrase], "equals"
                note = (
                    f"Samples {samples} with the equals rule: the phrase itself, "
                    "every word of which is in the model vocabulary."
                )
            entries.append(
                fighter(
                    VOSK_IDS.get((key, lang), f"vosk-ww-{slug}"),
                    "ovos-ww-plugin-vosk",
                    key,
                    {
                        "model": VOSK_MODELS[lang],
                        "lang": lang,
                        "samples": samples,
                        "full_vocab": False,
                        "rule": rule,
                        "threshold": 0.75,
                        "time_between_checks": 1.0,
                        "debug": False,
                    },
                    f"Vosk wake word ({phrase})",
                    "VoskWakeWordPlugin",
                    ["neural-net", "keyword-match"],
                    "Detects wake words by running the Vosk Kaldi recognizer in keyword mode and "
                    "matching the transcript against configured samples.",
                    f"{note} The language model is auto-downloaded.",
                    {"source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-vosk"},
                    langs=langs,
                )
            )
        if english and key in LOCAL_WAKE_REFERENCES:
            ref = LOCAL_WAKE_REFERENCES[key]
            entries.append(
                fighter(
                    f"local-wake-{slug}",
                    "ovos-ww-plugin-local-wake",
                    key,
                    {"samples_from": {"dataset_id": ref, "count": REFERENCE_COUNT}},
                    f"local-wake ({phrase})",
                    "LocalWakeHotwordPlugin",
                    ["template-matching", "few-shot"],
                    f"Dynamic time warping of speech features against {REFERENCE_COUNT} recordings "
                    f"of '{phrase}'.",
                    f"The references are {REFERENCE_COUNT} positive clips of {ref}, "
                    "so the fighter is not scored on that set. Threshold is the plugin default.",
                    {"source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-local-wake"},
                    trained_on=[ref],
                )
            )
    for key in EFFICIENTWORD_BUNDLED:
        phrase = key.replace("_", " ")
        entries.append(
            fighter(
                f"efficientword-net-{key.replace('_', '-')}",
                "ovos-ww-plugin-efficientword-net",
                key,
                {},
                f"EfficientWord-Net ({phrase})",
                "EfficientWordHotwordPlugin",
                ["neural-net", "few-shot"],
                f"EfficientWord-Net ResNet-50 embeddings compared against the plugin's bundled "
                f"references for '{phrase}'.",
                "Bundled reference embeddings and the plugin's default threshold (0.7).",
                {"source": "https://github.com/OpenVoiceOS/ovos-ww-plugin-efficientword-net"},
            )
        )
    for entry in entries:
        (OUT / f"{entry['competitor_id']}.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"{len(pairs)} phrases in, {len(entries)} fighters out")


if __name__ == "__main__":
    main()
