#!/usr/bin/env python3
"""Generate the wake-word fighters that need no trained model for their phrase.

- ``wakeforge-zeroshot-<phrase>``: ovos-ww-plugin-wakeforge-zeroshot, given the
  phrase as phones. The phones come from ``espeak-ng -q -v en-us --ipa`` with
  the stress marks removed, or from PHONE_OVERRIDES where espeak-ng misreads
  a name. One fighter per en-US phrase that has a test set.
- ``efficientword-net-<phrase>``: ovos-ww-plugin-efficientword-net with its
  bundled references, for the bundled phrases that have a test set.
- ``local-wake-<phrase>``: ovos-ww-plugin-local-wake, given four clips of the
  phrase from one registry dataset (``samples_from``). That dataset is in
  ``trained_on``, so the fighter is scored only on the other sets.

Output is deterministic and idempotent. Run from the repository root.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from registry.loaders import list_datasets

OUT = Path("registry/competitors/wake_word")
EFFICIENTWORD_BUNDLED = {"alexa", "computer"}
# phrase -> the dataset its local-wake references come from. A reference must
# hold only the phrase: the community clips carry seconds of room sound, and
# local-wake then matches every clip, so the references are synthetic.
LOCAL_WAKE_REFERENCES = {
    "alexa": "synthetic-wakewords-alexa",
    "hey_ziggy": "synthetic-wakewords-hey_ziggy",
}
REFERENCE_COUNT = 4
# espeak-ng reads the name Mycroft with the vowel of "mick"; it is said "MY-croft"
PHONE_OVERRIDES = {"hey_mycroft": "h eɪ m aɪ k ɹ ɔ f t"}
LANGS = ["en-US", "en-GB", "en-AU"]


def phones(phrase: str) -> str:
    out = subprocess.run(
        ["espeak-ng", "-q", "-v", "en-us", "--ipa", "--sep= ", phrase],
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
):
    entry = {
        "competitor_id": competitor_id,
        "modality": "wake_word",
        "plugin": plugin,
        "config": {
            "listener": {"wake_word": key},
            "hotwords": {key: {"module": plugin, "listen": True, **hotword}},
        },
        "langs": LANGS,
        "display_name": display_name,
        "species": species,
        "types": types,
        "description": description,
        "links": links,
        "notes": notes,
    }
    if trained_on:
        entry["trained_on"] = trained_on
    return entry


def main() -> None:
    keys = sorted(
        {
            d.wakeword
            for d in list_datasets("wake_word")
            if d.wakeword and d.lang == "en-US" and d.role == "eval" and d.wakeword != "positive"
        }
    )
    entries = []
    for key in keys:
        phrase = key.replace("_", " ")
        slug = key.replace("_", "-").replace(" ", "-")
        ipa = PHONE_OVERRIDES.get(key) or phones(phrase)
        source = ("written by hand, because espeak-ng reads the name with the vowel of 'mick'"
                  if key in PHONE_OVERRIDES else "from espeak-ng (en-us), stress marks removed")
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
            )
        )
        if key in EFFICIENTWORD_BUNDLED:
            entries.append(
                fighter(
                    f"efficientword-net-{slug}",
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
        if key in LOCAL_WAKE_REFERENCES:
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
    for entry in entries:
        (OUT / f"{entry['competitor_id']}.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"{len(keys)} phrases in, {len(entries)} fighters out")


if __name__ == "__main__":
    main()
