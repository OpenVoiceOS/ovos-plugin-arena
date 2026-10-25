#!/usr/bin/env python3
"""Generate the wake-word clip test sets of the TigreGotico synthetic wake-word collections.

One ``tigregotico-<phrase>[-<lang>]`` dataset per phrase and language:

- English phrase repos: the ``train/`` clips, without ``openvoiceos-synthetic-wakewords/``
  (the source of the ``synthetic-wakewords-*`` sets) and ``original/`` (no card section
  says what it holds). ``unchecked/`` is never read.
- Localised wake-up repos with a held-out ``test/`` split: those clips only, one set per
  language of the split's ``metadata.jsonl``.
- Train-only localised and Portuguese repos: their ``train/`` clips.
- A repo whose clips say several phrases (SPLIT_BY_TEXT) gives one set per phrase.
- Galician, Asturian and Kabyle have no set: no fighter runs in those languages
  (no espeak-ng voice and no small Vosk model), so their boards would stay empty.

Every repo is pinned to its head commit at generation time. A set with fewer than
MIN_POSITIVES clips with the phrase is not written. Run from the repository root with a
network connection; the output is the same for the same heads.
"""

from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path

from huggingface_hub import HfApi

from registry.loaders import list_datasets
from registry.schemas import DatasetDef
from runner import audio_io

OUT = Path("registry/datasets/wake_word")
ORG = "TigreGotico/synthetic-wakeword-"
MIN_POSITIVES = 30
SAMPLE_POLICY = {"max_samples": 100, "seed": 1337}
NEGATIVES_SOURCES = [
    "TigreGotico/not-wake-words-speech-en/not_wake_word_subset",
    "TigreGotico/public_domain_sounds_3secs/pd_3sec",
    "TigreGotico/ESC-50",
    "TigreGotico/FMA_3secs",
    "TigreGotico/ambient_noises",
]
# repo word -> the hotword key the fighters of the same phrase use
HOTWORD_KEYS = {"ok_nabu": "okay_nabu"}
# repos whose metadata ``text`` holds several phrases: one set per phrase
SPLIT_BY_TEXT = {"jarbas", "computador"}
# voices of another language in an English repo; their clips are byte copies of
# the Portuguese repo's clips of a different phrase
OTHER_LANGUAGE_VOICES = {"hey_jarbas": {"Cristiano", "Ines", "Ricardo", "Vitoria"}}
EXCLUDE = ["train/openvoiceos-synthetic-wakewords", "train/original"]
LANGUAGES = {
    "en-US": "English",
    "pt-PT": "Portuguese",
    "de-DE": "German",
    "es-ES": "Spanish",
    "an-ES": "Aragonese",
    "ca-ES": "Catalan",
    "eu-ES": "Basque",
    "it-IT": "Italian",
    "da-DK": "Danish",
    "nl-NL": "Dutch",
    "fr-FR": "French",
}
ENGLISH = [
    "alexa", "amelia", "android", "athena", "christopher", "computer", "firefox",
    "hello_nabu", "hey_android", "hey_chatterbox", "hey_christopher", "hey_computer",
    "hey_eggbert", "hey_firefox", "hey_floyd", "hey_jarbas", "hey_jarvis", "hey_k9",
    "hey_lazarus", "hey_marvin", "hey_mike", "hey_mycroft", "hey_neon", "hey_potato",
    "hey_rhasspy", "hey_robin", "hey_scout", "hey_siri", "hey_snapdragon", "hey_ziggy",
    "hi_galaxy", "hi_lumina", "home_assistant", "jarvis", "lights_off", "lights_on",
    "marvin", "no", "ok_nabu", "sheila", "snowboy", "stop", "view_glass",
    "voice_assistant", "wake_up", "yes",
]
# repo word -> [(arena lang, metadata lang or None when the split holds one language)]
HELD_OUT_TEST = {
    "acorda": [("pt-PT", None)],
    "aufwachen": [("de-DE", None)],
    "desperta": [("ca-ES", "ca")],
    "despierta": [("es-ES", "es"), ("an-ES", "an")],
    "esnatu": [("eu-ES", None)],
    "espierta": [("an-ES", "an")],
    "sveglia": [("it-IT", None)],
    "vagn_op": [("da-DK", None)],
    "wakker_worden": [("nl-NL", None)],
}
TRAIN_ONLY = {
    "desperta_ca": "ca-ES",
    "reveille_toi": "fr-FR",
    "ey_ordenador": "es-ES",
    **{word: "pt-PT" for word in (
        "afonsinho", "amalia", "ambrosio", "antonio", "apaga_a_luz", "computador", "elsa",
        "evaristo", "gervasio", "gloria", "jarbas", "joao", "julio", "liga_a_luz", "lurdes",
        "maria", "o_flor", "o_ines", "o_maria", "o_princesa",
    )},
}


def spoken(texts: list[str]) -> str:
    """The most common transcript, lower case, without punctuation."""
    norm = [re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", t.lower())).strip() for t in texts]
    return collections.Counter(n for n in norm if n).most_common(1)[0][0]


def key_of(text: str) -> str:
    """A phrase as a hotword key: lower case, no accents, words joined by underscores."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return "_".join(re.findall(r"\w+", plain))


def build(word: str, lang: str, split: str, meta_lang: str | None, sha: str,
          files: list[str], mlsw: set[str], text: str | None = None) -> dict | None:
    repo = ORG + word
    two = lang.split("-")[0]
    name = key_of(text) if text else word
    suffix = "" if lang == "en-US" or name.endswith(f"_{two}") else f"-{two}"
    dataset_id = f"tigregotico-{name}{suffix}"
    excluded = sorted(d for d in EXCLUDE if split == "train"
                      and any(f.startswith(d + "/") for f in files))
    rows = audio_io._jsonl_rows(repo, f"{split}/metadata.jsonl", sha)
    where: dict[str, str | list[str]] = {}
    if meta_lang:
        where["lang"] = meta_lang
    if text:
        where["text"] = text
    left_voices = sorted(OTHER_LANGUAGE_VOICES.get(word, set()))
    if left_voices:
        where["voice"] = sorted({r["voice"] for r in rows} - set(left_voices))
    rows = [r for r in rows if all(
        str(r.get(k)) in ([v] if isinstance(v, str) else v) for k, v in where.items())]
    rows = [r for r in rows if not any(f"{split}/{r['file_name']}".startswith(d + "/")
                                       for d in excluded)]
    phrase = spoken([r.get("text") or "" for r in rows]) if rows else word.replace("_", " ")
    definition = {
        "dataset_id": dataset_id,
        "modality": "wake_word",
        "source": {"type": "huggingface", "hf_id": repo, "revision": sha},
        "reference_fields": {},
        "wakeword": HOTWORD_KEYS.get(name, name.removesuffix(f"_{two}")),
        "positives_dir": split,
        **({"positives_filter": where} if where else {}),
        **({"exclude_dirs": excluded} if excluded else {}),
        "negatives_sources": NEGATIVES_SOURCES,
        "lang": lang,
        "license": "cc-by-4.0",
        "role": "eval",
        "predictions_hf": f"OpenVoiceOS/ovos-wake-word-bench-{dataset_id}",
        "sample_policy": SAMPLE_POLICY,
    }
    if f"mlsw-negatives-{lang}" in mlsw:
        definition["negatives_dataset_ids"] = [f"mlsw-negatives-{lang}"]
    n = len(audio_io.ww_positive_files(DatasetDef(**definition), sha))
    if n < MIN_POSITIVES:
        print(f"skip {dataset_id}: {n} clips with the phrase")
        return None
    language = LANGUAGES[lang]
    folders = ", ".join(sorted({f.split("/")[1] for f in files
                                if f.startswith(split + "/") and f.count("/") >= 2
                                and not any(f.startswith(d + "/") for d in excluded)}))
    left_out = f", leaving out {' and '.join(d + '/' for d in excluded)}" if excluded else ""
    which = "".join([
        f" whose metadata lang is '{meta_lang}'" if meta_lang else "",
        f" whose metadata text is '{text}'" if text else "",
        f", leaving out the voices {', '.join(left_voices)}, whose clips are copies of"
        " Portuguese clips of another phrase" if left_voices else "",
    ])
    if split == "test":
        origin = (f"Clips of “{phrase}” in {language} from the held-out test split of the "
                  "TigreGotico synthetic wake-up collection, spoken by OmniVoice voices whose "
                  "seeds the train split never uses, used as the audio a detector must wake on")
    else:
        origin = (f"Clips of “{phrase}” in {language} from the TigreGotico synthetic wake-word "
                  "collection, made with several text-to-speech engines and converted voices, "
                  "used as the audio a detector must wake on")
    negatives = ("the clips it must ignore are pooled from a shared not-wake-word collection "
                 "of English speech, public-domain sounds, environmental audio, music and "
                 "ambient noise")
    extra = (" and 100 more spoken-word clips in the language join the negatives"
             if "negatives_dataset_ids" in definition else "")
    definition["display_name"] = f"TigreGotico synthetic wake word — “{phrase}” ({language})"
    definition["summary"] = (
        f"{origin}; {negatives}. One row is one clip with its label. Synthetic clips carry no "
        "room acoustics and no microphone variation, so this board does not exercise either. "
        "Runs on this corpus are capped at 100 clips per class, drawn the same way for every "
        f"fighter{extra}."
    )
    definition["notes"] = (
        f"{repo}@{sha[:8]}: {n} clips with the phrase in {split}/{which}{left_out}, from the "
        f"folders {folders}. The spoken text is '{phrase}'."
    )
    return definition


def main() -> None:
    api = HfApi()
    mlsw = {d.dataset_id for d in list_datasets("wake_word")
            if d.dataset_id.startswith("mlsw-negatives-")}
    jobs = ([(w, "en-US", "train", None) for w in ENGLISH]
            + [(w, lang, "test", m) for w, pairs in HELD_OUT_TEST.items() for lang, m in pairs]
            + [(w, lang, "train", None) for w, lang in TRAIN_ONLY.items()])
    written = 0
    for word, lang, split, meta_lang in jobs:
        info = api.dataset_info(ORG + word)
        files = api.list_repo_files(ORG + word, repo_type="dataset", revision=info.sha)
        texts = [None]
        if word in SPLIT_BY_TEXT:
            rows = audio_io._jsonl_rows(ORG + word, f"{split}/metadata.jsonl", info.sha)
            texts = sorted({r["text"] for r in rows})
        for text in texts:
            definition = build(word, lang, split, meta_lang, info.sha, files, mlsw, text)
            if definition is None:
                continue
            write(definition)
            written += 1
    print(f"{len(jobs)} repos in, {written} sets written")


def write(definition: dict) -> None:
    ordered = {k: definition[k] for k in (
        "dataset_id", "display_name", "summary", "modality", "source", "reference_fields",
        "wakeword", "positives_dir", "positives_filter", "exclude_dirs", "negatives_sources",
        "lang", "license", "role", "predictions_hf", "notes", "negatives_dataset_ids",
        "sample_policy") if k in definition}
    (OUT / f"{definition['dataset_id']}.json").write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
