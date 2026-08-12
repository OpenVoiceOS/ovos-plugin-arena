#!/usr/bin/env python3
"""Generate arena TTS fighter files for every voice in the phoonnx voice_index.

Source of truth: ``phoonnx/voice_index/*.json`` (one file per engine family,
mapping voice_id -> voice metadata). This script covers the installed
phoonnx package if it ships a ``voice_index/`` dir; otherwise falls back to
the phoonnx dev checkout at
``/home/miro/AgentWorkspaces/ovos/phoonnx/phoonnx/voice_index``.

22 fighters for OpenVoiceOS/phoonnx_* voices (the OVOS.json family) already
exist by hand in registry/competitors/tts/ — those are left untouched and
matching voice_index entries are skipped (matched by ``model``/``voice_id``).

Output is deterministic (sorted by competitor_id) and idempotent.
"""
from __future__ import annotations

import json
import os
import re
import sys

try:
    import phoonnx  # type: ignore
    _INSTALLED_DIR = os.path.join(os.path.dirname(phoonnx.__file__), "voice_index")
except Exception:
    _INSTALLED_DIR = None

_FALLBACK_DIR = "/home/miro/AgentWorkspaces/ovos/phoonnx/phoonnx/voice_index"

if _INSTALLED_DIR and os.path.isdir(_INSTALLED_DIR):
    VOICE_INDEX_DIR = _INSTALLED_DIR
    SOURCE_NOTE = f"installed phoonnx package ({VOICE_INDEX_DIR})"
elif os.path.isdir(_FALLBACK_DIR):
    VOICE_INDEX_DIR = _FALLBACK_DIR
    SOURCE_NOTE = f"phoonnx dev checkout fallback ({VOICE_INDEX_DIR})"
else:
    print("ERROR: no voice_index directory found (installed or dev checkout)", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_DIR = os.path.join(REPO_ROOT, "registry", "competitors", "tts")

FAMILY_SPECIES = {
    "BSC": "PhoonnxBSC",
    "MMS": "PhoonnxMMS",
    "OVOS": "PhoonnxOVOS",
    "chatterbox": "PhoonnxChatterbox",
    "coqui_community": "PhoonnxCoquiCommunity",
    "coqui_vits": "PhoonnxCoquiVITS",
    "f5tts": "PhoonnxF5TTS",
    "fastpitch": "PhoonnxFastPitch",
    "glowtts": "PhoonnxGlowTTS",
    "mimic3": "PhoonnxMimic3",
    "mixertts": "PhoonnxMixerTTS",
    "neurlang": "PhoonnxNeurlang",
    "optispeech": "PhoonnxOptiSpeech",
    "phonikud": "PhoonnxPhonikud",
    "piper": "PhoonnxPiper",
    "piper_community": "PhoonnxPiperCommunity",
    "proxectonos": "PhoonnxProxectoNos",
    "shami": "PhoonnxShami",
    "styletts2": "PhoonnxStyleTTS2",
    "supertonic": "PhoonnxSuperTonic",
    "transformers_community": "PhoonnxTransformersCommunity",
    "vits2": "PhoonnxVITS2",
}

FAMILY_DISPLAY = {
    "BSC": "BSC",
    "MMS": "MMS",
    "OVOS": "OVOS",
    "chatterbox": "Chatterbox",
    "coqui_community": "Coqui Community",
    "coqui_vits": "Coqui VITS",
    "f5tts": "F5-TTS",
    "fastpitch": "FastPitch",
    "glowtts": "GlowTTS",
    "mimic3": "Mimic3",
    "mixertts": "MixerTTS",
    "neurlang": "Neurlang",
    "optispeech": "OptiSpeech",
    "phonikud": "Phonikud",
    "piper": "Piper",
    "piper_community": "Piper Community",
    "proxectonos": "Proxecto Nós",
    "shami": "Shami",
    "styletts2": "StyleTTS2",
    "supertonic": "SuperTonic",
    "transformers_community": "Transformers Community",
    "vits2": "VITS2",
}

DESCRIPTIONS = {
    "BSC": "ONNX-runtime TTS via the phoonnx adapter framework, running a BSC (Barcelona Supercomputing Center) voice.",
    "MMS": "ONNX-runtime TTS via the phoonnx adapter framework, running a Meta MMS (Massively Multilingual Speech) VITS voice.",
    "OVOS": "ONNX-runtime TTS via the phoonnx adapter framework, running an OpenVoiceOS community voice.",
    "chatterbox": "ONNX-runtime TTS via the phoonnx adapter framework, running a Chatterbox voice.",
    "coqui_community": "ONNX-runtime TTS via the phoonnx adapter framework, running a community-trained Coqui TTS voice.",
    "coqui_vits": "ONNX-runtime TTS via the phoonnx adapter framework, running a Coqui VITS voice.",
    "f5tts": "ONNX-runtime TTS via the phoonnx adapter framework, running an F5-TTS voice.",
    "fastpitch": "ONNX-runtime TTS via the phoonnx adapter framework, running a FastPitch voice.",
    "glowtts": "ONNX-runtime TTS via the phoonnx adapter framework, running a GlowTTS voice.",
    "mimic3": "ONNX-runtime TTS via the phoonnx adapter framework, running a Mycroft Mimic3 voice.",
    "mixertts": "ONNX-runtime TTS via the phoonnx adapter framework, running a MixerTTS voice.",
    "neurlang": "ONNX-runtime TTS via the phoonnx adapter framework, running a Neurlang voice.",
    "optispeech": "ONNX-runtime TTS via the phoonnx adapter framework, running an OptiSpeech voice.",
    "phonikud": "ONNX-runtime TTS via the phoonnx adapter framework, running a Phonikud voice.",
    "piper": "ONNX-runtime TTS via the phoonnx adapter framework, running a Piper VITS voice.",
    "piper_community": "ONNX-runtime TTS via the phoonnx adapter framework, running a community-trained Piper voice.",
    "proxectonos": "ONNX-runtime TTS via the phoonnx adapter framework, running a Proxecto Nós (Galician) voice.",
    "shami": "ONNX-runtime TTS via the phoonnx adapter framework, running a Shami voice.",
    "styletts2": "ONNX-runtime TTS via the phoonnx adapter framework, running a StyleTTS2 voice.",
    "supertonic": "ONNX-runtime TTS via the phoonnx adapter framework, running a SuperTonic voice.",
    "transformers_community": "ONNX-runtime TTS via the phoonnx adapter framework, running a community HF Transformers TTS voice.",
    "vits2": "ONNX-runtime TTS via the phoonnx adapter framework, running a VITS2 voice.",
}


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def short_voice_name(voice_id: str) -> str:
    """Human-friendly short form of a voice_id for display_name."""
    return voice_id.split("/")[-1]


_lang_cache: dict[str, list[str]] = {}


def normalize_langs(lang_field) -> list[str]:
    """Expand bare language codes to full BCP-47 tags via langcodes.

    Full tags (already contain '-') pass through unchanged. Multi-lang
    voices (lang_field is a list, or a '+'/','-separated string) keep all
    tags, each individually normalized.
    """
    if lang_field is None:
        return []
    if isinstance(lang_field, list):
        parts = lang_field
    else:
        parts = re.split(r"[,+]", str(lang_field))
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        out.append(_normalize_one(part))
    # dedupe, preserve order
    seen = set()
    result = []
    for t in out:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _normalize_one(tag: str) -> str:
    """Expand a bare language code to a full BCP-47 tag.

    langcodes' ``maximize()`` walks ``broader_tags()`` (self, then the
    macrolanguage, then ``und``) until it finds a CLDR likely-subtags
    match. When no entry exists for the language or its macrolanguage at
    all, the walk bottoms out at the generic ``und`` -> ``en-Latn-US``
    default — that region is a fabrication, not real CLDR data (e.g. this
    happened for 654/1150 MMS voices covering minority languages like
    Dinka or Garifuna, which have no likely-subtags entry). Only trust the
    maximized region when the match came from a real per-language/
    macrolanguage entry; otherwise emit the language with the generic
    ``ZZ`` "unknown region" tag, same as languages that already resolve
    there genuinely (e.g. 'aai' -> 'aai-Latn-ZZ' is real CLDR data, not a
    fallback).

    Any script subtag a source ``lang`` field already carries survives
    untouched (full tags pass straight through, above); bare codes are
    expanded to language-region only, matching every other full tag
    already in the registry (none currently carry a script subtag).
    """
    if "-" in tag:
        return tag
    if tag in _lang_cache:
        return _lang_cache[tag][0]
    import langcodes
    from langcodes.data_dicts import LIKELY_SUBTAGS

    try:
        lang = langcodes.Language.get(tag)
        genuine = False
        for broader in lang.broader_tags():
            if broader in LIKELY_SUBTAGS:
                genuine = broader != "und"
                break
        maxed = lang.maximize()
        if genuine:
            region = maxed.territory or "ZZ"
        else:
            region = "ZZ"
        full = f"{maxed.language}-{region}"
    except Exception:
        full = f"{tag}-ZZ"
    _lang_cache[tag] = [full]
    return full


GENERATED_NOTES_MARKER = "from the phoonnx voice_index"


def load_existing_models() -> set[str]:
    """model/voice values already covered by hand-authored fighter files.

    Files this script generated itself (tagged via ``GENERATED_NOTES_MARKER``
    in their ``notes``) are excluded from this set — otherwise a rerun would
    treat its own prior output as a dedupe barrier and skip regenerating
    every voice, even when the generation logic (e.g. lang normalization)
    changed. Only genuinely hand-authored fighters gate dedupe.
    """
    existing = set()
    for fn in os.listdir(TTS_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(TTS_DIR, fn)
        with open(path) as f:
            data = json.load(f)
        if data.get("plugin") != "ovos-tts-plugin-phoonnx":
            continue
        if GENERATED_NOTES_MARKER in (data.get("notes") or ""):
            continue
        model = data.get("model")
        if model:
            existing.add(model)
        voice = (
            data.get("config", {})
            .get("tts", {})
            .get("ovos-tts-plugin-phoonnx", {})
            .get("voice")
        )
        if voice:
            existing.add(voice)
    return existing


def main() -> None:
    os.makedirs(TTS_DIR, exist_ok=True)
    existing_models = load_existing_models()

    fighters: dict[str, dict] = {}
    skipped = 0
    per_family_count: dict[str, int] = {}
    per_family_skipped: dict[str, int] = {}

    for fname in sorted(os.listdir(VOICE_INDEX_DIR)):
        if not fname.endswith(".json"):
            continue
        family = fname[: -len(".json")]
        path = os.path.join(VOICE_INDEX_DIR, fname)
        with open(path) as f:
            voices = json.load(f)

        species = FAMILY_SPECIES.get(family, f"Phoonnx{family.title().replace('_', '')}")
        family_display = FAMILY_DISPLAY.get(family, family)
        description_base = DESCRIPTIONS.get(
            family,
            f"ONNX-runtime TTS via the phoonnx adapter framework, running a {family_display} voice.",
        )

        for voice_id, meta in voices.items():
            if voice_id in existing_models:
                skipped += 1
                per_family_skipped[family] = per_family_skipped.get(family, 0) + 1
                continue

            competitor_id = f"phoonnx-{slugify(family)}-{slugify(voice_id)}"
            langs = normalize_langs(meta.get("lang"))
            short_name = short_voice_name(voice_id)
            display_name = f"Phoonnx {family_display} ({short_name})"

            engine = meta.get("engine", family)
            phoneme_type = meta.get("phoneme_type")
            alphabet = meta.get("alphabet")
            model_url = meta.get("model_url")
            config_url = meta.get("config_url")

            notes_bits = [f"Voice '{voice_id}' from the phoonnx voice_index ({family} family)."]
            if phoneme_type:
                notes_bits.append(f"phoneme_type={phoneme_type}.")
            if alphabet:
                notes_bits.append(f"alphabet={alphabet}.")

            links = {"source": "https://github.com/TigreGotico/phoonnx"}
            if model_url:
                links["model"] = model_url
            if config_url:
                links["config"] = config_url

            fighter = {
                "competitor_id": competitor_id,
                "modality": "tts",
                "plugin": "ovos-tts-plugin-phoonnx",
                "config": {
                    "tts": {
                        "module": "ovos-tts-plugin-phoonnx",
                        "ovos-tts-plugin-phoonnx": {
                            "voice": voice_id,
                        },
                    }
                },
                "langs": langs,
                "display_name": display_name,
                "species": species,
                "types": ["neural-net"],
                "description": f"{description_base} Engine: {engine}.",
                "model": voice_id,
                "links": links,
                "notes": " ".join(notes_bits),
            }
            fighters[competitor_id] = fighter
            per_family_count[family] = per_family_count.get(family, 0) + 1

    written = 0
    for competitor_id in sorted(fighters):
        fighter = fighters[competitor_id]
        out_path = os.path.join(TTS_DIR, f"{competitor_id}.json")
        new_content = json.dumps(fighter, indent=2, sort_keys=False) + "\n"
        if os.path.exists(out_path):
            with open(out_path) as f:
                old_content = f.read()
            if old_content == new_content:
                continue
        with open(out_path, "w") as f:
            f.write(new_content)
        written += 1

    print(f"voice_index source: {VOICE_INDEX_DIR} ({SOURCE_NOTE})")
    print(f"Generated {len(fighters)} fighters, wrote/updated {written} files")
    print(f"Skipped {skipped} entries already covered by existing fighters")
    print("Per-family generated counts:")
    for fam in sorted(per_family_count):
        print(f"  {fam}: {per_family_count[fam]} (skipped {per_family_skipped.get(fam, 0)})")


if __name__ == "__main__":
    main()
