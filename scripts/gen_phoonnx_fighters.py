#!/usr/bin/env python3
"""Generate arena TTS fighter files for every voice in the phoonnx voice_index.

Source of truth: ``phoonnx/voice_index/*.json`` (one file per engine family,
mapping voice_id -> voice metadata). ``--voice-index DIR`` names the
directory to read; without it the script reads the installed phoonnx
package if it ships a ``voice_index/`` dir, otherwise the phoonnx dev
checkout at ``/home/miro/AgentWorkspaces/ovos/phoonnx/phoonnx/voice_index``.

22 fighters for OpenVoiceOS/phoonnx_* voices (the OVOS.json family) already
exist by hand in registry/competitors/tts/ — those are left untouched and
matching voice_index entries are skipped (matched by ``model``/``voice_id``).

One voice becomes one fighter. The index lists some model files under more
than one voice_id (a curated community entry, a stock piper entry and a
mechanical ``hf_community`` scrape of the same upload). Voices are keyed on
the Hugging Face LFS sha256 of their ``model_url`` plus their languages and
voice-selection fields, and only the preferred voice_id of each key becomes a
fighter (see ``voice_rank``). A voice whose model file cannot be resolved
anonymously on the Hub can never run, so it gets no fighter. Neither does a
voice whose languages no registered TTS prompt set holds: the bench has no
text for it to read, and the voice returns once a prompt set covers its
language (matched by ``runner.media_bench._lang_matches``). Generated files
that the current index no longer produces are removed.

Output is deterministic (sorted by competitor_id) and idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Iterable

_FALLBACK_DIR = "/home/miro/AgentWorkspaces/ovos/phoonnx/phoonnx/voice_index"


def resolve_voice_index_dir(override: str | None) -> str:
    """The voice_index directory to read: the override, the package, the checkout."""
    if override:
        return override
    try:
        import phoonnx  # type: ignore

        installed = os.path.join(os.path.dirname(phoonnx.__file__), "voice_index")
        if os.path.isdir(installed):
            return installed
    except ImportError:
        pass
    return _FALLBACK_DIR


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
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

_DESCRIPTION_PREFIX = "ONNX-runtime TTS via the phoonnx adapter framework, running "
DESCRIPTIONS = {
    "BSC": _DESCRIPTION_PREFIX + "a BSC (Barcelona Supercomputing Center) voice.",
    "MMS": _DESCRIPTION_PREFIX + "a Meta MMS (Massively Multilingual Speech) VITS voice.",
    "OVOS": _DESCRIPTION_PREFIX + "an OpenVoiceOS community voice.",
    "chatterbox": _DESCRIPTION_PREFIX + "a Chatterbox voice.",
    "coqui_community": _DESCRIPTION_PREFIX + "a community-trained Coqui TTS voice.",
    "coqui_vits": _DESCRIPTION_PREFIX + "a Coqui VITS voice.",
    "f5tts": _DESCRIPTION_PREFIX + "an F5-TTS voice.",
    "fastpitch": _DESCRIPTION_PREFIX + "a FastPitch voice.",
    "glowtts": _DESCRIPTION_PREFIX + "a GlowTTS voice.",
    "mimic3": _DESCRIPTION_PREFIX + "a Mycroft Mimic3 voice.",
    "mixertts": _DESCRIPTION_PREFIX + "a MixerTTS voice.",
    "neurlang": _DESCRIPTION_PREFIX + "a Neurlang voice.",
    "optispeech": _DESCRIPTION_PREFIX + "an OptiSpeech voice.",
    "phonikud": _DESCRIPTION_PREFIX + "a Phonikud voice.",
    "piper": _DESCRIPTION_PREFIX + "a Piper VITS voice.",
    "piper_community": _DESCRIPTION_PREFIX + "a community-trained Piper voice.",
    "proxectonos": _DESCRIPTION_PREFIX + "a Proxecto Nós (Galician) voice.",
    "shami": _DESCRIPTION_PREFIX + "a Shami voice.",
    "styletts2": _DESCRIPTION_PREFIX + "a StyleTTS2 voice.",
    "supertonic": _DESCRIPTION_PREFIX + "a SuperTonic voice.",
    "transformers_community": _DESCRIPTION_PREFIX + "a community HF Transformers TTS voice.",
    "vits2": _DESCRIPTION_PREFIX + "a VITS2 voice.",
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

# Index tags that miss the language a voice reads. The prompt sets use the
# written-standard code: Norwegian Bokmål for ``no``, Azerbaijani for the
# Latin-script North Azerbaijani voice, Swahili for the ``swh`` voice.
TAG_FIXES = {
    "no-NO": "nb-NO",
    "azj-Latn": "az-AZ",
    "swh": "sw-TZ",
    # CLDR aliases tl to fil; the prompt sets hold Tagalog as tl-PH.
    "tl": "tl-PH",
}

# Voices the index tags ``mul``: the languages the checkpoint is trained on.
VOICE_LANGS = {
    "f5tts/v1-base": ["en-US", "zh-CN"],
}


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
        out.append(TAG_FIXES.get(part) or _normalize_one(part))
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


# A curated community upload is the original; a stock ``piper/`` entry that
# claims the same file comes next; a re-upload of another author's voice and
# the mechanical ``hf_community`` scrape come last.
REUPLOADERS = frozenset({"brycebeattie"})

# Fields that name where a voice's files live or how it is labelled, not
# which voice it is.
_NON_IDENTITY_FIELDS = frozenset({"voice_id", "model_url", "config_url", "display_name", "lang"})

_HF_RESOLVE = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")


def voice_rank(voice_id: str) -> tuple[int, str]:
    """Sort key among voice_ids that share one model file: lowest is kept."""
    prefix, _, rest = voice_id.partition("/")
    if prefix == "hf_community":
        rank = 3
    elif prefix == "piper_community" and rest.split("/")[0].lower() in REUPLOADERS:
        rank = 2
    elif prefix == "piper":
        rank = 1
    else:
        rank = 0
    return rank, voice_id


def fetch_model_sha256(model_urls: Iterable[str]) -> dict[str, str | None]:
    """LFS sha256 of each Hub ``resolve`` URL; None when it is not reachable anonymously.

    Only a missing, private or gated repo counts as unreachable. Any other
    error (a rate limit, a server error, no network) propagates, so a
    transient failure stops the run before it removes any fighter.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    api = HfApi(token=False)
    wanted: dict[tuple[str, str], dict[str, str]] = {}
    shas: dict[str, str | None] = {}
    for url in model_urls:
        shas[url] = None
        match = _HF_RESOLVE.match(url)
        if match:
            repo, revision, path = match.groups()
            wanted.setdefault((repo, revision), {})[path] = url
    for (repo, revision), by_path in sorted(wanted.items()):
        paths = sorted(by_path)
        for start in range(0, len(paths), 100):
            try:
                infos = api.get_paths_info(repo, paths[start : start + 100], revision=revision)
            except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
                print(f"  {repo}@{revision}: not reachable ({type(exc).__name__})")
                break
            except HfHubHTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in (401, 403, 404):
                    raise
                print(f"  {repo}@{revision}: not reachable (HTTP {status})")
                break
            for info in infos:
                lfs = getattr(info, "lfs", None)
                shas[by_path[info.path]] = lfs.sha256 if lfs else info.blob_id
    return shas


def _primary_language(tag: str) -> str:
    """Primary subtag after folding individual languages into their macrolanguage."""
    import langcodes

    return langcodes.standardize_tag(tag, macro=True).split("-")[0]


def dedupe_voices(
    voices: list[tuple[str, dict, list[str]]],
    model_sha256: Callable[[str], str | None],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Split ``(voice_id, meta, langs)`` entries into kept, duplicate and unreachable.

    Two voices are one voice when their model file has the same sha256, they
    read the same languages and every voice-selection field matches. A piper
    model reads one language whatever its tag says, so piper voices compare
    on the primary language; other engines take the tag as a model input
    (one chatterbox model speaks ten Arabic dialects by region). Returns
    the kept voice_ids, a map from each dropped duplicate to the voice_id
    kept in its place, and the voice_ids whose model file is unreachable.
    """
    groups: dict[str, list[str]] = {}
    unreachable = []
    for voice_id, meta, langs in voices:
        sha = model_sha256(meta.get("model_url") or "")
        if sha is None:
            unreachable.append(voice_id)
            continue
        if meta.get("engine") == "piper":
            lang_key = sorted({_primary_language(tag) for tag in langs})
        else:
            lang_key = sorted(langs)
        selection = {k: v for k, v in meta.items() if k not in _NON_IDENTITY_FIELDS}
        key = json.dumps([sha, lang_key, selection], sort_keys=True)
        groups.setdefault(key, []).append(voice_id)
    kept, duplicates = [], {}
    for ids in groups.values():
        keep, *drops = sorted(ids, key=voice_rank)
        kept.append(keep)
        duplicates.update({drop: keep for drop in drops})
    return sorted(kept), duplicates, sorted(unreachable)


def prompt_languages() -> list[str]:
    """Every language a registered TTS prompt set holds text for."""
    from registry.loaders import list_datasets

    return sorted({lang for dataset in list_datasets("tts") for lang in dataset.langs or []})


def has_prompts(langs: list[str], prompt_langs: list[str]) -> bool:
    """True if the bench would run a voice with these langs on some prompt set."""
    from runner.media_bench import _lang_matches

    return any(_lang_matches(lang, p) for lang in langs for p in prompt_langs)


def build_fighter(family: str, voice_id: str, meta: dict, langs: list[str]) -> dict:
    """The registry fighter for one voice_index entry."""
    family_display = FAMILY_DISPLAY.get(family, family)
    description_base = DESCRIPTIONS.get(
        family,
        _DESCRIPTION_PREFIX + f"a {family_display} voice.",
    )
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

    return {
        "competitor_id": f"phoonnx-{slugify(family)}-{slugify(voice_id)}",
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
        "display_name": f"Phoonnx {family_display} ({short_voice_name(voice_id)})",
        "species": FAMILY_SPECIES.get(family, f"Phoonnx{family.title().replace('_', '')}"),
        "types": ["neural-net"],
        "description": f"{description_base} Engine: {engine}.",
        "model": voice_id,
        "links": links,
        "notes": " ".join(notes_bits),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--voice-index", help="phoonnx voice_index directory to read")
    args = parser.parse_args(argv)

    voice_index_dir = resolve_voice_index_dir(args.voice_index)
    if not os.path.isdir(voice_index_dir):
        print(f"ERROR: no voice_index directory at {voice_index_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(TTS_DIR, exist_ok=True)
    existing_models = load_existing_models()

    entries: dict[str, tuple[str, dict, list[str]]] = {}
    skipped = 0
    for fname in sorted(os.listdir(voice_index_dir)):
        if not fname.endswith(".json"):
            continue
        family = fname[: -len(".json")]
        with open(os.path.join(voice_index_dir, fname)) as f:
            voices = json.load(f)
        for voice_id, meta in voices.items():
            if voice_id in existing_models:
                skipped += 1
                continue
            langs = VOICE_LANGS.get(voice_id) or normalize_langs(meta.get("lang"))
            entries[voice_id] = (family, meta, langs)

    shas = fetch_model_sha256(meta.get("model_url") or "" for _, meta, _ in entries.values())
    kept, duplicates, unreachable = dedupe_voices(
        [(voice_id, meta, langs) for voice_id, (_, meta, langs) in entries.items()],
        shas.get,
    )

    prompt_langs = prompt_languages()
    no_prompts = [v for v in kept if not has_prompts(entries[v][2], prompt_langs)]

    fighters = {}
    for voice_id in sorted(set(kept) - set(no_prompts)):
        family, meta, langs = entries[voice_id]
        fighter = build_fighter(family, voice_id, meta, langs)
        fighters[fighter["competitor_id"]] = fighter

    written = 0
    for competitor_id in sorted(fighters):
        out_path = os.path.join(TTS_DIR, f"{competitor_id}.json")
        new_content = json.dumps(fighters[competitor_id], indent=2, sort_keys=False) + "\n"
        if os.path.exists(out_path):
            with open(out_path) as f:
                if f.read() == new_content:
                    continue
        with open(out_path, "w") as f:
            f.write(new_content)
        written += 1

    removed = 0
    for fn in sorted(os.listdir(TTS_DIR)):
        if not fn.endswith(".json") or fn[: -len(".json")] in fighters:
            continue
        path = os.path.join(TTS_DIR, fn)
        with open(path) as f:
            data = json.load(f)
        if GENERATED_NOTES_MARKER in (data.get("notes") or ""):
            os.remove(path)
            removed += 1

    print(f"voice_index source: {voice_index_dir}")
    print(f"Read {len(entries)} voices, skipped {skipped} covered by hand-authored fighters")
    print(f"Dropped {len(duplicates)} duplicate voices (same model file, langs and selection):")
    for drop in sorted(duplicates):
        print(f"  {drop} -> {duplicates[drop]}")
    print(f"Dropped {len(unreachable)} voices whose model file is not reachable anonymously:")
    for voice_id in unreachable:
        print(f"  {voice_id}")
    print(f"Dropped {len(no_prompts)} voices whose languages no TTS prompt set holds")
    print(f"Generated {len(fighters)} fighters, wrote/updated {written}, removed {removed} files")


if __name__ == "__main__":
    main()
