"""Per-language intelligibility judge resolution (§4 R16).

Owner directive: the TTS intelligibility judge is always the best OFFLINE
ASR for the clip's language — usually a conformer — loaded through
``onnx-asr``, never ``faster-whisper``. "Best for this language" is not
invented here: it mirrors what a real OVOS install would pick, by reading
the same source ovos-config itself uses to recommend an offline STT plugin
per language (``ovos_config/recommends/offline_stt/*.conf``, each a
``{"stt": {"module": "ovos-stt-plugin-onnx-asr", "ovos-stt-plugin-onnx-asr":
{"model": "..."}}}`` blob). That is not an installable Python API — it is a
small set of data files — so :data:`_RECOMMENDS` below is a pinned copy of
those files' models, refreshed by re-reading
``ovos_config/recommends/offline_stt/`` when ovos-config's recommends change.

ovos-config only recommends ~20 languages today. Every other language falls
back to :data:`_FALLBACK`, a small set of judges pinned to a known HF commit
(see :data:`_REVISIONS`), and then to onnx-asr's own bundled multilingual
``whisper-base`` wrapper. That is still the ``onnx-asr`` PACKAGE
(onnxruntime, MIT) — architecturally a Whisper checkpoint, but never
touching ``faster-whisper``.

Which languages can be judged at ALL is a separate question from which
model judges them, and it is NOT answered by the tables here.
:func:`judge_available` reads the installed ``ovos-stt-plugin-onnx-asr``
registry (``ovos_stt_plugin_onnxasr.defaults.LANG_DEFAULTS``) directly, so
a language the plugin has claimed since is never mistaken for one no ASR
model covers. Copying that registry into this module is what makes the
answer go stale.

``onnx_asr.load_model`` has no ``revision`` parameter (unlike
``faster_whisper.WhisperModel``) — it always resolves the model's *current*
HEAD on the Hub. :data:`_REVISIONS` therefore is NOT enforced at load time;
it is the HF commit sha pinned via ``HfApi().model_info(repo_id).sha`` at
authoring time and recorded on every scored row purely for provenance (§4
R16: "a judge upgrade must not silently reinterpret past scores" — if the
upstream repo moves, the recorded revision tells you the scores predate
that move even though the loader itself could not pin against it).
"""
from __future__ import annotations

# Mirrors ovos_config/recommends/offline_stt/*.conf's "model" values, keyed
# by the lowercase full BCP-47 tag (the .conf file's basename). Full-tag
# match wins; primary-subtag prefix match (mirroring ovos-config's own
# do_merge() lookup) is tried next in resolve_judge_model().
_RECOMMENDS: dict[str, str] = {
    "ca-es": "OpenVoiceOS/stt-ca-es-conformer-transducer-large-onnx",
    "da-dk": "nemo-parakeet-tdt-0.6b-v3",
    "de-de": "nemo-parakeet-tdt-0.6b-v3",
    "en-us": "nemo-parakeet-tdt-0.6b-v3",
    "es-es": "OpenVoiceOS/parakeet-rnnt-1.1b-cv17-es-ep18-1270h-onnx",
    "eu-es": "OpenVoiceOS/stt-eu-conformer-transducer-large-onnx",
    "fr-fr": "nemo-parakeet-tdt-0.6b-v3",
    "gl-es": "onnx-community/whisper-large-v3-turbo",
    "it-it": "nemo-parakeet-tdt-0.6b-v3",
    "nl-nl": "nemo-parakeet-tdt-0.6b-v3",
    "pt-pt": "OpenVoiceOS/whisper-medium-pt-onnx",
}

# Judges pinned to a known HF commit for languages ovos-config has no
# recommends/offline_stt/*.conf for. This is a revision-pinning override
# (see _REVISIONS), NOT a statement of which languages onnx-asr covers —
# judge_available() reads the plugin's own registry for that.
_FALLBACK: dict[str, str] = {
    # Best dedicated OVOS onnx export per language for langs ovos-config has
    # no recommends/offline_stt/*.conf for yet (verified against ovos-config
    # dev 2026-08-11: only the 11 langs above exist there). Citrinet appears
    # only where no better export exists (ko, zh).
    "ar": "OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pcd_v1.0_onnx",
    "fo": "OpenVoiceOS/carlosdanielhernandezmena-stt_fo_quartznet15x5_sp_ep163_100h_onnx",
    "hy": "OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx",
    "is": "OpenVoiceOS/carlosdanielhernandezmena-stt_is_quartznet15x5_ft_ep56_875h_onnx",
    "ka": "OpenVoiceOS/stt_ka_fastconformer_hybrid_large_pc_onnx",
    "kk": "OpenVoiceOS/stt_kk_ru_fastconformer_hybrid_large_onnx",
    "ko": "OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx",
    "mt": "OpenVoiceOS/carlosdanielhernandezmena-stt_mt_quartznet15x5_sp_ep255_64h_onnx",
    "uk": "OpenVoiceOS/stt_ua_fastconformer_hybrid_large_pc_onnx",
    "zh": "OpenVoiceOS/stt_zh_citrinet_1024_gamma_0_25_onnx",
    "fi": "nemo-parakeet-tdt-0.6b-v3",
    "hu": "nemo-parakeet-tdt-0.6b-v3",
    "lv": "nemo-parakeet-tdt-0.6b-v3",
    "ro": "nemo-parakeet-tdt-0.6b-v3",
    "sv": "nemo-parakeet-tdt-0.6b-v3",
    "sl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-sl-onnx",
    "fa": "OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx",
    "ja": "OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx",
    "vi": "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx",
    "tl": "OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx",
    # AI4Bharat IndicConformer per-language exports beat whisper-base on Indic langs.
    "bn": "OpenVoiceOS/ai4bharat-indicconformer-bn-onnx",
    "hi": "OpenVoiceOS/ai4bharat-indicconformer-hi-onnx",
    "kn": "OpenVoiceOS/ai4bharat-indicconformer-kn-onnx",
    "ta": "OpenVoiceOS/ai4bharat-indicconformer-ta-onnx",
    "te": "OpenVoiceOS/ai4bharat-indicconformer-te-onnx",
    "ur": "OpenVoiceOS/ai4bharat-indicconformer-ur-onnx",
}

#: onnx-asr's own multilingual Whisper wrapper (the `onnx-asr` PACKAGE, not
#: `faster-whisper`) — the panel's always-present multilingual member, and
#: the resolver's last resort for a language with no dedicated export.
_UNIVERSAL_FALLBACK = "whisper-base"

#: HF commit sha per model id, pinned via ``HfApi().model_info(id).sha`` at
#: authoring time (2026-08). Recorded for provenance only — see module
#: docstring for why it is not passed to ``onnx_asr.load_model``.
_REVISIONS: dict[str, str] = {
    "nemo-parakeet-tdt-0.6b-v3": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    "whisper-base": "998334d3bfe2deba3c8e6821f05388dbf2b706d2",
    "OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pcd_v1.0_onnx": "c5f78db4d5a8da706ab74cad73481c18b8d736b9",
    "OpenVoiceOS/stt-ca-es-conformer-transducer-large-onnx": "c43ddeda9f8bb739deb26162a6727cd47d52484a",
    "OpenVoiceOS/parakeet-rnnt-1.1b-cv17-es-ep18-1270h-onnx": "74c76d8c69e54472f2cd2a3507bd2c61d9874cf3",
    "OpenVoiceOS/stt-eu-conformer-transducer-large-onnx": "ee5f59fe21c70416988d24102a7a5fe22e128242",
    "OpenVoiceOS/carlosdanielhernandezmena-stt_fo_quartznet15x5_sp_ep163_100h_onnx": "e797c7df6602ae73b15e78cea32a40ba2ab6851b",
    "onnx-community/whisper-large-v3-turbo": "360ebcde2559d60bb474678be3c1de9ef347d01a",
    "OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx": "f7db1fad9af6bd8a7a9e08fca02068c403b35468",
    "OpenVoiceOS/carlosdanielhernandezmena-stt_is_quartznet15x5_ft_ep56_875h_onnx": "8c04e8b54f82c699e7457f18f71db74cd2bcdd1b",
    "OpenVoiceOS/stt_ka_fastconformer_hybrid_large_pc_onnx": "fa19e061165a0238fe6477c65cdc3357d56f5788",
    "OpenVoiceOS/stt_kk_ru_fastconformer_hybrid_large_onnx": "489c2d5b5671509737c64b44ee6b2a2b7d619558",
    "OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx": "74230b92a06ce3e7e6f8214492793a94c91bc59f",
    "OpenVoiceOS/carlosdanielhernandezmena-stt_mt_quartznet15x5_sp_ep255_64h_onnx": "cff51ea8349448abf5c8bcbd383a3ec29fd8fb75",
    "OpenVoiceOS/whisper-medium-pt-onnx": "7db38a22790ba3f831702db12cb19dd684642bf5",
    "OpenVoiceOS/stt_ua_fastconformer_hybrid_large_pc_onnx": "d7ffacec32e95786d22f3c7417348fa6f5a02c98",
    "OpenVoiceOS/stt_zh_citrinet_1024_gamma_0_25_onnx": "94147e6e63f09b133c4366f38892dca83d7cb30a",
    "OpenVoiceOS/ai4bharat-indicconformer-bn-onnx": "46053d8f1c1b1ed12eb1b34f3f7ccf8512fb08b1",
    "OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx": "d84de4ccefe28598d006d5827210352aa9053a0d",
    "OpenVoiceOS/ai4bharat-indicconformer-hi-onnx": "8960b8611af5b8c375d442d52907360176410c8b",
    "OpenVoiceOS/ai4bharat-indicconformer-kn-onnx": "55b6f618ade2fc7cf8127f1c6778ea069961fb8c",
    "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-sl-onnx": "29eaa10f01e70113d436b3460865db1779821c52",
    "OpenVoiceOS/ai4bharat-indicconformer-ta-onnx": "10e43940d12106763c3aebe1922b70d37fc0c6fd",
    "OpenVoiceOS/ai4bharat-indicconformer-te-onnx": "d952dec2a1d256a1569cdfc9a9615cba05a1a2e3",
    "OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx": "e7eb84062e138d2f531ef349f0c5a164846c019b",
    "OpenVoiceOS/ai4bharat-indicconformer-ur-onnx": "c93610bb4c08642b1e48b6f17b2141d5446898fc",
    "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx": "ed9f55ba980eb1c9eeba02a5733eba7cba02f6e7",
    "OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx": "4353e7b9e2e9ebdd35e85b8140c7351d03c2219c",
}


def resolve_judge_model(lang: str) -> tuple[str, str]:
    """Resolve a BCP-47 ``lang`` tag to ``(onnx-asr model id, pinned revision)``.

    Lookup order (mirrors ovos-config's ``do_merge()``): exact full-tag
    match in the ovos-config recommends copy, then a primary-subtag prefix
    match within it, then the onnx-asr LANG_DEFAULTS fallback table by
    primary subtag, then the universal onnx-asr ``whisper-base`` wrapper.
    """
    full = lang.lower()
    primary = full.split("-")[0]

    model_id = _RECOMMENDS.get(full)
    if model_id is None:
        prefix_matches = sorted(k for k in _RECOMMENDS if k.split("-")[0] == primary)
        if prefix_matches:
            model_id = _RECOMMENDS[prefix_matches[0]]
    if model_id is None:
        model_id = _FALLBACK.get(primary, _UNIVERSAL_FALLBACK)

    revision = _REVISIONS.get(model_id) or "main"
    return model_id, revision


def resolve_judge_panel(lang: str) -> list[tuple[str, str]]:
    """Resolve ``lang`` to a small panel of ``(model id, revision)`` judges
    for ROVER consensus intelligibility scoring (§4 R16 extension).

    Owner directive: "use onnx-asr lang specific models + whisper as
    judges" — the panel is every distinct language-specific onnx-asr model
    available for ``lang`` (the ovos-config recommends primary from
    :func:`resolve_judge_model`, plus any distinct onnx-asr LANG_DEFAULTS
    alternate for the same primary subtag), deduped by model id, PLUS the
    multilingual ``whisper-base`` wrapper, which is always a panel member —
    not merely a long-tail fallback. The primary judge (used for
    ``resolve_judge_model`` and as the ROVER tie-break) stays the
    language-specific model when one exists, or ``whisper-base`` itself for
    languages with no dedicated export — those get a panel of one, since
    there is nothing else to vote against.

    Never includes ``faster-whisper`` (see module docstring) — only
    ``onnx-asr``-loadable models.
    """
    primary_id, primary_rev = resolve_judge_model(lang)
    panel: list[tuple[str, str]] = [(primary_id, primary_rev)]
    seen = {primary_id}

    full = lang.lower()
    primary_tag = full.split("-")[0]

    candidate_ids: list[str] = []
    recommend_id = _RECOMMENDS.get(full)
    if recommend_id is None:
        prefix_matches = sorted(k for k in _RECOMMENDS if k.split("-")[0] == primary_tag)
        if prefix_matches:
            recommend_id = _RECOMMENDS[prefix_matches[0]]
    if recommend_id is not None:
        candidate_ids.append(recommend_id)
    fallback_id = _FALLBACK.get(primary_tag)
    if fallback_id is not None:
        candidate_ids.append(fallback_id)

    for model_id in candidate_ids:
        if model_id not in seen:
            seen.add(model_id)
            panel.append((model_id, _REVISIONS.get(model_id) or "main"))

    if _UNIVERSAL_FALLBACK not in seen:
        panel.append(
            (_UNIVERSAL_FALLBACK, _REVISIONS.get(_UNIVERSAL_FALLBACK) or "main")
        )

    return panel


def _plugin_claims(lang: str) -> bool:
    """Whether ``ovos-stt-plugin-onnx-asr``'s registry holds a model for ``lang``.

    Asked of the installed plugin rather than a copy kept here: a copy goes
    stale silently, and a language the plugin has claimed since would keep
    reading as unjudgeable. The plugin's own ``_match`` does the lookup so
    the answer matches what a real install resolves, nearest-tag matching
    included (``nb`` reaches the ``no`` entry) — reimplementing that here
    would be the same staleness in a different shape. Imported lazily:
    ``arena.metrics`` and the non-audio CLI paths stay importable on a base
    install.
    """
    try:
        from ovos_stt_plugin_onnxasr.defaults import LANG_DEFAULTS, _match
    except ImportError as exc:
        raise RuntimeError(
            "resolving TTS intelligibility judges requires the "
            "'ovos-stt-plugin-onnx-asr' package (it holds the per-language "
            "ASR registry) — install the 'audio' extra: "
            "pip install ovos-plugin-arena[audio]"
        ) from exc
    return _match(lang, LANG_DEFAULTS) is not None


def judge_available(lang: str) -> bool:
    """Whether an ASR model claims ``lang``, so a round trip can be judged.

    True when ovos-config recommends an offline STT for the language, when
    :data:`_FALLBACK` pins a dedicated judge for it, or when the installed
    ``ovos-stt-plugin-onnx-asr`` registry holds an entry for it — including
    the languages that registry deliberately assigns to ``whisper-base``,
    which Whisper does transcribe.

    False when nothing claims the language. Such a language reaches
    ``whisper-base`` only through the plugin's blanket ``DEFAULT_CPU_MODEL``,
    with no model ever trained on it: the panel transcribes noise, and the
    round-trip error rate that comes back measures the ASR fleet's coverage
    rather than the voice (§4 R16). Callers MUST skip intelligibility
    scoring for those languages instead of recording the number.

    ``lang`` is one clip's own language tag. The dataset-level ``"multi"``
    tag is not a language and raises — a multilingual corpus is benched one
    real language at a time, and silently reading ``"multi"`` as unjudgeable
    would drop intelligibility from every language on such a board.
    """
    full = lang.lower()
    if not full or full == "multi":
        raise ValueError(
            f"{lang!r} is not a language tag — judge availability is decided "
            f"per clip language, not per dataset"
        )
    primary = full.split("-")[0]
    if full in _RECOMMENDS or any(k.split("-")[0] == primary for k in _RECOMMENDS):
        return True
    if primary in _FALLBACK:
        return True
    return _plugin_claims(full) or _plugin_claims(primary)
