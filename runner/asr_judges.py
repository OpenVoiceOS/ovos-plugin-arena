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
back to :data:`_FALLBACK`, the best onnx-asr model per language from
``ovos-stt-plugin-onnx-asr``'s own built-in registry
(``ovos_stt_plugin_onnxasr.defaults.LANG_DEFAULTS``), itself sourced from the
``OpenVoiceOS/stt-asr-onnx`` HuggingFace collection. A handful of long-tail
languages (af, am, az, cy, he, id, jv, km, mn, ms, my, nb, sq, sw, th, tr)
have no dedicated onnx-asr fine-tune published yet anywhere in that
collection, so they fall back further to onnx-asr's own bundled
multilingual ``whisper-base`` wrapper. That is still the ``onnx-asr``
PACKAGE (onnxruntime, MIT) — architecturally a Whisper checkpoint, but never
touching ``faster-whisper``. Document here, not silently, if any of those
gets a dedicated export later.

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

# ovos-stt-plugin-onnx-asr's built-in LANG_DEFAULTS, keyed by primary
# subtag, for languages ovos-config has no recommends/offline_stt/*.conf
# for yet. Dedicated fine-tunes beat the multilingual fillers on their
# language; multilingual `nemo-parakeet-tdt-0.6b-v3` and `whisper-base`
# cover the rest.
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
#: `faster-whisper`) — last-resort coverage for languages with no dedicated
#: onnx-asr fine-tune anywhere yet: af, am, az, cy, he, id, jv, km, mn, ms,
#: my, nb, sq, sw, th, tr.
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
