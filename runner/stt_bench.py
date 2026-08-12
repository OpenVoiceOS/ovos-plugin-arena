"""
STT benchmark adapter (§3.2) for the shared :mod:`runner.media_bench` engine.

Instantiates each registry STT fighter's real OVOS plugin from its
``mycroft.conf`` ``stt`` block, transcribes every eval clip and records the
hypothesis next to the reference transcript.  WER is left for the arena to
compute on ingest (``arena.metrics.row_wer``) so the dataset stays the single
source of truth for the scoring formula.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from runner.audio_io import stream_audio_dataset, stream_manifest_audio
from runner.media_bench import MediaBenchAdapter, PredictContext, load_plugin_class

log = logging.getLogger("stt-bench")

# STM-format (NIST/Kaldi-style) sentinel reference values that mark a segment
# as excluded from scoring rather than an actual transcript — e.g. EdAcc
# (edinburghcstr/edacc) carries ~2% of its rows as
# "IGNORE_TIME_SEGMENT_IN_SCORING", the standard STM out-of-bounds marker also
# seen in TED-LIUM-family corpora. Left unfiltered, every fighter's WER on
# such a dataset is uniformly inflated by these unscoreable rows (there is no
# real reference text to compare a hypothesis against). Checked EdAcc's other
# short/all-caps values (YEAH, MM HMM, <OVERLAP>, <LAUGH>, ...) — those are
# genuine transcribed utterances/annotations, not sentinels, so they are left
# in place.
_SENTINEL_REFERENCES = {"IGNORE_TIME_SEGMENT_IN_SCORING"}


def _is_sentinel_reference(text) -> bool:
    return isinstance(text, str) and text.strip() in _SENTINEL_REFERENCES


class STTBench(MediaBenchAdapter):
    modality = "stt"
    card_tags = ("automatic-speech-recognition", "stt")
    card_task = "Per-clip transcripts"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[tuple[str, dict]]:
        fields = dataset_def.reference_fields or {}
        audio_key = fields.get("audio", "audio")
        gt_col = fields.get("ground_truth", "transcription")
        src = dataset_def.source
        # Multi-lang corpora (lang="multi") template {lang} into subset /
        # file_pattern so one registry entry covers every language, e.g.
        # source.subset="{lang}" for a per-language parquet config.
        if "{lang}" in (src.subset or "") or "{lang}" in (src.file_pattern or ""):
            update = {}
            if src.subset:
                update["subset"] = src.subset.format(lang=lang)
            if src.file_pattern:
                update["file_pattern"] = src.file_pattern.format(lang=lang)
            src = src.model_copy(update=update)
        # Manifest-backed corpora (metadata.csv / manifest.jsonl beside the
        # audio, e.g. speech_MASSIVE_pt-PT) vs parquet-embedded audio (MInDS-14).
        streamer = (stream_manifest_audio
                    if (src.file_pattern or "").endswith((".csv", ".tsv", ".jsonl"))
                    else stream_audio_dataset)
        # max_samples counts against the streamer's own cap, so pull one extra
        # sentinel-skip pass worth of slack isn't needed here: the streamer
        # yields raw rows and this generator filters after, same as any other
        # consumer — a caller asking for N samples may get fewer than N if
        # sentinel rows fall inside the requested window, same tradeoff the
        # resumable JSONL skip logic already accepts for failed samples.
        for sample_id, sample in streamer(
            src,
            audio_key=audio_key,
            extra_keys={"ground_truth": gt_col},
            revision=revision,
            max_samples=max_samples,
        ):
            if _is_sentinel_reference(sample.get("ground_truth")):
                log.debug("skipping sentinel-reference sample %s", sample_id)
                continue
            yield sample_id, sample

    def load_engine(self, competitor, lang: str):
        from ovos_plugin_manager.stt import load_stt_plugin

        stt_cfg = competitor.config.get("stt", {})
        module = stt_cfg.get("module") or competitor.plugin
        plugin_cfg = dict(stt_cfg.get(module, {}))
        clazz = load_plugin_class(load_stt_plugin, module)
        return clazz({"lang": lang, "module": module, **plugin_cfg})

    def predict(self, engine, sample: dict, ctx: PredictContext) -> dict:
        from ovos_plugin_manager.utils.audio import AudioData

        audio = AudioData.from_array(
            sample["array"], sample_rate=sample["sr"], sample_width=2
        )
        start = time.perf_counter()
        result = engine.transcribe(audio, lang=ctx.lang)
        latency_ms = (time.perf_counter() - start) * 1000

        text, conf = _first_hypothesis(result)
        return {
            "reference_text": sample.get("ground_truth"),
            "prediction": text,
            "confidence": conf,
            # parquet-embedded clips have no stable per-sample URL, so STT
            # battles fall back to text; a manifest-backed corpus would carry one
            "audio_url": sample.get("audio_url"),
            "latency_ms": round(latency_ms, 3),
            # input clip duration (RTF = elapsed_ms / 1000 / audio_secs, §M1)
            "audio_secs": round(len(sample["array"]) / sample["sr"], 3),
        }


def _first_hypothesis(result) -> tuple[str, float]:
    """Normalise the STT.transcribe return into (text, confidence)."""
    if isinstance(result, list) and result:
        head = result[0]
        if isinstance(head, (list, tuple)):
            return str(head[0]), float(head[1]) if len(head) > 1 else 1.0
        return str(head), 1.0
    if isinstance(result, tuple):
        return str(result[0]), float(result[1]) if len(result) > 1 else 1.0
    return str(result or ""), 1.0
