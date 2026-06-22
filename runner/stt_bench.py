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
from typing import Iterator, Tuple

from runner.audio_io import stream_audio_dataset
from runner.media_bench import MediaBenchAdapter, PredictContext

log = logging.getLogger("stt-bench")


class STTBench(MediaBenchAdapter):
    modality = "stt"
    card_tags = ("automatic-speech-recognition", "stt")
    card_task = "Per-clip transcripts"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[Tuple[str, dict]]:
        fields = dataset_def.reference_fields or {}
        audio_key = fields.get("audio", "audio")
        gt_col = fields.get("ground_truth", "transcription")
        yield from stream_audio_dataset(
            dataset_def.source,
            audio_key=audio_key,
            extra_keys={"ground_truth": gt_col},
            revision=revision,
            max_samples=max_samples,
        )

    def load_engine(self, competitor, lang: str):
        from ovos_plugin_manager.stt import load_stt_plugin

        stt_cfg = competitor.config.get("stt", {})
        module = stt_cfg.get("module") or competitor.plugin
        plugin_cfg = dict(stt_cfg.get(module, {}))
        clazz = load_stt_plugin(module)
        if clazz is None:
            raise RuntimeError(f"STT plugin not found: {module}")
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
        }


def _first_hypothesis(result) -> Tuple[str, float]:
    """Normalise the STT.transcribe return into (text, confidence)."""
    if isinstance(result, list) and result:
        head = result[0]
        if isinstance(head, (list, tuple)):
            return str(head[0]), float(head[1]) if len(head) > 1 else 1.0
        return str(head), 1.0
    if isinstance(result, tuple):
        return str(result[0]), float(result[1]) if len(result) > 1 else 1.0
    return str(result or ""), 1.0
