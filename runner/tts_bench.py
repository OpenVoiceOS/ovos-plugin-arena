"""
TTS benchmark adapter (§3.2) for :mod:`runner.media_bench`.

TTS has no *ground-truth* reference metric — there is no single correct
waveform for a given prompt — so blind human preference votes stay the
league's primary signal (spec §2.1, §3.2).  This benchmark synthesises: it
reads a prompt corpus, calls each registry fighter's real OVOS TTS plugin to
render every prompt, stores the clip and records its URL as the prediction.
The arena assembles those clips into blind A/B listening battles for human
voting *and* scores every clip with UTMOS (reference-free naturalness MOS,
§4 R14) to drive an objective benchmark board and benchmark-seeded ELO
votes, exactly like the other leagues.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from runner.media_bench import MediaBenchAdapter, PredictContext, load_plugin_class

log = logging.getLogger("tts-bench")

#: Judge identity recorded on every scored row (§4 R14) — provenance for
#: the objective TTS board, since a different judge revision is not directly
#: comparable to this one.
UTMOS_JUDGE = "TigreGotico/utmos-onnx"
UTMOS_JUDGE_REVISION = "ff41b8f440cb12ecda18261f9ff7326d058275ce"

_utmos_judge = None  # module-cached lazy singleton, one ONNX session per process


def _get_utmos_judge():
    """Lazily import and cache the UTMOS judge.

    ``speechonnxmetrics`` is an optional (``audio`` extra) dependency —
    importing it at module load would break every environment that doesn't
    run TTS benchmarks. Scoring is NOT optional for a TTS run though: if the
    package is missing when a clip is actually synthesised, this raises a
    clear, actionable error rather than silently skipping the score.
    """
    global _utmos_judge
    if _utmos_judge is None:
        try:
            from speechonnxmetrics.mos.utmos import UTMOS
        except ImportError as exc:
            raise RuntimeError(
                "TTS benchmarking requires the 'speechonnxmetrics' package "
                "(objective UTMOS scoring is not optional for TTS runs) — "
                "install the 'audio' extra: pip install ovos-plugin-arena[audio]"
            ) from exc
        _utmos_judge = UTMOS()
    return _utmos_judge


class TTSBench(MediaBenchAdapter):
    modality = "tts"
    card_tags = ("text-to-speech", "tts")
    card_task = "Synthesised clips (one per prompt)"

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[tuple[str, dict]]:
        text_col = (dataset_def.reference_fields or {}).get("text", "text")
        rows = _load_prompts(dataset_def, lang, revision, text_col)
        if max_samples:
            rows = rows[:max_samples]
        for i, text in enumerate(rows):
            yield f"{lang}/{i:05d}", {"input_text": text}

    def load_engine(self, competitor, lang: str):
        from ovos_plugin_manager.tts import load_tts_plugin

        tts_cfg = competitor.config.get("tts", {})
        module = tts_cfg.get("module") or competitor.plugin
        plugin_cfg = dict(tts_cfg.get(module, {}))
        clazz = load_plugin_class(load_tts_plugin, module)
        return clazz({"lang": lang, "module": module, **plugin_cfg})

    def predict(self, engine, sample: dict, ctx: PredictContext) -> dict:
        text = sample["input_text"]
        rel = (f"{ctx.lang}/{ctx.competitor.competitor_id}/"
               f"{_safe(text)}.wav")
        wav_path = ctx.audio_dir / rel
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        engine.get_tts(text, str(wav_path), lang=ctx.lang)
        latency_ms = (time.perf_counter() - start) * 1000

        judge = _get_utmos_judge()
        # ``sr`` here is the *input's* sample rate, not the model's — for a
        # path input it is ignored anyway (the loader reads the real rate
        # from the wav header), so this value is only a placeholder.
        score = judge(str(wav_path), judge.sample_rate)

        return {
            "input_text": text,
            "prediction": ctx.hf_audio_url(rel),
            "audio_url": ctx.hf_audio_url(rel),
            "latency_ms": round(latency_ms, 3),
            # PredictionRow has no modeled utmos field — these MUST be nested
            # under "extras" (§3.2) or pydantic silently drops them and the
            # objective board goes empty; see runner/media_bench.py:make_row
            # (row.update(fields)) and arena/predictions.py:parse_row.
            "extras": {
                "utmos": round(float(score), 4),
                "utmos_judge": UTMOS_JUDGE,
                "utmos_judge_revision": UTMOS_JUDGE_REVISION,
            },
        }


def _safe(text: str) -> str:
    """Stable, filesystem-safe clip name from the prompt text."""
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_prompts(dataset_def, lang: str, revision: str, text_col: str) -> list[str]:
    """Read prompt strings for one language (HF split or per-lang file)."""
    source = dataset_def.source
    if getattr(source, "file_pattern", None):
        from runner.intent_bench import fetch_rows

        rows = fetch_rows(dataset_def, lang, revision)
        return [r[text_col] for r in rows if r.get(text_col)]

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from runner.audio_io import _parquet_files

    out: list[str] = []
    for pfile in _parquet_files(source.hf_id, source.subset, source.split,
                                revision):
        local = hf_hub_download(source.hf_id, pfile, repo_type="dataset",
                                revision=revision)
        table = pq.read_table(local, columns=[text_col])
        for value in table.column(text_col).to_pylist():
            if value:
                out.append(str(value))
    return out
