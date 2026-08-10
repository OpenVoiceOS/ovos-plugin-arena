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

#: Judge identity recorded on every scored row (§4 R16) — the intelligibility
#: metric transcribes the rendered clip back to text with a *pinned*
#: faster-whisper model/revision and scores WER/CER against the prompt, so a
#: judge upgrade must not silently reinterpret past scores.
INTELLIGIBILITY_JUDGE = "Systran/faster-whisper-small"
INTELLIGIBILITY_JUDGE_REVISION = "536b0662742c02347bc0e980a01041f333bce120"

_utmos_judge = None  # module-cached lazy singleton, one ONNX session per process
_intelligibility_judge = None  # module-cached lazy singleton, one faster-whisper model per process


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


def _get_intelligibility_judge():
    """Lazily import and cache the pinned STT round-trip judge (§4 R16).

    ``faster-whisper`` is an optional (``audio`` extra) dependency, same
    reasoning as the UTMOS judge above: importing it at module load would
    break every environment that doesn't run TTS benchmarks, but scoring is
    NOT optional once a TTS run is actually synthesising clips.
    """
    global _intelligibility_judge
    if _intelligibility_judge is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "TTS benchmarking requires the 'faster-whisper' package "
                "(intelligibility WER/CER scoring is not optional for TTS "
                "runs) — install the 'audio' extra: "
                "pip install ovos-plugin-arena[audio]"
            ) from exc
        _intelligibility_judge = WhisperModel(
            INTELLIGIBILITY_JUDGE.rsplit("/", 1)[-1],
            revision=INTELLIGIBILITY_JUDGE_REVISION,
        )
    return _intelligibility_judge


def _transcribe(judge, array) -> str:
    """Run the pinned STT judge over a decoded 16 kHz mono float32 array."""
    segments, _info = judge.transcribe(array)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _score_intelligibility(wav_path, prompt_text: str) -> tuple[float, float]:
    """STT round-trip WER/CER (§4 R16) for one rendered clip.

    Reads the raw file bytes and decodes through
    ``runner.audio_io.decode_audio_bytes`` — which transcodes non-wav
    containers (mp3/opus/...) via ``soundfile``/``av`` and always resamples
    to 16 kHz mono — rather than assuming the on-disk file is already PCM at
    the right rate. Reading a 44.1 kHz (or stereo) file as if it already
    were 16 kHz mono is a known false-~1.7-WER footgun; going through the
    shared decoder sidesteps it here exactly like the STT/wake-word
    benchmarks.
    """
    from runner.audio_io import decode_audio_bytes

    with open(wav_path, "rb") as fh:
        array, _sr = decode_audio_bytes(fh.read())  # always 16k mono here
    judge = _get_intelligibility_judge()
    hypothesis = _transcribe(judge, array)
    from arena.metrics import intelligibility_scores

    return intelligibility_scores(prompt_text, hypothesis)


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

        # Direct get_tts path — no audio player / playback mode. A synthesis
        # crash MUST still produce a row (worst-case WER/CER, §4 R16) rather
        # than let the exception propagate up to media_bench.run_competitor_lang,
        # whose blanket try/except just skips the sample and silently drops
        # it from the board — that would hide the exact failures this metric
        # exists to catch.
        start = time.perf_counter()
        try:
            engine.get_tts(text, str(wav_path), lang=ctx.lang)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            log.warning("synthesis failed for %r (%s): %s",
                        text, ctx.competitor.competitor_id, exc)
            return {
                "input_text": text,
                "prediction": None,
                "audio_url": None,
                "latency_ms": round(latency_ms, 3),
                "extras": {
                    "synthesis_error": str(exc),
                    "intelligibility_wer": 1.0,
                    "intelligibility_cer": 1.0,
                    "intelligibility_judge": INTELLIGIBILITY_JUDGE,
                    "intelligibility_judge_revision": INTELLIGIBILITY_JUDGE_REVISION,
                },
            }
        latency_ms = (time.perf_counter() - start) * 1000

        judge = _get_utmos_judge()
        # ``sr`` here is the *input's* sample rate, not the model's — for a
        # path input it is ignored anyway (the loader reads the real rate
        # from the wav header), so this value is only a placeholder.
        score = judge(str(wav_path), judge.sample_rate)

        extras = {
            "utmos": round(float(score), 4),
            "utmos_judge": UTMOS_JUDGE,
            "utmos_judge_revision": UTMOS_JUDGE_REVISION,
            "intelligibility_judge": INTELLIGIBILITY_JUDGE,
            "intelligibility_judge_revision": INTELLIGIBILITY_JUDGE_REVISION,
        }
        try:
            wer, cer = _score_intelligibility(wav_path, text)
            extras["intelligibility_wer"] = wer
            extras["intelligibility_cer"] = cer
        except Exception as exc:
            # Low-resource languages the judge transcribes weakly are
            # warn-only (§4 R16) — the real WER is still recorded (never
            # gates), but the judge itself crashing (e.g. on silence/noise)
            # must not drop the row: force the worst-case score instead of
            # leaving the metric silently missing.
            log.warning("intelligibility scoring failed for %r (%s): %s",
                        text, ctx.competitor.competitor_id, exc)
            extras["intelligibility_wer"] = 1.0
            extras["intelligibility_cer"] = 1.0
            extras["intelligibility_error"] = str(exc)

        return {
            "input_text": text,
            "prediction": ctx.hf_audio_url(rel),
            "audio_url": ctx.hf_audio_url(rel),
            "latency_ms": round(latency_ms, 3),
            # PredictionRow has no modeled utmos/intelligibility fields —
            # these MUST be nested under "extras" (§3.2) or pydantic
            # silently drops them and the objective board goes empty; see
            # runner/media_bench.py:make_row (row.update(fields)) and
            # arena/predictions.py:parse_row.
            "extras": extras,
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
