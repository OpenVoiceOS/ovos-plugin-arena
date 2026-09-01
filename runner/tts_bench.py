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

from runner.asr_judges import resolve_judge_model, resolve_judge_panel
from runner.media_bench import MediaBenchAdapter, PredictContext, load_plugin_class
from runner.perf import rss_mb

log = logging.getLogger("tts-bench")

#: Judge identity recorded on every scored row (§4 R14) — provenance for
#: the objective TTS board, since a different judge revision is not directly
#: comparable to this one.
UTMOS_JUDGE = "TigreGotico/utmos-onnx"
UTMOS_JUDGE_REVISION = "ff41b8f440cb12ecda18261f9ff7326d058275ce"

#: SIGMOS (P.804), DNSMOS (P.835) and NISQA-v2 are objective per-dimension
#: quality judges — same provenance/pinning discipline as UTMOS above (§4
#: R14 extension). SIGMOS provides the headline P.804 dimensions (MIT-
#: licensed Microsoft weights); NISQA is recorded as a complementary
#: predictor alongside it (see docs/methodology.md for the license
#: rationale — this arena is a non-commercial OVOS project, so NISQA's CC
#: BY-NC-SA 4.0 weights are usable here).
SIGMOS_JUDGE = "TigreGotico/sigmos-onnx"
SIGMOS_JUDGE_REVISION = "33ccd4fca5b8ffe03828530753f0b35769b8e880"
DNSMOS_JUDGE = "TigreGotico/dnsmos-onnx"
DNSMOS_JUDGE_REVISION = "27691a53aa069b27be6ac957013d43b3c442da9d"
NISQA_JUDGE = "TigreGotico/nisqa-onnx"
NISQA_JUDGE_REVISION = "3de0221b7bb4919dc2ba9a891da7fba76b06e573"

_utmos_judge = None  # module-cached lazy singleton, one ONNX session per process
_sigmos_judge = None
_dnsmos_judge = None
_nisqa_judge = None
_intelligibility_judges: dict[str, tuple[object, str]] = {}  # model_id -> (onnx-asr model, revision), cached per model — several langs share one model


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


def _get_sigmos_judge():
    """Lazily import and cache the SIGMOS (P.804) judge — same optional-dep
    contract as :func:`_get_utmos_judge`."""
    global _sigmos_judge
    if _sigmos_judge is None:
        try:
            from speechonnxmetrics.mos.sigmos import SIGMOS
        except ImportError as exc:
            raise RuntimeError(
                "TTS benchmarking requires the 'speechonnxmetrics' package "
                "(objective SIGMOS scoring is not optional for TTS runs) — "
                "install the 'audio' extra: pip install ovos-plugin-arena[audio]"
            ) from exc
        _sigmos_judge = SIGMOS()
    return _sigmos_judge


def _get_dnsmos_judge():
    """Lazily import and cache the DNSMOS (P.835) judge — same optional-dep
    contract as :func:`_get_utmos_judge`."""
    global _dnsmos_judge
    if _dnsmos_judge is None:
        try:
            from speechonnxmetrics.mos.dnsmos import DNSMOS
        except ImportError as exc:
            raise RuntimeError(
                "TTS benchmarking requires the 'speechonnxmetrics' package "
                "(objective DNSMOS scoring is not optional for TTS runs) — "
                "install the 'audio' extra: pip install ovos-plugin-arena[audio]"
            ) from exc
        _dnsmos_judge = DNSMOS()
    return _dnsmos_judge


def _get_nisqa_judge():
    """Lazily import and cache the NISQA-v2 judge — same optional-dep
    contract as :func:`_get_utmos_judge`. NISQA's weights are CC BY-NC-SA
    4.0; usable here because this arena is a non-commercial OVOS project
    (see docs/methodology.md)."""
    global _nisqa_judge
    if _nisqa_judge is None:
        try:
            from speechonnxmetrics.mos.nisqa import NISQA
        except ImportError as exc:
            raise RuntimeError(
                "TTS benchmarking requires the 'speechonnxmetrics' package "
                "(objective NISQA scoring is not optional for TTS runs) — "
                "install the 'audio' extra: pip install ovos-plugin-arena[audio]"
            ) from exc
        _nisqa_judge = NISQA()
    return _nisqa_judge


def _get_intelligibility_judge(lang: str) -> tuple[object, str, str]:
    """Lazily import and cache the per-language onnx-asr round-trip judge (§4 R16).

    Owner directive: the judge is always the best OFFLINE ASR for ``lang``
    (usually a conformer), resolved from ovos-config's offline STT
    recommends with a built-in onnx-asr fallback table — see
    ``runner.asr_judges``. Never ``faster-whisper``.

    ``onnx-asr`` is an optional (``audio`` extra) dependency, same reasoning
    as the UTMOS judge above: importing it at module load would break every
    environment that doesn't run TTS benchmarks, but scoring is NOT optional
    once a TTS run is actually synthesising clips.

    Cached per resolved model id, not per language — several languages
    (e.g. every ``nemo-parakeet-tdt-0.6b-v3`` language) share one model, and
    that model should load into memory once per process, not once per lang.
    """
    model_id, revision = resolve_judge_model(lang)
    model, revision = _load_intelligibility_judge(model_id, revision)
    return model, revision, model_id


def _load_intelligibility_judge(model_id: str, revision: str) -> tuple[object, str]:
    """Lazily import and load one onnx-asr judge by model id, cached per
    model id — several panel members (or several languages) can share the
    same loaded model. See :func:`_get_intelligibility_judge` docstring for
    the optional-dependency contract."""
    if model_id not in _intelligibility_judges:
        try:
            import onnx_asr
        except ImportError as exc:
            raise RuntimeError(
                "TTS benchmarking requires the 'onnx-asr' package "
                "(intelligibility WER/CER scoring is not optional for TTS "
                "runs) — install the 'audio' extra: "
                "pip install ovos-plugin-arena[audio]"
            ) from exc
        model = onnx_asr.load_model(model_id)
        _intelligibility_judges[model_id] = (model, revision)
    return _intelligibility_judges[model_id]


def _score_quality_dimensions(wav_path) -> dict:
    """SIGMOS (col/disc/loud/noise/reverb/sig/ovrl) + DNSMOS (sig/bak/ovrl) +
    NISQA (mos/noi/dis/col/loud) for one rendered clip, flattened onto
    ``sigmos.<dim>``/``dnsmos.<dim>``/``nisqa.<dim>`` extras keys — same
    shape ``speechonnxmetrics.score()`` would produce (§4 R14 extension).
    SIGMOS provides the board's headline P.804 dimensions; NISQA rides
    along as a complementary predictor, not surfaced as its own board
    column (see docs/methodology.md for why both are used together here).

    Warn-only: a judge that fails to score a pathological clip (e.g. one
    the intelligibility judge above already tolerates) must not drop the
    row — the extras below simply stay absent for the failed judge, and
    aggregation (``score_tts``) already excludes rows missing a dimension.
    """
    extras: dict = {}
    try:
        sigmos_judge = _get_sigmos_judge()
        sigmos_scores = sigmos_judge(str(wav_path), sigmos_judge.sample_rate)
        for dim, value in sigmos_scores.items():
            extras[f"sigmos.{dim}"] = round(float(value), 4)
        extras["sigmos_judge"] = SIGMOS_JUDGE
        extras["sigmos_judge_revision"] = SIGMOS_JUDGE_REVISION
    except Exception as exc:
        log.warning("SIGMOS scoring failed for %s: %s", wav_path, exc)
    try:
        dnsmos_judge = _get_dnsmos_judge()
        dnsmos_scores = dnsmos_judge(str(wav_path), dnsmos_judge.sample_rate)
        for dim, value in dnsmos_scores.items():
            extras[f"dnsmos.{dim}"] = round(float(value), 4)
        extras["dnsmos_judge"] = DNSMOS_JUDGE
        extras["dnsmos_judge_revision"] = DNSMOS_JUDGE_REVISION
    except Exception as exc:
        log.warning("DNSMOS scoring failed for %s: %s", wav_path, exc)
    try:
        nisqa_judge = _get_nisqa_judge()
        nisqa_scores = nisqa_judge(str(wav_path), nisqa_judge.sample_rate)
        for dim, value in nisqa_scores.items():
            extras[f"nisqa.{dim}"] = round(float(value), 4)
        extras["nisqa_judge"] = NISQA_JUDGE
        extras["nisqa_judge_revision"] = NISQA_JUDGE_REVISION
    except Exception as exc:
        log.warning("NISQA scoring failed for %s: %s", wav_path, exc)
    return extras


def _transcribe(judge, array, sample_rate: int = 16000) -> str:
    """Run the resolved onnx-asr judge over a decoded 16 kHz mono float32 array."""
    return judge.recognize(array, sample_rate=sample_rate).strip()


def _score_intelligibility(wav_path, prompt_text: str, lang: str) -> dict:
    """STT round-trip WER/CER via ROVER panel consensus (§4 R16) for one
    rendered clip.

    Reads the raw file bytes and decodes through
    ``runner.audio_io.decode_audio_bytes`` — which transcodes non-wav
    containers (mp3/opus/...) via ``soundfile``/``av`` and always resamples
    to 16 kHz mono — rather than assuming the on-disk file is already PCM at
    the right rate. Reading a 44.1 kHz (or stereo) file as if it already
    were 16 kHz mono is a known false-~1.7-WER footgun; going through the
    shared decoder sidesteps it here exactly like the STT/wake-word
    benchmarks.

    Every panel judge (``runner.asr_judges.resolve_judge_panel``) transcribes
    the clip independently; ``wer``/``cer`` are ALWAYS derived from the
    ROVER consensus of those transcripts (:mod:`arena.rover`) — never from a
    single raw judge transcript, even for a panel of one, where the
    "consensus" is just that one hypothesis but still flows through the
    same ROVER path (owner directive). The per-judge raw transcripts are
    returned too and MUST be persisted on the row: they are what lets a
    future reweighted ROVER be recomputed purely from stored data, with no
    ASR re-run.
    """
    from runner.audio_io import decode_audio_bytes
    from arena.metrics import intelligibility_scores
    from arena.rover import rover_consensus_and_agreement_from_judges

    with open(wav_path, "rb") as fh:
        array, _sr = decode_audio_bytes(fh.read())  # always 16k mono here

    panel = resolve_judge_panel(lang)
    judges: list[dict] = []
    for model_id, revision in panel:
        judge, loaded_revision = _load_intelligibility_judge(model_id, revision)
        transcript = _transcribe(judge, array)
        judges.append({
            "model": model_id,
            "revision": loaded_revision,
            "transcript": transcript,
        })

    consensus, agreement = rover_consensus_and_agreement_from_judges(judges)
    wer, cer = intelligibility_scores(prompt_text, consensus)
    primary_model_id, primary_revision = panel[0]
    return {
        "wer": wer,
        "cer": cer,
        "judge_model_id": primary_model_id,
        "judge_revision": primary_revision,
        "judges": judges,
        "consensus": consensus,
        "agreement": agreement,
    }


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
        #
        # elapsed_ms/peak_rss_mb are measured around ONLY this synthesis call
        # (before/after RSS via runner.perf) and self-reported on the row
        # below — media_bench.run_competitor_lang's own measure_call wraps
        # the WHOLE predict() (synthesis + UTMOS + intelligibility judging),
        # and its fields.setdefault("elapsed_ms", ...) only fires when the
        # key is absent, so setting it explicitly here keeps RTF (the metric
        # this capture exists for) scoped to synthesis, never judging time.
        before_rss = rss_mb()
        start = time.perf_counter()
        try:
            engine.get_tts(text, str(wav_path), lang=ctx.lang)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            after_rss = rss_mb()
            peak_rss_mb = (max(before_rss, after_rss)
                           if before_rss is not None and after_rss is not None
                           else None)
            log.warning("synthesis failed for %r (%s): %s",
                        text, ctx.competitor.competitor_id, exc)
            judge_model_id, judge_revision = resolve_judge_model(ctx.lang)
            return {
                "input_text": text,
                "prediction": None,
                "audio_url": None,
                "latency_ms": round(latency_ms, 3),
                "elapsed_ms": round(latency_ms, 3),
                "peak_rss_mb": round(peak_rss_mb, 3) if peak_rss_mb is not None else None,
                "audio_secs": None,  # synthesis failed — no clip was produced
                "extras": {
                    "synthesis_error": str(exc),
                    "intelligibility_wer": 1.0,
                    "intelligibility_cer": 1.0,
                    "intelligibility_judge": judge_model_id,
                    "intelligibility_judge_revision": judge_revision,
                },
            }
        latency_ms = (time.perf_counter() - start) * 1000
        after_rss = rss_mb()
        peak_rss_mb = (max(before_rss, after_rss)
                       if before_rss is not None and after_rss is not None
                       else None)

        judge = _get_utmos_judge()
        # ``sr`` here is the *input's* sample rate, not the model's — for a
        # path input it is ignored anyway (the loader reads the real rate
        # from the wav header), so this value is only a placeholder.
        score = judge(str(wav_path), judge.sample_rate)

        extras = {
            "utmos": round(float(score), 4),
            "utmos_judge": UTMOS_JUDGE,
            "utmos_judge_revision": UTMOS_JUDGE_REVISION,
        }
        # SIGMOS/DNSMOS/NISQA quality dimensions ride alongside UTMOS (§4 R14
        # extension) — UTMOS stays the board's primary metric, these are
        # additional secondary columns. Warn-only on failure, same as
        # intelligibility scoring below: never drops the row.
        extras.update(_score_quality_dimensions(wav_path))
        try:
            result = _score_intelligibility(wav_path, text, ctx.lang)
            extras["intelligibility_wer"] = result["wer"]
            extras["intelligibility_cer"] = result["cer"]
            extras["intelligibility_judge"] = result["judge_model_id"]
            extras["intelligibility_judge_revision"] = result["judge_revision"]
            extras["intelligibility_judges"] = result["judges"]
            extras["intelligibility_consensus"] = result["consensus"]
            extras["intelligibility_agreement"] = result["agreement"]
            extras["intelligibility_rover"] = True
        except Exception as exc:
            # Low-resource languages the judge transcribes weakly are
            # warn-only (§4 R16) — the real WER is still recorded (never
            # gates), but the judge itself crashing (e.g. on silence/noise)
            # must not drop the row: force the worst-case score instead of
            # leaving the metric silently missing.
            log.warning("intelligibility scoring failed for %r (%s): %s",
                        text, ctx.competitor.competitor_id, exc)
            judge_model_id, judge_revision = resolve_judge_model(ctx.lang)
            extras["intelligibility_wer"] = 1.0
            extras["intelligibility_cer"] = 1.0
            extras["intelligibility_judge"] = judge_model_id
            extras["intelligibility_judge_revision"] = judge_revision
            extras["intelligibility_error"] = str(exc)

        return {
            "input_text": text,
            "prediction": ctx.hf_audio_url(rel),
            "audio_url": ctx.hf_audio_url(rel),
            "latency_ms": round(latency_ms, 3),
            # elapsed_ms/peak_rss_mb are self-reported here (synthesis-only
            # span) rather than left to media_bench's outer wrapper, which
            # would otherwise fold in UTMOS + intelligibility judging time —
            # see the long comment above the get_tts call.
            "elapsed_ms": round(latency_ms, 3),
            "peak_rss_mb": round(peak_rss_mb, 3) if peak_rss_mb is not None else None,
            # produced clip duration (RTF = elapsed_ms / 1000 / audio_secs, §M1)
            "audio_secs": _clip_duration_secs(wav_path),
            # PredictionRow has no modeled utmos/intelligibility fields —
            # these MUST be nested under "extras" (§3.2) or pydantic
            # silently drops them and the objective board goes empty; see
            # runner/media_bench.py:make_row (row.update(fields)) and
            # arena/predictions.py:parse_row.
            "extras": extras,
        }


def _clip_duration_secs(wav_path) -> float | None:
    """Duration in seconds of a just-synthesised clip, best-effort."""
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        return round(info.frames / info.samplerate, 3)
    except Exception as exc:
        log.warning("could not read duration of %s: %s", wav_path, exc)
        return None


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
