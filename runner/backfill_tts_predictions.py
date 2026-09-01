"""
Build missing TTS prediction rows from wavs that are already stored in a
predictions repo (§3.2 / §4 R14), without re-synthesising anything.

``runner.tts_bench`` writes one JSONL file per (lang, competitor) pair under
``predictions/<lang>/<competitor_id>.jsonl`` in the modality's HF repo
(``OpenVoiceOS/ovos-tts-bench-<dataset_id>``), alongside the rendered clips
under ``audio/<lang>/<competitor_id>/<hash>.wav``. Two things can leave a
pair short of rows even though its clips are all sitting on the hub:

1. the pair's JSONL never landed at all (a run interrupted, or an upload
   that crashed between the ``audio/`` and ``predictions/`` halves of
   ``media_bench.upload_predictions``);
2. the JSONL exists but is missing individual rows — ``TTSBench.predict``
   writes the wav to disk, then scores it (UTMOS, then the quality
   dimensions, then intelligibility), and only *then* does
   ``media_bench.run_competitor_lang`` append a row; a judge that raises
   during that scoring is caught by the outer ``except Exception`` in
   ``run_competitor_lang`` and the sample is skipped (``continue``) — so
   the wav exists on disk/on the hub but no row was ever written for it.

``runner.rescore_tts`` only iterates rows that already exist, so neither
case is visible to it: an entirely-missing JSONL has no rows to iterate,
and a JSONL missing some rows never surfaces the wavs that have none.

This script finds every (lang, competitor_id) pair that has audio on the
hub, and for each one:

1. ``find_pairs`` lists (lang, competitor_id) → the set of wav content
   hashes present under ``audio/`` (HF API file listing only, no
   downloads);
2. ``pair_status`` downloads that pair's ``predictions/<lang>/<id>.jsonl``
   if one exists (a small file) and computes which of the dataset's
   prompts have a stored wav but no row for it — matching prompt → wav by
   the same content-hash filename ``TTSBench.predict`` writes;
3. ``backfill_pair`` scores only those missing prompts (same UTMOS/SIGMOS/
   DNSMOS/NISQA/intelligibility judges and row shape
   ``TTSBench.predict``/``media_bench.make_row`` use for a live run) and
   returns the new rows;
4. the new rows are appended after any existing rows — which are copied
   through byte-for-byte, never re-serialised or reordered — and on
   ``--upload`` the merged file is pushed back to the repo.

Two things a live run captures that a backfill genuinely cannot:
``latency_ms``/``elapsed_ms`` (the wav was already rendered) and
``peak_rss_mb`` — all three are left ``None`` on a backfilled row rather
than faked.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from registry.loaders import load_competitor, load_dataset
from runner.asr_judges import resolve_judge_model
from runner.intent_bench import HF_OWNER, done_samples, resolve_revision, results_repo_for
from runner.media_bench import make_row
from runner.tts_bench import (
    UTMOS_JUDGE,
    UTMOS_JUDGE_REVISION,
    _clip_duration_secs,
    _get_utmos_judge,
    _load_prompts,
    _safe,
    _score_intelligibility,
    _score_quality_dimensions,
)

log = logging.getLogger("backfill-tts")

MODALITY = "tts"


def find_pairs(repo_id: str) -> dict[tuple[str, str], set[str]]:
    """(lang, competitor_id) → the set of wav content hashes under ``audio/``.

    Every pair that has at least one stored clip, whether or not it has a
    predictions JSONL at all.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[0] == "audio" and parts[3].endswith(".wav"):
            pairs[(parts[1], parts[2])].add(parts[3][: -len(".wav")])
    return dict(pairs)


def _existing_sample_ids(repo_id: str, lang: str, competitor_id: str,
                          revision: str = "main") -> set[str]:
    """sample_ids already present in this pair's predictions JSONL, if any."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        local = hf_hub_download(
            repo_id, f"predictions/{lang}/{competitor_id}.jsonl",
            repo_type="dataset", revision=revision,
        )
    except EntryNotFoundError:
        return set()
    return done_samples(Path(local))


def pair_status(
    dataset_def, dataset_revision: str, lang: str, competitor_id: str,
    wav_hashes: set[str], existing_sample_ids: set[str],
) -> dict:
    """Which of a pair's prompts have a stored wav but no row for it yet.

    Returns ``{"wavs": n, "existing": n, "missing": [(index, text), ...]}``
    — ``missing`` drives :func:`backfill_pair`, ``wavs``/``existing`` are
    for reporting.
    """
    text_col = (dataset_def.reference_fields or {}).get("text", "text")
    prompts = _load_prompts(dataset_def, lang, dataset_revision, text_col)
    missing = []
    for i, text in enumerate(prompts):
        if _safe(text) not in wav_hashes:
            continue  # never synthesised (or synthesis failed): no clip to score
        sample_id = f"{lang}/{i:05d}"
        if sample_id in existing_sample_ids:
            continue  # already has a row
        missing.append((i, text))
    return {"wavs": len(wav_hashes), "existing": len(existing_sample_ids),
            "missing": missing}


def _download_audio_dir(repo_id: str, lang: str, competitor_id: str,
                         revision: str = "main") -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"audio/{lang}/{competitor_id}/*.wav"],
    )
    return Path(local) / "audio" / lang / competitor_id


def backfill_pair(
    dataset_def, lang: str, competitor_id: str, audio_dir: Path,
    dataset_revision: str, missing: list[tuple[int, str]],
) -> list[dict]:
    """Score the wavs of ``missing`` (index, text) prompts and build their rows.

    One row per entry in ``missing``, same row shape
    ``TTSBench.predict``/``media_bench.make_row`` produce for a live run.
    """
    competitor = load_competitor(MODALITY, competitor_id)
    results_repo = results_repo_for(MODALITY, dataset_def.dataset_id, HF_OWNER)

    rows = []
    for i, text in missing:
        wav_path = audio_dir / f"{_safe(text)}.wav"
        sample_id = f"{lang}/{i:05d}"
        rel = f"{lang}/{competitor_id}/{wav_path.name}"
        audio_url = (f"https://huggingface.co/datasets/{results_repo}"
                     f"/resolve/main/audio/{rel}")

        judge = _get_utmos_judge()
        utmos_score = judge(str(wav_path), judge.sample_rate)
        extras = {
            "utmos": round(float(utmos_score), 4),
            "utmos_judge": UTMOS_JUDGE,
            "utmos_judge_revision": UTMOS_JUDGE_REVISION,
        }
        extras.update(_score_quality_dimensions(wav_path))
        try:
            result = _score_intelligibility(wav_path, text, lang)
            extras["intelligibility_wer"] = result["wer"]
            extras["intelligibility_cer"] = result["cer"]
            extras["intelligibility_judge"] = result["judge_model_id"]
            extras["intelligibility_judge_revision"] = result["judge_revision"]
            extras["intelligibility_judges"] = result["judges"]
            extras["intelligibility_consensus"] = result["consensus"]
            extras["intelligibility_agreement"] = result["agreement"]
            extras["intelligibility_rover"] = True
        except Exception as exc:
            log.warning("intelligibility scoring failed for %r (%s/%s): %s",
                        text, lang, competitor_id, exc)
            judge_model_id, judge_revision = resolve_judge_model(lang)
            extras["intelligibility_wer"] = 1.0
            extras["intelligibility_cer"] = 1.0
            extras["intelligibility_judge"] = judge_model_id
            extras["intelligibility_judge_revision"] = judge_revision
            extras["intelligibility_error"] = str(exc)

        fields = {
            "input_text": text,
            "prediction": audio_url,
            "audio_url": audio_url,
            "latency_ms": None,
            "elapsed_ms": None,
            "peak_rss_mb": None,
            "audio_secs": _clip_duration_secs(wav_path),
            "extras": extras,
        }
        rows.append(make_row(competitor, dataset_def.dataset_id, lang,
                              sample_id, dataset_revision, fields,
                              modality=MODALITY))
    return rows


def _existing_lines(repo_id: str, lang: str, competitor_id: str,
                     revision: str = "main") -> list[str]:
    """Raw existing JSONL lines for a pair, if it has a predictions file."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        local = hf_hub_download(
            repo_id, f"predictions/{lang}/{competitor_id}.jsonl",
            repo_type="dataset", revision=revision,
        )
    except EntryNotFoundError:
        return []
    return [line for line in Path(local).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_merged(existing_lines: list[str], new_rows: list[dict],
                  out_path: Path) -> None:
    """Write *existing_lines* verbatim, then *new_rows* appended after them.

    Existing rows are never re-serialised or reordered — only the new rows
    are freshly ``json.dumps``-ed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for line in existing_lines:
            fh.write(line + "\n")
        for row in new_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_pair(repo_id: str, jsonl_path: Path, lang: str,
                 competitor_id: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=f"predictions/{lang}/{competitor_id}.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=(f"backfill: reconstruct {lang}/{competitor_id} "
                         "prediction rows from stored audio"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True,
                         help="TTS dataset id, e.g. massive-prompts")
    parser.add_argument("--hf-owner", default=HF_OWNER)
    parser.add_argument("--output-dir", default="predictions",
                         help="Local root the rebuilt JSONLs are written under")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report wavs/existing/missing per pair and exit")
    parser.add_argument("--upload", action="store_true",
                         help="Upload the merged predictions/<lang>/<id>.jsonl files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dataset_def = load_dataset(MODALITY, args.dataset_id)
    repo_id = results_repo_for(MODALITY, args.dataset_id, args.hf_owner)
    pairs = find_pairs(repo_id)

    if not pairs:
        log.info("%s: no audio pairs found", repo_id)
        return

    dataset_revision = resolve_revision(dataset_def.source.hf_id,
                                        dataset_def.source.revision)
    statuses = {}
    for lang, competitor_id in sorted(pairs):
        existing_sample_ids = _existing_sample_ids(repo_id, lang, competitor_id)
        statuses[(lang, competitor_id)] = pair_status(
            dataset_def, dataset_revision, lang, competitor_id,
            pairs[(lang, competitor_id)], existing_sample_ids)

    incomplete = {k: v for k, v in statuses.items() if v["missing"]}
    log.info("%s: %d audio pairs, %d incomplete", repo_id, len(pairs),
              len(incomplete))
    for (lang, competitor_id), status in statuses.items():
        log.info("  %s/%s: %d wavs, %d existing rows, %d missing rows",
                  lang, competitor_id, status["wavs"], status["existing"],
                  len(status["missing"]))
    if args.dry_run or not incomplete:
        return

    out_root = Path(args.output_dir) / args.dataset_id / MODALITY / "predictions"
    for (lang, competitor_id), status in incomplete.items():
        audio_dir = _download_audio_dir(repo_id, lang, competitor_id)
        new_rows = backfill_pair(dataset_def, lang, competitor_id, audio_dir,
                                 dataset_revision, status["missing"])
        existing_lines = _existing_lines(repo_id, lang, competitor_id)
        jsonl_path = out_root / lang / f"{competitor_id}.jsonl"
        write_merged(existing_lines, new_rows, jsonl_path)
        log.info("  %s/%s: wrote %d existing + %d new rows to %s", lang,
                  competitor_id, len(existing_lines), len(new_rows), jsonl_path)
        if args.upload:
            upload_pair(repo_id, jsonl_path, lang, competitor_id)
            log.info("  %s/%s: uploaded", lang, competitor_id)


if __name__ == "__main__":
    main()
