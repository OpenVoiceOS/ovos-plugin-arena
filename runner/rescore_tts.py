"""
Backfill SIGMOS/DNSMOS/NISQA quality dimensions onto existing TTS prediction rows
(§4 R14 extension), without re-synthesising anything.

``runner.tts_bench`` renders a clip and uploads it under
``audio/<lang>/<competitor_id>/<hash>.wav`` inside the modality's HF
predictions repo (``OpenVoiceOS/ovos-tts-bench-<dataset_id>``) alongside the
JSONL prediction rows themselves (``predictions/<lang>/<competitor_id>.jsonl``,
one row per sample). The rendered wav is exactly what UTMOS/intelligibility
were scored from at bench time and it stays in the repo indefinitely — so a
metric that got added *after* a bench run (like this one) can be backfilled
by re-downloading the stored wav and scoring it, instead of re-running the
whole (often slow, plugin-dependent) synthesis benchmark.

This script:

1. downloads a TTS predictions repo's ``predictions/`` and ``audio/`` trees
   (or reads them from a local bench output dir);
2. for every row missing ``sigmos.*``/``dnsmos.*``/``nisqa.*`` extras,
   downloads/locates its ``audio_url`` wav and scores it with the same
   judges ``runner.tts_bench`` uses;
3. rewrites the JSONL file in place with the new extras merged in;
4. on ``--upload``, re-uploads the updated ``predictions/`` folder.

Rows that already carry ``sigmos.ovrl``, ``dnsmos.ovrl`` AND ``nisqa.mos``
are skipped (idempotent re-runs), and rows with no ``audio_url`` (a failed
synthesis) are left untouched — there is no clip to rescore.

``--rejudge-intelligibility`` additionally re-judges any row that predates
#143's ROVER judge panel (no ``intelligibility_rover: true`` extra) by
locating its stored wav the same way and re-running the panel — replacing
the row's single-judge ``intelligibility_wer/cer`` with the full panel
result (per-judge transcripts, ROVER consensus/agreement, and consensus-
derived wer/cer). A row whose wav can no longer be found is skipped and
logged, never crashes the run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from runner.asr_judges import (
    _UNIVERSAL_FALLBACK, judge_available, resolve_judge_model,
)
from runner.intent_bench import HF_OWNER, results_repo_for
from runner.tts_bench import (  # noqa: F401 (patchable at module level; optional-dep imports are inside tts_bench's own lazy judge getters)
    _score_intelligibility,
    _score_quality_dimensions,
    unjudgeable_intelligibility_extras,
)

log = logging.getLogger("rescore-tts")

MODALITY = "tts"

# Extras keys a fresh ``_score_intelligibility`` panel result replaces —
# both the legacy single-judge fields (§4 R16, pre-#143) and any error
# marker left by a previous failed judging attempt, or by a language that
# had no ASR judge when it was benched.
_INTELLIGIBILITY_EXTRAS_KEYS = (
    "intelligibility_wer", "intelligibility_cer", "intelligibility_judge",
    "intelligibility_judge_revision", "intelligibility_error",
    "intelligibility",
)


def _needs_rescoring(row: dict) -> bool:
    extras = row.get("extras") or {}
    return ("sigmos.ovrl" not in extras or "dnsmos.ovrl" not in extras
            or "nisqa.mos" not in extras)


def _needs_intelligibility_rejudge(row: dict) -> bool:
    """A row needs re-judging (``--rejudge-intelligibility``) unless it
    already carries a #143 ROVER panel result — legacy pre-#143 rows only
    ever have the single-judge ``intelligibility_wer/cer`` fields and no
    ``intelligibility_rover`` marker at all. A row already marked
    ``intelligibility: not_available`` stays that way only while its
    language still has no ASR judge; once the registry claims the language,
    the row is judgeable and MUST be judged, or a board ends up with one
    language split between judged and unjudged rows. Re-judging such a row
    is not stripping it: the never-strip rule protects rows that carry a
    real score.

    A row whose stored judge is not the one its language resolves to also
    needs re-judging: one board's rows for one language MUST all come from
    the same model (§4 R16), and a row left behind by a judge change is not
    comparable with the rows around it."""
    extras = row.get("extras") or {}
    lang = row.get("lang")
    if extras.get("intelligibility") == "not_available":
        return bool(lang) and judge_available(lang)
    if extras.get("intelligibility_rover") is not True:
        return True
    return _judge_is_stale(row)


def _judge_is_stale(row: dict) -> bool:
    """Whether the row's stored judge is no longer the one its language resolves to."""
    lang = row.get("lang")
    judge = (row.get("extras") or {}).get("intelligibility_judge")
    if not lang or judge in (None, "none") or not judge_available(lang):
        return False
    return judge != resolve_judge_model(lang)[0]


def _is_unjudgeable_row(row: dict, lang: str) -> bool:
    """Whether this row should be marked ``intelligibility: not_available``
    rather than re-judged.

    Two conditions, both required. The language must have no ASR judge
    (§4 R16), and the row's stored score must have come from the blanket
    ``whisper-base`` default or from no judge at all. A row scored by a
    dedicated model is a real measurement and is never overwritten with the
    marker — the marker deletes the WER/CER outright, so a wrong answer
    here destroys data no rescore can bring back.
    """
    if judge_available(lang):
        return False
    judge = (row.get("extras") or {}).get("intelligibility_judge")
    return judge in (None, "none", _UNIVERSAL_FALLBACK)


def _download_repo_tree(repo_id: str, revision: str = "main") -> Path:
    """Download ``predictions/`` and ``audio/`` from a TTS predictions repo."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["predictions/**/*.jsonl", "audio/**"],
    )
    return Path(local)


def _resolve_wav_path(repo_dir: Path, row: dict) -> Path | None:
    """Local path of the wav a row's ``audio_url`` resolves to, if present.

    ``audio_url`` is ``https://huggingface.co/datasets/<repo>/resolve/main/
    audio/<rel_path>`` (see ``runner.media_bench.PredictContext.hf_audio_url``)
    — the part after ``/audio/`` is the same relative path the repo snapshot
    was downloaded under.
    """
    url = row.get("audio_url")
    if not url:
        return None
    marker = "/resolve/main/audio/"
    idx = url.find(marker)
    if idx == -1:
        return None
    rel = url[idx + len(marker):]
    path = repo_dir / "audio" / rel
    return path if path.is_file() else None


def rescore_file(path: Path, repo_dir: Path,
                  rejudge_intelligibility: bool = False) -> tuple[int, int]:
    """Rescore one predictions JSONL file in place, one row at a time.

    Rows are read, scored and written straight back out through a sibling
    temp file rather than being parsed into a list held for the whole file
    — a prediction file can be thousands of rows, and each row's judge call
    only needs the one row and its wav on disk at a time. The temp file is
    atomically renamed onto ``path`` at the end, so a crash mid-run never
    leaves a partially-rewritten predictions file behind.

    ``rejudge_intelligibility=True`` additionally re-scores any row that
    doesn't yet carry a #143 ROVER panel result (``intelligibility_rover:
    true``) from its stored wav, and REPLACES its intelligibility extras
    wholesale — the judges transcripts, ROVER consensus/agreement and the
    consensus-derived wer/cer — rather than leaving the legacy single-judge
    fields sitting alongside them. Off by default, this leaves the quality-
    dims backfill behaviour byte-identical to before.
    """
    rescored = 0
    skipped = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    changed = False
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            needs_quality = _needs_rescoring(row)
            needs_intel = rejudge_intelligibility and _needs_intelligibility_rejudge(row)
            if needs_quality or needs_intel:
                wav_path = _resolve_wav_path(repo_dir, row)
                if wav_path is None:
                    skipped += 1
                    if needs_intel:
                        log.info(
                            "skipping intelligibility rejudge for %s (%s): "
                            "wav not found on disk", row.get("sample_id"),
                            row.get("competitor_id"))
                else:
                    row_changed = False
                    if needs_quality:
                        try:
                            new_extras = _score_quality_dimensions(wav_path)
                        except Exception as exc:
                            log.warning("rescoring failed for %s (%s): %s",
                                        row.get("sample_id"), row.get("competitor_id"), exc)
                            new_extras = None
                        if new_extras:
                            row.setdefault("extras", {}).update(new_extras)
                            row_changed = True
                    lang = row.get("lang")
                    if needs_intel and lang and _is_unjudgeable_row(row, lang):
                        extras = row.setdefault("extras", {})
                        for key in _INTELLIGIBILITY_EXTRAS_KEYS:
                            extras.pop(key, None)
                        extras.update(unjudgeable_intelligibility_extras())
                        row_changed = True
                    elif needs_intel:
                        try:
                            result = _score_intelligibility(
                                wav_path, row.get("input_text"), row.get("lang"))
                        except Exception as exc:
                            log.warning(
                                "intelligibility rejudge failed for %s (%s): %s",
                                row.get("sample_id"), row.get("competitor_id"), exc)
                            result = None
                        if result:
                            extras = row.setdefault("extras", {})
                            for key in _INTELLIGIBILITY_EXTRAS_KEYS:
                                extras.pop(key, None)
                            extras["intelligibility_wer"] = result["wer"]
                            extras["intelligibility_cer"] = result["cer"]
                            extras["intelligibility_judge"] = result["judge_model_id"]
                            extras["intelligibility_judge_revision"] = result["judge_revision"]
                            extras["intelligibility_judges"] = result["judges"]
                            extras["intelligibility_consensus"] = result["consensus"]
                            extras["intelligibility_agreement"] = result["agreement"]
                            extras["intelligibility_rover"] = True
                            row_changed = True
                    if row_changed:
                        rescored += 1
                        changed = True
                    else:
                        skipped += 1
            else:
                skipped += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    if changed:
        tmp_path.replace(path)
    else:
        tmp_path.unlink()
    return rescored, skipped


def rescore_repo(repo_id: str, revision: str = "main",
                  rejudge_intelligibility: bool = False) -> Path:
    """Download, rescore in place, and return the local repo dir for upload."""
    repo_dir = _download_repo_tree(repo_id, revision)
    predictions_dir = repo_dir / "predictions"
    total_rescored = 0
    total_skipped = 0
    for jsonl_path in sorted(predictions_dir.glob("**/*.jsonl")):
        rescored, skipped = rescore_file(jsonl_path, repo_dir, rejudge_intelligibility)
        total_rescored += rescored
        total_skipped += skipped
        log.info("  %s: rescored %d, skipped %d",
                  jsonl_path.relative_to(predictions_dir), rescored, skipped)
    log.info("%s: rescored %d rows total, skipped %d",
              repo_id, total_rescored, total_skipped)
    return repo_dir


def upload_rescored(repo_id: str, repo_dir: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(repo_dir / "predictions"),
        path_in_repo="predictions",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="rescore: backfill SIGMOS/DNSMOS/NISQA quality dimensions",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True,
                         help="TTS dataset id, e.g. massive-prompts-en-US")
    parser.add_argument("--hf-owner", default=HF_OWNER)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--upload", action="store_true",
                         help="Re-upload the rescored predictions/ folder")
    parser.add_argument(
        "--rejudge-intelligibility", action="store_true",
        help="Also re-judge intelligibility with the #143 ROVER panel for "
             "any row not yet carrying intelligibility_rover: true, from "
             "its stored wav (no re-synthesis)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repo_id = results_repo_for(MODALITY, args.dataset_id, args.hf_owner)
    repo_dir = rescore_repo(repo_id, args.revision, args.rejudge_intelligibility)
    if args.upload:
        upload_rescored(repo_id, repo_dir)
        log.info("Uploaded rescored predictions to %s", repo_id)


if __name__ == "__main__":
    main()
