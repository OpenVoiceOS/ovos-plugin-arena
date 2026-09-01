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
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from runner.intent_bench import HF_OWNER, results_repo_for
from runner.tts_bench import _score_quality_dimensions  # noqa: F401 (patchable at module level; optional-dep import is inside tts_bench's own lazy judge getters)

log = logging.getLogger("rescore-tts")

MODALITY = "tts"


def _needs_rescoring(row: dict) -> bool:
    extras = row.get("extras") or {}
    return ("sigmos.ovrl" not in extras or "dnsmos.ovrl" not in extras
            or "nisqa.mos" not in extras)


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


def rescore_file(path: Path, repo_dir: Path) -> tuple[int, int]:
    """Rescore one predictions JSONL file in place, one row at a time.

    Rows are read, scored and written straight back out through a sibling
    temp file rather than being parsed into a list held for the whole file
    — a prediction file can be thousands of rows, and each row's judge call
    only needs the one row and its wav on disk at a time. The temp file is
    atomically renamed onto ``path`` at the end, so a crash mid-run never
    leaves a partially-rewritten predictions file behind.

    Returns ``(n_rescored, n_skipped)``.
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
            if _needs_rescoring(row):
                wav_path = _resolve_wav_path(repo_dir, row)
                if wav_path is None:
                    skipped += 1
                else:
                    try:
                        new_extras = _score_quality_dimensions(wav_path)
                    except Exception as exc:
                        log.warning("rescoring failed for %s (%s): %s",
                                    row.get("sample_id"), row.get("competitor_id"), exc)
                        new_extras = None
                    if not new_extras:
                        skipped += 1
                    else:
                        row.setdefault("extras", {}).update(new_extras)
                        rescored += 1
                        changed = True
            else:
                skipped += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    if changed:
        tmp_path.replace(path)
    else:
        tmp_path.unlink()
    return rescored, skipped


def rescore_repo(repo_id: str, revision: str = "main") -> Path:
    """Download, rescore in place, and return the local repo dir for upload."""
    repo_dir = _download_repo_tree(repo_id, revision)
    predictions_dir = repo_dir / "predictions"
    total_rescored = 0
    total_skipped = 0
    for jsonl_path in sorted(predictions_dir.glob("**/*.jsonl")):
        rescored, skipped = rescore_file(jsonl_path, repo_dir)
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repo_id = results_repo_for(MODALITY, args.dataset_id, args.hf_owner)
    repo_dir = rescore_repo(repo_id, args.revision)
    if args.upload:
        upload_rescored(repo_id, repo_dir)
        log.info("Uploaded rescored predictions to %s", repo_id)


if __name__ == "__main__":
    main()
