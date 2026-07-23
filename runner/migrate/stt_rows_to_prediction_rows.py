"""Backfill legacy ``ovos-stt-bench-*`` HF datasets to the canonical §3.2
``PredictionRow`` schema (A2.1 step 2 of 3 — see ``docs/SPECIFICATION.md``).

Step 1 (merged) added a read-compat adapter: ``runner.schema.STTRow`` /
``arena.predictions.parse_row`` detect the legacy column layout
(``dataset_entry_id`` / ``plugin_name`` / ``prediction_transcript`` /
``transcript`` / ``prediction_confidence`` / ``prediction_type``, no
``sample_id``) and convert it to a §3.2 row on the fly at load time, tagging
it ``schema_version: 1`` for provenance.

This module is step 2: it rewrites the *published* legacy datasets
themselves so downstream tooling can eventually stop paying the read-compat
tax. It does **not** reimplement the field mapping — every row is converted
via ``arena.predictions.parse_row`` (which itself defers to
``STTRow.to_prediction_row_dict`` for the legacy shape), then re-stamped
``schema_version: 2`` to mark it as a source-of-truth §3.2 row rather than a
read-time shim conversion.

Rows are pushed as a **new revision** under a ``v2/`` path inside the same
HF dataset repo — old root-level shard files (``stt_<lang>_<plugin>_<n>.jsonl``,
see ``runner/publish.py``) are never touched or deleted; the migration only
*adds* a commit. ``--dry-run`` (the default) never calls the network-writing
entry point (``push_migrated``); ``--apply`` is required to push for real
and is meant to be maintainer-run only.

This migration never touches ``runner/schema.py:JobManifest`` state and
never re-runs or resumes a completed job — it only rewrites already-published
HF rows. Step 3 (removing the ``STTRow`` read-compat shim) follows in a later
release, once every legacy dataset has been migrated and republished.

Idempotent: a dataset whose rows are already ``schema_version: 2`` (or
already in the canonical shape, since ``PredictionRow.schema_version``
defaults to ``2``) is detected and skipped — re-running the script is a
no-op for already-migrated data.

Usage::

    python -m runner.migrate.stt_rows_to_prediction_rows \\
        --dataset OpenVoiceOS/ovos-stt-bench-pt-PT --out /tmp/backfill

    python -m runner.migrate.stt_rows_to_prediction_rows --all --out /tmp/backfill

    # maintainer-run only, after reviewing the dry-run output:
    python -m runner.migrate.stt_rows_to_prediction_rows \\
        --dataset OpenVoiceOS/ovos-stt-bench-pt-PT --apply
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("migrate-stt-rows")

HF_ORG = "OpenVoiceOS"
LEGACY_DATASET_PREFIX = "ovos-stt-bench-"


@dataclass
class MigrationResult:
    repo_id: str
    source_revision: str = ""
    new_revision: str = ""
    files_seen: int = 0
    files_skipped_idempotent: int = 0
    rows_seen: int = 0
    rows_migrated: int = 0
    output_files: list[str] = field(default_factory=list)
    applied: bool = False

    def summary(self) -> str:
        return (
            f"{self.repo_id}: {self.rows_migrated}/{self.rows_seen} rows migrated "
            f"across {self.files_seen - self.files_skipped_idempotent}/{self.files_seen} "
            f"file(s) (source_revision={self.source_revision or '?'}, "
            f"new_revision={self.new_revision or ('n/a (dry-run)' if not self.applied else '?')})"
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_legacy_datasets(org: str = HF_ORG) -> list[str]:
    """List every ``<org>/ovos-stt-bench-*`` dataset repo id on the Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    repos = api.list_datasets(author=org)
    return sorted(
        d.id for d in repos
        if d.id.split("/", 1)[-1].startswith(LEGACY_DATASET_PREFIX)
    )


# ---------------------------------------------------------------------------
# Download (legacy layout: flat *.jsonl shards at the repo root — see
# runner/publish.py — NOT under predictions/, unlike the canonical §3.2 layout)
# ---------------------------------------------------------------------------


def _source_revision(repo_id: str, revision: str = "main") -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision)
    return info.sha or revision


def download_legacy_files(repo_id: str, revision: str = "main") -> dict[str, list[dict]]:
    """Download every root-level ``*.jsonl`` shard of *repo_id*.

    Returns ``{filename: [raw_row, ...]}``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [
        f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if f.endswith(".jsonl") and "/" not in f
    ]
    out: dict[str, list[dict]] = {}
    for fname in sorted(files):
        local = hf_hub_download(
            repo_id, fname, repo_type="dataset", revision=revision,
        )
        rows = []
        for lineno, line in enumerate(Path(local).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d skipped (bad json): %s", fname, lineno, exc)
        out[fname] = rows
    return out


# ---------------------------------------------------------------------------
# Conversion — reuses arena.predictions.parse_row / STTRow, no reimplemented
# field mapping.
# ---------------------------------------------------------------------------


def is_already_migrated(raw: dict) -> bool:
    """True when *raw* is already a canonical schema_version 2 §3.2 row."""
    return "sample_id" in raw and raw.get("schema_version", 2) == 2


def convert_row(raw: dict, competitor_id_fallback: str) -> dict:
    """Convert one raw legacy (or already-canonical) row to a schema_version 2
    §3.2 row dict, via ``arena.predictions.parse_row`` (which defers to
    ``STTRow.to_prediction_row_dict`` for the legacy shape)."""
    from arena.predictions import parse_row

    row = parse_row(raw, competitor_id_fallback)
    data = row.model_dump()
    data["schema_version"] = 2  # source-of-truth §3.2 row, not a read-time shim
    return data


def migrate_file(fname: str, rows: list[dict]) -> tuple[list[dict], bool]:
    """Convert *rows* from one legacy shard.

    Returns ``(migrated_rows, skipped)`` — ``skipped`` is True when every row
    was already schema_version 2 (idempotent no-op; no output produced).
    """
    if rows and all(is_already_migrated(r) for r in rows):
        return [], True
    competitor_fallback = Path(fname).stem
    migrated = [convert_row(r, competitor_fallback) for r in rows]
    return migrated, False


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write *rows* with stable ``sort_keys`` ordering for byte-stable output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Push (network-writing entry point — never called on --dry-run; tests
# monkeypatch this symbol to assert dry-run makes zero network writes)
# ---------------------------------------------------------------------------


def push_migrated(repo_id: str, files: dict[str, Path], token: str | None = None) -> str:
    """Push migrated *files* (``{path_in_repo_relative_name: local_path}``)
    under ``v2/`` in *repo_id* as a new commit. Never deletes or overwrites
    the legacy root-level shards. Returns the new revision SHA.

    Maintainer-run only (``--apply``) — never invoked by ``--dry-run``.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for name, local_path in files.items():
        api.upload_file(
            path_or_fileobj=local_path.read_bytes(),
            path_in_repo=f"v2/{name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"migrate: backfill {name} to §3.2 schema_version 2 (v2/)",
        )
    info = api.dataset_info(repo_id, revision="main")
    return info.sha or "main"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def migrate_dataset(
    repo_id: str,
    out_dir: Path | None,
    apply: bool = False,
    revision: str = "main",
    token: str | None = None,
) -> MigrationResult:
    result = MigrationResult(repo_id=repo_id)
    result.source_revision = _source_revision(repo_id, revision)

    shards = download_legacy_files(repo_id, revision)
    result.files_seen = len(shards)

    written: dict[str, Path] = {}
    for fname, rows in shards.items():
        result.rows_seen += len(rows)
        migrated, skipped = migrate_file(fname, rows)
        if skipped:
            result.files_skipped_idempotent += 1
            logger.info("%s/%s already schema_version 2 — skipping (idempotent)",
                        repo_id, fname)
            continue
        result.rows_migrated += len(migrated)
        if out_dir is not None:
            out_path = out_dir / repo_id.replace("/", "__") / fname
            write_jsonl(migrated, out_path)
            written[fname] = out_path
            result.output_files.append(str(out_path))

    if apply and written:
        result.new_revision = push_migrated(repo_id, written, token=token)
        result.applied = True
    elif apply and not written:
        logger.info("%s: nothing to apply (already migrated)", repo_id)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help=f"single {HF_ORG}/ovos-stt-bench-<lang> repo id")
    group.add_argument("--all", action="store_true",
                        help=f"discover every {HF_ORG}/ovos-stt-bench-* repo on the Hub")
    parser.add_argument("--out", type=Path, default=None,
                         help="local dir to write migrated JSONL to")
    parser.add_argument("--revision", default="main", help="source HF revision to read")
    parser.add_argument("--dry-run", action="store_true", default=True,
                         help="(default) never pushes; only writes --out and prints stats")
    parser.add_argument("--apply", action="store_true",
                         help="push migrated rows as a new v2/ revision (maintainer-run only)")
    args = parser.parse_args(argv)

    apply = args.apply
    if apply:
        args.dry_run = False

    if args.all:
        repo_ids = discover_legacy_datasets()
        if not repo_ids:
            logger.info("no %s/%s* datasets found", HF_ORG, LEGACY_DATASET_PREFIX)
    else:
        repo_ids = [args.dataset]

    for repo_id in repo_ids:
        result = migrate_dataset(
            repo_id, out_dir=args.out, apply=apply, revision=args.revision,
        )
        print(result.summary())
        if apply:
            print(f"  before: {result.source_revision}  after: {result.new_revision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
