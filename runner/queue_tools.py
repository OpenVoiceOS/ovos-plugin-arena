"""Full-sweep queue generator.

Diffs the declarative registry (``registry/competitors/<modality>/*.json`` ×
``registry/datasets/<modality>/*.json``) against what is actually published
on HuggingFace for each eval dataset's ``predictions_hf`` repo, and emits
``runner/queue.yaml``-shaped job entries for every (fighter × dataset) pair
that is missing or incomplete.

This module never writes ``runner/queue.yaml`` and never runs a benchmark —
it only reports/generates job entries for a human to review and paste in.

HF layout assumed (matches ``arena.predictions``): one file per competitor
under ``predictions/<competitor_id>.jsonl`` inside the dataset's
``predictions_hf`` repo.

Listing is done with ``HfApi.list_repo_tree`` (metadata only — file names +
sizes), never a full ``snapshot_download``. Row counts are only fetched
(``count_rows``) for files that are non-empty, by downloading just that one
file (via ``hf_hub_download``) — still not "download everything".
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from registry.loaders import load_all_competitors, load_all_datasets
from registry.schemas import CompetitorDef, DatasetDef

MODALITIES = ("stt", "wake_word", "tts", "vad")

# Cheaper/smaller engines first — static weight heuristic (lower = cheaper).
# Matched by substring against competitor_id/plugin; unmatched falls back to
# _DEFAULT_WEIGHT.
ENGINE_WEIGHTS: dict[str, int] = {
    "vosk": 1,
    "webrtc": 1,
    "silero": 2,
    "citrinet": 3,
    "fasterwhisper-base": 3,
    "fasterwhisper-small": 4,
    "fasterwhisper": 5,
    "chromium": 2,
    "onnx-asr": 4,
    "whisper": 6,
    "azure": 7,
    "google": 7,
    "openai": 8,
    "parakeet": 9,
}
_DEFAULT_WEIGHT = 5


def engine_weight(competitor_id: str, plugin: str | None) -> int:
    """Static cheap-first ordering heuristic; longest match wins ties."""
    haystacks = [competitor_id.lower(), (plugin or "").lower()]
    best: int | None = None
    best_len = -1
    for key, weight in ENGINE_WEIGHTS.items():
        for h in haystacks:
            if key in h and len(key) > best_len:
                best = weight
                best_len = len(key)
    return _DEFAULT_WEIGHT if best is None else best


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def dataset_langs(dataset: DatasetDef) -> list[str]:
    """Every BCP-47 tag this dataset covers."""
    if dataset.lang == "multi" and dataset.langs:
        return list(dataset.langs)
    return [dataset.lang]


def is_compatible(competitor: CompetitorDef, dataset: DatasetDef) -> bool:
    """A fighter is compatible with a dataset when their language sets
    overlap. An *empty* ``langs`` list on the fighter means "any language"."""
    if not competitor.langs:
        return True
    return bool(set(competitor.langs) & set(dataset_langs(dataset)))


def enumerate_pairs(
    modality: str,
    registry_root: Path | None = None,
) -> list[tuple[CompetitorDef, DatasetDef]]:
    """Every compatible (competitor × eval dataset) pair for *modality*."""
    competitors = [
        c for c in load_all_competitors(registry_root) if c.modality == modality
    ]
    datasets = [
        d
        for d in load_all_datasets(registry_root)
        if d.modality == modality and d.role == "eval"
    ]
    pairs: list[tuple[CompetitorDef, DatasetDef]] = []
    for dataset in datasets:
        for competitor in competitors:
            if is_compatible(competitor, dataset):
                pairs.append((competitor, dataset))
    return pairs


# ---------------------------------------------------------------------------
# HF listing (injectable — tests never hit the network)
# ---------------------------------------------------------------------------


class HFLister(Protocol):
    def list_files(self, repo_id: str) -> dict[str, int]:
        """Return {relative_path_in_repo: size_bytes} for a dataset repo."""
        ...

    def count_rows(self, repo_id: str, path_in_repo: str) -> int:
        """Return the number of non-blank JSONL lines in one repo file."""
        ...


class HubLister:
    """Real ``huggingface_hub``-backed implementation."""

    def __init__(self) -> None:
        self._file_cache: dict[str, dict[str, int]] = {}

    def list_files(self, repo_id: str) -> dict[str, int]:
        if repo_id in self._file_cache:
            return self._file_cache[repo_id]
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

        files: dict[str, int] = {}
        try:
            for entry in HfApi().list_repo_tree(
                repo_id, path_in_repo="predictions", recursive=True, repo_type="dataset"
            ):
                size = getattr(entry, "size", None)
                if size is not None:
                    files[entry.path] = size
        except (RepositoryNotFoundError, RevisionNotFoundError):
            # The repo (or its "predictions" path) genuinely does not exist —
            # every competitor for this dataset is legitimately all-missing.
            files = {}
        # Anything else (network failure, rate limit, auth error, ...) is a
        # transient/ambient problem, not "nothing is published" — propagate
        # it rather than silently returning {} and mislabeling every
        # competitor in this repo as "no_file". A sweep generator must fail
        # loudly here, not emit a queue that re-runs work that's actually
        # already done.
        self._file_cache[repo_id] = files
        return files

    def count_rows(self, repo_id: str, path_in_repo: str) -> int:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=repo_id, filename=path_in_repo, repo_type="dataset"
        )
        count = 0
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


@dataclass
class MissingPair:
    modality: str
    competitor: CompetitorDef
    dataset: DatasetDef
    reason: str  # "no_repo" | "no_file" | "empty_file" | "low_rows"
    rows: int | None = field(default=None)


def find_missing_pairs(
    modality: str,
    registry_root: Path | None = None,
    min_rows: int = 1,
    lister: HFLister | None = None,
    check_rows: bool = True,
) -> list[MissingPair]:
    """Diff registry pairs for *modality* against HF prediction state."""
    lister = lister or HubLister()
    missing: list[MissingPair] = []

    for competitor, dataset in enumerate_pairs(modality, registry_root):
        if not dataset.predictions_hf:
            missing.append(MissingPair(modality, competitor, dataset, "no_repo"))
            continue

        files = lister.list_files(dataset.predictions_hf)
        rel_path = f"predictions/{competitor.competitor_id}.jsonl"
        size = files.get(rel_path)

        if size is None:
            missing.append(MissingPair(modality, competitor, dataset, "no_file"))
            continue
        if size == 0:
            missing.append(MissingPair(modality, competitor, dataset, "empty_file"))
            continue
        if check_rows:
            rows = lister.count_rows(dataset.predictions_hf, rel_path)
            if rows < min_rows:
                missing.append(
                    MissingPair(modality, competitor, dataset, "low_rows", rows=rows)
                )

    def sort_key(mp: MissingPair):
        return (
            engine_weight(mp.competitor.competitor_id, mp.competitor.plugin),
            mp.competitor.competitor_id,
            mp.dataset.dataset_id,
        )

    return sorted(missing, key=sort_key)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_dry_run_table(missing: list[MissingPair]) -> str:
    if not missing:
        return "No missing or incomplete pairs found.\n"
    header = f"{'modality':<10} {'competitor_id':<32} {'dataset_id':<28} {'lang':<8} {'reason':<12} rows"
    lines = [header, "-" * len(header)]
    for mp in missing:
        rows = "" if mp.rows is None else str(mp.rows)
        lines.append(
            f"{mp.modality:<10} {mp.competitor.competitor_id:<32} "
            f"{mp.dataset.dataset_id:<28} {mp.dataset.lang:<8} {mp.reason:<12} {rows}"
        )
    lines.append(f"\n{len(missing)} missing/incomplete pair(s).")
    return "\n".join(lines) + "\n"


def render_queue_yaml(missing: list[MissingPair]) -> str:
    lines = [
        "# Auto-generated full-sweep queue entries.",
        "# Generated by `python -m runner.queue_tools` — review before merging",
        "# into runner/queue.yaml.",
        "jobs:",
    ]
    for mp in missing:
        comp = mp.competitor
        ds = mp.dataset
        hf_out = ds.predictions_hf or f"OpenVoiceOS/ovos-{mp.modality}-bench-{ds.dataset_id}"
        lines.append(f"  # {mp.reason}" + (f" (rows={mp.rows})" if mp.rows is not None else ""))
        lines.append(f"  - competitor: {comp.competitor_id}")
        lines.append(f"    dataset_ref: {ds.dataset_id}")
        lines.append(f"    hf_output_dataset: {hf_out}")
        lines.append("    max_samples: 0")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m runner.queue_tools",
        description="Generate full-sweep runner/queue.yaml job entries by "
        "diffing the registry against published HF predictions.",
    )
    parser.add_argument(
        "--modality",
        choices=[*MODALITIES, "all"],
        default="all",
        help="Restrict to one modality (default: all)",
    )
    parser.add_argument("--out", default=None, help="Write YAML to this file instead of stdout")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a human-readable table of missing pairs instead of YAML",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="A published file with fewer rows than this is 'incomplete' (default: 1)",
    )
    parser.add_argument(
        "--no-row-check",
        action="store_true",
        help="Skip per-file row counting (size/existence checks only — no downloads)",
    )
    parser.add_argument(
        "--registry-root",
        default=None,
        help="Override registry root (default: repo's registry/ dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    registry_root = Path(args.registry_root) if args.registry_root else None
    modalities: Iterable[str] = MODALITIES if args.modality == "all" else [args.modality]

    all_missing: list[MissingPair] = []
    lister = HubLister()
    try:
        for modality in modalities:
            all_missing.extend(
                find_missing_pairs(
                    modality,
                    registry_root=registry_root,
                    min_rows=args.min_rows,
                    lister=lister,
                    check_rows=not args.no_row_check,
                )
            )
    except Exception as exc:
        # A transient HF failure (network error, rate limit, auth error, ...)
        # must abort generation, not be swallowed into a queue that
        # re-schedules work that's actually already done — see HubLister
        # .list_files.
        print(f"error: failed to query HuggingFace state: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.dry_run:
        sys.stdout.write(render_dry_run_table(all_missing))
        return

    output = render_queue_yaml(all_missing)
    if args.out:
        Path(args.out).write_text(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
