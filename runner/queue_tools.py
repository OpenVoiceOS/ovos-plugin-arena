"""Full-sweep queue generator.

Diffs the declarative registry (``registry/competitors/<modality>/*.json`` ×
``registry/datasets/<modality>/*.json``) against what is actually published
on HuggingFace for each eval dataset's ``predictions_hf`` repo, and emits
``runner/queue.yaml``-shaped job entries for every (fighter × dataset) pair
that is missing or incomplete.

This module never writes ``runner/queue.yaml`` and never runs a benchmark —
it only reports/generates job entries for a human to review and paste in.

HF layout assumed (matches ``arena.predictions``): one file per competitor
per lang under ``predictions/<lang>/<competitor_id>.jsonl`` inside the
dataset's ``predictions_hf`` repo (the flat legacy
``predictions/<competitor_id>.jsonl`` form is still accepted).

Listing is done with ``HfApi.list_repo_tree`` (metadata only — file names +
sizes), never a full ``snapshot_download``. Row counts are only fetched
(``count_rows``) for files that are non-empty, by downloading just that one
file (via ``hf_hub_download``) — still not "download everything".
"""
from __future__ import annotations

import argparse
import re
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
}
_DEFAULT_WEIGHT = 5

# Model-family weights, matched only against the competitor_id remainder
# AFTER stripping the engine-wrapper prefix (e.g. "parakeet-tdt-11b" out of
# "onnx-asr-parakeet-tdt-11b", or "parakeet-tdt-1.1b-fp16" out of
# "coreml-parakeet-tdt-1.1b-fp16") — see engine_weight's docstring for the
# bug this fixes. Longest match among these wins; a match here always
# overrides the wrapper's own base weight in ENGINE_WEIGHTS, since it is
# strictly more specific by construction (the wrapper prefix has already
# been stripped off), not a coincidental same-length substring collision.
FAMILY_WEIGHTS: dict[str, int] = {
    # Size-tiered within a family, largest/most-expensive first — a bare
    # family name (no size token in the id) falls to that family's own
    # generic entry near the bottom, cheaper than any sized variant above
    # it: an unsized id is never assumed to be the biggest member.
    "parakeet-tdt-11b": 10,
    "cohere-transcribe-2b": 9,
    "parakeet-rnnt-1.1b": 8,
    "parakeet-tdt-1.1b": 8,
    "parakeet-ctc-1.1b": 8,
    "granite": 8,
    "voxtral": 8,
    "cohere": 7,
    "qwen3": 7,
    "canary": 7,
    "parakeet": 6,
}

# A wrapper's engine-family prefix, derived from the registry ``plugin`` id
# (e.g. "ovos-stt-plugin-onnx-asr" -> "onnx-asr") rather than a hand-
# maintained list — every competitor's ``plugin`` field is already registry
# data, so this reads it straight off what's passed to engine_weight.
_PLUGIN_PREFIX_RE = re.compile(r"^ovos-[a-z0-9]+-plugin-", re.IGNORECASE)


def _wrapper_prefix(plugin: str | None) -> str | None:
    if not plugin:
        return None
    match = _PLUGIN_PREFIX_RE.match(plugin)
    if not match:
        return None
    return plugin[match.end():].lower() or None


def engine_weight(competitor_id: str, plugin: str | None) -> int:
    """Static cheap-first ordering heuristic; longest match wins ties.

    Regression this guards: matching :data:`ENGINE_WEIGHTS` by plain
    substring against the full ``competitor_id`` let a generic engine-
    wrapper prefix (``onnx-asr``) tie in length against an unrelated,
    coincidentally-same-length model-family key (``parakeet``) and win the
    tie by dict insertion order — so every ``onnx-asr-parakeet-*`` id
    (including the 11B ``onnx-asr-parakeet-tdt-11b``) got the wrapper's
    cheap weight (4) instead of the family's expensive one (9). Model-
    family keys (:data:`FAMILY_WEIGHTS`) are now matched only against the
    remainder of ``competitor_id`` after stripping the wrapper prefix
    derived from ``plugin`` (:func:`_wrapper_prefix`), and always win over
    the wrapper's own base weight when they match — no length-tie
    coincidence possible between the two dicts.
    """
    remainder = competitor_id.lower()
    wrapper = _wrapper_prefix(plugin)
    if wrapper and remainder.startswith(wrapper + "-"):
        remainder = remainder[len(wrapper) + 1:]

    family_best: int | None = None
    family_len = -1
    for key, weight in FAMILY_WEIGHTS.items():
        if key in remainder and len(key) > family_len:
            family_best = weight
            family_len = len(key)
    if family_best is not None:
        return family_best

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


_TRAILING_LOCALE_RE = re.compile(r"-([a-zA-Z]{2,3}(?:-[a-zA-Z]{2,3})?)$")


def resolved_dataset_lang(dataset: DatasetDef) -> str | None:
    """The single concrete BCP-47 tag this dataset runs jobs under, or
    ``None`` when the dataset is genuinely multilingual/unknown.

    A queued job that omits ``lang`` resolves against the *fighter's*
    default lang (``queue_config._plugin_from_competitor``), not the
    dataset's — for a multilingual fighter that silently runs the wrong
    lang and publishes into the wrong ``predictions/<lang>/`` path (e.g.
    onnx-asr-canary queued against ``speech-massive-de-DE`` running as
    ``en`` instead of ``de-DE``). Every generated entry for a
    single-language dataset must pin ``lang`` explicitly to this value.
    """
    lang = getattr(dataset, "lang", None)
    if lang and lang != "multi":
        return lang
    # Registry lang is missing/multi/unknown — fall back to parsing a
    # trailing "-xx-XX" (or "-xx") locale suffix off the dataset id itself,
    # e.g. "speech-massive-de-DE" -> "de-DE".
    match = _TRAILING_LOCALE_RE.search(dataset.dataset_id)
    if match:
        return match.group(1)
    return None


def is_compatible(competitor: CompetitorDef, dataset: DatasetDef) -> bool:
    """A fighter is compatible with a dataset when their language sets
    overlap. An *empty* ``langs`` list on the fighter means "any language".

    Registry fighters commonly pin bare primary subtags (``de``) while
    datasets carry full BCP-47 tags (``de-DE``) — comparison goes through
    :func:`runner.media_bench._lang_matches` (primary-subtag equality), the
    same rule every bench adapter uses, so the queue diff and the benches
    agree on what "compatible" means."""
    from runner.media_bench import _lang_matches

    if not competitor.langs:
        return True
    return any(
        _lang_matches(cl, dl)
        for cl in competitor.langs
        for dl in dataset_langs(dataset)
    )


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
        from huggingface_hub.utils import (
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        files: dict[str, int] = {}
        try:
            for entry in HfApi().list_repo_tree(
                repo_id, path_in_repo="predictions", recursive=True, repo_type="dataset"
            ):
                size = getattr(entry, "size", None)
                if size is not None:
                    files[entry.path] = size
        except (EntryNotFoundError, RepositoryNotFoundError,
                RevisionNotFoundError):
            # The repo, its revision, or its "predictions" path genuinely
            # does not exist (EntryNotFoundError = repo exists but the
            # predictions/ folder was never written — every freshly created
            # prediction repo looks like this) — every competitor for this
            # dataset is legitimately all-missing.
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
    present_by_competitor: dict[str, int] | None = None,
    present_by_dataset: dict[str, int] | None = None,
) -> list[MissingPair]:
    """Diff registry pairs for *modality* against HF prediction state.

    *present_by_competitor*/*present_by_dataset*, if given, are populated
    in place with the same presence counts used internally for breadth-first
    tiering (competitor_id/dataset_id -> number of already-published pairs)
    — callers outside this module (``runner.autorun``) that need the raw
    presence signal for their own tiering pass these in instead of
    re-deriving the HF-listing/lang-matching logic above.
    """
    lister = lister or HubLister()
    missing: list[MissingPair] = []
    if present_by_competitor is None:
        present_by_competitor = {}
    if present_by_dataset is None:
        present_by_dataset = {}

    def _mark_present(competitor: CompetitorDef, dataset: DatasetDef) -> None:
        present_by_competitor[competitor.competitor_id] = (
            present_by_competitor.get(competitor.competitor_id, 0) + 1
        )
        present_by_dataset[dataset.dataset_id] = (
            present_by_dataset.get(dataset.dataset_id, 0) + 1
        )

    for competitor, dataset in enumerate_pairs(modality, registry_root):
        if not dataset.predictions_hf:
            missing.append(MissingPair(modality, competitor, dataset, "no_repo"))
            continue

        files = lister.list_files(dataset.predictions_hf)
        # The published layout nests per lang (predictions/<lang>/<id>.jsonl,
        # what media_bench and the daemon write — spec §3.2); the flat
        # predictions/<id>.jsonl form predates it and is always accepted too.
        #
        # For a dataset with a concrete lang, only the shard published under
        # THAT lang counts: matching any lang dir by suffix let a
        # wrong-lang run (e.g. a multilingual fighter defaulting to "en"
        # against a de-DE dataset) satisfy the pair and never get re-queued
        # — the shard existed, just under the wrong language and wrong
        # transcription conditioning. Only a dataset that is genuinely
        # multi/unknown-lang keeps the old any-lang suffix matching.
        lang = resolved_dataset_lang(dataset)
        flat_path = f"predictions/{competitor.competitor_id}.jsonl"
        if lang:
            lang_path = f"predictions/{lang}/{competitor.competitor_id}.jsonl"
            matches = {
                path: size for path, size in files.items()
                if path == flat_path or path == lang_path
            }
        else:
            suffix = f"/{competitor.competitor_id}.jsonl"
            matches = {
                path: size for path, size in files.items()
                if path == flat_path
                or (path.startswith("predictions/") and path.endswith(suffix))
            }

        if not matches:
            missing.append(MissingPair(modality, competitor, dataset, "no_file"))
            continue
        if sum(matches.values()) == 0:
            missing.append(MissingPair(modality, competitor, dataset, "empty_file"))
            continue
        if check_rows:
            rows = sum(
                lister.count_rows(dataset.predictions_hf, path)
                for path, size in matches.items() if size
            )
            if rows < min_rows:
                missing.append(
                    MissingPair(modality, competitor, dataset, "low_rows", rows=rows)
                )
            else:
                _mark_present(competitor, dataset)
        else:
            _mark_present(competitor, dataset)

    return breadth_first_order(missing, present_by_competitor, present_by_dataset)


# ---------------------------------------------------------------------------
# Breadth-first ordering
# ---------------------------------------------------------------------------
#
# Coverage is lumpy when missing pairs are sorted fighter-major (every pair
# for one competitor before the next): a fighter compatible with many
# datasets soaks up the front of the queue while a fighter compatible with
# only one or two sits at the back, so a long sweep run can finish with most
# fighters still at zero predictions. Breadth-first instead tiers missing
# pairs by how many (dataset, lang) pairs a fighter *already has* published
# — every fighter gets its first pair before any fighter gets a second, then
# a second before any third, and so on — so partial coverage spreads across
# every fighter before any one of them goes deep.


def breadth_tier_sort(
    items: list,
    competitor_id_of,
    dataset_id_of,
    weight_of,
    present_by_competitor: dict[str, int],
    present_by_dataset: dict[str, int],
) -> list:
    """Order arbitrary *items* breadth-first across fighters.

    Generic engine behind :func:`breadth_first_order` — factored out so a
    caller outside this module with its own item shape (``runner.autorun``'s
    ``PairKey``, not :class:`MissingPair`) can reuse the exact same tiering
    instead of re-deriving it. *competitor_id_of*/*dataset_id_of*/
    *weight_of* are accessor callables over one item.

    *present_by_competitor* and *present_by_dataset* are presence counts —
    how many (dataset, lang) pairs each competitor/dataset already has
    published predictions for — computed from the same HF listing the
    sweep diff already does (see :func:`find_missing_pairs`). Pure function,
    no I/O: both maps are plain ``dict[str, int]`` so this is unit-testable
    against a synthetic registry without touching HuggingFace.

    Within a tier, cheaper jobs go first (*weight_of* — smaller/CPU
    fighters before slow/expensive ones), and among equally-cheap jobs the
    dataset with the most fighters already present is preferred, so fresh
    predictions land on datasets that already have enough competitors to
    pair up into battles rather than spreading onto a dataset no one else
    has touched yet.
    """
    by_competitor: dict[str, list] = {}
    for item in items:
        by_competitor.setdefault(competitor_id_of(item), []).append(item)

    tiered: list[tuple[int, object]] = []
    for competitor_id, group in by_competitor.items():
        base_tier = present_by_competitor.get(competitor_id, 0)
        # Deterministic per-competitor order before tier assignment: the
        # dataset with the most fighters already present goes first (so the
        # fighter's earliest tiers land on datasets that pair up into
        # battles), tie-broken by dataset_id for reproducibility.
        group.sort(
            key=lambda it: (
                -present_by_dataset.get(dataset_id_of(it), 0),
                dataset_id_of(it),
            )
        )
        for offset, item in enumerate(group):
            tiered.append((base_tier + offset, item))

    def sort_key(entry: tuple[int, object]):
        tier, item = entry
        return (
            tier,
            weight_of(item),
            -present_by_dataset.get(dataset_id_of(item), 0),
            competitor_id_of(item),
            dataset_id_of(item),
        )

    return [item for _tier, item in sorted(tiered, key=sort_key)]


def breadth_first_order(
    missing: list[MissingPair],
    present_by_competitor: dict[str, int],
    present_by_dataset: dict[str, int],
) -> list[MissingPair]:
    """Order *missing* pairs breadth-first across fighters.

    See :func:`breadth_tier_sort` for the tiering rules — this is that
    generic engine specialized to :class:`MissingPair`.
    """
    return breadth_tier_sort(
        missing,
        competitor_id_of=lambda mp: mp.competitor.competitor_id,
        dataset_id_of=lambda mp: mp.dataset.dataset_id,
        weight_of=lambda mp: engine_weight(
            mp.competitor.competitor_id, mp.competitor.plugin
        ),
        present_by_competitor=present_by_competitor,
        present_by_dataset=present_by_dataset,
    )


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
        lang = resolved_dataset_lang(ds)
        if lang:
            # Pin the dataset's own lang so the job doesn't silently fall
            # back to the fighter's default lang (queue_config resolves
            # lang_override or cfg_lang or langs[0]) — see
            # resolved_dataset_lang's docstring for the production bug this
            # closes.
            lines.append(f"    lang: {lang}")
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
