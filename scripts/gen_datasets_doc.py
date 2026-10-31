#!/usr/bin/env python3
"""Write ``docs/datasets.md``: one table per modality of every registry dataset.

The page is generated from ``registry/datasets/``, so it never disagrees with
the registry. ``tests/test_datasets_doc.py`` fails when the committed page
differs from this script's output.

Usage, from the repository root::

    python scripts/gen_datasets_doc.py           # rewrite docs/datasets.md
    python scripts/gen_datasets_doc.py --check   # exit 1 when the page is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.metrics import is_pinned_revision  # noqa: E402
from registry.loaders import load_all_datasets  # noqa: E402
from registry.schemas import DatasetDef  # noqa: E402

OUTPUT = ROOT / "docs" / "datasets.md"

SECTIONS = {
    "intent": "Intent test sets",
    "intent_template": "Intent training sets: templates",
    "intent_keyword": "Intent training sets: keywords",
    "wake_word": "Wake word clip sets",
    "ww_stream": "Wake word streams and false wake-up sets",
    "stt": "Speech to text",
    "tts": "Text to speech",
    "vad": "Voice activity detection",
}

TABLE_HEAD = (
    "| Dataset | Source | Revision | Licence | Languages | Samples | What it tests |\n"
    "|---|---|---|---|---|---|---|\n"
)
FOOTER = (
    "\n---\n[← Methodology](methodology.md) · [Home](index.md)"
    " · [Reproduce a row →](reproduce-a-row.md)\n"
)

HEADER = """\
# Datasets

This page lists every dataset in `registry/datasets/`. The script
`scripts/gen_datasets_doc.py` writes it from the registry. Do not edit it by
hand: change the registry entry, then run the script again.

Each table has these columns:

- **Dataset**: the registry `dataset_id`.
- **Source**: the Hugging Face dataset, with its subset or file pattern.
- **Revision**: the first 12 characters of the pinned commit. A branch or
  ref name, such as `main`, is not a pin and shows "(not a commit)". On
  such a dataset the board cannot drop rows that came from another revision
  of the corpus.
- **Licence**: the `license` field of the entry.
- **Languages**: the `lang` or `langs` field.
- **Samples**: the `sample_policy`. "all" means that a run reads every row.
- **What it tests**: the first sentence of the `summary`, or of the `notes`
  when the entry has no summary.

The [leagues](leagues.md) page tells how each league scores these sets.
"""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u201c\"(])")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _first_sentence(text: str | None) -> str:
    if not text:
        return ""
    return _SENTENCE_END.split(text.strip(), maxsplit=1)[0]


def _source(d: DatasetDef) -> str:
    src = d.source
    cell = f"[`{src.hf_id}`](https://huggingface.co/datasets/{src.hf_id})"
    if src.subset:
        cell += f" subset `{src.subset}`"
    if src.file_pattern:
        cell += f" `{src.file_pattern}`"
    return cell


def _revision(d: DatasetDef) -> str:
    rev = d.source.revision
    if is_pinned_revision(rev):
        return f"`{rev[:12]}`"
    return f"`{rev}` (not a commit)"


def _langs(d: DatasetDef) -> str:
    return ", ".join(d.langs) if d.langs else d.lang


def _samples(d: DatasetDef) -> str:
    policy = d.sample_policy
    if policy is None or policy.max_samples is None:
        return "all"
    if d.wakeword or d.positives_dir:
        return f"{policy.max_samples} per class"
    return f"{policy.max_samples} per language, seed {policy.seed}"


def _row(d: DatasetDef) -> str:
    cells = [
        f"`{d.dataset_id}`",
        _source(d),
        _revision(d),
        d.license or "not stated",
        _langs(d),
        _samples(d),
        _first_sentence(d.summary or d.notes),
    ]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def render(datasets: list[DatasetDef] | None = None) -> str:
    """The full text of ``docs/datasets.md`` for *datasets* (default: the registry)."""
    if datasets is None:
        datasets = load_all_datasets()
    by_modality: dict[str, list[DatasetDef]] = defaultdict(list)
    for d in datasets:
        by_modality[d.modality.value].append(d)
    unknown = sorted(set(by_modality) - set(SECTIONS))
    if unknown:
        raise SystemExit(f"no section title for modality {unknown}; add one to SECTIONS")

    parts = [HEADER]
    for modality, title in SECTIONS.items():
        entries = sorted(by_modality.get(modality, []), key=lambda d: d.dataset_id)
        if not entries:
            continue
        parts.append(f"\n## {title} (`{modality}`)\n\n{len(entries)} datasets.\n\n")
        parts.append(TABLE_HEAD)
        parts.extend(_row(d) + "\n" for d in entries)
    parts.append(FOOTER)
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 when docs/datasets.md differs from the registry",
    )
    args = parser.parse_args(argv)
    datasets = load_all_datasets()
    text = render(datasets)
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT.relative_to(ROOT)} is stale: run python scripts/gen_datasets_doc.py")
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} matches the registry")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(datasets)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
