"""Spec coverage (§A7.1): the R-numbers cited in code and the ones defined in
the spec must match in both directions.

The specification numbers its normative requirements R1, R2, … (with letter
suffixes like R5a). Modules and tests cite those numbers in comments.

- **No orphaned citations.** If a requirement is renumbered or removed
  without updating the citation, the code points at a rule that no longer
  exists.
- **No uncited requirements.** If a requirement is implemented but nothing
  in the codebase cites its R-number, a future reader has no way to find
  where it lives, and a change to that code has no textual link back to the
  rule it must keep satisfying.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "SPECIFICATION.md"

# Definitions look like: "- **R5a — Significance gate.** …"
_DEF = re.compile(r"\*\*(R\d+[a-z]?)\b")
# Citations are bare tokens like "§4 R5" or "**R1**".
_REF = re.compile(r"\bR\d+[a-z]?\b")

# Every source directory that may legitimately implement a spec requirement.
_SOURCE_DIRS = ("arena", "runner", "registry", "tests")


def _defined() -> set[str]:
    return set(_DEF.findall(SPEC.read_text(encoding="utf-8")))


def _source_files():
    for dirname in _SOURCE_DIRS:
        yield from (ROOT / dirname).glob("*.py")


def _citations() -> dict[str, set[str]]:
    cites: dict[str, set[str]] = {}
    for path in _source_files():
        if path.name == "test_spec_coverage.py":
            continue
        found = set(_REF.findall(path.read_text(encoding="utf-8")))
        if found:
            cites[str(path.relative_to(ROOT))] = found
    return cites


def _all_cited() -> set[str]:
    cited: set[str] = set()
    for refs in _citations().values():
        cited |= refs
    return cited


def test_spec_defines_requirements():
    defined = _defined()
    # Sanity: the spec should define at least the core rules.
    assert {"R1", "R5"} <= defined, f"spec missing core R-numbers, got {sorted(defined)}"


def test_no_orphaned_requirement_citations():
    defined = _defined()
    orphans = {
        path: sorted(refs - defined)
        for path, refs in _citations().items()
        if refs - defined
    }
    assert not orphans, f"citations reference undefined spec R-numbers: {orphans}"


def test_every_requirement_is_cited():
    defined = _defined()
    cited = _all_cited()
    uncited = sorted(defined - cited)
    assert not uncited, (
        f"spec R-numbers with no citation anywhere in {_SOURCE_DIRS}: {uncited} "
        "— add a short '# R<n> <name>' comment at the implementing site"
    )


def test_no_modality_with_a_scorer_is_called_unscored():
    """A modality the code aggregates a board metric for is never described
    in the spec as having no objective metric, board or ELO seed.

    ``arena.metrics`` grows one ``score_<modality>`` aggregator per modality
    that has a benchmark board. The spec's modality walkthrough is prose and
    drifts silently when a scorer lands, which understates the system to
    exactly the reader least able to check it against the code.
    """
    from arena import metrics

    scored = {name[len("score_"):] for name in dir(metrics)
              if name.startswith("score_") and callable(getattr(metrics, name))}
    assert "tts" in scored, "expected score_tts to exist"

    text = SPEC.read_text(encoding="utf-8")
    denials = ("no objective metric", "no benchmark board", "no ELO seed")
    for paragraph in re.split(r"\n(?=\d+\. \*\*)", text):
        named = {m for m in scored if f"**{m.upper()}**" in paragraph
                 or f"**{m}**" in paragraph}
        if not named:
            continue
        for denial in denials:
            assert denial not in paragraph, (
                f"spec says '{denial}' about {sorted(named)}, "
                f"which arena.metrics scores"
            )
