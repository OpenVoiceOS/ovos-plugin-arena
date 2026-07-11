"""Spec coverage (§A7.1): every R-number a module cites must exist in the spec.

The specification numbers its normative requirements R1, R2, … (with letter
suffixes like R5a). Modules and tests cite those numbers in comments. If a
requirement is renumbered or removed without updating the citation, the code
points at a rule that no longer exists — this test catches that orphan.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "SPECIFICATION.md"

# Definitions look like: "- **R5a — Significance gate.** …"
_DEF = re.compile(r"\*\*(R\d+[a-z]?)\b")
# Citations are bare tokens like "§4 R5" or "**R1**".
_REF = re.compile(r"\bR\d+[a-z]?\b")


def _defined() -> set[str]:
    return set(_DEF.findall(SPEC.read_text(encoding="utf-8")))


def _citations() -> dict[str, set[str]]:
    cites: dict[str, set[str]] = {}
    for path in list((ROOT / "arena").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        if path.name == "test_spec_coverage.py":
            continue
        found = set(_REF.findall(path.read_text(encoding="utf-8")))
        if found:
            cites[str(path.relative_to(ROOT))] = found
    return cites


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
