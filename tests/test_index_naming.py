"""Guard against the stale-board-duplication defect (chore/purge-stale-boards).

A pre-CI generation era once wrote benchmark boards without a dataset segment
in the filename (``benchmark-intent_template-<lang>.json``), which silently
duplicated the properly-named ``benchmark-intent_template-massive-templates-
<lang>.json`` board under a different key with stale (non-CI) metrics. 43 such
files were purged. This test locks two invariants so the generator can never
recreate that class of bug:

  (a) no two ``index.json`` benchmark entries share (modality, dataset_id, lang)
  (b) every benchmark filename contains its own dataset_id segment
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "frontend-static" / "public" / "data" / "index.json"


def _benchmarks() -> list[dict]:
    payload = json.loads(INDEX.read_text(encoding="utf-8"))
    return payload["benchmarks"]


def test_no_duplicate_benchmark_keys():
    benchmarks = _benchmarks()
    seen: dict[tuple, str] = {}
    dupes = []
    for entry in benchmarks:
        key = (entry.get("modality"), entry.get("dataset_id"), entry.get("lang"))
        if key in seen:
            dupes.append((key, seen[key], entry["file"]))
        else:
            seen[key] = entry["file"]
    assert not dupes, f"duplicate (modality, dataset_id, lang) benchmark entries: {dupes}"


def test_every_benchmark_filename_contains_its_dataset_segment():
    benchmarks = _benchmarks()
    missing = [
        entry["file"]
        for entry in benchmarks
        if entry.get("dataset_id") and entry["dataset_id"] not in entry["file"]
    ]
    assert not missing, (
        f"benchmark filenames missing their dataset_id segment "
        f"(reintroduces the un-suffixed naming bug): {missing}"
    )
