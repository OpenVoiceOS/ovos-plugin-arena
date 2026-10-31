"""docs/datasets.md is the output of scripts/gen_datasets_doc.py over the registry.

A registry change that is not followed by a regeneration fails here, so the
published dataset list never disagrees with the registry.
"""

import importlib.util
import json
import sys
from pathlib import Path

from registry.loaders import load_all_datasets
from registry.schemas import DatasetDef

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gen_datasets_doc", ROOT / "scripts" / "gen_datasets_doc.py"
)
gen = importlib.util.module_from_spec(_spec)
sys.modules["gen_datasets_doc"] = gen
_spec.loader.exec_module(gen)


def test_committed_page_matches_the_registry():
    committed = (ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")
    assert committed == gen.render(), (
        "docs/datasets.md is stale: run python scripts/gen_datasets_doc.py"
    )


def test_every_registry_dataset_has_one_row():
    page = gen.render()
    ids = [d.dataset_id for d in load_all_datasets()]
    rows = [line for line in page.splitlines() if line.startswith("| `")]
    assert len(rows) == len(ids)
    # a dataset id can exist once per modality (speech-massive is stt and intent)
    for dataset_id in set(ids):
        assert page.count(f"| `{dataset_id}` |") == ids.count(dataset_id), dataset_id


def test_row_shows_pin_prefix_and_branch():
    base = json.loads(
        (ROOT / "registry/datasets/wake_word/synthetic-wakewords-hey_mycroft.json").read_text()
    )
    pinned = DatasetDef.model_validate(
        {**base, "dataset_id": "pinned", "source": {**base["source"], "revision": "ab" * 20}}
    )
    branch = DatasetDef.model_validate(
        {**base, "dataset_id": "branch", "source": {**base["source"], "revision": "main"}}
    )
    page = gen.render([pinned, branch])
    assert "| `pinned` |" in page and "`abababababab` |" in page
    assert "ab" * 20 not in page
    assert "| `branch` |" in page and "`main` (not a commit) |" in page
    assert "| 100 per class |" in page
