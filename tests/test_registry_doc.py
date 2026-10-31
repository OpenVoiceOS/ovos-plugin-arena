"""docs/registry.md is the output of scripts/gen_registry_doc.py over the schema.

A schema change that is not followed by a regeneration fails here, and so does
a new schema field that has no description, so the published field reference
never disagrees with ``registry/schemas.py``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gen_registry_doc", ROOT / "scripts" / "gen_registry_doc.py"
)
gen = importlib.util.module_from_spec(_spec)
sys.modules["gen_registry_doc"] = gen
_spec.loader.exec_module(gen)


def test_committed_page_matches_the_schema():
    committed = (ROOT / "docs" / "registry.md").read_text(encoding="utf-8")
    assert committed == gen.render(), (
        "docs/registry.md is stale: run python scripts/gen_registry_doc.py"
    )


def test_every_schema_field_has_a_description():
    assert gen.undocumented_fields() == []


def test_no_description_names_a_removed_field():
    assert gen.stale_descriptions() == []


def test_every_field_has_one_row_in_its_table():
    page = gen.render()
    for model, title in gen.MODELS:
        section = page.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]
        rows = [line for line in section.splitlines() if line.startswith("| `")]
        assert [row.split("`")[1] for row in rows] == list(model.model_fields), title


def test_a_new_field_without_a_description_is_refused(monkeypatch):
    class Grown(BaseModel):
        added: int = Field(0)

    monkeypatch.setattr(gen, "MODELS", ((Grown, "Grown"),))
    assert gen.undocumented_fields() == ["Grown.added"]
    with pytest.raises(SystemExit, match="Grown.added"):
        gen.render()


def test_types_and_defaults_come_from_the_schema():
    page = gen.render()
    # values read off registry/schemas.py by hand
    assert '| `seed` | `int` | `1337` |' in page
    assert '| `role` | `"eval" \\| "train" \\| "unrestricted"` | `"eval"` |' in page
    assert '| `competitor_id` | `str` | `required` |' in page
    assert '| `capabilities` | `list["clip" \\| "stream"]` | `["clip"]` |' in page
    assert '| `trained_on` | `list[str]` | `[]` |' in page


def test_examples_are_the_registry_files():
    page = gen.render()
    for rel in (gen.EXAMPLE_COMPETITOR, gen.EXAMPLE_DATASET):
        text = (ROOT / rel).read_text(encoding="utf-8").rstrip()
        assert f"```json\n{text}\n```" in page
