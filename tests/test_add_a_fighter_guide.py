"""Anti-rot test: the JSON examples in docs/add-a-fighter.md must stay valid.

If the CompetitorDef schema changes and the guide is not updated, this fails —
so the onboarding docs can never silently drift from the real schema.
"""

import json
import re
from pathlib import Path

from registry.schemas import CompetitorDef

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "add-a-fighter.md"
_JSON_BLOCK = re.compile(r"```json\n(.*?)```", re.DOTALL)


def _examples():
    return _JSON_BLOCK.findall(GUIDE.read_text(encoding="utf-8"))


def test_guide_has_examples():
    assert len(_examples()) >= 2


def test_every_json_example_validates():
    for block in _examples():
        data = json.loads(block)  # must be valid JSON
        CompetitorDef.model_validate(data)  # must satisfy the live schema
