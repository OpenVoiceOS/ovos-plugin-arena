"""Every published number is explained in plain language somewhere.

A leaderboard column is a code name (``ood_fpr``, ``sigmos.disc``) unless
something translates it, and a new metric that ships without an entry in
``arena.metrics_legend`` reaches visitors unexplained. These tests hold that
line against the committed boards and against the export command.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.metrics_legend import LEAGUE_INTROS, METRICS, metrics_legend
from arena.models import leagues

DATA_DIR = Path(__file__).resolve().parent.parent / "frontend-static" / "public" / "data"


def _published_metric_keys() -> set[str]:
    keys: set[str] = set()
    for path in DATA_DIR.glob("benchmark-*.json"):
        for entry in json.loads(path.read_text()).get("entries", []):
            keys.update(entry.get("metrics") or {})
    return keys


class TestCoverage:
    def test_committed_boards_have_metric_keys(self):
        # Guards the glob above: an empty set would make the next test pass
        # for the wrong reason.
        assert len(_published_metric_keys()) > 20

    def test_every_published_metric_is_explained(self):
        missing = sorted(_published_metric_keys() - set(METRICS))
        assert not missing, f"metrics with no legend entry: {missing}"

    def test_every_league_has_an_intro(self):
        missing = sorted({le["id"] for le in leagues()} - set(LEAGUE_INTROS))
        assert not missing, f"leagues with no intro: {missing}"


class TestShape:
    @pytest.mark.parametrize("key", sorted(METRICS))
    def test_entry_is_complete(self, key):
        entry = METRICS[key]
        assert entry["label"] and entry["meaning"] and entry["unit"]
        assert entry["direction"] in ("higher", "lower", "neutral")
        assert entry["meaning"].endswith(".")

    def test_payload_is_json_serialisable(self):
        payload = metrics_legend()
        assert json.loads(json.dumps(payload)) == payload


class TestExport:
    def test_export_index_writes_the_legend(self, tmp_path):
        from arena.cli import main

        out = tmp_path / "data"
        out.mkdir()
        with pytest.raises(SystemExit) as exc:
            main(["export-index", "--data-dir", str(out),
                  "--output", str(out / "index.json")])
        assert exc.value.code == 0
        payload = json.loads((out / "metrics-legend.json").read_text())
        assert payload == metrics_legend()

    def test_committed_legend_matches_the_source(self):
        published = json.loads((DATA_DIR / "metrics-legend.json").read_text())
        assert published == metrics_legend()
