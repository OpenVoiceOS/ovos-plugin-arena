"""``arena`` is the read-only assembler/API layer — it never runs plugins,
so it must never need ``runner`` (which does, and pulls in modality-only
extras like ``ovos_spec_tools`` that the assemble/commit CI legs never
install). A regression here silently degraded every ``sample_policy``
dataset's benchmark board to ``sample_set="unmanaged"`` because a *lazy*
``from runner.intent_bench import results_repo_for`` inside
``arena.cli._load_sample_set`` dragged in the whole intent-pipeline import
chain, which failed on a missing extra and was swallowed by a bare
``except Exception``. A plain "import every module" check would not have
caught this — the offending import was inside a function body, never run
at import time — so this test blocks ``runner`` at the meta-path level and
then actually EXERCISES the code paths that used to reach it.
"""
from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import arena

REPO_ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = '''
import json
import sys
from pathlib import Path
from types import SimpleNamespace


class _BlockRunner:
    def find_spec(self, name, path=None, target=None):
        if name == "runner" or name.startswith("runner."):
            raise ImportError(f"runner is blocked in this test: {name}")
        return None


sys.meta_path.insert(0, _BlockRunner())

import importlib
for name in MODULE_NAMES:
    importlib.import_module(name)

from arena import cli, predictions
from registry.schemas import SamplePolicy

# Exercise arena.cli._load_sample_set's manifest-fetch path (this is where
# the boundary violation lived: a lazy import of a runner helper).
cli._SAMPLE_SET_CACHE.clear()


def fake_load_dataset(modality, dataset_id):
    return SimpleNamespace(
        sample_policy=SamplePolicy(max_samples=2, seed=1),
        predictions_hf="OpenVoiceOS/ovos-stt-bench-fake",
    )


import registry.loaders
registry.loaders.load_dataset = fake_load_dataset

manifest_path = Path("boundary_test_manifest.json")
manifest_path.write_text(json.dumps({"sample_ids": ["a", "b"]}))

import huggingface_hub
huggingface_hub.hf_hub_download = lambda *a, **k: str(manifest_path)

result = cli._load_sample_set("stt", "fake", "en-US")
assert result == {"a", "b"}, result

# Exercise arena.predictions.parse_row's legacy-shape conversion (the
# other lazy runner import this boundary covers).
row = predictions.parse_row(
    {
        "dataset_entry_id": "e0",
        "plugin_name": "p",
        "model_id": "m",
        "prediction_transcript": "hi",
        "transcript": "hi",
        "prediction_confidence": 0.9,
        "dataset_id": "d",
        "lang": "en-US",
    },
    "c",
)
assert row.sample_id == "e0", row

print("OK")
'''


def _arena_module_names() -> list[str]:
    return [
        modinfo.name
        for modinfo in pkgutil.walk_packages(arena.__path__, prefix="arena.")
    ]


def test_arena_never_imports_runner_even_for_its_lazy_helpers(tmp_path):
    """Blocks ``runner`` from being importable at all — not just checking
    it isn't in ``sys.modules`` afterwards, which a deferred import inside
    a function would slip past — imports every module under ``arena/``,
    and then actually calls the two code paths (``cli._load_sample_set``,
    ``predictions.parse_row``) that used to reach ``runner`` lazily."""
    modules = _arena_module_names()
    assert modules, "expected at least one module under arena/"

    script = _SCRIPT.replace("MODULE_NAMES", repr(modules))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, (
        f"arena/ reaches runner (boundary violation):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
