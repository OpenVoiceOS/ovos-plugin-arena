"""Every benchmarks/*.py script's default dataset id must resolve in the registry.

Each ``benchmarks/*.py`` script is a thin wrapper that hardcodes a default
dataset id (either via ``run_benchmark(adapter, dataset_id, ...)`` or
``run_benchmark(dataset_id, ...)`` for the intent league). A default that
does not exist in the
registry silently breaks the plain ``python benchmarks/<script>.py`` smoke
run documented in every script's own docstring. This regression-tested
``speech-vs-nonspeech`` (no locale suffix) for VAD.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from registry.loaders import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

# Adapter class name (as imported in the script) -> registry modality.
_ADAPTER_MODALITY = {
    "STTBench": "stt",
    "VADBench": "vad",
    "TTSBench": "tts",
    "WakeWordBench": "wake_word",
    "WakeWordStreamBench": "ww_stream",
}


def _imported_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _run_benchmark_string_args(tree: ast.Module) -> list[str]:
    """String-literal positional args passed to any ``run_benchmark(...)`` call."""
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_benchmark"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append(arg.value)
    return found


def _resolve_modality_and_dataset(path: Path) -> tuple[str, str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = _imported_names(tree)

    adapter_names = imported & set(_ADAPTER_MODALITY)
    if adapter_names:
        modality = _ADAPTER_MODALITY[sorted(adapter_names)[0]]
        str_args = _run_benchmark_string_args(tree)
        assert str_args, f"{path.name}: no string dataset id found in run_benchmark(...) call"
        return modality, str_args[0]

    str_args = _run_benchmark_string_args(tree)
    if str_args:
        # intent league: run_benchmark(dataset_id, description).
        return "intent", str_args[0]

    pytest.fail(f"{path.name}: could not determine default dataset id")


BENCHMARK_SCRIPTS = sorted(BENCHMARKS_DIR.glob("*.py"))


@pytest.mark.parametrize("path", BENCHMARK_SCRIPTS, ids=lambda p: p.name)
def test_default_dataset_resolves(path: Path) -> None:
    modality, dataset_id = _resolve_modality_and_dataset(path)
    load_dataset(modality, dataset_id)  # raises FileNotFoundError if missing
