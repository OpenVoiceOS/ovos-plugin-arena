"""Build a wheel and verify the package installs and imports cleanly.

Guards against packaging regressions (e.g. `runner` dropped from
`[tool.setuptools] packages`, a module that only works via the repo-root
sys.path hack, or missing `ovos-arena` console-script metadata) that unit
tests running from the repo checkout would never catch.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.network  # builds + installs into a scratch venv


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_wheel_builds_and_installs_cleanly(tmp_path):
    dist_dir = tmp_path / "dist"
    build = _run(["uv", "build", "--wheel", "--out-dir", str(dist_dir), str(REPO_ROOT)])
    assert build.returncode == 0, build.stderr

    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    env_dir = tmp_path / "scratch-venv"
    venv_create = _run(["uv", "venv", str(env_dir)])
    assert venv_create.returncode == 0, venv_create.stderr

    venv_python = env_dir / "bin" / "python"
    install = _run(["uv", "pip", "install", "--python", str(venv_python), str(wheels[0])])
    assert install.returncode == 0, install.stderr

    imports = _run([str(venv_python), "-c", "import arena, registry, runner"])
    assert imports.returncode == 0, imports.stderr

    cli = _run([str(env_dir / "bin" / "ovos-arena"), "--help"])
    assert cli.returncode == 0, cli.stderr
