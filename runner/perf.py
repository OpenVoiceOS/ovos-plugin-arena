"""Per-row performance capture: elapsed time, RSS and a hardware fingerprint.

M1 of the performance-metrics campaign — every prediction row SHOULD carry
enough performance data to compute RTF (real-time factor, ``elapsed_ms /
audio_secs``) downstream, without breaking any row written before this
module existed. All fields here are additive and optional: a loader MUST
NOT require them (see ``arena.models.PredictionRow`` — every new field
defaults to ``None``) since the vast majority of already-published rows
predate this capture and will never carry it.

Two helpers:

- :func:`measure_call` times one inference call and samples process RSS
  immediately before/after it.
- :func:`hw_fingerprint` captures a compact, cached-per-process hardware
  descriptor to stamp onto every row of a run, so a shard stays
  self-contained when merged with shards produced on a different machine.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Timing + RSS
# ---------------------------------------------------------------------------


def rss_mb() -> float | None:
    """Current process RSS in MB via ``psutil``, or ``None`` if unavailable.

    ``psutil`` is an optional dependency of the runner's audio extras — a
    process without it simply gets no RSS reading rather than a crash.
    Public: callers that need to bracket a narrower span than a whole
    ``measure_call`` (e.g. ``runner.tts_bench.TTSBench.predict``, which must
    exclude UTMOS/intelligibility judging from its own elapsed/RSS numbers)
    sample it directly instead.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


# Backward-compat private alias (kept in case anything already imported the
# old private name during this same development cycle).
_rss_mb = rss_mb


def measure_call(fn: Callable[[], T]) -> tuple[T, float, float | None]:
    """Time one inference call and bracket it with an RSS sample.

    Returns ``(result, elapsed_ms, peak_rss_mb)``.

    ``elapsed_ms`` is the wall-clock time of the single call, measured with
    ``time.perf_counter`` (monotonic, unaffected by system clock changes).

    ``peak_rss_mb`` is the higher of the RSS sampled immediately before and
    immediately after the call. This is **not** a true peak — nothing
    samples memory *during* the call, so a short-lived spike between the two
    samples is invisible — and it is **process-wide**: any other thread's
    allocations (or GC pauses reclaiming memory) in the same window are
    folded in, not attributed to this call alone. Treat it as a rough
    per-row signal for spotting gross regressions, not a precise memory
    profile. ``None`` when ``psutil`` is not installed.
    """
    before = _rss_mb()
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    after = _rss_mb()
    peak_rss_mb: float | None = None
    if before is not None and after is not None:
        peak_rss_mb = max(before, after)
    return result, elapsed_ms, peak_rss_mb


# ---------------------------------------------------------------------------
# Hardware fingerprint
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    """Best-effort CPU model string from ``/proc/cpuinfo`` (Linux only)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _accelerator() -> str | None:
    """Name of an available GPU accelerator, or ``None`` on a CPU-only host.

    Checked in order: torch CUDA, then onnxruntime's non-CPU execution
    providers. Either import failing (package not installed) is normal in
    a CPU-only runner environment, not an error.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import onnxruntime as ort

        providers = [p for p in ort.get_available_providers()
                     if p != "CPUExecutionProvider"]
        if providers:
            return providers[0]
    except Exception:
        pass
    return None


def _host_class(accelerator: str | None) -> str:
    if accelerator:
        return "gpu"
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "arm64"
    return "cpu-x86"


_HW_FINGERPRINT: dict[str, Any] | None = None


def hw_fingerprint() -> dict[str, Any]:
    """Per-run hardware fingerprint, captured once and cached per process.

    Compact dict merged onto every row of a run so shards stay
    self-contained when predictions from several machines are merged into
    one competitor's ``.jsonl`` file: ``{host_class, cpu_model, threads,
    accelerator, hostname}``. ``host_class`` is one of ``"cpu-x86"``,
    ``"arm64"`` or ``"gpu"`` — decided from ``platform.machine()`` plus
    torch/onnxruntime accelerator availability, never hand-configured.
    """
    global _HW_FINGERPRINT
    if _HW_FINGERPRINT is None:
        accelerator = _accelerator()
        _HW_FINGERPRINT = {
            "host_class": _host_class(accelerator),
            "cpu_model": _cpu_model(),
            "threads": os.cpu_count() or 0,
            "accelerator": accelerator,
            "hostname": platform.node(),
        }
    return _HW_FINGERPRINT
