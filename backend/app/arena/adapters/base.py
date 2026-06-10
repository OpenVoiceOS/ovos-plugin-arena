"""
Base adapter interface for OVOS Plugin Arena.

Each adapter knows how to:
1. Load a specific plugin with a given config
2. Run it over a list of input references
3. Return a list of Sample objects with output artifacts and metrics

Adapters are responsible for loading/unloading the underlying plugin.
They should be used as context managers to ensure proper cleanup.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.arena.models import EvalRun, PluginFamily, Sample


class BaseAdapter(ABC):
    """
    Abstract base class for all arena plugin adapters.

    Subclasses must implement ``_run_one`` and declare ``family``.

    Usage::

        adapter = TtsAdapter(plugin_name="ovos-tts-plugin-phoonnx", config={...})
        with adapter:
            samples = adapter.run_eval(run, prompts)
    """

    family: PluginFamily  # must be set by subclass

    def __init__(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.plugin_name = plugin_name
        self.config: Dict[str, Any] = config or {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseAdapter":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the plugin. Called once before run_eval."""
        self._load_plugin()
        self._loaded = True

    def unload(self) -> None:
        """Release plugin resources."""
        self._unload_plugin()
        self._loaded = False

    def _load_plugin(self) -> None:
        """Override to load the plugin instance."""

    def _unload_plugin(self) -> None:
        """Override to release resources."""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_eval(
        self,
        run: EvalRun,
        inputs: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Sample]:
        """
        Run plugin over *inputs* and return Sample records.

        Parameters
        ----------
        run        : the EvalRun this belongs to
        inputs     : list of input references (text prompts or audio paths)
        output_dir : directory to write artifacts; None = in-memory only

        Returns
        -------
        List of Sample objects (not persisted — caller decides)
        """
        if not self._loaded:
            raise RuntimeError("Adapter not loaded. Use 'with adapter:' or call load() first.")

        samples: List[Sample] = []
        for input_ref in inputs:
            try:
                output_ref, metrics = self._run_one(input_ref, output_dir)
            except Exception as e:
                # Record failure as a sample with error metric
                output_ref = None
                metrics = {"error": 1.0, "error_msg": str(e)[:200]}

            sample = Sample(
                run_id=run.id,
                plugin_id=run.plugin_id,
                family=self.family,
                input_ref=input_ref,
                output_ref=output_ref,
                metrics=metrics,
                produced_at=datetime.utcnow(),
            )
            samples.append(sample)

        return samples

    @abstractmethod
    def _run_one(
        self,
        input_ref: str,
        output_dir: Optional[str],
    ) -> tuple[Optional[str], Dict[str, float]]:
        """
        Process a single input.

        Returns
        -------
        (output_ref, metrics)
        output_ref : path to artifact file, or None for in-memory only
        metrics    : dict of float metrics (e.g. {"rtf": 0.12, "wer": 0.05})
        """
