"""Row and manifest types for the STT prediction runner.

``STTRow`` (the legacy ``ovos-stt-bench-*`` column layout, read-compat only
per §4 A2) lives in ``arena.legacy_schema`` — its only reader is
``arena.predictions`` — and is re-exported here since every runner script
and test already imports it from this module.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from arena.legacy_schema import STTRow

__all__ = ["STTRow", "JobManifest"]

log = logging.getLogger(__name__)


@dataclass
class JobManifest:
    """Tracks which (sample_id) have been completed for one job."""

    job_key: str           # "{plugin_name}|{model_name}|{dataset_id}"
    done_ids: set[str] = field(default_factory=set)
    output_file: str = ""  # path to the .jsonl output

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _manifest_path(base_dir: Path, job_key: str) -> Path:
        safe = job_key.replace("/", "__").replace("|", "_")
        return base_dir / f"manifest_{safe}.json"

    @classmethod
    def load(cls, base_dir: Path, job_key: str) -> JobManifest:
        path = cls._manifest_path(base_dir, job_key)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                manifest = cls(
                    job_key=data["job_key"],
                    done_ids=set(data.get("done_ids", [])),
                    output_file=data.get("output_file", ""),
                )
            except (json.JSONDecodeError, OSError, KeyError, TypeError,
                     ValueError) as exc:
                log.error("corrupt manifest %s (%s), moving aside and "
                          "starting the job from scratch", path, exc)
                path.rename(path.with_suffix(path.suffix + ".corrupt"))
                return cls(job_key=job_key)
            return manifest
        return cls(job_key=job_key)

    def save(self, base_dir: Path) -> None:
        path = self._manifest_path(base_dir, self.job_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "job_key": self.job_key,
                    "done_ids": sorted(self.done_ids),
                    "output_file": self.output_file,
                },
                indent=2,
            )
        )
        os.replace(tmp_path, path)

    def mark_done(self, sample_id: str, base_dir: Path) -> None:
        self.done_ids.add(sample_id)
        self.save(base_dir)

    def is_done(self, sample_id: str) -> bool:
        return sample_id in self.done_ids
