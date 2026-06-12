"""Row schema and manifest types for the STT prediction runner."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set


@dataclass
class STTRow:
    """One prediction row in the ``ovos-stt-bench-*`` column layout."""

    dataset_entry_id: str
    plugin_name: str
    model_id: str
    prediction_transcript: str
    transcript: str
    prediction_confidence: float
    prediction_type: str = "STT"
    dataset_id: str = ""
    lang: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_entry_id": self.dataset_entry_id,
            "plugin_name": self.plugin_name,
            "model_id": self.model_id,
            "prediction_transcript": self.prediction_transcript,
            "transcript": self.transcript,
            "prediction_confidence": self.prediction_confidence,
            "prediction_type": self.prediction_type,
            "dataset_id": self.dataset_id,
            "lang": self.lang,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def fields(cls):
        return [
            "dataset_entry_id",
            "plugin_name",
            "model_id",
            "prediction_transcript",
            "transcript",
            "prediction_confidence",
            "prediction_type",
            "dataset_id",
            "lang",
        ]


@dataclass
class JobManifest:
    """Tracks which (sample_id) have been completed for one job."""

    job_key: str           # "{plugin_name}|{model_name}|{dataset_id}"
    done_ids: Set[str] = field(default_factory=set)
    output_file: str = ""  # path to the .jsonl output

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _manifest_path(base_dir: Path, job_key: str) -> Path:
        safe = job_key.replace("/", "__").replace("|", "_")
        return base_dir / f"manifest_{safe}.json"

    @classmethod
    def load(cls, base_dir: Path, job_key: str) -> "JobManifest":
        path = cls._manifest_path(base_dir, job_key)
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                job_key=data["job_key"],
                done_ids=set(data.get("done_ids", [])),
                output_file=data.get("output_file", ""),
            )
        return cls(job_key=job_key)

    def save(self, base_dir: Path) -> None:
        path = self._manifest_path(base_dir, self.job_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "job_key": self.job_key,
                    "done_ids": sorted(self.done_ids),
                    "output_file": self.output_file,
                },
                indent=2,
            )
        )

    def mark_done(self, sample_id: str, base_dir: Path) -> None:
        self.done_ids.add(sample_id)
        self.save(base_dir)

    def is_done(self, sample_id: str) -> bool:
        return sample_id in self.done_ids
