"""Row schema and manifest types for the STT prediction runner.

§4 A2 schema convergence: ``STTRow`` is the legacy ``ovos-stt-bench-*``
column layout. It is kept **read-compat only** — to interpret already-
published legacy-format HF data (``arena.predictions`` detects the shape and
converts via ``to_prediction_row_dict``) — new runs never construct one; see
``runner/plugin_runner.py``, which writes the canonical §3.2
``PredictionRow`` shape directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class STTRow:
    """One prediction row in the legacy ``ovos-stt-bench-*`` column layout.

    Read-compat only (§4 A2) — see the module docstring.
    """

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

    def to_prediction_row_dict(self, competitor_id: str) -> dict[str, Any]:
        """Convert to the canonical §3.2 ``PredictionRow`` field shape.

        *competitor_id* MUST already be resolved (typically via
        ``registry.loaders.get_competitor_by_alias(modality, self.plugin_name)``
        — the legacy layout has no ``competitor_id`` column, only
        ``plugin_name``, so the caller owns re-keying).
        """
        return {
            "competitor_id": competitor_id,
            "sample_id": self.dataset_entry_id,
            "dataset_id": self.dataset_id,
            "lang": self.lang,
            "plugin_id": self.plugin_name,
            "modality": "stt",
            "prediction": self.prediction_transcript,
            "reference_text": self.transcript,
            "confidence": self.prediction_confidence,
            "extras": {"model_id": self.model_id, "legacy_schema": "STTRow"},
        }

    @classmethod
    def is_legacy_shape(cls, raw: dict) -> bool:
        """True when *raw* looks like a legacy ``STTRow``-shaped JSON row
        (has the old column names, lacks the canonical ``sample_id``)."""
        return "dataset_entry_id" in raw and "sample_id" not in raw

    @classmethod
    def from_dict(cls, raw: dict) -> STTRow:
        return cls(
            dataset_entry_id=raw["dataset_entry_id"],
            plugin_name=raw["plugin_name"],
            model_id=raw.get("model_id", ""),
            prediction_transcript=raw.get("prediction_transcript", ""),
            transcript=raw.get("transcript", ""),
            prediction_confidence=raw.get("prediction_confidence", 0.0),
            prediction_type=raw.get("prediction_type", "STT"),
            dataset_id=raw.get("dataset_id", ""),
            lang=raw.get("lang", ""),
        )

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
