"""Queue configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetSpec:
    """Identifies a source HF dataset to transcribe."""

    hf_repo: str
    split: str = "train"
    subset: Optional[str] = None
    ground_truth_key: str = "transcription"
    audio_key: str = "audio"
    entry_id_key: Optional[str] = None  # if None, derived from audio path
    trust_remote_code: bool = False
    # Maximum samples to process per run (0 = all)
    max_samples: int = 0

    @property
    def dataset_id(self) -> str:
        """Mirrors ovos_plugin_bench.stt.STTDataset.dataset_id convention."""
        did = self.hf_repo
        if self.subset:
            did += f"/{self.subset}"
        if self.split:
            did += f"/{self.split}"
        return did


@dataclass
class PluginSpec:
    """One plugin/model combination to run."""

    plugin_name: str
    model_name: str
    lang: str
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobSpec:
    """A single (plugin × dataset) job."""

    plugin: PluginSpec
    dataset: DatasetSpec
    hf_output_dataset: str  # e.g. "OpenVoiceOS/ovos-stt-bench-pt-PT"


def _load_plugin(d: dict) -> PluginSpec:
    return PluginSpec(
        plugin_name=d["plugin_name"],
        model_name=d["model_name"],
        lang=d["lang"],
        extra_config=d.get("extra_config", {}),
    )


def _load_dataset(d: dict) -> DatasetSpec:
    return DatasetSpec(
        hf_repo=d["hf_repo"],
        split=d.get("split", "train"),
        subset=d.get("subset"),
        ground_truth_key=d.get("ground_truth_key", "transcription"),
        audio_key=d.get("audio_key", "audio"),
        entry_id_key=d.get("entry_id_key"),
        trust_remote_code=d.get("trust_remote_code", False),
        max_samples=d.get("max_samples", 0),
    )


def load_queue(path: str | Path) -> List[JobSpec]:
    """Parse a queue YAML file and return a list of JobSpec objects."""
    raw = yaml.safe_load(Path(path).read_text())
    jobs: List[JobSpec] = []
    for entry in raw.get("jobs", []):
        plugin = _load_plugin(entry["plugin"])
        dataset = _load_dataset(entry["dataset"])
        hf_out = entry.get(
            "hf_output_dataset",
            f"OpenVoiceOS/ovos-stt-bench-{plugin.lang}",
        )
        jobs.append(JobSpec(plugin=plugin, dataset=dataset, hf_output_dataset=hf_out))
    return jobs
