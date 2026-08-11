"""Queue configuration loader.

Backward-compatible with the original inline plugin/dataset YAML format.
Also accepts the declarative registry format where a job references a
competitor file and/or a dataset file instead of inlining the plugin config.

Supported job shapes (all produce a ``JobSpec``):

1. **Fully inline** (original format — unchanged):

   .. code-block:: yaml

       jobs:
         - plugin:
             plugin_name: ovos-stt-plugin-fasterwhisper
             model_name: small
             lang: pt-PT
             extra_config: {compute_type: int8}
           dataset:
             hf_repo: PolyAI/minds14
             subset: pt-PT
             split: train
           hf_output_dataset: OpenVoiceOS/ovos-stt-bench-pt-PT

2. **Registry-referenced** (new declarative format):

   .. code-block:: yaml

       jobs:
         - competitor: fasterwhisper-small-pt    # competitors/stt/<id>.json
           dataset_ref: minds14-pt-PT            # datasets/stt/<id>.json
           hf_output_dataset: OpenVoiceOS/ovos-stt-bench-pt-PT
           max_samples: 200                       # optional cap

3. **Mixed** — ``competitor`` + inline ``dataset:`` block or vice-versa.

When ``competitor`` is given the plugin block (``plugin_name``, ``model_name``,
``lang``, ``extra_config``) is derived from the competitor JSON.  The
``model_name`` field is taken from ``config["model"]`` if present, else from
the ``competitor_id`` suffix; ``plugin_name`` maps to ``plugin``; ``lang``
defaults to the first entry in ``langs``; ``extra_config`` receives the rest
of ``config``.

When ``dataset_ref`` is given the dataset block is derived from the dataset
JSON (HuggingFace source only; ``reference_fields`` maps into
``ground_truth_key`` / ``audio_key``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetSpec:
    """Identifies a source HF dataset to transcribe."""

    hf_repo: str
    split: str = "train"
    subset: str | None = None
    ground_truth_key: str = "transcription"
    audio_key: str = "audio"
    entry_id_key: str | None = None  # if None, derived from audio path
    trust_remote_code: bool = False
    # Maximum samples to process per run (0 = all)
    max_samples: int = 0
    # Canonical registry dataset id (registry/datasets/<mod>/<id>.json). Set
    # for registry-referenced jobs; rows and boards must be keyed by THIS,
    # not the raw hf path — a dataset_id with slashes explodes the
    # assembler's board filename into nonexistent directories.
    registry_id: str | None = None

    @property
    def dataset_id(self) -> str:
        """Canonical registry id when known, else the legacy hf-path form."""
        if self.registry_id:
            return self.registry_id
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
    extra_config: dict[str, Any] = field(default_factory=dict)
    # Populated when loaded from a competitor file (None for inline jobs)
    competitor_id: str | None = None


@dataclass
class JobSpec:
    """A single (plugin × dataset) job."""

    plugin: PluginSpec
    dataset: DatasetSpec
    hf_output_dataset: str  # e.g. "OpenVoiceOS/ovos-stt-bench-pt-PT"
    # Set when this job was loaded from a competitor registry file
    competitor_id: str | None = None
    # Set when this job was loaded from a dataset registry file
    dataset_registry_id: str | None = None


# ---------------------------------------------------------------------------
# Inline loaders (original format — unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry-derived loaders
# ---------------------------------------------------------------------------


def _plugin_from_competitor(
    competitor_id: str,
    registry_root: Path | None = None,
    lang_override: str | None = None,
) -> PluginSpec:
    """Load a PluginSpec from ``registry/competitors/<modality>/<id>.json``.

    The modality is inferred by scanning all modality sub-directories.
    """
    root = registry_root or _default_registry_root()
    comp_dir = root / "competitors"
    path: Path | None = None
    for mod_dir in sorted(comp_dir.iterdir()) if comp_dir.exists() else []:
        candidate = mod_dir / f"{competitor_id}.json"
        if candidate.exists():
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            f"Competitor '{competitor_id}' not found under {comp_dir}"
        )

    import json
    data = json.loads(path.read_text())
    cfg = dict(data.get("config", {}))
    # Fighters are complete mycroft.conf snippets: the plugin's own settings
    # live nested under config.stt.<module>, not at the top level. Flatten
    # that section into the config handed to the plugin class — otherwise a
    # fighter whose model is only defined there (e.g. the vosk model URLs)
    # falls back to model_name=competitor_id and the plugin rejects it
    # ("Invalid model: vosk-small-it").
    stt_section = cfg.pop("stt", None)
    if isinstance(stt_section, dict):
        module = stt_section.get("module") or data["plugin"]
        plugin_section = stt_section.get(module)
        if isinstance(plugin_section, dict):
            cfg = {**cfg, **plugin_section}
    model_name = cfg.pop("model", competitor_id)
    # Pop the fighter's top-level config.lang instead of letting it ride in
    # extra_config: _load_plugin builds {"lang": spec.lang, **extra_config},
    # so a leaked "lang" key silently overrides a queue-supplied
    # lang_override. The popped value still beats the langs[] fallback.
    cfg_lang = cfg.pop("lang", None)
    lang = lang_override or cfg_lang or (data.get("langs") or ["en-US"])[0]

    return PluginSpec(
        plugin_name=data["plugin"],
        model_name=model_name,
        lang=lang,
        extra_config=cfg,
        competitor_id=competitor_id,
    )


def _dataset_spec_from_registry(
    dataset_id: str,
    registry_root: Path | None = None,
    max_samples: int = 0,
) -> DatasetSpec:
    """Load a DatasetSpec from ``registry/datasets/<modality>/<id>.json``."""
    root = registry_root or _default_registry_root()
    ds_dir = root / "datasets"
    path: Path | None = None
    for mod_dir in sorted(ds_dir.iterdir()) if ds_dir.exists() else []:
        candidate = mod_dir / f"{dataset_id}.json"
        if candidate.exists():
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found under {ds_dir}"
        )

    import json
    data = json.loads(path.read_text())
    src = data.get("source", {})
    if src.get("type") != "huggingface":
        raise ValueError(
            f"Dataset '{dataset_id}' uses source type '{src.get('type')}'; "
            "only 'huggingface' is supported by the runner"
        )

    ref = data.get("reference_fields", {})
    return DatasetSpec(
        hf_repo=src["hf_id"],
        split=src.get("split", "train"),
        subset=src.get("subset"),
        ground_truth_key=ref.get("ground_truth", ref.get("transcription", "transcription")),
        audio_key=ref.get("audio", "audio"),
        max_samples=max_samples,
        registry_id=data.get("dataset_id", dataset_id),
    )


def _default_registry_root() -> Path:
    """Return the registry/ directory relative to this file (repo root)."""
    return Path(__file__).parent.parent / "registry"


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_queue(
    path: str | Path,
    registry_root: Path | None = None,
) -> list[JobSpec]:
    """Parse a queue YAML file and return a list of JobSpec objects.

    Accepts both the original inline format and the new registry-reference
    format.  A job may mix the two (e.g. ``competitor`` + inline ``dataset``).
    """
    raw = yaml.safe_load(Path(path).read_text())
    jobs: list[JobSpec] = []
    rr = registry_root  # may be None — loaders fall back to default

    for entry in raw.get("jobs", []):
        competitor_id: str | None = entry.get("competitor")
        dataset_ref: str | None = entry.get("dataset_ref")
        max_samples: int = int(entry.get("max_samples", 0))

        # Resolve plugin spec
        if competitor_id is not None:
            lang_override = entry.get("lang")
            plugin = _plugin_from_competitor(competitor_id, rr, lang_override)
        else:
            plugin = _load_plugin(entry["plugin"])

        # Resolve dataset spec
        if dataset_ref is not None:
            dataset = _dataset_spec_from_registry(dataset_ref, rr, max_samples)
            # Inline max_samples on competitor jobs
            if max_samples and not dataset.max_samples:
                dataset.max_samples = max_samples
        else:
            dataset = _load_dataset(entry["dataset"])
            # Allow top-level max_samples to override inline dataset value
            if max_samples:
                dataset.max_samples = max_samples

        hf_out = entry.get(
            "hf_output_dataset",
            f"OpenVoiceOS/ovos-stt-bench-{plugin.lang}",
        )

        jobs.append(
            JobSpec(
                plugin=plugin,
                dataset=dataset,
                hf_output_dataset=hf_out,
                competitor_id=competitor_id,
                dataset_registry_id=dataset_ref,
            )
        )

    return jobs
