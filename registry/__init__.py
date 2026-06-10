"""Declarative evaluation registry for OVOS Plugin Arena.

Competitors and datasets are described as JSON files under
``registry/competitors/<modality>/<id>.json`` and
``registry/datasets/<modality>/<id>.json`` respectively.

Usage::

    from registry import load_competitor, load_dataset, list_competitors

    comp = load_competitor("stt", "fasterwhisper-small-pt")
    ds   = load_dataset("stt", "minds14-pt-PT")
    all_stt = list_competitors("stt")
"""

from registry.schemas import CompetitorDef, DatasetDef, DatasetSource
from registry.loaders import (
    load_competitor,
    load_dataset,
    list_competitors,
    list_datasets,
    REGISTRY_ROOT,
)

__all__ = [
    "CompetitorDef",
    "DatasetDef",
    "DatasetSource",
    "load_competitor",
    "load_dataset",
    "list_competitors",
    "list_datasets",
    "REGISTRY_ROOT",
]
