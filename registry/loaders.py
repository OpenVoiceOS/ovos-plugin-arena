"""Registry file loaders.

Reads competitor and dataset definitions from the JSON files under
``registry/competitors/<modality>/`` and ``registry/datasets/<modality>/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from registry.schemas import INTENT_MODALITIES, CompetitorDef, DatasetDef

# Root of the registry tree — two levels up from this file (repo root)
REGISTRY_ROOT: Path = Path(__file__).parent

_COMPETITORS_DIR = REGISTRY_ROOT / "competitors"
_DATASETS_DIR = REGISTRY_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Competitors
# ---------------------------------------------------------------------------


def load_competitor(modality: str, competitor_id: str) -> CompetitorDef:
    """Load a single competitor definition by modality and id.

    Parameters
    ----------
    modality:       e.g. "stt", "intent"
    competitor_id:  the id portion of the filename (without .json)

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValidationError   if the JSON does not match the schema.
    """
    path = _COMPETITORS_DIR / modality / f"{competitor_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Competitor '{competitor_id}' not found for modality '{modality}' "
            f"(expected {path})"
        )
    return CompetitorDef.model_validate(json.loads(path.read_text()))


def list_competitors(modality: str | None = None) -> list[CompetitorDef]:
    """Return all competitor definitions, optionally filtered by modality."""
    results: list[CompetitorDef] = []
    search_root = _COMPETITORS_DIR if modality is None else _COMPETITORS_DIR / modality
    if not search_root.exists():
        return results
    for path in sorted(search_root.glob("**/*.json")):
        try:
            results.append(CompetitorDef.model_validate(json.loads(path.read_text())))
        except Exception as exc:
            import warnings
            warnings.warn(f"Skipping invalid competitor file {path}: {exc}", stacklevel=2)
    return results


def load_all_competitors(
    registry_root: Path | None = None,
) -> list[CompetitorDef]:
    """Return every competitor definition across all modalities.

    *registry_root* overrides the default registry location (used by the
    CLI when run from outside the repo root).
    """
    root = (registry_root or REGISTRY_ROOT) / "competitors"
    results: list[CompetitorDef] = []
    if not root.exists():
        return results
    for path in sorted(root.glob("**/*.json")):
        try:
            results.append(CompetitorDef.model_validate(json.loads(path.read_text())))
        except Exception as exc:
            import warnings
            warnings.warn(f"Skipping invalid competitor file {path}: {exc}", stacklevel=2)
    return results


def get_competitor_by_alias(
    modality: str,
    plugin_id: str,
) -> CompetitorDef | None:
    """Find a competitor whose plugin field or alias list matches *plugin_id*.

    Used by the ingestion layer to re-key legacy ``plugin_id`` values.
    Returns the first match or None if no match found.
    """
    for comp in list_competitors(modality):
        if comp.plugin == plugin_id:
            return comp
        if comp.alias and plugin_id in comp.alias:
            return comp
    return None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def load_dataset(modality: str, dataset_id: str) -> DatasetDef:
    """Load a single dataset definition by modality and id.

    Parameters
    ----------
    modality:   e.g. "stt", "intent"
    dataset_id: the id portion of the filename (without .json)

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValidationError   if the JSON does not match the schema.
    """
    path = _DATASETS_DIR / modality / f"{dataset_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found for modality '{modality}' "
            f"(expected {path})"
        )
    return DatasetDef.model_validate(json.loads(path.read_text()))


def list_datasets(modality: str | None = None) -> list[DatasetDef]:
    """Return all dataset definitions, optionally filtered by modality."""
    results: list[DatasetDef] = []
    search_root = _DATASETS_DIR if modality is None else _DATASETS_DIR / modality
    if not search_root.exists():
        return results
    for path in sorted(search_root.glob("**/*.json")):
        try:
            results.append(DatasetDef.model_validate(json.loads(path.read_text())))
        except Exception as exc:
            import warnings
            warnings.warn(f"Skipping invalid dataset file {path}: {exc}", stacklevel=2)
    return results


def load_all_datasets(
    registry_root: Path | None = None,
) -> list[DatasetDef]:
    """Return every dataset definition across all modalities.

    *registry_root* overrides the default registry location (used by the
    CLI when run from outside the repo root).
    """
    root = (registry_root or REGISTRY_ROOT) / "datasets"
    results: list[DatasetDef] = []
    if not root.exists():
        return results
    for path in sorted(root.glob("**/*.json")):
        try:
            results.append(DatasetDef.model_validate(json.loads(path.read_text())))
        except Exception as exc:
            import warnings
            warnings.warn(f"Skipping invalid dataset file {path}: {exc}", stacklevel=2)
    return results


def validate_registry(registry_root: Path | None = None) -> list[str]:
    """Strictly validate every registry JSON file.

    Unlike ``list_competitors``/``list_datasets`` (which warn and skip bad
    files so runtime code keeps working off the good ones), this collects
    every validation failure — including unknown/typo'd keys, now that the
    schemas are closed (``extra="forbid"``) — and returns them as
    ``"<path>: <message>"`` strings. An empty list means every file in the
    registry validates cleanly.
    """
    root = registry_root or REGISTRY_ROOT
    errors: list[str] = []

    competitors_dir = root / "competitors"
    if competitors_dir.exists():
        for path in sorted(competitors_dir.glob("**/*.json")):
            try:
                CompetitorDef.model_validate(json.loads(path.read_text()))
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    datasets_dir = root / "datasets"
    if datasets_dir.exists():
        for path in sorted(datasets_dir.glob("**/*.json")):
            try:
                DatasetDef.model_validate(json.loads(path.read_text()))
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    return errors


def paradigm_league_repo(dataset: DatasetDef, paradigm: str) -> str:
    """The real HF repo backing *dataset*'s ``<paradigm>`` sub-league
    predictions: ``<owner>/ovos-intent-<paradigm>-bench-<dataset_id>`` (the
    ``runner.intent_bench.results_repo_for`` convention).

    *dataset* is the eval corpus (e.g. ``banking77``, or ``meteocat`` which
    borrows another corpus's training data); the repo is always keyed by
    *this* eval dataset's own id, never the training corpus's — several eval
    datasets can share one ``intent_<paradigm>/`` training corpus (meteocat
    and intents-for-eval both train from ``intents-for-eval-templates``) and
    each still publishes its own predictions repo. This only confirms the
    training corpus is genuinely registered under ``intent_<paradigm>/`` —
    the directory that makes the league real — before naming the repo.
    """
    train_id = (dataset.train_datasets or {}).get(paradigm)
    if not train_id:
        raise KeyError(
            f"{dataset.dataset_id}: no train_datasets entry for paradigm "
            f"{paradigm!r}"
        )
    load_dataset(f"intent_{paradigm}", train_id)  # raises if misfiled
    owner = (dataset.predictions_hf or "OpenVoiceOS/x").split("/")[0]
    return f"{owner}/ovos-intent-{paradigm}-bench-{dataset.dataset_id}"


def list_prediction_repos(modality: str | None = None) -> list[str]:
    """Sorted unique HF prediction repos across all eval datasets.

    Each eval dataset names its predictions repo via ``predictions_hf`` —
    one dedicated repo per benchmark modality, following the runner
    convention ``<owner>/ovos-<modality>-bench-<dataset_id>``.  Intent eval
    corpora additionally feed the paradigm sub-leagues, whose fighters
    publish to their own ``ovos-intent-<paradigm>-bench-<dataset_id>``
    repos — one per paradigm the corpus provides training data for.

    *modality* scopes the result to only the repos an ``assemble
    --modality <modality>`` run actually reads (§assemble scalability: an
    unscoped ``assemble`` was resolving+downloading every one of the ~120
    registry prediction repos across every modality even when the caller
    only wanted one board type — see arena.cli.cmd_assemble). ``None``
    keeps the full unscoped set (the registry default). A base intent
    league modality (``intent``) matches only a dataset's own
    ``predictions_hf`` repo; a paradigm sub-league modality (e.g.
    ``intent_template``) matches only the paradigm repo whose name encodes
    that paradigm (``ovos-intent-template-bench-*``), from ANY intent-
    family eval dataset — the sub-repo's paradigm, not the eval dataset's
    own base modality, is what the caller cares about. Any other modality
    matches a dataset's own ``predictions_hf`` when
    ``dataset.modality == modality``.
    """
    repos: set = set()
    for dataset in list_datasets():
        if dataset.role != "eval" or not dataset.predictions_hf:
            continue
        if dataset.modality in INTENT_MODALITIES:
            if modality is None or modality == dataset.modality:
                repos.add(dataset.predictions_hf)
            for paradigm in dataset.train_datasets or {}:
                if modality is None or modality == f"intent_{paradigm}":
                    repos.add(paradigm_league_repo(dataset, paradigm))
        elif modality is None or modality == dataset.modality:
            repos.add(dataset.predictions_hf)
    return sorted(repos)
