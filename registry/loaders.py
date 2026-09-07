"""Registry file loaders.

Reads competitor and dataset definitions from the JSON files under
``registry/competitors/<modality>/`` and ``registry/datasets/<modality>/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from registry.schemas import INTENT_MODALITIES, CompetitorDef, DatasetDef, Modality

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
    competitors: dict[Path, CompetitorDef] = {}
    all_competitors: dict[str, CompetitorDef] = {}
    competitor_paths: dict[str, Path] = {}
    if competitors_dir.exists():
        for path in sorted(competitors_dir.glob("**/*.json")):
            try:
                competitor = CompetitorDef.model_validate(json.loads(path.read_text()))
                competitors[path] = competitor
                all_competitors[competitor.competitor_id] = competitor
                competitor_paths[competitor.competitor_id] = path
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    datasets_dir = root / "datasets"
    all_datasets: dict[str, DatasetDef] = {}
    dataset_paths: dict[str, Path] = {}
    if datasets_dir.exists():
        for path in sorted(datasets_dir.glob("**/*.json")):
            try:
                dataset = DatasetDef.model_validate(json.loads(path.read_text()))
                all_datasets[dataset.dataset_id] = dataset
                dataset_paths[dataset.dataset_id] = path
            except Exception as exc:
                errors.append(f"{path}: {exc}")

    # Every corpus needs a human name, and every eval corpus also needs the
    # plain-language summary the leaderboard prints above its board — a
    # dataset that reaches the site as a bare code name with no explanation
    # is a defect, not a cosmetic gap.
    for dataset_id, dataset in all_datasets.items():
        path = dataset_paths[dataset_id]
        if not (dataset.display_name or "").strip():
            errors.append(f"{path}: display_name is required")
        if dataset.role == "eval" and not (dataset.summary or "").strip():
            errors.append(f"{path}: summary is required on a role=eval corpus")
    # An offline fighter's label_set names the corpora whose labels its
    # pretrained artefact can emit; it is what makes the fighter eligible on
    # a board at all (runner.intent_bench.trained_on), so a typo'd id would
    # silently drop it from every sweep.
    for path, competitor in competitors.items():
        for dataset_id in competitor.label_set or []:
            if dataset_id not in all_datasets:
                errors.append(
                    f"{path}: label_set references unknown dataset_id "
                    f"{dataset_id!r}"
                )

    # negatives_dataset_ids must resolve to a registered wake_word dataset —
    # _pooled_dataset_negatives (runner/audio_io.py) loads each id via
    # load_dataset("wake_word", did), so a missing or wrong-modality id only
    # surfaces at benchmark-sweep time otherwise. A lang mismatch is not an
    # error: cross-lingual hard negatives can be intentional, so it's only a
    # warning (via the ``warnings`` module, matching the skip-and-warn
    # convention used by ``list_datasets``/``load_all_datasets`` above).
    for dataset_id, dataset in all_datasets.items():
        if not dataset.negatives_dataset_ids:
            continue
        path = dataset_paths[dataset_id]
        for neg_id in dataset.negatives_dataset_ids:
            neg = all_datasets.get(neg_id)
            if neg is None:
                errors.append(
                    f"{path}: negatives_dataset_ids references unknown "
                    f"dataset_id {neg_id!r}"
                )
            elif neg.modality != Modality.WAKE_WORD:
                errors.append(
                    f"{path}: negatives_dataset_ids entry {neg_id!r} has "
                    f"modality={neg.modality.value!r}, expected 'wake_word'"
                )
            elif neg.lang != dataset.lang:
                import warnings
                warnings.warn(
                    f"{path}: negatives_dataset_ids entry {neg_id!r} has "
                    f"lang={neg.lang!r}, differs from this dataset's "
                    f"lang={dataset.lang!r}",
                    stacklevel=2,
                )

    # trained_on must resolve to a registered dataset for the fighter's OWN
    # modality — the pair the entry excludes from scoring
    # (runner.media_bench, arena.predictions.group_rows, runner.autorun).
    # A typo'd or cross-modality id here would silently never exclude
    # anything, so it fails the registry gate instead of being discovered
    # at benchmark time.
    for competitor_id, competitor in all_competitors.items():
        if not competitor.trained_on:
            continue
        path = competitor_paths[competitor_id]
        for dataset_id in competitor.trained_on:
            trained_dataset = all_datasets.get(dataset_id)
            if trained_dataset is None:
                errors.append(
                    f"{path}: trained_on references unknown dataset_id "
                    f"{dataset_id!r}"
                )
            elif trained_dataset.modality != competitor.modality:
                errors.append(
                    f"{path}: trained_on entry {dataset_id!r} has "
                    f"modality={trained_dataset.modality.value!r}, expected "
                    f"{competitor.modality.value!r}"
                )

    return errors


HF_OWNER = "OpenVoiceOS"


def results_repo_for(modality: str, dataset_id: str, owner: str = HF_OWNER) -> str:
    """One dedicated HF repo per benchmark modality.

    Lives here rather than in ``runner`` so that ``arena`` (which only
    reads published results, never runs plugins) can name a dataset's
    results repo without importing the runner package — pulling in
    ``runner`` for this one naming convention used to drag the whole
    plugin-adapter stack, including audio-only dependencies, into every
    ``arena`` code path.
    """
    return f"{owner}/ovos-{modality.replace('_', '-')}-bench-{dataset_id}"


_TRAILING_LOCALE_RE = re.compile(r"-([a-zA-Z]{2,3}(?:-[a-zA-Z]{2,3})?)$")


def resolved_dataset_lang(dataset: DatasetDef) -> str | None:
    """The single concrete BCP-47 tag this dataset runs jobs under, or
    ``None`` when the dataset is genuinely multilingual/unknown.

    A queued job that omits ``lang`` resolves against the *fighter's*
    default lang (``runner.queue_config._plugin_from_competitor``), not the
    dataset's — for a multilingual fighter that silently runs the wrong
    lang and publishes into the wrong ``predictions/<lang>/`` path (e.g.
    onnx-asr-canary queued against ``speech-massive-de-DE`` running as
    ``en`` instead of ``de-DE``). Every generated entry for a
    single-language dataset must pin ``lang`` explicitly to this value.
    """
    lang = getattr(dataset, "lang", None)
    if lang and lang != "multi":
        return lang
    # Registry lang is missing/multi/unknown — fall back to parsing a
    # trailing "-xx-XX" (or "-xx") locale suffix off the dataset id itself,
    # e.g. "speech-massive-de-DE" -> "de-DE".
    match = _TRAILING_LOCALE_RE.search(dataset.dataset_id)
    if match:
        return match.group(1)
    return None


def paradigm_league_repo(dataset: DatasetDef, paradigm: str) -> str:
    """The real HF repo backing *dataset*'s ``<paradigm>``-supervised
    predictions: ``<owner>/ovos-intent-<paradigm>-bench-<dataset_id>`` (the
    ``results_repo_for`` convention).

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
    corpora additionally carry one repo per training datashape,
    ``ovos-intent-<paradigm>-bench-<dataset_id>``, holding the rows of the
    fighters that consume it.

    *modality* scopes the result to only the repos an ``assemble
    --modality <modality>`` run actually reads (§assemble scalability: an
    unscoped ``assemble`` was resolving+downloading every one of the ~120
    registry prediction repos across every modality even when the caller
    only wanted one board type — see arena.cli.cmd_assemble). ``None``
    keeps the full unscoped set (the registry default). Any intent league
    matches every intent prediction repo: repos are keyed by the training
    datashape a sweep consumed, not by league, and one repo carries rows
    from fighters of several leagues (a row's league is resolved from the
    registry at assemble time). Any other modality matches a dataset's own
    ``predictions_hf`` when ``dataset.modality == modality``.
    """
    repos: set = set()
    for dataset in list_datasets():
        if dataset.role != "eval" or not dataset.predictions_hf:
            continue
        if dataset.modality in INTENT_MODALITIES:
            if modality is None or modality in INTENT_MODALITIES:
                repos.add(dataset.predictions_hf)
                for paradigm in dataset.train_datasets or {}:
                    repos.add(paradigm_league_repo(dataset, paradigm))
        elif modality is None or modality == dataset.modality:
            repos.add(dataset.predictions_hf)
    return sorted(repos)
