"""
Shared engine for the intent benchmark scripts.

Every benchmark stays one dedicated script under ``benchmarks/`` (P4); this
module is the machinery they share.  A benchmark run:

1. loads the eval dataset definition + its paradigm-specific training
   corpora from the registry, pinning the dataset revision;
2. selects the eligible fighters — every stage's paradigm must have a
   training corpus in this benchmark (a keyword engine cannot train where
   only templates exist);
3. trains each fighter per language and predicts the eval split, writing
   resumable §3.2 rows to
   ``<output_dir>/<dataset_id>/<modality>/<lang>/<competitor_id>.jsonl``;
4. on ``--upload``, publishes **one HF dataset repo per benchmark
   modality** — ``OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`` — with
   files at ``predictions/<lang>/<competitor_id>.jsonl`` and a generated
   dataset card declaring one split per language.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import multiprocessing as mp
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from arena.metrics import domain_of, is_pinned_revision
from arena.version import __version__ as ARENA_VERSION
from registry.loaders import HF_OWNER, load_all_competitors, load_dataset, results_repo_for
from registry.schemas import (
    ENGINE_TRAITS,
    TrainingRegime,
    engine_paradigm,
    expected_league,
)
from runner.audio_io import resolve_sample_cap, stream_audio_dataset, stream_manifest_audio
from runner.dataset_cards import FUNDING_BLOCK, check_funding_block
from runner.intent_pipeline import IntentPipeline, plugin_version
from runner.perf import hw_fingerprint, measure_call
from runner.queue_tools import is_trained_on

# HF_OWNER / results_repo_for live in registry.loaders (they're a naming
# convention over the registry, not runner-specific) and are re-exported
# here since every caller in this module — and every other runner script —
# already imports them from here.

log = logging.getLogger("intent-bench")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_name(lang: str) -> str:
    """HF split names allow only word characters."""
    return lang.replace("-", "_")


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def resolve_revision(hf_id: str, revision: str, timeout: float | None = None) -> str:
    """Pin a branch name to the commit sha it points at right now.

    *timeout* (seconds), when given, is passed straight through to
    ``HfApi.dataset_info`` — the underlying request library's own default
    can otherwise hang far longer than ``HF_HUB_ETAG_TIMEOUT``/
    ``HF_HUB_DOWNLOAD_TIMEOUT`` cover (those only bound file downloads, not
    every metadata call).
    """
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(hf_id, revision=revision, timeout=timeout)
    return info.sha or revision


def fetch_rows(dataset_def, lang: str, revision: str) -> list:
    """Fetch one language's rows for this dataset.

    Two source shapes exist. ``source.file_pattern`` datasets are plain
    JSONL files under the HF repo (one per language) whose rows already
    carry the canonical row shape (``utterance``/``expected_intent`` for
    eval, ``intent_id``/``template`` for train) — used by datasets the
    arena curated itself (e.g. massive-templates). Datasets absorbed
    straight from a public HF *classification* dataset (SNIPS, BANKING77,
    CLINC150) have no such file and are read through :mod:`datasets`
    instead, with ``reference_fields`` mapping the source columns onto
    the same canonical row shape.
    """
    pattern = dataset_def.source.file_pattern
    if pattern:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            dataset_def.source.hf_id,
            pattern.format(lang=lang),
            repo_type="dataset",
            revision=revision,
        )
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()
                if line.strip()]
    else:
        rows = fetch_hf_classification_rows(dataset_def, lang, revision)
    if dataset_def.role == "eval":
        reject_unexpanded_utterances(dataset_def, lang, rows)
    return rows


#: Template markup an OVOS intent file may carry: alternation ``(a|b)``,
#: optional ``[word]``, and ``{slot}`` placeholders. A *training* row is
#: expected to contain these — expanding them is what
#: ``runner.intent_pipeline.expand_template`` is for. An *eval* row must not:
#: it stands for something a person said, and an engine scored against
#: ``(create|add) list [items]`` is being asked to match a string no user
#: would ever utter.
_TEMPLATE_MARKUP = re.compile(r"\([^()]*\|[^()]*\)|\[[^\[\]]+\]|\{[^{}]+\}")


class UnexpandedUtterances(ValueError):
    """An eval corpus ships template markup where utterances belong."""


def reject_unexpanded_utterances(dataset_def, lang: str, rows: list) -> None:
    """Refuse an eval corpus whose utterances were never expanded.

    Feeding raw templates to an engine measures nothing: every fighter is
    scored on a string no speaker produces, and a template-parsing engine is
    handed its own training syntax as input. Fail on the corpus rather than
    publish rows that look like results.
    """
    field = dataset_def.reference_fields.get("utterance", "utterance")
    for index, row in enumerate(rows):
        text = row.get(field)
        if isinstance(text, str) and _TEMPLATE_MARKUP.search(text):
            raise UnexpandedUtterances(
                f"{dataset_def.dataset_id} ({dataset_def.source.hf_id}, "
                f"{lang}) row {index} carries unexpanded template markup "
                f"where an utterance belongs: {text!r}. Republish the corpus "
                "with the templates expanded."
            )


def stt_plugin_version(plugin_id: str) -> str:
    """``<plugin_id>==<installed dist version>``, mirrors intent_pipeline's

    plugin_version() but for OVOS STT plugin distributions (the pip name
    equals the OPM module id for every currently-registered fighter, e.g.
    ``ovos-stt-plugin-onnx-asr``).
    """
    try:
        return f"{plugin_id}=={importlib.metadata.version(plugin_id)}"
    except importlib.metadata.PackageNotFoundError:
        return plugin_id


def transcript_cache_path(cache_dir: Path, dataset_id: str, lang: str) -> Path:
    return Path(cache_dir) / dataset_id / f"{lang}.jsonl"


def load_stt_engine(dataset_def, lang: str):
    """Instantiate the dataset's pinned STT plugin (§ audio-input intent).

    Raises ``RuntimeError`` (not a bare ``None``) when the plugin isn't
    installed — ``load_stt_plugin`` logs and returns ``None`` on a missing
    entry point, which used to reach the caller as ``clazz({...})`` raising
    an opaque ``TypeError: 'NoneType' object is not callable``.
    """
    from ovos_plugin_manager.stt import load_stt_plugin

    module = dataset_def.stt_plugin
    clazz = load_stt_plugin(module)
    if clazz is None:
        raise RuntimeError(f"STT plugin {module} is not installed")
    return clazz({"lang": lang, "module": module, **dict(dataset_def.stt_config)})


def _iter_audio_samples(dataset_def, lang: str, revision: str, max_samples: int):
    fields = dataset_def.reference_fields or {}
    audio_key = fields.get("audio", "audio")
    intent_key = fields.get("intent", "intent_str")
    src = dataset_def.source
    if "{lang}" in (src.subset or "") or "{lang}" in (src.file_pattern or ""):
        update = {}
        if src.subset:
            update["subset"] = src.subset.format(lang=lang)
        if src.file_pattern:
            update["file_pattern"] = src.file_pattern.format(lang=lang)
        src = src.model_copy(update=update)
    is_manifest = (src.file_pattern or "").endswith((".csv", ".tsv", ".jsonl"))
    streamer = stream_manifest_audio if is_manifest else stream_audio_dataset
    effective_max_samples, seed = resolve_sample_cap(dataset_def, max_samples)
    stream_kwargs = dict(
        audio_key=audio_key, extra_keys={"expected_intent": intent_key},
        revision=revision, max_samples=effective_max_samples,
    )
    if not is_manifest:
        stream_kwargs["seed"] = seed
    yield from streamer(src, **stream_kwargs)


def transcribe_dataset(
    dataset_def, lang: str, revision: str, cache_dir: Path, max_samples: int = 0,
) -> tuple[list[dict], str]:
    """Transcribe every clip ONCE per (dataset, lang) with the dataset's
    pinned STT and cache the result — every intent fighter trains/predicts
    off this SAME cached transcript, isolating intent ranking from STT
    variance (see registry.schemas.DatasetDef.input and
    docs/SPECIFICATION.md §3.2).

    Returns ``(rows, stt_revision)`` where ``rows`` follow the same
    canonical shape as a text-input eval set (``utterance``/
    ``expected_intent``/``split``), so ``run_competitor_lang`` treats an
    audio dataset identically to a text one downstream of this call.
    """
    out_path = transcript_cache_path(cache_dir, dataset_def.dataset_id, lang)
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["sample_id"]] = row

    stt_revision = stt_plugin_version(dataset_def.stt_plugin)
    engine = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    count = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for sample_id, sample in _iter_audio_samples(
                dataset_def, lang, revision, max_samples):
            if sample_id in done:
                rows.append(done[sample_id])
                count += 1
                continue
            if engine is None:
                engine = load_stt_engine(dataset_def, lang)
            from ovos_plugin_manager.utils.audio import AudioData

            audio = AudioData.from_array(
                sample["array"], sample_rate=sample["sr"], sample_width=2
            )
            result = engine.transcribe(audio, lang=lang)
            text = _first_transcript(result)
            row = {
                "sample_id": sample_id,
                "utterance": text,
                "expected_intent": sample.get("expected_intent"),
                "split": "test",
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            count += 1
            if max_samples and count >= max_samples:
                break
    return rows, stt_revision


def _first_transcript(result) -> str:
    if isinstance(result, list) and result:
        head = result[0]
        if isinstance(head, (list, tuple)):
            return str(head[0])
        return str(head)
    if isinstance(result, tuple):
        return str(result[0])
    return str(result or "")


def normalize_hierarchical_label(domain: str, intent: str) -> str:
    """Collapse a source's own domain + intent columns into the arena's
    'domain:intent' label convention (see intents-for-eval's
    expected_intent shape).

    Source intent values are frequently prefixed with a generic parser
    tag rather than the real domain (MTOP: 'IN:SEND_MESSAGE', domain
    column 'messaging') and/or shout-cased. Only the text after the last
    ':' is kept from the intent side; both sides are lowercased.
    """
    local = intent.rsplit(":", 1)[-1]
    return f"{domain.strip().lower()}:{local.strip().lower()}"


def fetch_hf_classification_rows(dataset_def, lang: str, revision: str) -> list:
    """Read a plain HF text-classification dataset into canonical rows.

    ``reference_fields`` must provide ``utterance`` (the text column) and
    ``intent`` (the label column — an int ``ClassLabel`` or a plain string
    column, either works). Integer labels are decoded through the source
    dataset's own feature metadata, never guessed. When ``source.id_field``
    is set, rows are deduplicated on that column, keeping the first
    occurrence — some HF mirrors ship the same row more than once;
    deduping is a no-op on clean sources. ``lang`` is accepted for
    interface parity with ``fetch_rows`` (a future per-language-column
    source would filter on it here) — every currently-registered
    classification source is single-language or split per language via
    ``source.split``/``source.subset`` instead.
    """
    from datasets import load_dataset

    src = dataset_def.source
    fields = dataset_def.reference_fields
    text_col = fields.get("utterance")
    intent_col = fields.get("intent")
    domain_col = fields.get("domain")
    domain_label = dataset_def.domain_label
    if not text_col or (not intent_col and domain_label is None):
        raise ValueError(
            f"{dataset_def.dataset_id}: reference_fields must map "
            "'utterance' and 'intent' for a plain HF classification source "
            "(or set domain_label for a corpus with no label column)"
        )
    ds = load_dataset(src.hf_id, name=src.subset, split=src.split, revision=revision)

    if src.lang_field:
        ds = ds.filter(lambda r: r[src.lang_field] == src.lang_value)

    intent_feature = ds.features.get(intent_col)
    def _decode(value):
        if intent_feature is not None and hasattr(intent_feature, "int2str"):
            return intent_feature.int2str(value)
        return value

    id_col = src.id_field
    oos = dataset_def.oos_label
    is_train = dataset_def.role == "train"
    rows = []
    seen_ids: set = set()
    for r in ds:
        if id_col is not None:
            row_id = r.get(id_col)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
        label = domain_label if intent_col is None else _decode(r[intent_col])
        if domain_col is not None and label is not None:
            label = normalize_hierarchical_label(r[domain_col], label)
        text = r[text_col]
        if oos and label == oos:
            if is_train:
                continue  # engines don't train on the negative class
            rows.append({"utterance": text, "expected_intent": None, "split": "far_ood"})
            continue
        if is_train:
            rows.append({"intent_id": label, "template": text})
        else:
            rows.append({"utterance": text, "expected_intent": label, "split": "test"})
    return rows


# ---------------------------------------------------------------------------
# Fighter selection
# ---------------------------------------------------------------------------


def needed_paradigms(competitor) -> set:
    return {engine_paradigm(p) for p in competitor.pipeline_plugins}


def predictions_store(competitor) -> str:
    """Which predictions repo a fighter's rows are published to.

    Prediction repos are keyed by the *datashape* a sweep consumed, not by
    the league: ``ovos-intent-template-bench-<dataset>`` for template-only
    fighters, ``ovos-intent-keyword-bench-<dataset>`` for keyword-only ones,
    and the eval corpus's own ``ovos-intent-bench-<dataset>`` for fighters
    that mix both. A row's league is read from the registry at assemble time
    (``arena.predictions.group_rows``), so a fighter that changes league
    keeps its published rows.
    """
    paradigms = needed_paradigms(competitor)
    if paradigms == {"keyword"}:
        return "intent_keyword"
    if paradigms == {"template"}:
        return "intent_template"
    return "intent"


def check_league(competitor) -> None:
    """A fighter's league must follow from its stages.

    Leagues are pure: an ``intent_online`` fighter may not carry a stage that
    needs a pretrained artefact, an ``intent_zero_shot`` fighter may not
    carry a stage that trains, and a keyword fighter never shares a board
    with template supervision.
    """
    intents = competitor.config.get("intents") or {}
    league = expected_league(competitor.pipeline_plugins, intents)
    if competitor.modality != league:
        raise ValueError(
            f"{competitor.competitor_id}: stages put this fighter in the "
            f"{league.value} league, but it is filed under "
            f"{competitor.modality.value}"
        )


def trained_on(competitor, dataset_id: str) -> bool:
    """Whether an offline fighter's artefact can emit *dataset_id*'s labels.

    Only offline fighters are restricted: their pretrained head has a fixed
    label set, so on any other corpus every answer is wrong for a reason
    that says nothing about the engine. Zero-shot and online fighters learn
    the corpus's labels from its training rows and always compete.
    """
    if competitor.training_regime is not TrainingRegime.OFFLINE:
        return True
    return dataset_id in (competitor.label_set or [])


def eligible_competitors(train_paradigms: set, dataset_id: str = "") -> list:
    """Runnable intent fighters: known engines, pure leagues, trainable here."""
    eligible = []
    for comp in load_all_competitors():
        if not comp.modality.value.startswith("intent"):
            continue
        if not all(p in ENGINE_TRAITS for p in comp.pipeline_plugins):
            continue
        check_league(comp)
        if dataset_id and not trained_on(comp, dataset_id):
            log.info("Skipping %s — not trained on this label set (%s)",
                     comp.competitor_id, dataset_id)
            continue
        if needed_paradigms(comp) - train_paradigms:
            log.info("Skipping %s — needs %s training data this benchmark "
                     "does not provide", comp.competitor_id,
                     ", ".join(sorted(needed_paradigms(comp) - train_paradigms)))
            continue
        eligible.append(comp)
    return eligible


# ---------------------------------------------------------------------------
# Prediction rows
# ---------------------------------------------------------------------------


def make_row(
    competitor,
    dataset_id: str,
    lang: str,
    sample_index: int,
    test_row: dict,
    prediction: str | None,
    slots: dict,
    confidence: float | None,
    latency_ms: float,
    stage: str | None,
    dataset_revision: str,
    granularity: str = "intent",
    peak_rss_mb: float | None = None,
    stt_provenance: dict | None = None,
    model_revision: str | None = None,
    label_overlap: int | None = None,
) -> dict:
    """Build one §3.2 prediction row.

    ``granularity`` mirrors the eval dataset's ``reference_granularity``:
    'intent' (default) requires an exact string match; 'domain' compares
    only the text before the first ':' on both sides, for corpora that
    only carry a domain-level reference (e.g. meteocat).

    ``stt_provenance`` (audio-input datasets only) carries
    ``{"stt_plugin", "stt_config", "stt_revision"}`` — which pinned STT
    produced ``test_row["utterance"]`` (the cached transcript every
    fighter is scored against). Absent for text-input datasets.
    """
    reference_intent = test_row.get("expected_intent")
    if reference_intent is None:
        exact = prediction is None  # OOD: correct behaviour is no match
    elif granularity == "domain":
        exact = domain_of(prediction) == domain_of(reference_intent)
    else:
        exact = prediction == reference_intent
    versions = ";".join(
        plugin_version(p) for p in competitor.pipeline_plugins
    )
    if model_revision:
        # A pretrained fighter's weights are half its identity: a row that
        # names only the plugin version cannot be reproduced. The sha here is
        # the one the snapshot resolved to, never the declared one.
        versions = f"{versions};{competitor.model}@{model_revision}"
    row = {
        "competitor_id": competitor.competitor_id,
        "sample_id": f"{lang}/{sample_index:05d}",
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "lang": lang,
        "modality": competitor.modality.value,
        "plugin_id": competitor.plugin or "ensemble",
        "plugin_version": versions,
        "pipeline": competitor.pipeline,
        "stage": stage,
        "utterance": test_row["utterance"],
        "reference_intent": reference_intent,
        "reference_slots": test_row.get("expected_slots") or None,
        "prediction": prediction,
        "predicted_slots": slots or None,
        "exact_match": exact,
        "confidence": confidence,
        "bucket": test_row.get("split"),
        "model_revision": model_revision,
        "latency_ms": round(latency_ms, 3),
        # performance-metrics campaign M1 (runner.perf) — additive, optional;
        # no audio_secs here, intent rows are text-in/text-out.
        "elapsed_ms": round(latency_ms, 3),
        "peak_rss_mb": round(peak_rss_mb, 3) if peak_rss_mb is not None else None,
        "hw": hw_fingerprint(),
        "runner_version": f"ovos-plugin-arena=={ARENA_VERSION}",
        "created_at": _now_iso(),
    }
    if stt_provenance:
        row.update(stt_provenance)
    if label_overlap is not None:
        row["extras"] = {"label_overlap": label_overlap}
    return row


def done_samples(out_path: Path, dataset_revision: str | None = None) -> set:
    """sample_ids already present in a (resumable) output file.

    *dataset_revision* is the dataset's DECLARED ``source.revision``, never
    the sha a branch happens to resolve to today. Only when it pins an
    immutable commit does a row have to carry that same revision to count as
    done: a ``sample_id`` is an index into whichever revision produced it,
    so re-pinning a corpus must make its old rows regenerate rather than
    read as complete. On a sha-pinned dataset a row with no
    ``dataset_revision`` at all cannot be shown to belong to the pin, so it
    is not done either.

    A branch-pinned dataset has no fixed revision to compare against — its
    rows legitimately carry whatever sha the branch held when they were
    swept — so every row counts, exactly as before this check existed.
    """
    if not is_pinned_revision(dataset_revision):
        dataset_revision = None
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if (dataset_revision is None
                        or row.get("dataset_revision") == dataset_revision):
                    done.add(row["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


#: Seed for the ``--max-samples`` bucket sample. Fixed so two runs of the
#: same cap draw the same rows and a resumed shard keeps growing rather than
#: restarting on a fresh subset.
SAMPLE_SEED = 0


def stratified_sample(
    indexed_rows: list[tuple[int, dict]], max_samples: int,
) -> list[tuple[int, dict]]:
    """Take *max_samples* of *indexed_rows*, proportionally per bucket.

    ``intents-for-eval`` stores its test split grouped by bucket, so a head
    slice takes whole leading buckets and nothing from the trailing ones —
    ``--max-samples 1000`` of 1,381 rows yields no ``far_ood``,
    ``asr_noise`` or ``typos`` at all, and ``generalization_accuracy``
    degenerates into a paraphrase-only score. Sampling proportionally keeps
    every bucket represented at roughly its share of the corpus.

    Each bucket contributes at least one row, so a bucket too small to earn
    a proportional slot is still measured. Rows keep their original index,
    which is what ``sample_id`` is built from: the same row has the same id
    whether it was drawn under a cap or in a full sweep. A corpus with a
    single bucket (or none) has nothing to stratify and keeps the head
    slice, so caps on unbucketed corpora draw exactly the rows they always
    did.
    """
    if max_samples >= len(indexed_rows):
        return indexed_rows
    buckets: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in indexed_rows:
        buckets[row.get("split") or "test"].append((index, row))
    if len(buckets) < 2:
        return indexed_rows[:max_samples]
    if max_samples <= len(buckets):
        # Too small to apportion: spend the whole cap on breadth, taking one
        # row from each of the largest buckets.
        widest = sorted(buckets, key=lambda name: (-len(buckets[name]), name))
        return sorted(
            (buckets[name][0] for name in widest[:max_samples]),
            key=lambda pair: pair[0],
        )

    quotas = {name: 1 for name in buckets}
    remaining = max_samples - len(quotas)
    # Largest-remainder apportionment over what is left after the floor, so
    # the quotas sum to exactly max_samples without drifting toward the
    # buckets that happen to be visited first.
    shares = {
        name: remaining * (len(rows) - 1) / (len(indexed_rows) - len(quotas))
        for name, rows in buckets.items()
    }
    for name, share in shares.items():
        quotas[name] += int(share)
    leftover = max_samples - sum(quotas.values())
    by_remainder = sorted(
        shares, key=lambda name: (-(shares[name] % 1), name),
    )
    for name in by_remainder[:leftover]:
        quotas[name] += 1

    picked: list[tuple[int, dict]] = []
    for name, rows in buckets.items():
        take = min(quotas[name], len(rows))
        rng = random.Random(f"{SAMPLE_SEED}:{name}")
        picked.extend(rng.sample(rows, take))
    return sorted(picked, key=lambda pair: pair[0])


def prune_other_revisions(out_path: Path, dataset_revision: str | None) -> int:
    """Drop rows not produced against *dataset_revision* from *out_path*.

    Returns how many rows were removed. A shard is appended to across runs,
    so a re-pinned dataset would otherwise leave one file holding two
    revisions' rows — which then publishes as one shard and gets scored as
    though it were a single sweep. Rewritten via a temporary file and an
    atomic replace, so an interrupted prune leaves the original intact.

    *dataset_revision* is the dataset's DECLARED ``source.revision``. Nothing
    is pruned unless it pins an immutable commit: a branch pin resolves to a
    different sha every time the branch moves, so comparing rows against the
    resolved sha would delete the entire shard of every branch-pinned
    dataset on the first run after any upstream commit — including rows
    published before ``dataset_revision`` existed as a column at all.
    """
    if not is_pinned_revision(dataset_revision) or not out_path.exists():
        return 0
    kept: list[str] = []
    dropped = 0
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if row.get("dataset_revision") == dataset_revision:
            kept.append(line)
        else:
            dropped += 1
    if not dropped:
        return 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    tmp.replace(out_path)
    log.info("  %s: dropped %d row(s) from another dataset revision",
             out_path.name, dropped)
    return dropped


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def _child_rss_mb(pid: int) -> float | None:
    """Resident set size of another process, in MB, via ``/proc/<pid>/status``.

    Linux-only, like the rest of this repo's process introspection
    (``runner/perf.py``'s ``_cpu_model``). Returns ``None`` once the process
    has exited or ``/proc`` has nothing for it, rather than raising.
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _train_and_predict(
    competitor,
    train_data: dict,
    intents_config: dict,
    lang: str,
    todo: list,
    out_path: Path,
    dataset_id: str,
    revision: str,
    granularity: str,
    stt_provenance: dict | None,
    model_revision: str | None = None,
    reference_labels: set | None = None,
) -> int:
    """Train one fighter and predict its eval rows, appending them to ``out_path``.

    The body of one benchmark cell — split out so it can run either
    in-process (guard disabled) or as the target of a child process the
    parent's watchdog can kill outright (guard enabled).
    """
    pipeline = IntentPipeline(intents_config, lang=lang)
    log.info("  training %s for %s (stages: %s)",
             competitor.competitor_id, lang, ", ".join(pipeline.stage_names))
    pipeline.train(train_data)

    # A pretrained head either shares labels with this corpus or it does not,
    # and only the loaded model can say. Without this check a zero-overlap
    # fighter publishes a whole board of nulls that reads as a terrible
    # engine rather than as an ineligible one.
    label_overlap = None
    if reference_labels:
        label_overlap = model_label_overlap(pipeline, reference_labels)
        if label_overlap == 0:
            log.error(
                "  %s/%s: the loaded model emits none of %s's %d labels — "
                "its label_set claims a corpus its artefact was not trained "
                "on; no rows written",
                competitor.competitor_id, lang, dataset_id,
                len(reference_labels),
            )
            return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    errored = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, test_row in todo:
            try:
                (prediction, slots, confidence, latency_ms, stage), _, peak_rss_mb = (
                    measure_call(
                        lambda test_row=test_row: pipeline.predict(test_row["utterance"])
                    )
                )
            except Exception as exc:
                log.warning("    %s/%s sample %s failed: %s",
                            competitor.competitor_id, lang, i, exc)
                errored += 1
                continue
            row = make_row(
                competitor, dataset_id, lang, i, test_row,
                prediction, slots, confidence, latency_ms, stage, revision,
                granularity=granularity,
                peak_rss_mb=peak_rss_mb,
                stt_provenance=stt_provenance,
                model_revision=model_revision,
                label_overlap=label_overlap,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 500 == 0:
                fh.flush()
                log.info("    %s/%s: %d/%d", competitor.competitor_id, lang,
                         written, len(todo))
    log.info("  %s/%s: wrote %d rows (%d errored)", competitor.competitor_id,
              lang, written, errored)
    return written


def _train_and_predict_child(result_queue, *args) -> None:
    """``multiprocessing`` target: run one cell, put the row count on the queue."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    written = _train_and_predict(*args)
    result_queue.put(written)


class UnresolvableModelPin(RuntimeError):
    """A fighter declares a model revision that cannot be fetched."""


def resolve_model_pin(competitor, intents_config: dict) -> str | None:
    """Download the exact model commit a fighter declares and point the
    plugin at it.

    Neither the OVOS pipeline plugins nor model2vec take a revision, so a
    declared pin would resolve to whatever ``main`` holds today while the row
    claimed the pinned sha — provenance that is worse than none. The arena
    resolves the snapshot itself and hands the plugin a local path, which
    every one of these plugins accepts in place of a repo id. ``revision``
    stays in the config for plugins that grow support for it.

    Returns the sha the snapshot actually resolved to, or None for a fighter
    that declares no pin.
    """
    if not competitor.model_revision:
        return None
    if not competitor.model:
        raise UnresolvableModelPin(
            f"{competitor.competitor_id}: model_revision is set but no model "
            "repo is named"
        )
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=competitor.model, revision=competitor.model_revision,
        )
    except Exception as exc:
        raise UnresolvableModelPin(
            f"{competitor.competitor_id}: cannot resolve "
            f"{competitor.model}@{competitor.model_revision} — {exc}"
        ) from exc
    resolved = Path(path).name
    for key, block in intents_config.items():
        if key != "pipeline" and isinstance(block, dict) and block.get("model"):
            block["model"] = path
    log.info("  %s: pinned %s @ %s", competitor.competitor_id,
             competitor.model, resolved[:12])
    return resolved


def model_label_overlap(pipeline, reference_labels: set) -> int | None:
    """How many of a corpus's labels the loaded model can actually emit.

    A pretrained head has a fixed class list. ``label_set`` is the registry's
    *claim* that it matches a corpus; this is the measurement. Returns None
    when no stage exposes a class list (nothing to check).
    """
    classes: set = set()
    for plugin in pipeline.plugins.values():
        model = getattr(plugin, "model", None)
        model_classes = getattr(model, "classes_", None)
        if model_classes is not None:
            classes.update(str(c) for c in model_classes)
    if not classes:
        return None
    return len(classes & reference_labels)


def run_competitor_lang(
    competitor,
    dataset_id: str,
    lang: str,
    eval_def,
    train_defs: dict,
    revision: str,
    out_path: Path,
    max_samples: int = 0,
    transcript_cache_dir: Path | None = None,
    train_timeout_secs: float = 0,
    train_max_rss_mb: float = 0,
) -> int:
    """Train one fighter for one language and predict the eval split."""
    stt_provenance = None
    if eval_def.input == "audio":
        test_rows, stt_revision = transcribe_dataset(
            eval_def, lang, revision,
            transcript_cache_dir or Path("transcript_cache"),
            max_samples=max_samples,
        )
        stt_provenance = {
            "stt_plugin": eval_def.stt_plugin,
            "stt_config": eval_def.stt_config,
            "stt_revision": stt_revision,
        }
        indexed = list(enumerate(test_rows))
    else:
        indexed = list(enumerate(fetch_rows(eval_def, lang, revision)))
        if max_samples:
            indexed = stratified_sample(indexed, max_samples)

    # The DECLARED pin, not ``revision`` (the sha the pin resolves to today):
    # comparing a branch-pinned dataset's rows against a moving branch tip
    # would wipe its shard on every upstream commit.
    declared_revision = eval_def.source.revision
    prune_other_revisions(out_path, declared_revision)
    done = done_samples(out_path, declared_revision)
    todo = [(i, row) for i, row in indexed if f"{lang}/{i:05d}" not in done]
    if not todo:
        log.info("  %s/%s: already complete", competitor.competitor_id, lang)
        return 0

    needed = needed_paradigms(competitor)
    # The eval revision sha belongs to the EVAL repo. Train corpora may live
    # in a different HF repo (meteocat trains from intents-for-eval), so each
    # train dataset pins its own branch to its own repo's sha.
    train_data = {
        paradigm: fetch_rows(
            train_def, lang,
            resolve_revision(train_def.source.hf_id,
                             train_def.source.revision))
        for paradigm, train_def in train_defs.items()
        if paradigm in needed
    }

    intents_config = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in competitor.config["intents"].items()
    }
    model_revision = resolve_model_pin(competitor, intents_config)
    if "intent_transformers" in competitor.config:
        # Fighter-declared, config-gated exactly like production's
        # mycroft.conf ``intent_transformers`` section — carried alongside
        # (not inside) ``intents`` since it is not a pipeline stage.
        intents_config["intent_transformers"] = competitor.config["intent_transformers"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Only a pretrained fighter's label set is in question: an engine that
    # learns this corpus's labels from its training rows always shares them.
    reference_labels = None
    if competitor.training_regime is TrainingRegime.OFFLINE:
        reference_labels = {
            row["expected_intent"] for _, row in indexed
            if row.get("expected_intent")
        }
    cell_args = (
        competitor, train_data, intents_config, lang, todo, out_path,
        dataset_id, revision, eval_def.reference_granularity, stt_provenance,
        model_revision, reference_labels,
    )

    if not train_timeout_secs and not train_max_rss_mb:
        # Guard disabled: exact previous in-process behaviour, no
        # subprocess overhead.
        return _train_and_predict(*cell_args)

    # A runaway ``pipeline.train()`` (e.g. padatious/padaos compiling an
    # entity-heavy corpus) can hold the GIL solid, so a watchdog *thread* in
    # the same process could itself starve and never get to check anything.
    # The only reliable stop is a separate process the parent can SIGKILL.
    # ``fork`` (not ``spawn``) so tests can monkeypatch ``IntentPipeline``
    # in the parent and have the child inherit the patched module state.
    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_train_and_predict_child, args=(result_queue, *cell_args))
    proc.start()
    start = time.monotonic()
    reason = None
    while proc.is_alive():
        elapsed = time.monotonic() - start
        if train_timeout_secs and elapsed > train_timeout_secs:
            reason = "train_timeout"
            break
        if train_max_rss_mb:
            rss = _child_rss_mb(proc.pid)
            if rss is not None and rss > train_max_rss_mb:
                reason = "train_memory"
                break
        proc.join(timeout=0.2)

    if reason is not None:
        proc.kill()
        proc.join(timeout=5)
        limit = (f"{train_timeout_secs}s" if reason == "train_timeout"
                 else f"{train_max_rss_mb}MB")
        log.error(
            "  %s/%s: aborted training on dataset %s, limit %s exceeded "
            "- trained=False reason=%s",
            competitor.competitor_id, lang, dataset_id, limit, reason,
        )
    else:
        proc.join()

    try:
        return result_queue.get_nowait()
    except Exception:
        # Killed (or crashed) before it could report a count; any rows it
        # did write are already flushed on disk (``fh.flush()`` every 500
        # rows) and will be skipped as ``done`` on the next run.
        return 0


# ---------------------------------------------------------------------------
# Publishing — one repo per benchmark modality, splits per language
# ---------------------------------------------------------------------------


def _dataset_card(modality: str, dataset_id: str, eval_def, langs: list[str]) -> str:
    configs = "\n".join(
        f"  - split: {split_name(lang)}\n"
        f"    path: predictions/{lang}/*.jsonl"
        for lang in sorted(langs)
    )
    league = {
        "intent": "mixed-supervision",
        "intent_template": "template-supervised",
        "intent_keyword": "keyword-supervised",
    }.get(modality, modality)
    return f"""---
license: apache-2.0
tags:
  - openvoiceos
  - intent-classification
  - benchmark
  - predictions
pretty_name: OVOS {modality} bench — {dataset_id}
configs:
- config_name: default
  data_files:
{configs}
---

# OVOS `{modality}` bench — `{dataset_id}`

Per-sample predictions of the {league} intent fighters of the
[OVOS Plugin Arena](https://github.com/OpenVoiceOS/ovos-plugin-arena) over
[`{eval_def.source.hf_id}`](https://huggingface.co/datasets/{eval_def.source.hf_id}).

One dedicated repo per benchmark modality; one dataset split per language;
one JSONL file per fighter under `predictions/<lang>/<competitor_id>.jsonl`.
Rows follow the arena §3.2 contract (pinned `dataset_revision`,
`plugin_version`, fired pipeline `stage`, `exact_match` with correct-OOD
semantics). Produced by the reproducible benchmark script in the arena repo;
the arena's `assemble` workflow turns these rows into benchmark boards,
blind battle pools and a benchmark-seeded ELO ladder.

{FUNDING_BLOCK}
"""


def upload_predictions(
    bench_dir: Path,
    dataset_id: str,
    eval_def,
    owner: str = HF_OWNER,
) -> None:
    """Upload ``<bench_dir>/<modality>/…`` to the per-modality HF repos."""
    from huggingface_hub import HfApi

    api = HfApi()
    for modality_dir in sorted(d for d in bench_dir.iterdir() if d.is_dir()):
        modality = modality_dir.name
        repo = results_repo_for(modality, dataset_id, owner)
        langs = sorted(d.name for d in modality_dir.iterdir() if d.is_dir())
        if not langs:
            continue
        try:
            api.create_repo(repo, repo_type="dataset", exist_ok=True)
        except Exception as exc:
            # Restricted tokens may write to existing repos but not create
            # new ones — proceed and let the upload itself decide.
            log.warning("create_repo(%s) refused (%s) — uploading anyway",
                        repo, exc)
        api.upload_file(
            path_or_fileobj=check_funding_block(
                _dataset_card(modality, dataset_id, eval_def, langs),
                repo).encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        log.info("Uploading %s → %s (%d langs)", modality_dir, repo, len(langs))
        api.upload_folder(
            folder_path=str(modality_dir),
            path_in_repo="predictions",
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=["**/*.jsonl"],
            commit_message=f"bench: refresh {modality} predictions",
        )


# ---------------------------------------------------------------------------
# Entry point shared by the benchmark scripts
# ---------------------------------------------------------------------------


def run_benchmark(dataset_id: str, description: str, argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--competitors", default="",
                        help="Comma-separated competitor ids (default: all eligible)")
    parser.add_argument("--langs", default="",
                        help="Comma-separated languages (default: all in dataset)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap test rows per language (smoke runs)")
    parser.add_argument("--output-dir", default="predictions",
                        help="Local root for prediction JSONLs")
    parser.add_argument("--upload", action="store_true",
                        help="Upload predictions to the per-modality HF repos")
    parser.add_argument("--hf-owner", default=HF_OWNER)
    parser.add_argument(
        "--train-timeout-secs", type=float, default=1200,
        help=(
            "Kill a fighter's train+predict cell if it runs longer than "
            "this many wall-clock seconds (default 20 minutes); 0 disables "
            "the timeout guard."
        ),
    )
    parser.add_argument(
        "--train-max-rss-mb", type=float, default=8192,
        help=(
            "Kill a fighter's train+predict cell if its resident memory "
            "exceeds this many MB (default 8 GB); 0 disables the memory "
            "guard."
        ),
    )
    parser.add_argument(
        "--transcript-cache-dir", default="transcript_cache",
        help=(
            "Audio-input datasets only: where cached per-(dataset,lang) "
            "STT transcripts are read/written. One transcription pass per "
            "language, shared by every fighter (see DatasetDef.input)."
        ),
    )
    args = parser.parse_args(argv)

    eval_def = load_dataset("intent", dataset_id)
    if eval_def.input == "audio":
        from runner.media_bench import plugin_is_installed

        if not plugin_is_installed("stt", eval_def.stt_plugin):
            log.error(
                "Dataset %s requires STT plugin %s, which is not installed "
                "— skipping every fighter/lang cell",
                dataset_id, eval_def.stt_plugin,
            )
            log.info("run summary: skipped_datasets_missing_stt=1")
            return 0
    train_defs = {
        paradigm: load_dataset(f"intent_{paradigm}", train_id)
        for paradigm, train_id in (eval_def.train_datasets or {}).items()
    }
    revision = resolve_revision(eval_def.source.hf_id, eval_def.source.revision)
    log.info("Dataset %s @ %s (train paradigms: %s)",
             eval_def.source.hf_id, revision[:12],
             ", ".join(train_defs) or "none")

    competitors = eligible_competitors(set(train_defs), dataset_id)
    if args.competitors:
        wanted = {c.strip() for c in args.competitors.split(",") if c.strip()}
        competitors = [c for c in competitors if c.competitor_id in wanted]
        missing = wanted - {c.competitor_id for c in competitors}
        if missing:
            log.error("Unknown/ineligible competitors: %s",
                      ", ".join(sorted(missing)))
            return 1

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()] or (
        eval_def.langs or [eval_def.lang]
    )

    from runner.media_bench import plugin_is_installed

    bench_dir = Path(args.output_dir) / dataset_id
    skipped_missing_plugin = 0
    for competitor in competitors:
        modality = competitor.modality.value
        if competitor.plugin and not plugin_is_installed(modality, competitor.plugin):
            log.error("Fighter %s [%s]: plugin %s is not installed — skipping",
                      competitor.competitor_id, modality, competitor.plugin)
            skipped_missing_plugin += 1
            continue
        store = predictions_store(competitor)
        log.info("Fighter %s [%s]", competitor.competitor_id, modality)
        if is_trained_on(competitor, eval_def):
            log.info("  %s: trained on this corpus, not scored",
                     competitor.competitor_id)
            continue
        for lang in langs:
            if competitor.langs and lang not in competitor.langs:
                continue
            out_path = (bench_dir / store / lang
                        / f"{competitor.competitor_id}.jsonl")
            try:
                run_competitor_lang(
                    competitor, dataset_id, lang, eval_def, train_defs,
                    revision, out_path, max_samples=args.max_samples,
                    transcript_cache_dir=Path(args.transcript_cache_dir),
                    train_timeout_secs=args.train_timeout_secs,
                    train_max_rss_mb=args.train_max_rss_mb,
                )
            except UnresolvableModelPin as exc:
                # A refusal, not a crash: the fighter cannot be run as
                # declared and says so in one line, with no rows written.
                log.error("  %s", exc)
            except Exception:
                log.exception("  %s/%s failed", competitor.competitor_id, lang)

    if skipped_missing_plugin:
        log.info("run summary: skipped_missing_plugin=%d", skipped_missing_plugin)

    if args.upload:
        upload_predictions(bench_dir, dataset_id, eval_def, owner=args.hf_owner)
    return 0
