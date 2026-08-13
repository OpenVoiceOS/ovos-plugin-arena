"""Autonomous round-robin prediction runner.

``python -m runner.autorun`` is a fleet-friendly companion to the manual
``runner.<modality>_bench`` scripts and the STT-only queue daemon
(``runner.daemon``): instead of a human curating ``runner/queue.yaml`` and
running one dataset to completion at a time, it keeps a process alive that

1. reloads the declarative registry every sweep, so a new fighter or dataset
   json dropped into ``registry/`` is picked up without a restart;
2. enumerates every eligible ``(fighter, dataset, lang)`` pair per modality,
   reusing ``runner.queue_tools``' compatibility/eligibility rules rather
   than re-deriving them;
3. round-robins across pairs: one bounded *batch* of new samples per pair,
   then moves on — never drains a single pair to completion in one sweep,
   so N fighters against a big dataset all make visible progress instead of
   one fighter hogging the process for hours;
4. persists progress two ways: the local per-(dataset, modality, lang,
   competitor) JSONL shard IS the resume state (identical file the manual
   bench scripts write/append — ``runner.media_bench.run_competitor_lang``'s
   own ``done_samples`` resume), and a small ``autorun_state.json`` records
   which pairs are already fully complete / which fighters are quarantined,
   so a restart doesn't have to re-stream an already-finished dataset just
   to rediscover "nothing new here";
5. batches HF uploads instead of publishing every 10-row batch: a pair's
   shard is pushed immediately when that pair finishes, and everything else
   flushes on a timer (``--flush-every`` minutes);
6. is safe to run one instance per fleet host with disjoint fighter subsets
   via ``--include``/``--exclude``/``--langs``/``--min-size``/``--max-size``/
   ``--heavy``/``--light``/``--host-class``.

Run one instance per modality-class per host, e.g.:

    python -m runner.autorun --modalities stt,tts --batch 10 --host-class auto
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import random
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from registry.loaders import load_all_competitors, load_all_datasets
from runner import media_bench as mb
from runner.intent_bench import HF_OWNER, resolve_revision, results_repo_for
from runner.queue_tools import HFLister, HubLister, dataset_langs, is_compatible

log = logging.getLogger("autorun")

# Intent leagues are NOT covered here — they run through the separate
# ``runner.intent_bench`` script/registry-modality (a different row shape,
# different train+eval flow) and are not wired into this scheduler.
# Autonomous, fleet-wide round-robin coverage for intent is a follow-up.
MODALITIES = ("stt", "wake_word", "tts", "vad")

SIZE_ORDER = [
    "micro", "tiny", "small", "base", "medium", "large", "x-large", "giant", "titan",
]
HEAVY_SIZES = {"large", "x-large", "giant", "titan"}
# Fallback name-hints for fighters with no registry ``size`` field: engine
# families that are heavy regardless of a param count in their name/id.
HEAVY_NAME_HINTS = ("speech-llm", "whisper-large", "canary")
# Any "<number>b" token (param count in billions) at/above this threshold is
# heavy — catches names the hint list above doesn't enumerate (e.g.
# "cohere-transcribe-2b") without having to hand-maintain a growing list of
# model names. This is a heuristic on the *id/plugin string*, not a real
# param count lookup — a fighter named e.g. "modelx-2bit" would false-match
# ("2b" + word boundary) if "bit" didn't push the boundary past "b"; the
# regex requires a non-word character or string end right after the "b" to
# guard against that. Genuinely uncertain classifications are logged once
# per fighter at startup (see ``log_classifications``) specifically so a
# misclassification is visible rather than silent.
HEAVY_PARAM_THRESHOLD_B = 1.0
# Deliberately does NOT match a leading-zero integer with no decimal point
# (e.g. "06b"): several registry ids drop the "." from a real "0.6b" model
# name (see "onnx-asr-parakeet-tdt-06b-v3", NVIDIA Parakeet TDT 0.6B) —
# parsed as a bare integer that would read as "6b" and false-positive as
# heavy. Excluding that shape means it falls through to no match instead
# (still correctly "light" here, since the real model is sub-1B either
# way) rather than guessing at the dropped decimal position.
_PARAM_SIZE_RE = re.compile(
    r"(?<!\d)(?:(0\.\d+)|([1-9]\d*(?:\.\d+)?))b(?:\W|$)", re.IGNORECASE
)


def _param_size_billion(text: str) -> float | None:
    """Largest "<N>b" (billions of params) token found in *text*, or None."""
    best: float | None = None
    for match in _PARAM_SIZE_RE.finditer(text.lower()):
        token = match.group(1) or match.group(2)
        try:
            value = float(token)
        except (ValueError, TypeError):
            continue
        if best is None or value > best:
            best = value
    return best


def is_heavy(competitor) -> bool:
    """A GPU-class fighter.

    Heavy when any of: the registry ``size`` field is >= large; the
    id/plugin string carries a "<N>b" param-count token >=
    :data:`HEAVY_PARAM_THRESHOLD_B` billion (e.g. "cohere-transcribe-2b",
    "onnx-asr-canary-qwen-2.5b", "coreml-parakeet-tdt-1.1b-fp16"); or the
    string matches a known heavy engine family name
    (:data:`HEAVY_NAME_HINTS`, for names like "whisper-large-v3-turbo" that
    carry no explicit param count). This is a heuristic over the fighter's
    id/plugin name, not a verified parameter count — see
    ``log_classifications`` for making it auditable.
    """
    if getattr(competitor, "size", None) in HEAVY_SIZES:
        return True
    haystack = f"{competitor.competitor_id} {competitor.plugin or ''}".lower()
    param_b = _param_size_billion(haystack)
    if param_b is not None and param_b >= HEAVY_PARAM_THRESHOLD_B:
        return True
    return any(hint in haystack for hint in HEAVY_NAME_HINTS)


def log_classifications(entries: list[tuple[str, object, object, str]]) -> None:
    """Log each distinct fighter's heavy/light classification once.

    Called at startup so a misclassification (the heuristic in
    :func:`is_heavy` is a name/id pattern match, not a verified parameter
    count) is visible in the log instead of silently routing a fighter to
    the wrong host class.
    """
    seen: set[str] = set()
    for _modality, competitor, _dataset, _lang in entries:
        cid = competitor.competitor_id
        if cid in seen:
            continue
        seen.add(cid)
        size = getattr(competitor, "size", None)
        log.info("classified %s as %s (registry size=%s)",
                  cid, "heavy" if is_heavy(competitor) else "light", size)


def size_rank(size: str | None) -> int | None:
    return SIZE_ORDER.index(size) if size in SIZE_ORDER else None


# ---------------------------------------------------------------------------
# Adapters (lazy-imported — the real ones pull in audio/plugin deps)
# ---------------------------------------------------------------------------

_ADAPTER_FACTORIES: dict[str, Callable[[], mb.MediaBenchAdapter]] = {}


def adapter_factories() -> dict[str, Callable[[], mb.MediaBenchAdapter]]:
    global _ADAPTER_FACTORIES
    if not _ADAPTER_FACTORIES:
        from runner.stt_bench import STTBench
        from runner.tts_bench import TTSBench
        from runner.vad_bench import VADBench
        from runner.ww_bench import WakeWordBench

        _ADAPTER_FACTORIES = {
            "stt": STTBench,
            "wake_word": WakeWordBench,
            "tts": TTSBench,
            "vad": VADBench,
        }
    return _ADAPTER_FACTORIES


# ---------------------------------------------------------------------------
# Eligibility (reuses runner.queue_tools — never re-derives compatibility)
# ---------------------------------------------------------------------------


def enumerate_all_pairs(
    modalities: Iterable[str],
    registry_root: Path | None = None,
) -> list[tuple[str, object, object, str]]:
    """Every ``(modality, competitor, dataset, lang)`` tuple eligible to run.

    Fighter x dataset compatibility is ``runner.queue_tools.is_compatible``
    (primary-subtag language overlap); per-pair language expansion is
    ``MediaBenchAdapter.competitor_langs`` (the same rule ``run_benchmark``
    uses) so autorun and a manual bench run agree on what gets scheduled.
    """
    out: list[tuple[str, object, object, str]] = []
    factories = adapter_factories()
    for modality in modalities:
        if modality not in factories:
            log.warning("skipping unknown modality %r (not in %s)",
                        modality, sorted(factories))
            continue
        competitors = [
            c for c in load_all_competitors(registry_root) if c.modality == modality
        ]
        datasets = [
            d for d in load_all_datasets(registry_root)
            if d.modality == modality and d.role == "eval"
        ]
        adapter = factories[modality]()
        competitors = adapter.filter_competitors(competitors)
        for dataset in datasets:
            langs = dataset_langs(dataset)
            for competitor in competitors:
                if not is_compatible(competitor, dataset):
                    continue
                for lang in adapter.competitor_langs(competitor, langs):
                    out.append((modality, competitor, dataset, lang))
    return out


def apply_filters(
    entries: list[tuple[str, object, object, str]],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    datasets: list[str] | None = None,
    langs: list[str] | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    heavy_only: bool = False,
    light_only: bool = False,
) -> list[tuple[str, object, object, str]]:
    out = []
    for modality, competitor, dataset, lang in entries:
        cid = competitor.competitor_id
        if include and not any(fnmatch.fnmatch(cid, pat) for pat in include):
            continue
        if exclude and any(fnmatch.fnmatch(cid, pat) for pat in exclude):
            continue
        if datasets and not any(
            fnmatch.fnmatch(dataset.dataset_id, pat) for pat in datasets
        ):
            continue
        if langs and lang not in langs:
            continue
        if heavy_only and not is_heavy(competitor):
            continue
        if light_only and is_heavy(competitor):
            continue
        rank = size_rank(getattr(competitor, "size", None))
        if min_size is not None and (rank is None or rank < min_size):
            continue
        if max_size is not None and rank is not None and rank > max_size:
            continue
        out.append((modality, competitor, dataset, lang))
    return out


# ---------------------------------------------------------------------------
# Pair identity + round-robin scheduler (pure, network-free, unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairKey:
    modality: str
    competitor_id: str
    dataset_id: str
    lang: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.modality}|{self.competitor_id}|{self.dataset_id}|{self.lang}"

    def to_str(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, s: str) -> "PairKey":
        modality, competitor_id, dataset_id, lang = s.split("|", 3)
        return cls(modality, competitor_id, dataset_id, lang)


# Backoff schedule for a quarantined fighter: 30min * 2^(attempts-1),
# capped at 24h. A weeks-long daemon must recover from a transient load
# failure (network blip fetching a model, a momentarily-locked file, ...)
# on its own instead of quarantining a fighter forever after one bad
# attempt — but must also not hammer a genuinely-broken fighter (missing
# dependency) every sweep.
QUARANTINE_BASE_BACKOFF_SECONDS = 30 * 60
QUARANTINE_MAX_BACKOFF_SECONDS = 24 * 60 * 60


def _quarantine_backoff_seconds(attempts: int) -> float:
    return min(
        QUARANTINE_BASE_BACKOFF_SECONDS * (2 ** max(attempts - 1, 0)),
        QUARANTINE_MAX_BACKOFF_SECONDS,
    )


@dataclass
class QuarantineEntry:
    reason: str
    attempts: int = 1
    quarantined_at: float = 0.0  # time.time() — persisted, must survive restarts
    retry_after_seconds: float = QUARANTINE_BASE_BACKOFF_SECONDS

    def retry_due(self, now: float) -> bool:
        return now >= self.quarantined_at + self.retry_after_seconds

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "attempts": self.attempts,
            "quarantined_at": self.quarantined_at,
            "retry_after_seconds": self.retry_after_seconds,
        }

    @classmethod
    def from_dict(cls, d) -> "QuarantineEntry":
        if isinstance(d, str):
            # Pre-backoff state file shape (reason string only) — treat as
            # immediately eligible for a retry rather than refusing to load.
            return cls(reason=d, attempts=1, quarantined_at=0.0,
                       retry_after_seconds=QUARANTINE_BASE_BACKOFF_SECONDS)
        return cls(
            reason=d.get("reason", ""),
            attempts=int(d.get("attempts", 1)),
            quarantined_at=float(d.get("quarantined_at", 0.0)),
            retry_after_seconds=float(
                d.get("retry_after_seconds", QUARANTINE_BASE_BACKOFF_SECONDS)
            ),
        )


class RoundRobinScheduler:
    """Cycles through a fixed set of pairs, one batch per pair per sweep.

    :meth:`sweep` yields every pair that is neither completed nor currently
    quarantined, in stable original order. Every pair with work left gets
    exactly one turn before any pair gets a second — a pair that keeps
    reporting "not complete yet" can never crowd out the others within a
    sweep, which is the starvation-freedom property the caller relies on to
    make visible progress on every fighter instead of draining one to
    completion first.

    A pair is marked complete ONLY by an explicit :meth:`mark_complete`
    call — the scheduler itself never infers completion from a short batch;
    that judgment (exhausted iterator vs. every remaining sample erroring)
    belongs to the caller, which has the ``errored`` count
    ``run_competitor_lang`` returns (see ``runner.media_bench.BatchResult``
    and ``AutoRunner.process_pair``).
    """

    def __init__(self, pairs: Iterable[PairKey] = ()):
        self._pairs: list[PairKey] = list(pairs)
        self.completed: set[PairKey] = set()
        self.quarantined: dict[str, QuarantineEntry] = {}  # competitor_id -> entry
        # Consecutive all-error (written == 0, errored > 0) batches per
        # pair — reset to 0 the moment a batch makes ANY progress (writes a
        # row) or reports zero errors. Used to escalate to quarantine after
        # repeated total failures instead of looping forever on a pair that
        # never completes and never quarantines.
        self.error_streak: dict[PairKey, int] = {}

    def set_pairs(self, pairs: Iterable[PairKey]) -> None:
        """Reconcile against a freshly reloaded registry.

        Pairs no longer present (dataset/fighter removed from the registry)
        are dropped; pairs still present keep their completed/quarantine
        state; genuinely new pairs (new fighter or dataset json) are
        appended at the end in the order given, so they get folded into the
        round-robin without disturbing the order already in flight.
        """
        pairs = list(pairs)
        pair_set = set(pairs)
        kept = [p for p in self._pairs if p in pair_set]
        kept_set = set(kept)
        new_ones = [p for p in pairs if p not in kept_set]
        self._pairs = kept + new_ones

    def mark_complete(self, pair: PairKey) -> None:
        self.completed.add(pair)
        self.error_streak.pop(pair, None)

    def record_error_streak(self, pair: PairKey, written: int, errored: int) -> int:
        """Update and return the pair's consecutive all-error batch count.

        A batch that writes nothing AND has errors resets no progress was
        made this turn — increment. Any progress (a row written) or an
        error-free batch resets the streak to 0.
        """
        if written == 0 and errored > 0:
            streak = self.error_streak.get(pair, 0) + 1
        else:
            streak = 0
        self.error_streak[pair] = streak
        return streak

    def quarantine(self, competitor_id: str, reason: str, now: float | None = None) -> None:
        """Quarantine a fighter with an exponential-backoff retry window.

        Each call is a genuine failure event (the scheduler only calls this
        again for an already-quarantined fighter after its backoff expired
        and a retry attempt failed — see :meth:`sweep`), so each call is
        logged: that's the "recovers from network blips instead of being
        quarantined forever" behaviour — a transient failure gets a bounded
        number of silent retries over hours/days, each one logged as it
        happens, not replayed every sweep while still within backoff.
        """
        now = now if now is not None else time.time()
        existing = self.quarantined.get(competitor_id)
        attempts = existing.attempts + 1 if existing else 1
        entry = QuarantineEntry(
            reason=reason,
            attempts=attempts,
            quarantined_at=now,
            retry_after_seconds=_quarantine_backoff_seconds(attempts),
        )
        self.quarantined[competitor_id] = entry
        log.error("quarantined fighter %s (attempt %d, retry in %.0fmin): %s",
                  competitor_id, attempts, entry.retry_after_seconds / 60, reason)

    def is_quarantined(self, competitor_id: str, now: float | None = None) -> bool:
        entry = self.quarantined.get(competitor_id)
        if entry is None:
            return False
        now = now if now is not None else time.time()
        return not entry.retry_due(now)

    def sweep(self, now: float | None = None) -> Iterator[PairKey]:
        now = now if now is not None else time.time()
        for pair in self._pairs:
            if pair in self.completed:
                continue
            if self.is_quarantined(pair.competitor_id, now):
                continue
            yield pair

    def is_exhausted(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return all(
            p in self.completed or self.is_quarantined(p.competitor_id, now)
            for p in self._pairs
        )

    # -- state persistence ---------------------------------------------

    def to_state(self) -> dict:
        return {
            "completed": sorted(p.to_str() for p in self.completed),
            "quarantined": {
                cid: entry.to_dict() for cid, entry in self.quarantined.items()
            },
        }

    def apply_state(self, state: dict) -> None:
        self.completed = {PairKey.from_str(s) for s in state.get("completed", [])}
        self.quarantined = {
            cid: QuarantineEntry.from_dict(entry)
            for cid, entry in state.get("quarantined", {}).items()
        }


def save_state(path: Path, scheduler: RoundRobinScheduler) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(scheduler.to_state(), indent=2, sort_keys=True))
    tmp.replace(path)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("could not read state file %s — starting fresh", path)
        return {}


# ---------------------------------------------------------------------------
# Batch execution against the real bench machinery
# ---------------------------------------------------------------------------

_REVISION_CACHE: dict[str, str] = {}


def _revision_for(dataset) -> str:
    key = f"{dataset.source.hf_id}@{dataset.source.revision}"
    if key not in _REVISION_CACHE:
        _REVISION_CACHE[key] = resolve_revision(
            dataset.source.hf_id, dataset.source.revision
        )
    return _REVISION_CACHE[key]


def seed_from_hf(
    out_path: Path, repo: str, lang: str, competitor_id: str, lister: HFLister
) -> None:
    """Seed the local pending shard from the already-published HF shard.

    Only fires when there is no local file yet: a local file is always
    treated as at-least-as-fresh as the last publish (the runner's local
    file is the accumulated source of truth an upload snapshots FROM, not
    the other way round — see ``runner.plugin_runner``'s ``publish``
    notes). Without this, two fleet hosts racing the same pair (or one host
    restarting into a fresh ``--output-dir``) would redo samples another
    host already published.
    """
    if out_path.exists():
        return
    files = lister.list_files(repo)
    path_in_repo = f"predictions/{lang}/{competitor_id}.jsonl"
    size = files.get(path_in_repo)
    if not size:
        return
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo, path_in_repo, repo_type="dataset")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(Path(local).read_bytes())
    log.info("seeded %s <- %s (%d bytes)", out_path, repo, size)


@dataclass
class AutoRunConfig:
    output_dir: Path
    batch: int = 10
    flush_every_minutes: float = 15.0
    sleep_when_idle: int = 300
    hf_owner: str = HF_OWNER
    upload: bool = True
    seed_from_hf: bool = True
    # A pair that returns an all-error (written == 0, errored > 0) batch
    # this many sweeps in a row escalates to quarantine — it is neither
    # marked complete (that would silently drop real, never-collected
    # samples) nor left retried forever every single sweep.
    max_consecutive_error_batches: int = 5


class AutoRunner:
    """Drives :class:`RoundRobinScheduler` against the real bench machinery.

    All network/registry access is behind small seams
    (:func:`enumerate_all_pairs`, :data:`HFLister`,
    ``runner.media_bench.run_competitor_lang``) so tests can inject fakes
    for :attr:`process_fn` and never touch a plugin, an audio file or HF.
    """

    def __init__(
        self,
        config: AutoRunConfig,
        scheduler: RoundRobinScheduler | None = None,
        lister: HFLister | None = None,
        process_fn: Callable[[str, object, object, str, int], object] | None = None,
    ):
        self.config = config
        self.scheduler = scheduler or RoundRobinScheduler()
        self.lister = lister
        # Injectable for tests: (modality, competitor, dataset, lang, batch)
        # -> rows newly written. Defaults to the real bench engine.
        self.process_fn = process_fn or self._run_pair_batch
        self._dirty: set[tuple[str, str]] = set()
        self._dataset_cache: dict[tuple[str, str], object] = {}
        self._last_flush = time.monotonic()
        self._stop = False

    @property
    def state_path(self) -> Path:
        return self.config.output_dir / "autorun_state.json"

    def request_stop(self, *_a) -> None:
        if not self._stop:
            log.info("stop requested — finishing current pair then flushing")
        self._stop = True

    def load_state(self) -> None:
        state = load_state(self.state_path)
        if state:
            self.scheduler.apply_state(state)
            log.info("resumed state: %d complete, %d quarantined",
                      len(self.scheduler.completed), len(self.scheduler.quarantined))

    def save_state(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        save_state(self.state_path, self.scheduler)

    # -- real bench execution -------------------------------------------

    def _run_pair_batch(self, modality, competitor, dataset, lang, batch,
                         deadline: float | None = None) -> mb.BatchResult:
        adapter = adapter_factories()[modality]()
        bench_dir = self.config.output_dir / dataset.dataset_id
        out_path = (bench_dir / modality / "predictions" / lang
                    / f"{competitor.competitor_id}.jsonl")
        audio_dir = bench_dir / modality / "audio"
        repo = results_repo_for(modality, dataset.dataset_id, self.config.hf_owner)

        if self.config.seed_from_hf and self.lister is not None:
            try:
                seed_from_hf(out_path, repo, lang, competitor.competitor_id, self.lister)
            except Exception as exc:
                log.warning("seed-from-HF failed for %s/%s/%s: %s — local-only",
                            competitor.competitor_id, dataset.dataset_id, lang, exc)

        revision = _revision_for(dataset)
        return mb.run_competitor_lang(
            adapter, competitor, dataset.dataset_id, lang, dataset, revision,
            out_path, audio_dir, repo, max_new_samples=batch, deadline=deadline,
        )

    def pair_is_complete(self, modality, competitor, dataset, lang,
                          deadline: float | None = None) -> bool:
        """Best-effort completeness check that never runs inference.

        Seeds the local shard from the published HF shard (the same
        mechanics :meth:`_run_pair_batch` uses), then compares the sample
        ids already written against every sample id the eval set would
        yield — ``adapter.iter_samples`` reads dataset metadata only, it
        never calls ``adapter.predict``/``load_engine``. Used by
        :meth:`run_one_shot` to skip an already-finished pair once it is
        drawn — NOT to prescan every candidate (that made discovery cost
        O(all eligible pairs) instead of O(pairs actually drawn); see the
        module-level note on :meth:`run_one_shot`).

        *deadline* bounds the ``iter_samples`` scan itself for a huge
        dataset: if the deadline elapses mid-scan, this returns ``False``
        (never claim "complete" off a partial scan — the caller then
        attempts the pair for real, where the same deadline immediately
        short-circuits ``run_competitor_lang`` almost as soon as it starts).
        """
        adapter = adapter_factories()[modality]()
        bench_dir = self.config.output_dir / dataset.dataset_id
        out_path = (bench_dir / modality / "predictions" / lang
                    / f"{competitor.competitor_id}.jsonl")
        repo = results_repo_for(modality, dataset.dataset_id, self.config.hf_owner)
        if self.config.seed_from_hf and self.lister is not None:
            try:
                seed_from_hf(out_path, repo, lang, competitor.competitor_id, self.lister)
            except Exception as exc:
                log.warning("seed-from-HF failed for %s/%s/%s: %s — treating as incomplete",
                            competitor.competitor_id, dataset.dataset_id, lang, exc)
        done = mb.done_samples(out_path)
        if not done:
            return False
        revision = _revision_for(dataset)
        total = 0
        for sample_id, _sample in adapter.iter_samples(dataset, lang, revision, 0):
            if deadline is not None and time.monotonic() >= deadline:
                return False
            total += 1
            if sample_id not in done:
                return False
        return total > 0

    def _call_process_fn(self, modality, competitor, dataset, lang, batch,
                          deadline: float | None):
        # process_fn is injectable for tests; older/simpler fakes only take
        # the original 5-arg shape (no deadline) — fall back rather than
        # forcing every test double to grow a parameter it doesn't need.
        try:
            return self.process_fn(modality, competitor, dataset, lang, batch,
                                    deadline=deadline)
        except TypeError:
            return self.process_fn(modality, competitor, dataset, lang, batch)

    # -- orchestration -----------------------------------------------------

    def process_pair(self, modality, competitor, dataset, lang,
                      batch: int | None = None,
                      deadline: float | None = None) -> mb.BatchResult | None:
        """Run one batch for *pair*. Returns the :class:`BatchResult`, or
        ``None`` if the fighter hard-failed to load (already quarantined)."""
        pair = PairKey(modality, competitor.competitor_id, dataset.dataset_id, lang)
        batch = self.config.batch if batch is None else batch
        self._dataset_cache[(modality, dataset.dataset_id)] = dataset
        try:
            result = self._call_process_fn(modality, competitor, dataset, lang,
                                            batch, deadline)
        except Exception as exc:
            # A hard failure BEFORE/OUTSIDE the per-sample try/except inside
            # run_competitor_lang — e.g. the plugin failing to import/load
            # at all. This always quarantines immediately: there is no
            # partial-progress signal to distinguish from here.
            log.exception("pair %s failed", pair)
            self.scheduler.quarantine(competitor.competitor_id, str(exc))
            return None

        written = getattr(result, "written", result)
        errored = getattr(result, "errored", 0)
        last_error = getattr(result, "last_error", None)
        deadline_hit = getattr(result, "deadline_hit", False)

        if written:
            self._dirty.add((modality, dataset.dataset_id))
            log.info("%s: +%d rows (%d errored)", pair, written, errored)

        streak = self.scheduler.record_error_streak(pair, written, errored)

        # CRITICAL: a short batch is NOT proof the pair is exhausted — every
        # remaining sample raising in adapter.predict() (model crash, OOM,
        # a transient network blip fetching a sample) produces the exact
        # same "written < batch" shape as a genuinely finished dataset, and
        # so does a time-budget deadline cutting the batch short. Only mark
        # complete when the shortfall carried zero errors AND wasn't a
        # deadline cutoff.
        if written < batch and errored == 0 and not deadline_hit:
            log.info("pair complete: %s", pair)
            self.scheduler.mark_complete(pair)
            if self.config.upload:
                self._flush_pair(modality, dataset.dataset_id)
        elif streak >= self.config.max_consecutive_error_batches:
            reason = (f"{streak} consecutive all-error batches"
                      + (f"; last error: {last_error}" if last_error else ""))
            self.scheduler.quarantine(competitor.competitor_id, reason)

        return result

    def _flush_pair(self, modality: str, dataset_id: str) -> None:
        key = (modality, dataset_id)
        if key not in self._dirty:
            return
        dataset = self._dataset_cache.get(key)
        if dataset is None:
            return
        try:
            adapter = adapter_factories()[modality]()
            bench_dir = self.config.output_dir / dataset_id
            mb.upload_predictions(adapter, bench_dir, dataset_id, dataset,
                                  owner=self.config.hf_owner)
        except Exception:
            log.exception("upload failed for %s/%s — will retry next flush",
                          modality, dataset_id)
            return
        self._dirty.discard(key)

    def flush_all(self, force: bool = False) -> None:
        now = time.monotonic()
        due = force or (
            (now - self._last_flush) >= self.config.flush_every_minutes * 60
        )
        if not due:
            return
        if self.config.upload:
            for modality, dataset_id in list(self._dirty):
                self._flush_pair(modality, dataset_id)
        self._last_flush = now
        self.save_state()

    # -- main loop -----------------------------------------------------

    def run_forever(
        self,
        modalities: Iterable[str],
        filters: dict | None = None,
        registry_root: Path | None = None,
    ) -> None:
        filters = filters or {}
        self.load_state()
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        try:
            while not self._stop:
                self._sweep_once(modalities, filters, registry_root)
                if self._stop:
                    break
                if self.scheduler.is_exhausted():
                    log.info("nothing left to do — sleeping %ds",
                              self.config.sleep_when_idle)
                    self._sleep_interruptibly(self.config.sleep_when_idle)
        finally:
            self.flush_all(force=True)

    def _sleep_interruptibly(self, seconds: int) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    def _sweep_once(self, modalities, filters, registry_root) -> None:
        entries = enumerate_all_pairs(modalities, registry_root)
        entries = apply_filters(entries, **filters)
        for modality, competitor, dataset, lang in entries:
            self._dataset_cache[(modality, dataset.dataset_id)] = dataset
        lookup = {
            PairKey(m, c.competitor_id, d.dataset_id, l): (m, c, d, l)
            for m, c, d, l in entries
        }
        self.scheduler.set_pairs(lookup.keys())

        for pair in self.scheduler.sweep():
            if self._stop:
                break
            m, c, d, l = lookup[pair]
            self.process_pair(m, c, d, l)
            self.flush_all()

    # -- one-shot (CI-friendly) mode -------------------------------------

    #: Once less than this much budget remains after finishing a pair,
    #: stop instead of drawing another one — there isn't enough runway
    #: left for a fresh pair (model load + a meaningful slice of samples).
    ONE_SHOT_MIN_CONTINUE_SECONDS = 5 * 60

    def run_one_shot(
        self,
        modalities: Iterable[str],
        filters: dict | None = None,
        registry_root: Path | None = None,
        max_samples: int = 100,
        time_budget_secs: float | None = None,
        seed: object | None = None,
        max_attempts: int = 3,
    ) -> dict:
        """Run one (or, budget permitting, several) randomly chosen pairs.

        Picks ONE eligible, not-yet-complete ``(fighter, dataset, lang)``
        pair uniformly at random (seeded by *seed* for a reproducible
        pick; entropy if *seed* is ``None``), processes up to
        *max_samples* new samples for it, uploads the shard, and returns
        a summary dict.

        **Discovery is lazy, not exhaustive.** The candidate pool for each
        round is *shuffled* (seeded) and then walked one at a time:
        :meth:`pair_is_complete` — a network round-trip plus a full
        ``iter_samples`` scan of the dataset — is only ever called for the
        candidate actually being considered, and stops being called the
        moment a usable (not-complete, not-quarantined) pair is found.
        Discovery cost is O(pairs looked at before landing on one), never
        O(every eligible pair) — a registry with hundreds of pairs must
        not have to complete-check all of them before drawing one.

        *time_budget_secs* (``None`` = unbounded) is wall-clock counted
        from the start of THIS call — it covers the ENTIRE call, including
        discovery (the shuffle-and-walk above) and model load (which
        happens inside the first ``process_pair`` call), not just
        inference. The same deadline is checked between candidates during
        discovery, threaded into :meth:`pair_is_complete` to bound its
        per-candidate scan, and threaded down into ``run_competitor_lang``
        to be checked between samples. If the deadline elapses before any
        pair is ever actually drawn, the call returns with
        ``summary["discovery_bound"] = True`` — a run that found nothing
        to do because it ran out of time to LOOK, which is a distinct
        outcome from ``summary["nothing_to_do"]`` (every candidate was
        checked and genuinely is complete).

        If the chosen pair finishes (completes OR the deadline was still
        far off) with more than :data:`ONE_SHOT_MIN_CONTINUE_SECONDS` of
        budget left, another random pair is drawn and processed too —
        this is what lets a CI job use its whole time slot instead of
        idling after one small pair. With no *time_budget_secs*, runs
        exactly one pair.

        A pair whose fighter hard-fails to load is quarantined (as in the
        forever-mode scheduler) and a different pair is drawn instead —
        the shuffled walk simply continues onto the next candidate — up to
        *max_attempts* real draws (candidates that passed the completeness
        check and were actually handed to ``process_pair``), so one broken
        fighter never fails the whole run — the caller (the CI job) still
        exits 0.
        """
        self.load_state()
        filters = dict(filters or {})
        rng = random.Random(seed) if seed is not None else random.Random()
        deadline = (time.monotonic() + time_budget_secs
                    if time_budget_secs else None)

        summary: dict = {"pairs": [], "written": 0, "errored": 0,
                          "nothing_to_do": False, "discovery_bound": False,
                          "candidates_probed": 0}

        def deadline_passed() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        while True:
            if (deadline is not None and summary["pairs"]
                    and (deadline - time.monotonic())
                    < self.ONE_SHOT_MIN_CONTINUE_SECONDS):
                break
            if deadline_passed():
                # No budget left even to start a fresh discovery round.
                if not summary["pairs"]:
                    summary["discovery_bound"] = True
                    log.info("one-shot: discovery-bound, 0 pairs drawn, "
                              "%d candidate(s) probed", summary["candidates_probed"])
                break

            entries = enumerate_all_pairs(modalities, registry_root)
            entries = apply_filters(entries, **filters)
            lookup = {
                PairKey(m, c.competitor_id, d.dataset_id, l): (m, c, d, l)
                for m, c, d, l in entries
            }
            for _m, _c, d, _l in entries:
                self._dataset_cache[(_m, d.dataset_id)] = d
            self.scheduler.set_pairs(lookup.keys())

            now = time.time()
            candidates = [
                p for p in lookup
                if p not in self.scheduler.completed
                and not self.scheduler.is_quarantined(p.competitor_id, now)
            ]
            if not candidates:
                if not summary["pairs"]:
                    summary["nothing_to_do"] = True
                    log.info("one-shot: nothing to do — every eligible pair is complete")
                break
            rng.shuffle(candidates)

            result = None
            chosen = None
            m = c = d = l = None
            attempts_made = 0
            for cand in candidates:
                if deadline_passed():
                    break
                if self.scheduler.is_quarantined(cand.competitor_id, time.time()):
                    continue  # quarantined by an earlier draw THIS round
                summary["candidates_probed"] += 1
                if self.pair_is_complete(*lookup[cand], deadline=deadline):
                    self.scheduler.mark_complete(cand)
                    continue
                attempts_made += 1
                chosen = cand
                m, c, d, l = lookup[cand]
                log.info("one-shot: attempt %d/%d — chosen pair %s",
                          attempts_made, max_attempts, chosen)
                pair_start = time.monotonic()
                result = self.process_pair(m, c, d, l, batch=max_samples,
                                            deadline=deadline)
                elapsed = time.monotonic() - pair_start
                if result is not None:
                    break
                # process_pair already quarantined this fighter on a hard
                # load failure — the shuffled walk continues onto the next
                # candidate (any remaining pairs for this same fighter are
                # skipped above via the quarantine check).
                chosen = None
                if attempts_made >= max_attempts:
                    break

            if chosen is None:
                # Decide the reason from the deadline NOW, after the walk —
                # not from whether the deadline check inside the loop was
                # what broke it: the deadline can also elapse DURING the
                # last probe itself (pair_is_complete's own scan is what
                # ate the remaining budget) and the walk simply runs out of
                # candidates right after, with no separate "deadline broke
                # me" branch taken. Either way, time genuinely ran out
                # before a usable pair was found — that must still read as
                # discovery-bound, not "nothing to do".
                if not summary["pairs"] and attempts_made == 0 and deadline_passed():
                    summary["discovery_bound"] = True
                    log.info("one-shot: discovery-bound, 0 pairs drawn, "
                              "%d candidate(s) probed", summary["candidates_probed"])
                elif not summary["pairs"] and attempts_made == 0:
                    # Every candidate in this round was complete/quarantined
                    # and none was ever a real draw attempt.
                    summary["nothing_to_do"] = True
                    log.info("one-shot: nothing to do — every eligible pair is complete")
                elif attempts_made:
                    log.warning("one-shot: exhausted %d attempt(s) without a "
                                "successful pair this round", max_attempts)
                break

            written = getattr(result, "written", result)
            errored = getattr(result, "errored", 0)
            repo = results_repo_for(m, d.dataset_id, self.config.hf_owner)
            summary["pairs"].append({
                "pair": str(chosen),
                "written": written,
                "errored": errored,
                "elapsed_secs": round(elapsed, 1),
                "upload_url": f"https://huggingface.co/datasets/{repo}",
            })
            summary["written"] += written
            summary["errored"] += errored

            if deadline is None:
                break  # no time budget to spend on a second pair

        self.save_state()
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m runner.autorun",
        description="Autonomous round-robin prediction runner: cycles every "
        "eligible (fighter, dataset, lang) pair, --batch new samples at a "
        "time, uploading in batches — designed to run forever on a fleet "
        "of hosts while new fighters/datasets get added to the registry.",
    )
    parser.add_argument("--modalities", default=",".join(MODALITIES),
                        help=f"Comma-separated modalities (default: all — {', '.join(MODALITIES)})")
    parser.add_argument("--batch", type=int, default=10,
                        help="New samples per pair per turn (default: 10)")
    parser.add_argument("--flush-every", type=float, default=15.0,
                        help="Minutes between periodic HF uploads of dirty "
                        "shards, besides the immediate upload on pair "
                        "completion (default: 15)")
    parser.add_argument("--sleep-when-idle", type=int, default=300,
                        help="Seconds to sleep when every pair is complete "
                        "or quarantined, before re-checking the registry "
                        "(default: 300)")
    parser.add_argument("--max-consecutive-error-batches", type=int, default=5,
                        help="Consecutive all-error (0 written, some "
                        "errored) batches before a pair's fighter is "
                        "quarantined instead of retried forever (default: 5)")
    parser.add_argument("--output-dir", default="predictions",
                        help="Local root for prediction JSONLs / audio / "
                        "state file (default: predictions)")
    parser.add_argument("--hf-owner", default=HF_OWNER)
    parser.add_argument("--no-upload", action="store_true",
                        help="Write local shards only, never touch HF")
    parser.add_argument("--no-seed", action="store_true",
                        help="Never seed a local shard from an already-"
                        "published HF shard (each host redoes its own work)")
    parser.add_argument("--include", default="",
                        help="Comma-separated globs — only these competitor_ids")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated globs — never these competitor_ids")
    parser.add_argument("--datasets", default="",
                        help="Comma-separated globs — only these dataset_ids "
                        "(default: any eligible dataset)")
    parser.add_argument("--langs", default="",
                        help="Comma-separated BCP-47 tags — only these languages")
    parser.add_argument("--min-size", default=None,
                        choices=SIZE_ORDER, help="Only fighters >= this registry size class")
    parser.add_argument("--max-size", default=None,
                        choices=SIZE_ORDER, help="Only fighters <= this registry size class")
    parser.add_argument("--heavy", action="store_true",
                        help="Only heavyweight fighters (registry size >= "
                        "large, a '<N>b' param-count token >= 1B in the "
                        "id/plugin name, or a known heavy family like "
                        "whisper-large/canary) — for a GPU host")
    parser.add_argument("--light", action="store_true",
                        help="Only non-heavyweight fighters — for a CPU host")
    parser.add_argument("--host-class", choices=("auto", "gpu", "cpu"), default="auto",
                        help="'auto' infers heavy/light from this machine's "
                        "hardware fingerprint when neither --heavy nor "
                        "--light is given explicitly (default: auto)")
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the eligible pairs and exit — no network, no runs")
    parser.add_argument("--one-shot", action="store_true",
                        help="Run ONE randomly-chosen eligible/incomplete pair "
                        "(more, budget permitting) and exit 0 — for a "
                        "scheduled CI job instead of a long-lived daemon")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="[--one-shot] New samples cap for the chosen "
                        "pair, per pair drawn (default: 100)")
    parser.add_argument("--time-budget-secs", type=float, default=None,
                        help="[--one-shot] Wall-clock budget in seconds, "
                        "counted from process start including model load; "
                        "unset = unbounded (single pair, --max-samples cap "
                        "only). When set, another random pair is drawn with "
                        "any remaining budget after one finishes.")
    parser.add_argument("--seed", default=None,
                        help="[--one-shot] Seed for the random pair pick "
                        "(reproducible/debuggable); default: entropy")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="[--one-shot] Random pair draws to try before "
                        "giving up for this round if a fighter hard-fails "
                        "to load (default: 3)")
    return parser


def _resolve_heavy_light(args) -> tuple[bool, bool]:
    if args.heavy or args.light:
        return args.heavy, args.light
    if args.host_class == "gpu":
        return True, False
    if args.host_class == "cpu":
        return False, True
    if args.host_class == "auto":
        from runner.perf import hw_fingerprint
        is_gpu = hw_fingerprint().get("host_class") == "gpu"
        return is_gpu, not is_gpu
    return False, False


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        stream=sys.stdout,
    )

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    heavy_only, light_only = _resolve_heavy_light(args)
    filters = dict(
        include=[s.strip() for s in args.include.split(",") if s.strip()] or None,
        exclude=[s.strip() for s in args.exclude.split(",") if s.strip()] or None,
        datasets=[s.strip() for s in args.datasets.split(",") if s.strip()] or None,
        langs=[s.strip() for s in args.langs.split(",") if s.strip()] or None,
        min_size=size_rank(args.min_size),
        max_size=size_rank(args.max_size),
        heavy_only=heavy_only,
        light_only=light_only,
    )
    registry_root = Path(args.registry_root) if args.registry_root else None

    # Log every fighter's heavy/light classification once, before any
    # filtering — the heuristic is a name/id pattern match (see
    # is_heavy's docstring), so a misclassification must be visible in the
    # log rather than silently routing a fighter to the wrong host class.
    log_classifications(enumerate_all_pairs(modalities, registry_root))

    if args.dry_run:
        entries = enumerate_all_pairs(modalities, registry_root)
        entries = apply_filters(entries, **filters)
        for modality, competitor, dataset, lang in entries:
            print(f"{modality:<10} {competitor.competitor_id:<32} "
                  f"{dataset.dataset_id:<28} {lang}")
        print(f"\n{len(entries)} eligible pair(s).")
        return 0

    config = AutoRunConfig(
        output_dir=Path(args.output_dir),
        batch=args.batch,
        flush_every_minutes=args.flush_every,
        sleep_when_idle=args.sleep_when_idle,
        hf_owner=args.hf_owner,
        upload=not args.no_upload,
        seed_from_hf=not args.no_seed,
        max_consecutive_error_batches=args.max_consecutive_error_batches,
    )
    lister = None if args.no_seed else HubLister()
    runner = AutoRunner(config, lister=lister)

    if args.one_shot:
        summary = runner.run_one_shot(
            modalities, filters, registry_root,
            max_samples=args.max_samples,
            time_budget_secs=args.time_budget_secs,
            seed=args.seed,
            max_attempts=args.max_attempts,
        )
        if summary["nothing_to_do"]:
            print("one-shot: nothing to do — every eligible pair is already complete")
            return 0
        if summary["discovery_bound"]:
            print(f"one-shot: discovery-bound, 0 pairs drawn, "
                  f"{summary['candidates_probed']} candidate(s) probed "
                  f"— ran out of time-budget before finding a usable pair")
            return 0
        for entry in summary["pairs"]:
            print(f"one-shot: pair={entry['pair']} written={entry['written']} "
                  f"errored={entry['errored']} elapsed={entry['elapsed_secs']}s "
                  f"upload={entry['upload_url']}")
        print(f"one-shot: {len(summary['pairs'])} pair(s), "
              f"{summary['written']} written, {summary['errored']} errored")
        return 0

    runner.run_forever(modalities, filters, registry_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
