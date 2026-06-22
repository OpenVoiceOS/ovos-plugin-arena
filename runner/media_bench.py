"""
Shared engine for the audio-modality benchmark scripts (STT, wake word, TTS).

Every benchmark stays one dedicated script under ``benchmarks/`` (§P4); this
module is the machinery they share, mirroring ``runner.intent_bench`` for the
intent leagues.  An audio benchmark run:

1. loads the eval dataset definition and the registry competitors for the
   modality, pinning the dataset revision;
2. instantiates each competitor's real OVOS plugin from its shippable
   ``mycroft.conf`` fragment (P2/P4 — the arena owns no thresholds; the
   plugin decides), one instance per (competitor, language);
3. runs it over the eval samples, writing resumable §3.2 rows to
   ``<output_dir>/<dataset_id>/<modality>/<lang>/<competitor_id>.jsonl``;
4. on ``--upload``, publishes **one HF dataset repo per modality** —
   ``OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`` — with prediction files
   at ``predictions/<lang>/<competitor_id>.jsonl`` (plus any synthesised audio
   for TTS) and a generated dataset card declaring one split per language.

The modality-specific behaviour — how a sample is loaded and how a plugin
turns it into a prediction — lives in a :class:`MediaBenchAdapter`; the STT,
wake-word and TTS scripts each supply one.  Everything in this module is pure
Python with no audio/plugin imports at module load, so the row contract,
resume and publishing logic stay unit-testable without the engines installed.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from arena.version import __version__ as ARENA_VERSION
from registry.loaders import list_competitors, load_dataset
from runner.intent_bench import (
    HF_OWNER,
    _now_iso,
    done_samples,
    resolve_revision,
    results_repo_for,
    split_name,
)

log = logging.getLogger("media-bench")


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------


class MediaBenchAdapter:
    """How one audio modality loads samples and runs its plugins.

    Subclasses set :attr:`modality` and implement :meth:`iter_samples`,
    :meth:`load_engine` and :meth:`predict`.  The heavy audio / OVOS imports
    MUST stay inside these methods (lazy) so this module imports cleanly in CI.
    """

    modality: str = ""
    #: HF dataset card tags and one-line task description for the modality.
    card_tags: Tuple[str, ...] = ()
    card_task: str = ""

    def competitor_langs(self, competitor, dataset_langs: List[str]) -> List[str]:
        """Languages to benchmark this competitor in (intersection with data).

        Matches on the primary subtag too, so a plugin advertising ``en`` runs
        on an ``en-US`` corpus (and vice-versa) — STT/TTS plugins commonly
        declare primary-language tags while datasets use full BCP-47 tags.
        """
        if not competitor.langs:
            return list(dataset_langs)
        return [dl for dl in dataset_langs
                if any(_lang_matches(cl, dl) for cl in competitor.langs)]

    def iter_samples(
        self, dataset_def, lang: str, revision: str, max_samples: int
    ) -> Iterator[Tuple[str, dict]]:
        """Yield ``(sample_id, sample)`` for one language of the eval set."""
        raise NotImplementedError

    def load_engine(self, competitor, lang: str):
        """Instantiate the competitor's OVOS plugin for one language."""
        raise NotImplementedError

    def predict(self, engine, sample: dict, ctx: "PredictContext") -> dict:
        """Run one sample → modality-specific §3.2 row fields.

        Returns a dict merged into the base row: e.g. ``prediction``,
        ``reference_text``/``label``/``input_text``, ``audio_url``,
        ``confidence``, ``latency_ms``.
        """
        raise NotImplementedError


class PredictContext:
    """Per-(competitor, lang) context handed to :meth:`MediaBenchAdapter.predict`.

    Carries the bits an adapter needs to name and locate side artifacts (e.g.
    where TTS writes a synthesised clip and the HF URL it will resolve to).
    """

    def __init__(
        self,
        competitor,
        lang: str,
        dataset_id: str,
        modality: str,
        audio_dir: Path,
        results_repo: str,
    ):
        self.competitor = competitor
        self.lang = lang
        self.dataset_id = dataset_id
        self.modality = modality
        self.audio_dir = audio_dir
        self.results_repo = results_repo

    def hf_audio_url(self, rel_path: str) -> str:
        """Resolve URL for an audio file uploaded under ``audio/<rel_path>``."""
        return (f"https://huggingface.co/datasets/{self.results_repo}"
                f"/resolve/main/audio/{rel_path}")


# ---------------------------------------------------------------------------
# Row contract (§3.2)
# ---------------------------------------------------------------------------


def make_row(
    competitor,
    dataset_id: str,
    lang: str,
    sample_id: str,
    dataset_revision: str,
    fields: dict,
) -> dict:
    """Build one §3.2 prediction row from base metadata + adapter ``fields``."""
    row = {
        "competitor_id": competitor.competitor_id,
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "lang": lang,
        "modality": competitor.modality.value,
        "plugin_id": competitor.plugin or competitor.competitor_id,
        "plugin_version": _plugin_version(competitor),
        "runner_version": f"ovos-plugin-arena=={ARENA_VERSION}",
        "created_at": _now_iso(),
    }
    row.update(fields)
    return row


def _plugin_version(competitor) -> str:
    """``<plugin>==<installed version>`` for §3.2 reproducibility, best-effort."""
    import importlib.metadata

    plugin = competitor.plugin
    if not plugin:
        return ""
    try:
        return f"{plugin}=={importlib.metadata.version(plugin)}"
    except importlib.metadata.PackageNotFoundError:
        return plugin


# ---------------------------------------------------------------------------
# Fighter selection
# ---------------------------------------------------------------------------


def load_plugin_class(loader, name: str):
    """Resolve a plugin class, tolerating dash/underscore entry-point naming.

    Some OVOS plugins register their entry point with underscores
    (``ovos_tts_plugin_espeakng``) while the registry / pip name uses dashes;
    try both forms before giving up.
    """
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        clazz = loader(candidate)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"plugin not found: {name}")


def _primary(tag: str) -> str:
    """Primary language subtag, lowercased: ``pt-BR`` → ``pt``."""
    return tag.replace("_", "-").split("-")[0].lower()


def _lang_matches(a: str, b: str) -> bool:
    """True if two BCP-47 tags are equal or share a primary subtag."""
    return a.lower() == b.lower() or _primary(a) == _primary(b)


def competitors_for(modality: str, wanted: Optional[set] = None) -> list:
    """Registry competitors for *modality*, optionally filtered to *wanted* ids."""
    comps = list_competitors(modality)
    if wanted:
        comps = [c for c in comps if c.competitor_id in wanted]
    return comps


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def run_competitor_lang(
    adapter: MediaBenchAdapter,
    competitor,
    dataset_id: str,
    lang: str,
    eval_def,
    revision: str,
    out_path: Path,
    audio_dir: Path,
    results_repo: str,
    max_samples: int = 0,
) -> int:
    """Run one fighter over one language of the eval set, resumably."""
    done = done_samples(out_path)
    engine = None
    ctx = PredictContext(
        competitor, lang, dataset_id, adapter.modality, audio_dir, results_repo
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for sample_id, sample in adapter.iter_samples(
            eval_def, lang, revision, max_samples
        ):
            if sample_id in done:
                continue
            if engine is None:  # lazy: only pay model load if work remains
                log.info("  loading %s for %s", competitor.competitor_id, lang)
                engine = adapter.load_engine(competitor, lang)
            try:
                fields = adapter.predict(engine, sample, ctx)
            except Exception as exc:
                log.warning("    %s/%s sample %s failed: %s",
                            competitor.competitor_id, lang, sample_id, exc)
                continue
            row = make_row(
                competitor, dataset_id, lang, sample_id, revision, fields
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                fh.flush()
                log.info("    %s/%s: %d rows", competitor.competitor_id, lang,
                         written)
    log.info("  %s/%s: wrote %d rows", competitor.competitor_id, lang, written)
    return written


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def dataset_card(
    adapter: MediaBenchAdapter, dataset_id: str, eval_def, langs: List[str]
) -> str:
    """Generate the HF dataset card for one modality's predictions repo."""
    configs = "\n".join(
        f"  - split: {split_name(lang)}\n"
        f"    path: predictions/{lang}/*.jsonl"
        for lang in sorted(langs)
    )
    tags = "\n".join(f"  - {t}" for t in
                     ("openvoiceos", "benchmark", "predictions", *adapter.card_tags))
    return f"""---
license: apache-2.0
tags:
{tags}
pretty_name: OVOS {adapter.modality} bench — {dataset_id}
configs:
- config_name: default
  data_files:
{configs}
---

# OVOS `{adapter.modality}` bench — `{dataset_id}`

{adapter.card_task} predictions of the registered
[OVOS Plugin Arena](https://github.com/OpenVoiceOS/ovos-plugin-arena)
`{adapter.modality}` fighters over
[`{eval_def.source.hf_id}`](https://huggingface.co/datasets/{eval_def.source.hf_id}).

One dedicated repo per modality; one dataset split per language; one JSONL
file per fighter under `predictions/<lang>/<competitor_id>.jsonl`. Rows follow
the arena §3.2 contract (pinned `dataset_revision`, `plugin_version`,
`latency_ms`). Produced by the reproducible benchmark script in the arena repo;
the arena's `assemble` workflow turns these rows into benchmark boards, blind
battle pools and a benchmark-seeded ELO ladder.

Funded by the [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme.
"""


def upload_predictions(
    adapter: MediaBenchAdapter,
    bench_dir: Path,
    dataset_id: str,
    eval_def,
    owner: str = HF_OWNER,
) -> None:
    """Upload ``<bench_dir>/<modality>/…`` (predictions + audio) to its HF repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    modality_dir = bench_dir / adapter.modality
    if not modality_dir.is_dir():
        log.warning("Nothing to upload: %s does not exist", modality_dir)
        return
    repo = results_repo_for(adapter.modality, dataset_id, owner)
    langs = sorted(d.name for d in (modality_dir / "predictions").iterdir()
                   if d.is_dir()) if (modality_dir / "predictions").is_dir() else []
    try:
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
    except Exception as exc:
        log.warning("create_repo(%s) refused (%s) — uploading anyway", repo, exc)
    api.upload_file(
        path_or_fileobj=dataset_card(adapter, dataset_id, eval_def, langs).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    log.info("Uploading %s → %s (%d langs)", modality_dir, repo, len(langs))
    api.upload_folder(
        folder_path=str(modality_dir),
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"bench: refresh {adapter.modality} predictions",
    )


# ---------------------------------------------------------------------------
# Entry point shared by the benchmark scripts
# ---------------------------------------------------------------------------


def run_benchmark(
    adapter: MediaBenchAdapter,
    dataset_id: str,
    description: str,
    argv=None,
) -> int:
    """Drive a full audio-modality benchmark run for one dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default=dataset_id,
                        help=f"Registry dataset id (default: {dataset_id})")
    parser.add_argument("--competitors", default="",
                        help="Comma-separated competitor ids (default: all)")
    parser.add_argument("--langs", default="",
                        help="Comma-separated languages (default: dataset langs)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap samples per language (smoke runs)")
    parser.add_argument("--output-dir", default="predictions",
                        help="Local root for prediction JSONLs / audio")
    parser.add_argument("--upload", action="store_true",
                        help="Upload predictions to the per-modality HF repo")
    parser.add_argument("--hf-owner", default=HF_OWNER)
    args = parser.parse_args(argv)
    dataset_id = args.dataset

    eval_def = load_dataset(adapter.modality, dataset_id)
    revision = resolve_revision(eval_def.source.hf_id, eval_def.source.revision)
    log.info("Dataset %s @ %s", eval_def.source.hf_id, revision[:12])

    wanted = {c.strip() for c in args.competitors.split(",") if c.strip()} or None
    competitors = competitors_for(adapter.modality, wanted)
    if wanted:
        missing = wanted - {c.competitor_id for c in competitors}
        if missing:
            log.error("Unknown competitors: %s", ", ".join(sorted(missing)))
            return 1

    dataset_langs = [s.strip() for s in args.langs.split(",") if s.strip()] or (
        eval_def.langs or [eval_def.lang]
    )

    bench_dir = Path(args.output_dir) / dataset_id
    results_repo = results_repo_for(adapter.modality, dataset_id, args.hf_owner)
    for competitor in competitors:
        log.info("Fighter %s [%s]", competitor.competitor_id, adapter.modality)
        for lang in adapter.competitor_langs(competitor, dataset_langs):
            out_path = (bench_dir / adapter.modality / "predictions" / lang
                        / f"{competitor.competitor_id}.jsonl")
            audio_dir = bench_dir / adapter.modality / "audio"
            try:
                run_competitor_lang(
                    adapter, competitor, dataset_id, lang, eval_def, revision,
                    out_path, audio_dir, results_repo,
                    max_samples=args.max_samples,
                )
            except Exception:
                log.exception("  %s/%s failed", competitor.competitor_id, lang)

    if args.upload:
        upload_predictions(adapter, bench_dir, dataset_id, eval_def,
                           owner=args.hf_owner)
    return 0
