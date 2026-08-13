"""
G2P benchmark adapter (§A6) — text-in/IPA-out, closest in shape to
:mod:`runner.stt_bench` (STT is audio-in/text-out; g2p drops the audio leg
entirely — a word or short utterance in, an IPA string out).

Unlike the audio modalities, g2p samples are plain (grapheme, gold_ipa) pairs
pulled straight from a lexicon file (CMUdict-style flat dict, or a HF CSV
lexicon), so this module does not reuse :mod:`runner.media_bench`'s
audio-streaming machinery. It borrows the same conventions instead:
resumable JSONL rows, ``runner/perf.py`` latency/perf capture, and one HF
dataset repo per benchmark (``OpenVoiceOS/ovos-g2p-bench-<dataset_id>``).

A g2p run:

1. loads the eval dataset definition + eligible registry ``g2p`` fighters
   (a fighter competes on a language only when its own ``langs`` matches,
   §P2 — the arena owns no thresholds, the plugin declares its own support);
2. instantiates each fighter's real OVOS ``Grapheme2PhonemePlugin`` from its
   ``config.g2p`` mycroft.conf fragment;
3. calls ``get_ipa`` (falling back to ``utterance2ipa`` for multi-word
   entries) on every sample, writing resumable §3.2 rows to
   ``<output_dir>/<dataset_id>/g2p/<lang>/<competitor_id>.jsonl``
   (``reference_text`` = gold IPA, ``prediction`` = hypothesis IPA — PER is
   left for the arena to compute on ingest, ``arena.metrics.score_g2p``, so
   the dataset stays the single source of truth for the scoring formula,
   same convention as STT's WER);
4. on ``--upload``, publishes to HF exactly like the other benches.

This is a vote-less league (§A6): there is no battle/ELO wiring here at all,
only the benchmark board path.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

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
from runner.perf import hw_fingerprint, measure_call

log = logging.getLogger("g2p-bench")


# ---------------------------------------------------------------------------
# Sample access
# ---------------------------------------------------------------------------


def _load_arpabet2ipa() -> dict:
    from ovos_utils.lang.phonemes import arpabet2ipa

    return arpabet2ipa


def _arpa_to_ipa(phones: str) -> str:
    """Space-separated ARPABET phones (CMUdict style, e.g. 'HH AH0 L OW1')
    -> a bare IPA string, via the same ovos_utils table every installed
    ARPA-native g2p fighter uses on its own output
    (``Grapheme2PhonemePlugin.get_ipa`` fallback)."""
    table = _load_arpabet2ipa()
    norm = lambda p: "".join(c for c in p if not c.isdigit())  # strip stress digits
    out = []
    for phone in phones.split():
        key = norm(phone)
        if key in table:
            out.append(table[key])
    return "".join(out)


def iter_cmudict_samples(url: str, max_samples: int) -> Iterator[tuple[str, dict]]:
    """CMUdict-style flat file: 'word<space/tab>PHONE1 PHONE2 ...' per line,
    '#'-prefixed comments, '(2)'-suffixed alternate pronunciations."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    n = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";;;")):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        word, phones = parts
        # Drop CMUdict's "(2)"/"(3)" alternate-pronunciation suffix — keep
        # only the primary entry per grapheme (the eval gold is one gold IPA
        # per word, not every attested variant).
        base_word = word.split("(", 1)[0]
        if "(" in word:
            continue
        ipa = _arpa_to_ipa(phones)
        if not ipa:
            continue
        sample_id = base_word
        yield sample_id, {"grapheme": base_word, "reference_ipa": ipa}
        n += 1
        if n >= max_samples:
            return


# orthography2ipa language tag -> tugalex/portuguese_phonetic_lexicon
# ``region_code`` value. Mirrors g2p_bench's own ``_TUGALEX_REGION_MAP``
# (~/AgentWorkspaces/ml/g2p_bench/g2p_bench/datasets.py) so every one of the
# HF dataset's 5 wired dialect variants filters to the same rows g2p_bench
# scores.
_TUGALEX_REGION_MAP = {
    "pt-PT": "lbx",
    "pt-BR": "rjx",
    "pt-AO": "lda",
    "pt-MZ": "mpx",
    "pt-TL": "dli",
}


def iter_hf_lexicon_samples(
    dataset_def, lang: str, revision: str, max_samples: int
) -> Iterator[tuple[str, dict]]:
    """Plain HF-hosted CSV lexicon (e.g. TigreGotico/portuguese_phonetic_lexicon):
    ``reference_fields`` maps 'grapheme'/'ipa'/'region' onto source columns."""
    from huggingface_hub import hf_hub_download

    fields = dataset_def.reference_fields or {}
    grapheme_col = fields.get("grapheme", "word")
    ipa_col = fields.get("ipa", "phones")
    region_col = fields.get("region")
    src = dataset_def.source

    # TigreGotico/portuguese_phonetic_lexicon-shaped repos ship a single
    # dataset.csv rather than a `datasets`-loadable config; download it
    # directly and filter by region_code client-side, per g2p_bench's own
    # per-dialect mapping (see the dataset's registry notes).
    path = hf_hub_download(
        src.hf_id, "dataset.csv", repo_type="dataset", revision=revision,
    )
    region = _TUGALEX_REGION_MAP.get(lang) if region_col else None

    n = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if region_col and row.get(region_col) != region:
                continue
            grapheme = row.get(grapheme_col)
            ipa = row.get(ipa_col)
            if not grapheme or not ipa:
                continue
            sample_id = row.get("id") or f"{grapheme}-{i}"
            yield sample_id, {"grapheme": grapheme, "reference_ipa": ipa}
            n += 1
            if n >= max_samples:
                return


def iter_wikipron_samples(url: str, max_samples: int) -> Iterator[tuple[str, dict]]:
    """WikiPron TSV export (CUNY-CL/wikipron): ``word<TAB>ipa`` per line, no
    header. Community-curated (Wiktionary editors), crowd-scraped tier --
    see :mod:`orthography2ipa.scripts.benchmark`'s ``_WIKIPRON_FILES``."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    n = 0
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word = parts[0].strip()
        ipa = parts[1].strip()
        if not word or not ipa:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        yield word, {"grapheme": word, "reference_ipa": ipa}
        n += 1
        if n >= max_samples:
            return


def iter_mirandese_samples(
    url: str, dialect: str, max_samples: int
) -> Iterator[tuple[str, dict]]:
    """TigreGotico/mirandese_g2p (``mwl_dataset.tsv``): ``dialect<TAB>word<TAB>ipa``
    with a header row, filtered to *dialect* ("central" | "raiano" |
    "sendinese"). Native-speaker-collected gold (MdMV) -- see g2p_bench's
    ``_MirandeseG2PDataset``."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    n = 0
    for line in text.strip().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        row_dialect, word, ipa = (p.strip() for p in parts)
        if row_dialect != dialect or not word or not ipa:
            continue
        yield word, {"grapheme": word, "reference_ipa": ipa}
        n += 1
        if n >= max_samples:
            return


def iter_samples(
    dataset_def, lang: str, revision: str, max_samples: int
) -> Iterator[tuple[str, dict]]:
    src = dataset_def.source
    if src.type == "url" and src.format == "wikipron_tsv":
        yield from iter_wikipron_samples(src.url, max_samples)
    elif src.type == "url" and src.format == "mirandese_tsv":
        dialect = (dataset_def.reference_fields or {}).get("dialect")
        if not dialect:
            raise ValueError(
                "g2p bench: mirandese_tsv source requires reference_fields.dialect"
            )
        yield from iter_mirandese_samples(src.url, dialect, max_samples)
    elif src.type == "url":
        yield from iter_cmudict_samples(src.url, max_samples)
    elif src.type == "huggingface":
        yield from iter_hf_lexicon_samples(dataset_def, lang, revision, max_samples)
    else:
        raise ValueError(f"g2p bench: unsupported source type {src.type!r}")


# ---------------------------------------------------------------------------
# Fighter
# ---------------------------------------------------------------------------


def load_engine(competitor, lang: str):
    from ovos_plugin_manager.g2p import load_g2p_plugin

    g2p_cfg = competitor.config.get("g2p", {})
    module = g2p_cfg.get("module") or competitor.plugin
    plugin_cfg = dict(g2p_cfg.get(module, {}))
    clazz = load_g2p_plugin(module)
    return clazz({"lang": lang, "module": module, **plugin_cfg})


def predict(engine, sample: dict, lang: str) -> dict:
    """Run one grapheme through the fighter -> hypothesis IPA + timing."""
    from ovos_plugin_manager.templates.g2p import OutOfVocabulary

    grapheme = sample["grapheme"]
    start = time.perf_counter()
    try:
        if " " in grapheme.strip():
            phones = engine.utterance2ipa(grapheme, lang, ignore_oov=True) or []
        else:
            phones = engine.get_ipa(grapheme, lang, ignore_oov=True) or []
        prediction = "".join(phones)
        oov = not phones
    except OutOfVocabulary:
        prediction = ""
        oov = True
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "reference_text": sample["reference_ipa"],
        "prediction": prediction,
        "extras": {"oov": oov},
        "latency_ms": round(latency_ms, 3),
    }


# ---------------------------------------------------------------------------
# Row contract
# ---------------------------------------------------------------------------


def make_row(competitor, dataset_id: str, lang: str, sample_id: str,
             dataset_revision: str, fields: dict) -> dict:
    row = {
        "competitor_id": competitor.competitor_id,
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "lang": lang,
        "modality": "g2p",
        "plugin_id": competitor.plugin or competitor.competitor_id,
        "plugin_version": _plugin_version(competitor),
        "runner_version": f"ovos-plugin-arena=={ARENA_VERSION}",
        "created_at": _now_iso(),
    }
    row.update(fields)
    return row


def _plugin_version(competitor) -> str:
    import importlib.metadata

    plugin = competitor.plugin
    if not plugin:
        return ""
    try:
        return f"{plugin}=={importlib.metadata.version(plugin)}"
    except importlib.metadata.PackageNotFoundError:
        return plugin


def _lang_matches(competitor_lang: str, dataset_lang: str) -> bool:
    a, b = competitor_lang.lower(), dataset_lang.lower()
    return a == b or a.split("-")[0] == b.split("-")[0]


def competitor_langs(competitor, dataset_lang: str) -> list[str]:
    if not competitor.langs:
        return [dataset_lang]
    return [dataset_lang] if any(
        _lang_matches(cl, dataset_lang) for cl in competitor.langs
    ) else []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace, dataset_id: str) -> int:
    dataset_def = load_dataset("g2p", dataset_id)
    lang = dataset_def.lang
    revision = args.revision
    if dataset_def.source.type == "huggingface":
        revision = resolve_revision(dataset_def.source.hf_id, args.revision)

    competitors = [c for c in list_competitors("g2p")
                   if not args.competitors or c.competitor_id in args.competitors]

    out_dir = Path(args.output) / dataset_id / "g2p" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    ran_any = False
    for competitor in competitors:
        if not competitor_langs(competitor, lang):
            log.info("Skipping %s: no lang match for %s",
                      competitor.competitor_id, lang)
            continue
        out_path = out_dir / f"{competitor.competitor_id}.jsonl"
        already = done_samples(out_path)
        try:
            engine = load_engine(competitor, lang)
        except Exception as exc:
            log.error("Could not load %s: %s", competitor.competitor_id, exc)
            continue

        hw = hw_fingerprint()
        with open(out_path, "a", encoding="utf-8") as out:
            for sample_id, sample in iter_samples(
                dataset_def, lang, revision, args.max_samples
            ):
                if sample_id in already:
                    continue
                try:
                    result, elapsed_ms, peak_rss_mb = measure_call(
                        lambda: predict(engine, sample, lang)
                    )
                except Exception as exc:
                    log.warning("%s failed on %r: %s",
                                competitor.competitor_id, sample_id, exc)
                    continue
                fields = dict(result)
                fields["elapsed_ms"] = round(elapsed_ms, 3)
                if peak_rss_mb is not None:
                    fields["peak_rss_mb"] = peak_rss_mb
                fields.setdefault("hw", hw)
                row = make_row(competitor, dataset_id, lang, sample_id,
                                revision, fields)
                out.write(json.dumps(row) + "\n")
                ran_any = True

    if args.upload:
        _upload(dataset_id, lang, out_dir, args)
    return 0 if ran_any or not competitors else 1


def _upload(dataset_id: str, lang: str, out_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    repo = results_repo_for("g2p", dataset_id, args.hf_owner or HF_OWNER)
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    for path in sorted(out_dir.glob("*.jsonl")):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"predictions/{split_name(lang)}/{path.name}",
            repo_id=repo,
            repo_type="dataset",
        )
        log.info("Uploaded %s -> %s", path, repo)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a g2p benchmark (§A6)")
    p.add_argument("--dataset", required=True, help="registry g2p dataset_id")
    p.add_argument("--competitors", nargs="*", default=None,
                    help="competitor_ids to run (default: all eligible)")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--output", default="results")
    p.add_argument("--revision", default="main")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-owner", default=None)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = build_arg_parser().parse_args(argv)
    return run(args, args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
