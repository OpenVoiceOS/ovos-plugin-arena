"""Which wake-word boards the site lists.

A board that compares real models is listed. A board where one trained model
meets the generic fighters, or where the test set is too small to separate
anyone, stays on disk and is left out of ``index.json``.

Everything derives from the registry and the assembled boards:

* A trained model is a ``wake_word`` fighter that is not generic. A generic
  fighter works for any phrase without training on it, and says so in its
  registry ``types``: zero-shot, few-shot or keyword-match.
* A clip board qualifies when its test set holds at least ``MIN_POSITIVES``
  positive clips and ``MIN_TRAINED`` trained models are ranked on it. Rows of a
  model trained on the corpus never reach a board, so ranked means ranked after
  that exclusion.
* A phrase is featured when a qualifying clip board ranks a trained model whose
  engine is not WakeHuBERT.
* A clip board is listed when it qualifies and its phrase is featured.
* The ``ww_stream`` league holds only false wake-up sets: streams with no wake
  word. Their board ranks only the fighters of featured phrases, and is listed
  when a listed clip board shares its language. Any other ``ww_stream`` board
  is hidden.
* A clip set with no wake word (``mlsw-negatives-<lang>``) is a pool of
  negatives for the clip boards. It has no positives, so its board is never
  listed.

Before any board of a phrase exists, ``registry_featured_phrases`` applies the
same rule to the registry alone, so the generators write generic fighters only
for the phrases the site will show.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from registry.loaders import wakeword_key
from registry.schemas import CompetitorDef, DatasetDef

MIN_POSITIVES = 50
MIN_TRAINED = 2

GENERIC_TYPES = frozenset({"zero-shot", "few-shot", "keyword-match"})
WAKEHUBERT_PLUGIN = "ovos-ww-plugin-wakeforge"

BoardKey = tuple[str, str]


def is_trained_model(competitor: CompetitorDef) -> bool:
    return not GENERIC_TYPES.intersection(competitor.types)


def phrase_of(competitor: CompetitorDef) -> str | None:
    hotwords = (competitor.config or {}).get("hotwords") or {}
    return wakeword_key(next(iter(hotwords))) if hotwords else None


def is_false_wake_set(dataset: DatasetDef) -> bool:
    """A stream set with no wake word: it measures false wake-ups only."""
    return dataset.modality == "ww_stream" and not dataset.wakeword


def primary_language(lang: str) -> str:
    return lang.split("-")[0].lower()


def _qualifies(positives: int | None, trained: list[CompetitorDef]) -> bool:
    return (positives or 0) >= MIN_POSITIVES and len(trained) >= MIN_TRAINED


def _non_wakehubert_phrases(trained: Iterable[CompetitorDef]) -> set[str]:
    return {
        phrase for c in trained
        if c.plugin != WAKEHUBERT_PLUGIN and (phrase := phrase_of(c))
    }


@dataclass(frozen=True)
class WakeWordListing:
    """``(dataset_id, lang)`` keys of the listed and unlisted clip boards,
    and the featured phrases."""

    clip_boards: frozenset[BoardKey]
    unlisted_clip_boards: frozenset[BoardKey]
    featured_phrases: frozenset[str]
    false_wake_sets: frozenset[str] = frozenset()

    def is_hidden(self, modality: str, dataset_id: str, lang: str) -> bool:
        if modality == "wake_word":
            return (dataset_id, lang) in self.unlisted_clip_boards
        if modality == "ww_stream":
            return dataset_id not in self.false_wake_sets or primary_language(lang) not in {
                primary_language(board_lang) for _, board_lang in self.clip_boards}
        return False

    def ranks_on_false_wake_set(self, competitor: CompetitorDef) -> bool:
        """Whether a false wake-up board ranks *competitor*."""
        return phrase_of(competitor) in self.featured_phrases


def build_listing(
    clip_boards: Iterable[dict[str, Any]],
    competitors: Iterable[CompetitorDef],
    false_wake_sets: Iterable[str] = (),
) -> WakeWordListing:
    """*clip_boards* are ``benchmark-wake_word-*`` payloads; *false_wake_sets*
    are the ids of the registered false wake-up sets."""
    by_id = {c.competitor_id: c for c in competitors}
    qualifying: dict[BoardKey, set[str]] = {}
    unlisted: set[BoardKey] = set()
    featured: set[str] = set()
    for board in clip_boards:
        key = (board["dataset_id"], board["lang"])
        trained = [
            by_id[e["competitor_id"]] for e in board.get("entries", [])
            if not e.get("unranked") and e["competitor_id"] in by_id
            and is_trained_model(by_id[e["competitor_id"]])
        ]
        if not _qualifies(board.get("positives"), trained):
            unlisted.add(key)
            continue
        qualifying[key] = {phrase for c in trained if (phrase := phrase_of(c))}
        featured.update(_non_wakehubert_phrases(trained))
    listed = {key for key, phrases in qualifying.items() if phrases & featured}
    unlisted.update(qualifying.keys() - listed)
    return WakeWordListing(
        clip_boards=frozenset(listed),
        unlisted_clip_boards=frozenset(unlisted),
        featured_phrases=frozenset(featured),
        false_wake_sets=frozenset(false_wake_sets),
    )


def load_listing(
    data_dir: Path,
    competitors: Iterable[CompetitorDef],
    datasets: Iterable[DatasetDef] = (),
) -> WakeWordListing:
    """The listing of the boards in *data_dir*; *datasets* name the false
    wake-up sets."""
    paths = sorted(data_dir.glob("benchmark-wake_word-*.json"))
    boards = [json.loads(path.read_text()) for path in paths]
    return build_listing(
        boards, competitors, (d.dataset_id for d in datasets if is_false_wake_set(d)))


def registry_featured_phrases(
    datasets: Iterable[DatasetDef],
    competitors: Iterable[CompetitorDef],
    positives: dict[BoardKey, int | None],
) -> frozenset[str]:
    """The featured phrases the boards will give, predicted from the registry.

    A clip set stands in for its board: its trained models are the registered
    ones of its phrase and language that are not trained on it. *positives*
    holds the count of each board that exists; a set with no board yet counts
    its sample cap, which the set generators only write for sets that hold at
    least ``MIN_POSITIVES`` clips.
    """
    trained_models = [c for c in competitors if is_trained_model(c)]
    featured: set[str] = set()
    for dataset in datasets:
        if dataset.role != "eval" or not dataset.wakeword:
            continue
        phrase = wakeword_key(dataset.wakeword)
        language = primary_language(dataset.lang)
        trained = [
            c for c in trained_models
            if phrase_of(c) == phrase and dataset.dataset_id not in c.trained_on
            and (not c.langs or language in {primary_language(lang) for lang in c.langs})
        ]
        key = (dataset.dataset_id, dataset.lang)
        cap = dataset.sample_policy.max_samples if dataset.sample_policy else None
        if _qualifies(positives[key] if key in positives else cap, trained):
            featured.update(_non_wakehubert_phrases(trained))
    return frozenset(featured)


def board_positives(data_dir: Path) -> dict[BoardKey, int | None]:
    """The positive count of every clip board in *data_dir*."""
    counts: dict[BoardKey, int | None] = {}
    for path in sorted(data_dir.glob("benchmark-wake_word-*.json")):
        board = json.loads(path.read_text())
        counts[(board["dataset_id"], board["lang"])] = board.get("positives")
    return counts
