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
* A streaming board is listed when its clip set's board is listed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from registry.loaders import wakeword_key
from registry.schemas import CompetitorDef

MIN_POSITIVES = 50
MIN_TRAINED = 2

GENERIC_TYPES = frozenset({"zero-shot", "few-shot", "keyword-match"})
WAKEHUBERT_PLUGIN = "ovos-ww-plugin-wakeforge"
STREAM_PREFIX = "ww_stream_"

BoardKey = tuple[str, str]


def is_trained_model(competitor: CompetitorDef) -> bool:
    return not GENERIC_TYPES.intersection(competitor.types)


def phrase_of(competitor: CompetitorDef) -> str | None:
    hotwords = (competitor.config or {}).get("hotwords") or {}
    return wakeword_key(next(iter(hotwords))) if hotwords else None


@dataclass(frozen=True)
class WakeWordListing:
    """``(dataset_id, lang)`` keys of the listed and unlisted clip boards,
    and the featured phrases."""

    clip_boards: frozenset[BoardKey]
    unlisted_clip_boards: frozenset[BoardKey]
    featured_phrases: frozenset[str]

    def is_hidden(self, modality: str, dataset_id: str, lang: str) -> bool:
        if modality == "wake_word":
            return (dataset_id, lang) in self.unlisted_clip_boards
        if modality == "ww_stream" and dataset_id.startswith(STREAM_PREFIX):
            return (dataset_id.removeprefix(STREAM_PREFIX), lang) not in self.clip_boards
        return False


def build_listing(
    clip_boards: Iterable[dict[str, Any]], competitors: Iterable[CompetitorDef],
) -> WakeWordListing:
    """*clip_boards* are ``benchmark-wake_word-*`` payloads."""
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
        if (board.get("positives") or 0) < MIN_POSITIVES or len(trained) < MIN_TRAINED:
            unlisted.add(key)
            continue
        qualifying[key] = {phrase for c in trained if (phrase := phrase_of(c))}
        featured.update(
            phrase for c in trained
            if c.plugin != WAKEHUBERT_PLUGIN and (phrase := phrase_of(c))
        )
    listed = {key for key, phrases in qualifying.items() if phrases & featured}
    unlisted.update(qualifying.keys() - listed)
    return WakeWordListing(
        clip_boards=frozenset(listed),
        unlisted_clip_boards=frozenset(unlisted),
        featured_phrases=frozenset(featured),
    )


def load_listing(data_dir: Path, competitors: Iterable[CompetitorDef]) -> WakeWordListing:
    paths = sorted(data_dir.glob("benchmark-wake_word-*.json"))
    boards = [json.loads(path.read_text()) for path in paths]
    return build_listing(boards, competitors)
