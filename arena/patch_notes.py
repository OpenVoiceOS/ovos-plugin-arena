"""Patch notes — a human-readable changelog of leaderboard movement (§A5.4).

At tally time we diff the previous board (the leaderboard JSON already on disk,
which is under git) against the freshly-computed one and emit ``patch-notes.json``:
rank changes, new fighters, and the day's biggest upset. The frontend renders it
as a "what changed" feed — a cheap, no-LLM growth hook.

All copy is templated here in one place; there are no timestamps inside an
individual note (the file's own ``generated_at`` carries the time), so an
unchanged board produces the same notes.
"""

from __future__ import annotations

import json
from pathlib import Path

from arena.models import EloBoard, EloEntry


def load_board(path: Path) -> EloBoard | None:
    """Load an existing ``leaderboard-*.json`` into an EloBoard, or None."""
    if not path.is_file():
        return None
    try:
        return EloBoard.model_validate(json.loads(path.read_text()))
    except Exception:
        return None


def _rank_map(board: EloBoard | None) -> dict[str, EloEntry]:
    if board is None:
        return {}
    return {e.competitor_id: e for e in board.entries}


def diff_board(old: EloBoard | None, new: EloBoard) -> list[dict]:
    """Return per-fighter change notes for one (modality, lang) board.

    Note kinds: ``new`` (first appearance), ``up``/``down`` (rank moved),
    ``hold`` is omitted (only movement is newsworthy).
    """
    old_entries = _rank_map(old)
    notes: list[dict] = []
    modality = getattr(new.modality, "value", new.modality)
    for entry in new.entries:
        prev = old_entries.get(entry.competitor_id)
        base = {
            "modality": modality,
            "lang": new.lang,
            "fighter": entry.competitor_id,
            "rank": entry.rank,
        }
        if prev is None:
            notes.append({**base, "kind": "new",
                          "text": f"{entry.competitor_id} enters the {modality} "
                                  f"arena at #{entry.rank}"})
            continue
        delta = prev.rank - entry.rank  # positive = climbed
        if delta > 0:
            notes.append({**base, "kind": "up", "delta": delta, "from_rank": prev.rank,
                          "text": f"{entry.competitor_id} climbs {delta} to "
                                  f"#{entry.rank} in {modality}"})
        elif delta < 0:
            notes.append({**base, "kind": "down", "delta": delta, "from_rank": prev.rank,
                          "text": f"{entry.competitor_id} drops {-delta} to "
                                  f"#{entry.rank} in {modality}"})
    return notes


def build_patch_notes(all_notes: list[dict], generated_at: str) -> dict:
    """Aggregate per-board notes into the ``patch-notes.json`` payload.

    The "upset of the day" is the single largest upward move (biggest ``delta``);
    ties break deterministically by (fighter, modality) so the output is stable.
    """
    climbs = [n for n in all_notes if n.get("kind") == "up"]
    upset = None
    if climbs:
        upset = max(climbs, key=lambda n: (n["delta"], n["fighter"], n["modality"]))
    return {
        "generated_at": generated_at,
        "count": len(all_notes),
        "upset_of_the_day": upset,
        "notes": all_notes,
    }
