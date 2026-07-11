"""Embeddable SVG rank badges — the arena's growth loop.

At assemble/tally time we emit one small, self-contained SVG per fighter per
league (``badges/<modality>/<competitor>.svg``). Plugin authors embed their
badge in a README; every embed links back to the arena.

The SVG is hand-written in the flat "shields" style so there is no external
dependency and no build step. It contains **no timestamps or generated dates**
so the daily rebuild produces byte-identical output when nothing changed (a
churn-free diff, and GitHub's camo image cache stays warm).
"""

from __future__ import annotations

import html
from pathlib import Path

# Colour per modality — matches the frontend league palette.
_LEAGUE_COLOR = {
    "stt": "#1f6feb",
    "tts": "#238636",
    "ww": "#9e6a03",
    "vad": "#1f7a8c",
    "intent": "#8250df",
}
_DEFAULT_COLOR = "#57606a"
_LABEL_BG = "#24292f"

# Character-width heuristic (px) for the 11px Verdana/DejaVu the badge uses.
# A fixed average keeps the module dependency-free; slightly generous so text
# never clips.
_CHAR_PX = 6.5
_PAD = 8.0


def _text_width(text: str) -> float:
    """Approximate rendered width of ``text`` in the badge font."""
    return len(text) * _CHAR_PX


def render_rank_badge(
    fighter: str,
    modality: str,
    rank: int,
    rating: float,
    *,
    provisional: bool = False,
) -> str:
    """Return a self-contained flat-style SVG rank badge.

    Args:
        fighter: Competitor id shown as the badge label.
        modality: League key (``stt``/``tts``/``ww``/``vad``/``intent``).
        rank: 1-based rank within the league.
        rating: Bradley-Terry (or ELO) rating, rounded for display.
        provisional: If True, the value reads ``#N?`` to signal too few votes.

    Returns:
        An SVG document string.
    """
    label = fighter
    q = "?" if provisional else ""
    value = f"#{rank}{q} · {round(rating)}"
    color = _LEAGUE_COLOR.get(modality.lower(), _DEFAULT_COLOR)

    label_w = _text_width(label) + 2 * _PAD
    value_w = _text_width(value) + 2 * _PAD
    total_w = label_w + value_w
    label_mid = label_w / 2
    value_mid = label_w + value_w / 2

    el = html.escape(label)
    ev = html.escape(value)
    aria = html.escape(f"{label}: rank {rank} in {modality} arena, rating {round(rating)}")

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="20" '
        f'role="img" aria-label="{aria}">'
        f'<title>{aria}</title>'
        f'<linearGradient id="s" x2="0" y2="100%">'
        f'<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        f'<stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<rect rx="3" width="{total_w:.0f}" height="20" fill="{_LABEL_BG}"/>'
        f'<rect rx="3" x="{label_w:.0f}" width="{value_w:.0f}" height="20" fill="{color}"/>'
        f'<rect rx="3" width="{total_w:.0f}" height="20" fill="url(#s)"/>'
        f'<g fill="#fff" text-anchor="middle" '
        f'font-family="Verdana,DejaVu Sans,Geneva,sans-serif" font-size="11">'
        f'<text x="{label_mid:.0f}" y="14">{el}</text>'
        f'<text x="{value_mid:.0f}" y="14">{ev}</text>'
        f'</g></svg>\n'
    )


def emit_badges(board, out_dir: Path) -> int:
    """Write one badge SVG per entry of an ``EloBoard`` under ``out_dir``.

    Files land at ``<out_dir>/badges/<modality>/<competitor_id>.svg``. The
    rating is the Bradley-Terry rating when present, else the legacy ELO.

    Args:
        board: An ``EloBoard`` (duck-typed: ``.modality``, ``.provisional``,
            ``.entries`` with ``.rank``/``.competitor_id``/``.bt_rating``/``.elo``).
        out_dir: The data output directory (badges/ is created beneath it).

    Returns:
        Number of badge files written.
    """
    modality = getattr(board.modality, "value", board.modality)
    badge_dir = Path(out_dir) / "badges" / modality
    badge_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in board.entries:
        rating = entry.bt_rating if entry.bt_rating is not None else entry.elo
        svg = render_rank_badge(
            entry.competitor_id, modality, entry.rank, rating,
            provisional=board.provisional,
        )
        (badge_dir / f"{entry.competitor_id}.svg").write_text(svg, encoding="utf-8")
        written += 1
    return written
