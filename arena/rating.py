"""
Bradley-Terry batch rating with bootstrap confidence intervals (§4).

An order-independent alternative to the sequential ELO ladder in
``arena/elo.py`` (kept as a secondary/legacy display column — see
``docs/methodology.md`` for the rationale and the "why not sequential ELO
alone" / "why not TrueSkill" discussion). Fit via minorization-maximization
(Hunter, 2004) over aggregated weighted pairwise win/game totals — given the
same totals, the fit is deterministic and reproducible (§P5).

No numpy/scipy dependency: ``arena/`` stays pydantic-only per AGENTS.md, and
the vote volumes here (hundreds to low thousands of battles per board) are
comfortably fast in pure Python.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

ANCHOR_RATING = 1200.0
SCALE = 400.0 / math.log(10.0)
PRIOR_WEIGHT = 1.0  # one virtual tie vs. the field average per competitor
MM_MAX_ITER = 200
MM_TOLERANCE = 1e-9
DEFAULT_BOOTSTRAP_ROUNDS = 100
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5
PROVISIONAL_MIN_HUMAN_VOTES = 10

# Not a valid competitor_id (registry ids are alphanumeric/hyphen) — safe as
# an internal sentinel key in the pairwise dicts.
_PHANTOM = "\0__field_average__"

PairwiseWins = dict[str, dict[str, float]]
"""wins[i][j] = weighted count of i beating j (ties contribute 0.5)."""

PairwiseGames = dict[str, dict[str, float]]
"""games[i][j] == games[j][i] = weighted count of games played between i, j."""


@dataclass(frozen=True)
class PairResult:
    """One weighted pairwise comparison, from ``a``'s perspective."""

    a: str
    b: str
    score_a: float  # 1.0 win / 0.5 tie / 0.0 loss
    weight: float = 1.0


def accumulate(
    wins: PairwiseWins, games: PairwiseGames, a: str, b: str, score_a: float, weight: float = 1.0
) -> None:
    """Add one weighted pairwise result into running win/game totals, in place."""
    wins.setdefault(a, {})
    wins.setdefault(b, {})
    games.setdefault(a, {})
    games.setdefault(b, {})
    wins[a][b] = wins[a].get(b, 0.0) + weight * score_a
    wins[b][a] = wins[b].get(a, 0.0) + weight * (1.0 - score_a)
    games[a][b] = games[a].get(b, 0.0) + weight
    games[b][a] = games[b].get(a, 0.0) + weight


def pairwise_from_results(results: list[PairResult]) -> tuple[PairwiseWins, PairwiseGames]:
    """Aggregate a flat result list into pairwise win/game totals."""
    wins: PairwiseWins = {}
    games: PairwiseGames = {}
    for r in results:
        accumulate(wins, games, r.a, r.b, r.score_a, r.weight)
    return wins, games


def merge_pairwise(
    base_wins: PairwiseWins,
    base_games: PairwiseGames,
    extra_wins: PairwiseWins,
    extra_games: PairwiseGames,
) -> tuple[PairwiseWins, PairwiseGames]:
    """Return new (wins, games) = base + extra, without mutating either input."""
    wins = {i: dict(js) for i, js in base_wins.items()}
    games = {i: dict(js) for i, js in base_games.items()}
    for i, js in extra_wins.items():
        wins.setdefault(i, {})
        for j, w in js.items():
            wins[i][j] = wins[i].get(j, 0.0) + w
    for i, js in extra_games.items():
        games.setdefault(i, {})
        for j, g in js.items():
            games[i][j] = games[i].get(j, 0.0) + g
    return wins, games


def fit_bradley_terry(
    wins: PairwiseWins,
    games: PairwiseGames,
    competitors: list[str],
    prior_weight: float = PRIOR_WEIGHT,
    max_iter: int = MM_MAX_ITER,
    tolerance: float = MM_TOLERANCE,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths via minorization-maximization (Zermelo/Hunter).

    Returns ``{competitor_id: strength}``, strengths strictly positive, on an
    arbitrary multiplicative scale — use :func:`to_rating_scale` to anchor a
    display scale. Every competitor gets one virtual weighted tie against a
    fixed-strength "field average" phantom opponent (``prior_weight``): this
    connects the comparison graph (competitors that never played each other,
    directly or transitively, still get a well-defined relative order) and
    guarantees every real competitor has a nonzero recorded win fraction, so
    the MM update never collapses a 0-win or 0-loss competitor to 0 or
    infinity.
    """
    if not competitors:
        return {}
    strength: dict[str, float] = dict.fromkeys(competitors, 1.0)
    if len(competitors) == 1:
        return strength

    w = {i: dict(js) for i, js in wins.items()}
    g = {i: dict(js) for i, js in games.items()}

    if prior_weight > 0:
        for c in competitors:
            w.setdefault(c, {})
            g.setdefault(c, {})
            w[c][_PHANTOM] = w[c].get(_PHANTOM, 0.0) + prior_weight * 0.5
            g[c][_PHANTOM] = g[c].get(_PHANTOM, 0.0) + prior_weight
        strength[_PHANTOM] = 1.0

    ids = list(strength.keys())
    for _ in range(max_iter):
        prev = dict(strength)
        for i in ids:
            if i == _PHANTOM:
                continue
            total_wins_i = sum(w.get(i, {}).values())
            denom = 0.0
            for j, g_ij in g.get(i, {}).items():
                if g_ij <= 0:
                    continue
                denom += g_ij / (prev[i] + prev[j])
            if denom > 0:
                strength[i] = max(total_wins_i / denom, 1e-12)
        strength[_PHANTOM] = 1.0

        real_ids = [c for c in ids if c != _PHANTOM]
        max_delta = max(
            abs(math.log(strength[c]) - math.log(prev[c])) for c in real_ids
        )
        if max_delta < tolerance:
            break

    return {c: strength[c] for c in competitors}


def to_rating_scale(
    strengths: dict[str, float], anchor: float = ANCHOR_RATING, scale: float = SCALE
) -> dict[str, float]:
    """Map positive BT strengths onto an ELO-like display scale.

    Anchored so the geometric mean of the given strengths lands at
    ``anchor`` — the absolute scale of a BT fit is arbitrary (only ratios
    matter), so this fixes a convention for display.
    """
    if not strengths:
        return {}
    values = list(strengths.values())
    gmean_log = sum(math.log(max(v, 1e-12)) for v in values) / len(values)
    return {c: anchor + scale * (math.log(max(v, 1e-12)) - gmean_log) for c, v in strengths.items()}


def bootstrap_confidence_intervals(
    human_results: list[PairResult],
    fixed_wins: PairwiseWins,
    fixed_games: PairwiseGames,
    competitors: list[str],
    rounds: int = DEFAULT_BOOTSTRAP_ROUNDS,
    seed: int = 0,
    prior_weight: float = PRIOR_WEIGHT,
    anchor: float = ANCHOR_RATING,
    scale: float = SCALE,
) -> dict[str, tuple[float, float]]:
    """Seeded bootstrap over *human_results* — returns ``{competitor: (ci_lower, ci_upper)}``.

    Only the human vote list is resampled. *fixed_wins* / *fixed_games*
    (typically the benchmark-derived auto-vote seed, already capped per §4
    R5/A1.3) are held constant in every round: they are a deterministic
    function of a fixed benchmark corpus, not an i.i.d. sample, so treating
    them as a random draw would not model a real source of uncertainty.
    These CIs describe uncertainty from the number of human votes cast so
    far — with zero human votes every competitor's interval collapses to a
    single point (the seed-only rating).

    Deterministic for a fixed ``seed`` (§P5): reruns of ``assemble``/``tally``
    over the same vote log produce byte-identical output.
    """
    if not competitors:
        return {}
    rng = random.Random(seed)
    n = len(human_results)
    samples: dict[str, list[float]] = {c: [] for c in competitors}

    for _ in range(rounds):
        if n:
            resample = [human_results[rng.randrange(n)] for _ in range(n)]
            resample_wins, resample_games = pairwise_from_results(resample)
        else:
            resample_wins, resample_games = {}, {}
        wins, games = merge_pairwise(fixed_wins, fixed_games, resample_wins, resample_games)
        strengths = fit_bradley_terry(wins, games, competitors, prior_weight=prior_weight)
        ratings = to_rating_scale(strengths, anchor=anchor, scale=scale)
        for c in competitors:
            samples[c].append(ratings.get(c, anchor))

    out: dict[str, tuple[float, float]] = {}
    for c in competitors:
        vals = sorted(samples[c])
        out[c] = (_percentile(vals, CI_LOWER_PCT), _percentile(vals, CI_UPPER_PCT))
    return out


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100.0) * (len(sorted_values) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(lo)]
    frac = k - lo
    return sorted_values[int(lo)] * (1 - frac) + sorted_values[int(hi)] * frac
