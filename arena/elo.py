"""
Deterministic ELO rating engine for the OVOS Plugin Arena.

Ratings are fully replayable (§P5): given the same ordered vote log the
functions here always produce the same standings.  Two vote classes exist:

- **auto votes** (``system:benchmark``) — derived from benchmark metrics at
  assemble time, reduced K-factor (§4 R5).  They seed the initial ELO so a
  fresh arena starts with a meaningful ranking.
- **human votes** — GitHub vote issues, full K-factor, replayed on top of
  the seed in issue-number order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arena.models import VoteOutcome
from arena.rating import PairwiseGames, PairwiseWins, accumulate

INITIAL_ELO: float = 1200.0  # R7 sequential ELO is a secondary display column
K_FACTOR: float = 32.0
K_FACTOR_VETERAN: float = 16.0
VETERAN_THRESHOLD: int = 30  # battles before using the lower K
AUTO_K_DIVISOR: float = 4.0  # §4 R5 — auto votes carry K/4 weight

# Bradley-Terry (arena/rating.py) weight for a benchmark-derived auto vote,
# relative to a human vote's weight of 1.0. Same §4 R5 intent as
# AUTO_K_DIVISOR for sequential ELO, expressed as a BT pairwise weight.
BT_AUTO_WEIGHT: float = 1.0 / AUTO_K_DIVISOR  # R9 auto/human weighting


def _outcome_score_a(outcome: VoteOutcome) -> float:
    if outcome == VoteOutcome.CANDIDATE_A:
        return 1.0
    if outcome == VoteOutcome.CANDIDATE_B:
        return 0.0
    return 0.5


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that player A beats player B under the ELO model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(battles: int, auto: bool = False) -> float:
    k = K_FACTOR if battles < VETERAN_THRESHOLD else K_FACTOR_VETERAN
    return k / AUTO_K_DIVISOR if auto else k


def update_ratings(
    rating_a: float,
    rating_b: float,
    outcome: VoteOutcome,
    battles_a: int = 0,
    battles_b: int = 0,
    auto: bool = False,
) -> tuple[float, float]:
    """Compute updated ELO ratings for a single vote outcome.

    ``outcome`` is from A's perspective; TIE and BOTH_WRONG score 0.5/0.5.
    ``auto`` selects the reduced K-factor used for benchmark-derived votes.
    """
    e_a = expected_score(rating_a, rating_b)
    e_b = 1.0 - e_a

    if outcome == VoteOutcome.CANDIDATE_A:
        s_a, s_b = 1.0, 0.0
    elif outcome == VoteOutcome.CANDIDATE_B:
        s_a, s_b = 0.0, 1.0
    else:
        s_a, s_b = 0.5, 0.5

    new_a = rating_a + k_factor(battles_a, auto) * (s_a - e_a)
    new_b = rating_b + k_factor(battles_b, auto) * (s_b - e_b)
    return new_a, new_b


@dataclass
class EloLedger:
    """Mutable ELO state — ratings plus win/loss/tie bookkeeping.

    One ledger per (modality, lang) board.  Apply votes in deterministic
    order; the ledger never reorders anything itself.
    """

    ratings: dict[str, float] = field(default_factory=dict)
    battles: dict[str, int] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    ties: dict[str, int] = field(default_factory=dict)
    auto_votes: dict[str, int] = field(default_factory=dict)
    human_votes: dict[str, int] = field(default_factory=dict)

    # Bradley-Terry sufficient statistics (arena/rating.py) — every vote
    # applied here also accumulates into these, weighted by BT_AUTO_WEIGHT
    # for auto votes and 1.0 for human votes. Kept alongside the sequential
    # ELO bookkeeping above rather than computed separately so both rating
    # systems are always derived from exactly the same replayed vote log.
    pairwise_wins: PairwiseWins = field(default_factory=dict)
    pairwise_games: PairwiseGames = field(default_factory=dict)

    def ensure(self, competitor_id: str) -> None:
        if competitor_id not in self.ratings:
            self.ratings[competitor_id] = INITIAL_ELO
            for table in (self.battles, self.wins, self.losses,
                          self.ties, self.auto_votes, self.human_votes):
                table[competitor_id] = 0

    def apply(
        self,
        competitor_a: str,
        competitor_b: str,
        outcome: VoteOutcome,
        auto: bool = False,
        bt_weight: float | None = None,
    ) -> None:
        """Apply one vote (from A's perspective) to the ledger.

        ``bt_weight`` overrides the Bradley-Terry pairwise weight
        (``arena/rating.py``) for this vote — used by the vote fraud rules
        (§4 A1.4) to down-weight or zero out a vote's effect on the
        statistically-rigorous BT rating while the legacy sequential ELO
        column (secondary/display-only) still records the vote at full
        strength. Defaults to ``BT_AUTO_WEIGHT`` for auto votes, ``1.0`` for
        human votes, same as before this parameter existed.
        """
        self.ensure(competitor_a)
        self.ensure(competitor_b)

        new_a, new_b = update_ratings(
            self.ratings[competitor_a],
            self.ratings[competitor_b],
            outcome,
            self.battles[competitor_a],
            self.battles[competitor_b],
            auto=auto,
        )
        self.ratings[competitor_a] = new_a
        self.ratings[competitor_b] = new_b
        self.battles[competitor_a] += 1
        self.battles[competitor_b] += 1

        if outcome == VoteOutcome.CANDIDATE_A:
            self.wins[competitor_a] += 1
            self.losses[competitor_b] += 1
        elif outcome == VoteOutcome.CANDIDATE_B:
            self.wins[competitor_b] += 1
            self.losses[competitor_a] += 1
        else:
            self.ties[competitor_a] += 1
            self.ties[competitor_b] += 1

        counter = self.auto_votes if auto else self.human_votes
        counter[competitor_a] += 1
        counter[competitor_b] += 1

        weight = bt_weight if bt_weight is not None else (BT_AUTO_WEIGHT if auto else 1.0)
        if weight > 0:
            accumulate(
                self.pairwise_wins, self.pairwise_games,
                competitor_a, competitor_b, _outcome_score_a(outcome), weight,
            )

    def apply_pairwise_only(
        self,
        competitor_a: str,
        competitor_b: str,
        outcome: VoteOutcome,
        auto: bool = False,
        bt_weight: float | None = None,
    ) -> None:
        """Like :meth:`apply`, but skips the sequential-ELO ``update_ratings``
        math entirely — only battle/win/loss/tie counters and the BT
        pairwise accumulation are updated.

        Per-metric secondary ladders (§ per-metric ladders campaign) never
        display or read ``self.ratings`` — that legacy sequential column is
        display-only on the primary board. Fitting it for every extra
        metric on a large roster (many fighters × many samples × many
        metrics) is pure wasted work: the ``update_ratings`` call is the
        dominant per-comparison cost, so skipping it cuts secondary-metric
        seeding time roughly in half with byte-identical BT/counter output.
        """
        self.ensure(competitor_a)
        self.ensure(competitor_b)

        self.battles[competitor_a] += 1
        self.battles[competitor_b] += 1

        if outcome == VoteOutcome.CANDIDATE_A:
            self.wins[competitor_a] += 1
            self.losses[competitor_b] += 1
        elif outcome == VoteOutcome.CANDIDATE_B:
            self.wins[competitor_b] += 1
            self.losses[competitor_a] += 1
        else:
            self.ties[competitor_a] += 1
            self.ties[competitor_b] += 1

        counter = self.auto_votes if auto else self.human_votes
        counter[competitor_a] += 1
        counter[competitor_b] += 1

        weight = bt_weight if bt_weight is not None else (BT_AUTO_WEIGHT if auto else 1.0)
        if weight > 0:
            accumulate(
                self.pairwise_wins, self.pairwise_games,
                competitor_a, competitor_b, _outcome_score_a(outcome), weight,
            )
