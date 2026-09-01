"""ROVER (Recognizer Output Voting Error Reduction) consensus, in pure Python.

Owner directive: TTS ELO seeding should combine UTMOS with intelligibility,
and intelligibility should come from a small *panel* of per-language ASR
judges rather than a single model — one judge's quirks (a habit of dropping
articles, a weak accent model) should not single-handedly decide whether a
synthesised clip "counts" as intelligible. ROVER is the standard way to turn
several independent transcripts of the same audio into one consensus
transcript: align the hypotheses word-for-word, then vote per aligned slot.

There is no ``sclite``/NIST ROVER dependency here (external binary, not
pip-installable) — this reimplements the core two ideas in pure Python:

1. **Alignment.** True N-way multiple sequence alignment is expensive; this
   module instead does *pairwise-progressive* alignment, the same
   approximation ``sclite`` itself defaults to for more than two hypotheses —
   the first hypothesis seeds a running alignment "backbone", and every
   subsequent hypothesis is aligned onto that backbone one at a time with
   ordinary two-sequence dynamic-programming (Levenshtein-style) alignment,
   insertions extending the backbone with empty slots for the hypotheses
   already merged. This is an approximation of true N-way alignment, not
   optimal for every input, but is deterministic, cheap, and matches
   ROVER's own common-case behaviour closely enough for judge panels of two
   or three ASR models.
2. **Voting.** Once every hypothesis has a token (or a gap) in every
   backbone slot, the consensus token per slot is the plurality winner
   among the non-gap tokens; ties are broken in favour of whichever token
   the *primary* judge (hypothesis index 0) contributed for that slot, and
   if the primary judge itself has a gap there, the first hypothesis (in
   panel order) offering a non-gap token wins.
"""
from __future__ import annotations

_GAP = None


def _align_pair(backbone: list[list[str | None]], hypothesis: list[str]) -> list[list[str | None]]:
    """Align ``hypothesis`` onto ``backbone`` (a list of slots, each slot a
    list of per-prior-hypothesis tokens/gaps), returning a new, possibly
    longer, backbone with one more column appended to every slot.

    Alignment cost is computed against the backbone's *first* row (the
    original seed hypothesis), which is always present and gap-free — using
    it as the reference keeps this an ordinary word-level Levenshtein
    alignment.
    """
    ref = [slot[0] for slot in backbone]
    hyp = hypothesis
    n, m = len(ref), len(hyp)

    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrack to a sequence of (ref_index_or_None, hyp_index_or_None) ops.
    ops: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append((i - 1, j - 1))  # substitution
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((i - 1, None))  # deletion (ref has it, hyp doesn't)
            i -= 1
        else:
            ops.append((None, j - 1))  # insertion (hyp has it, ref doesn't)
            j -= 1
    ops.reverse()

    width = len(backbone[0]) if backbone else 1
    new_backbone: list[list[str | None]] = []
    for ref_idx, hyp_idx in ops:
        if ref_idx is not None:
            slot = list(backbone[ref_idx])
        else:
            slot = [_GAP] * width
        slot.append(hyp[hyp_idx] if hyp_idx is not None else _GAP)
        new_backbone.append(slot)
    return new_backbone


def _vote_slot(slot: list[str | None]) -> tuple[str | None, float]:
    """Plurality vote for one aligned slot, treating a gap (``None`` — "this
    judge did not have a word here") as a candidate in its own right, not
    merely an abstention: an insertion slot where only one of three judges
    contributed a word is a majority vote *against* that word, and must
    come out as a deletion, not a spurious keep. Ties favour the primary
    judge (index 0)'s contribution to the slot (word or gap alike),
    falling back to the first contributor (in panel order) among the tied
    candidates.

    Returns ``(winning_token_or_None, vote_share)`` — ``winning_token`` is
    ``None`` when the gap itself wins (nothing is emitted into the
    consensus for that slot). ``vote_share`` is the fraction of the panel
    that voted for the winner, used to compute ``intelligibility_agreement``
    (owner directive): judges disagreeing on what was said (including on
    *whether* a word was said at all) is itself evidence the speech was
    unclear.
    """
    counts: dict[str | None, int] = {}
    for token in slot:
        counts[token] = counts.get(token, 0) + 1
    best_count = max(counts.values())
    winners = {tok for tok, c in counts.items() if c == best_count}
    share = best_count / len(slot)
    if len(winners) == 1:
        return next(iter(winners)), share
    if slot[0] in winners:
        return slot[0], share
    for token in slot:
        if token in winners:
            return token, share
    return None, share  # unreachable — winners is non-empty


def rover_consensus_from_judges(judges: list[dict]) -> str:
    """Recompute ROVER consensus directly from stored per-judge records.

    ``judges`` is the same shape persisted on a scored row's
    ``extras["intelligibility_judges"]`` — a list of mappings, each
    carrying at least a ``"transcript"`` key (``"model"``/``"revision"``
    are provenance only, ignored here). Owner directive: the panel's raw
    transcripts must be durable, so ROVER (or a future reweighted variant
    of it) can be recomputed purely from stored rows without re-running
    any ASR. This is a pure function of ``judges`` — no I/O, no model
    loading — so recomputing it from a stored row reproduces the row's
    original ``intelligibility_consensus`` byte-for-byte.
    """
    return rover_consensus_and_agreement_from_judges(judges)[0]


def rover_consensus_and_agreement_from_judges(judges: list[dict]) -> tuple[str, float]:
    """Same as :func:`rover_consensus_from_judges`, also returning
    ``intelligibility_agreement`` (see :func:`rover_consensus_and_agreement`)."""
    return rover_consensus_and_agreement([j["transcript"] for j in judges])


def rover_consensus(hypotheses: list[str]) -> str:
    """Combine several ASR transcripts of the same clip into one consensus
    transcript via word-level ROVER (see module docstring).

    A single hypothesis is returned unchanged (panel-of-one judges, or a
    panel where every other judge failed to transcribe). An empty list
    returns the empty string.
    """
    return rover_consensus_and_agreement(hypotheses)[0]


def rover_consensus_and_agreement(hypotheses: list[str]) -> tuple[str, float]:
    """:func:`rover_consensus`, plus ``intelligibility_agreement`` — the
    mean per-slot vote share of the winning token across the aligned
    backbone, i.e. how much the panel agreed on what was said.

    ``1.0`` for a panel of one (nothing to disagree with) and for an empty
    hypothesis list (nothing to disagree about either).
    """
    non_empty = [h for h in hypotheses if h is not None]
    if not non_empty:
        return "", 1.0
    if len(non_empty) == 1:
        return non_empty[0], 1.0

    tokenized = [h.split() for h in non_empty]
    seed = tokenized[0]
    backbone: list[list[str | None]] = [[tok] for tok in seed] if seed else []

    for hyp in tokenized[1:]:
        backbone = _align_pair(backbone, hyp)

    if not backbone:
        return "", 1.0

    consensus_tokens: list[str] = []
    shares: list[float] = []
    for slot in backbone:
        token, share = _vote_slot(slot)
        if token is not None:
            consensus_tokens.append(token)
        shares.append(share)

    agreement = sum(shares) / len(shares) if shares else 1.0
    return " ".join(consensus_tokens), agreement
