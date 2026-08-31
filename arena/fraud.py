"""
Vote fraud / dedup resistance (§4, A1.4).

Every rule here is a **pure function of the vote log** — no network calls —
so replaying the identical vote log (plus the account-age cache, fetched
once at ingest in ``arena/cli.py`` and persisted to disk) always produces
identical discards (§P5: the ingest step, which touches the network, is
kept strictly separate from the replay step, which is pure). Duplicate
votes (one per author per battle) are removed upstream by
``arena.cli.dedupe_votes`` before any of these rules run.

Discards are never silently dropped: ``resolve_vote_weights`` returns one
``VoteDecision`` per input vote, so every discard can be reported (see
``arena.cli.cmd_tally``'s vote-audit output) — the arena's anti-fraud rules
are themselves public and auditable, same as the vote log.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

# R13 vote fraud rules
DAILY_VOTE_CAP = 50
NEW_ACCOUNT_MIN_DAYS = 7
ONE_SIDED_MIN_VOTES = 20
ONE_SIDED_THRESHOLD = 0.95
ONE_SIDED_WEIGHT = 0.5


@dataclass
class VoteDecision:
    """One vote's outcome after the fraud pipeline: full weight (1.0),
    down-weighted (e.g. 0.5 for one-sided voting), or discarded (0.0,
    with ``discarded_reason`` set — the vote log entry itself is never
    deleted, only its rating influence)."""

    vote: dict
    weight: float
    discarded_reason: str | None = None


def _vote_day(vote: dict) -> str:
    """Deterministic calendar-day bucket from an ISO8601 ``created_at`` —
    string slicing, no timezone-library dependency or ambiguity."""
    created = vote.get("created_at") or ""
    return created[:10] if len(created) >= 10 else ""


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def apply_daily_cap(
    votes: list[dict],
    modality_by_battle: dict[str, str],
    cap: int = DAILY_VOTE_CAP,
) -> list[VoteDecision]:
    """§4 — per-voter, per-league (modality), per-UTC-day cap.

    *votes* MUST already be in deterministic order (dedupe_votes sorts by
    issue number) — later votes in the same bucket are the ones capped, so
    the cap is itself replay-deterministic.
    """
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    decisions = []
    for vote in votes:
        modality = modality_by_battle.get(vote["battle_id"], "")
        key = (vote["author"], modality, _vote_day(vote))
        counts[key] += 1
        if counts[key] > cap:
            decisions.append(VoteDecision(vote, 0.0, "daily_vote_cap_exceeded"))
        else:
            decisions.append(VoteDecision(vote, 1.0))
    return decisions


def apply_account_age_gate(
    decisions: list[VoteDecision],
    account_created_at: dict[str, str],
    min_days: int = NEW_ACCOUNT_MIN_DAYS,
) -> list[VoteDecision]:
    """§4 — accounts created less than *min_days* before the vote get
    weight 0. *account_created_at* maps GitHub login to an ISO8601 account
    creation timestamp — fetched once per author in ``arena.cli.cmd_tally``
    and persisted in the committed data dir, never re-fetched on replay.
    A login absent from the cache (fetch failed, or not yet ingested) is
    not gated — it is not this function's job to fetch data.
    """
    out = []
    for d in decisions:
        if d.discarded_reason:
            out.append(d)
            continue
        created = account_created_at.get(d.vote["author"])
        voted_at = d.vote.get("created_at")
        if created and voted_at:
            age_days = (_parse_iso(voted_at) - _parse_iso(created)).total_seconds() / 86400.0
            if age_days < min_days:
                out.append(VoteDecision(d.vote, 0.0, "account_too_new"))
                continue
        out.append(d)
    return out


def apply_one_sided_downweight(
    decisions: list[VoteDecision],
    min_votes: int = ONE_SIDED_MIN_VOTES,
    threshold: float = ONE_SIDED_THRESHOLD,
    downweight: float = ONE_SIDED_WEIGHT,
) -> list[VoteDecision]:
    """§4 — a voter whose surviving votes are more than *threshold* for one
    literal A/B side, across at least *min_votes* votes, has every one of
    those votes down-weighted to *downweight*.

    This is deliberately a function of the **literal a/b choice**, not
    competitor identity: blind battles randomize which competitor is shown
    as "A" per battle (derived from the battle-id hash, §4 R4), so "always
    picks A" is the low-effort/bot-like signal — "always prefers competitor
    X" is not measurable this way and would be the *correct* behavior for
    someone who genuinely prefers a plugin. Ties/both-wrong votes are
    excluded from the ratio (neither side).
    """
    by_author: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        if d.discarded_reason is None and d.vote["choice"] in ("a", "b"):
            by_author[d.vote["author"]].append(i)

    out = list(decisions)
    for _author, idxs in by_author.items():
        if len(idxs) < min_votes:
            continue
        a_count = sum(1 for i in idxs if decisions[i].vote["choice"] == "a")
        dominant = max(a_count, len(idxs) - a_count)
        if dominant / len(idxs) > threshold:
            for i in idxs:
                d = out[i]
                out[i] = VoteDecision(d.vote, downweight, d.discarded_reason)
    return out


def resolve_vote_weights(
    votes: list[dict],
    modality_by_battle: dict[str, str],
    account_created_at: dict[str, str],
) -> list[VoteDecision]:
    """Apply every §4 A1.4 rule, in order, to an already-deduped vote list.

    Pure — no network, deterministic given the same inputs (§P5). Order:
    daily cap → account-age gate → one-sided downweight (each rule only
    acts on votes not already discarded by an earlier rule; the one-sided
    downweight is the exception — it evaluates a voter's full surviving
    history, so it must run last).
    """
    decisions = apply_daily_cap(votes, modality_by_battle)
    decisions = apply_account_age_gate(decisions, account_created_at)
    decisions = apply_one_sided_downweight(decisions)
    return decisions
