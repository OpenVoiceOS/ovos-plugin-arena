// Head-to-head records for one fighter pair: a real one from cast human
// votes (`humanPairRecord`), and a benchmark-replay signal
// (`pairRecord`/`battleOutcome`) for pairs or languages a real vote hasn't
// reached yet.

// Benchmark auto-battle record, replayed client-side from the same battles
// pools the battle page already has in memory. This is the same
// reference/prediction exact-match signal `arena/assembler.py
// auto_outcome()` uses to seed the ELO ladder — a real per-pair record, not
// a fabricated one, but it is not a human vote. Battles with no reference
// (freeform pools) contribute no signal.
export function battleOutcome(b) {
  if (b.reference == null || b.prediction_a == null || b.prediction_b == null) return null;
  const norm = v => {
    if (v && typeof v === 'object') v = v.intent ?? v.text ?? JSON.stringify(v);
    return String(v ?? '').trim().toLowerCase();
  };
  const ref = norm(b.reference);
  const correctA = norm(b.prediction_a) === ref;
  const correctB = norm(b.prediction_b) === ref;
  if (correctA && !correctB) return 'a';
  if (correctB && !correctA) return 'b';
  if (correctA && correctB) return 'tie';
  return 'both_wrong';
}

// Benchmark record for `a` against `b` across the given battles:
// {wins, losses, ties, total}.
export function pairRecord(battles, a, b) {
  const rec = { wins: 0, losses: 0, ties: 0, total: 0 };
  for (const battle of battles) {
    const isAB = battle.competitor_a === a && battle.competitor_b === b;
    const isBA = battle.competitor_a === b && battle.competitor_b === a;
    if (!isAB && !isBA) continue;
    const outcome = battleOutcome(battle);
    if (outcome == null) continue;
    rec.total++;
    if (outcome === 'tie' || outcome === 'both_wrong') { rec.ties++; continue; }
    const aWon = (isAB && outcome === 'a') || (isBA && outcome === 'b');
    if (aWon) rec.wins++; else rec.losses++;
  }
  return rec;
}

// --- Real human votes (votes.jsonl + vote-audit.json) ----------------------
//
// `votes.jsonl` is the append-only, publicly-cast vote record written by
// `arena/cli.py append_vote_records` — one JSON object per GitHub issue
// ingested, keyed by `issue`, carrying `battle_id`, `choice`
// (a|b|tie|both_wrong), `competitor_a`, `competitor_b`, `modality`, `lang`.
// An issue that never became a countable vote has no `competitor_a`/`b` at
// all (`invalid` title, or `discarded_reason` — battle no longer in the
// pool at ingest time). `vote-audit.json`'s `discarded` list additionally
// names issues the fraud rules (arena/fraud.py) rejected after ingest —
// duplicates, retired competitors, one-sided-voting zero-weight rejects —
// each entry carrying `issue_number`. A down-weighted vote (partial trust,
// still > 0) is not in `discarded` and counts here at full weight; this is
// a display count, not the Bradley-Terry rating itself.
let votesPromise = null;

function loadVotes(base) {
  if (!votesPromise) {
    votesPromise = Promise.all([
      fetch(`${base}data/votes.jsonl`).then(r => (r.ok ? r.text() : '')).catch(() => ''),
      fetch(`${base}data/vote-audit.json`).then(r => (r.ok ? r.json() : null)).catch(() => null),
    ]).then(([text, audit]) => {
      const discardedIssues = new Set((audit?.discarded || []).map(e => e.issue_number));
      return text
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => { try { return JSON.parse(line); } catch { return null; } })
        .filter(v => v
          && !v.invalid
          && !v.discarded_reason
          && !discardedIssues.has(v.issue)
          && v.competitor_a && v.competitor_b);
    });
  }
  return votesPromise;
}

// Real human head-to-head for `a` against `b`, optionally scoped to `lang`.
export async function humanPairRecord(base, a, b, { lang } = {}) {
  const votes = await loadVotes(base);
  const rec = { wins: 0, losses: 0, ties: 0, total: 0 };
  for (const v of votes) {
    if (lang && v.lang !== lang) continue;
    const isAB = v.competitor_a === a && v.competitor_b === b;
    const isBA = v.competitor_a === b && v.competitor_b === a;
    if (!isAB && !isBA) continue;
    rec.total++;
    if (v.choice === 'tie' || v.choice === 'both_wrong') { rec.ties++; continue; }
    const aWon = (isAB && v.choice === 'a') || (isBA && v.choice === 'b');
    if (aWon) rec.wins++; else rec.losses++;
  }
  return rec;
}

// Count of counted human votes, optionally scoped to `modality` and/or `lang`.
export async function humanVoteCount(base, { modality, lang } = {}) {
  const votes = await loadVotes(base);
  return votes.filter(v =>
    (!modality || v.modality === modality) && (!lang || v.lang === lang)).length;
}
