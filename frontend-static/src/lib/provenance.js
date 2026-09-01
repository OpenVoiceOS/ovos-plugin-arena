// Rating-provenance line shared by the home page, the leaderboard page, and
// the battle page: readers see a plain plugin ranking, but a lot of rating
// movement comes from auto-judged pairwise comparisons (benchmark scores
// turned into synthetic battles) seeded at reduced weight, not blind human
// votes. This module makes that split visible wherever a rating is shown,
// sourced only from published data files — never hardcoded — so it can't
// drift from what actually fed the boards.

// Sums the human/auto split across every leaderboard entry in index.json
// (site-wide total) and reads the auto-vote weight index.json itself
// carries (arena/elo.py BT_AUTO_WEIGHT, threaded through by
// `export-index`). Returns null if index.json carries no leaderboards with
// the split at all — never fabricates a "0" total from absent data.
export function totalVoteProvenance(index) {
  const boards = (index && index.leaderboards) || [];
  const withSplit = boards.filter(
    b => b.human_vote_count != null && b.vote_count != null);
  if (!withSplit.length) return null;
  const human = withSplit.reduce((n, b) => n + b.human_vote_count, 0);
  const total = withSplit.reduce((n, b) => n + b.vote_count, 0);
  const auto = Math.max(0, total - human);
  const weight = typeof index?.auto_vote_weight === 'number' ? index.auto_vote_weight : null;
  return { human, auto, weight };
}

// Same split for one board (a leaderboard-*.json payload, or the matching
// entry out of index.json's `leaderboards` array — both carry the same two
// keys). Returns null when either field is absent from the data, so the
// caller can render nothing instead of a misleading "0 human votes".
export function boardVoteProvenance(board, index) {
  if (!board || board.human_vote_count == null || board.vote_count == null) return null;
  const auto = Math.max(0, board.vote_count - board.human_vote_count);
  const weight = typeof index?.auto_vote_weight === 'number' ? index.auto_vote_weight : null;
  return { human: board.human_vote_count, auto, weight };
}

export async function fetchVoteProvenance(base) {
  const index = await fetch(`${base}data/index.json`)
    .then(r => (r.ok ? r.json() : null)).catch(() => null);
  if (!index) return null;
  return totalVoteProvenance(index);
}

// Compact provenance line, e.g. "Ratings from 1 human vote and 26,451,854
// auto-judged pairwise comparisons across all boards (auto-battles weighted
// ×0.25).", linking to the methodology section that explains it. `scope`
// is an optional trailing qualifier ("across all boards"); omit it for a
// single board's own line. Returns '' (render nothing) when `provenance`
// is null — the caller should skip showing the line entirely in that case.
export function provenanceHtml(provenance, methodologyHref, scope) {
  if (!provenance) return '';
  const { human, auto, weight } = provenance;
  const fmt = n => n.toLocaleString('en-US');
  const humanLabel = human === 1 ? 'human vote' : 'human votes';
  const scopeNote = scope ? ` ${scope}` : '';
  const weightNote = typeof weight === 'number'
    ? ` (auto-battles weighted ×${weight})` : '';
  return `Ratings from ${fmt(human)} ${humanLabel} and ${fmt(auto)} auto-judged pairwise comparisons${scopeNote}${weightNote}. `
    + `<a href="${methodologyHref}">Where ratings come from</a>`;
}
