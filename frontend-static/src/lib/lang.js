// Language-tag grouping shared by the leaderboard, matchups and fighters
// pages. Boards/fighters are keyed by whatever tag their data used — some
// bare (`ca`, `de`), some full (`de-DE`) — and both must stay independently
// selectable (a bare-tag board is not the same dataset as a regional one).
// What we group here is purely presentational: related tags are folded
// under one <optgroup> so `ca` and `ca-ES` read as siblings instead of
// unrelated entries scattered alphabetically.

export const primaryLang = tag => String(tag).split(/[-_]/)[0].toLowerCase();

const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' })[c]);

// langs: array of raw lang tags (bare or full, already deduped+sorted or
// not — this sorts internally). Returns <optgroup>/<option> HTML for a
// <select>, grouping by primary subtag. A group with only one tag is
// rendered as a plain (ungrouped) <option> to avoid visual noise for
// languages that only ever appear as one tag.
export function groupedLangOptionsHtml(langs) {
  const byPrimary = new Map();
  for (const l of langs) {
    const p = primaryLang(l);
    if (!byPrimary.has(p)) byPrimary.set(p, []);
    byPrimary.get(p).push(l);
  }
  const primaries = [...byPrimary.keys()].sort();
  return primaries.map(p => {
    const tags = byPrimary.get(p).sort();
    if (tags.length === 1) return `<option value="${esc(tags[0])}">${esc(tags[0])}</option>`;
    const label = esc(p.toUpperCase());
    const opts = tags.map(t => t === p
      ? `<option value="${esc(t)}">${esc(t)} (unspecified region)</option>`
      : `<option value="${esc(t)}">${esc(t)}</option>`).join('');
    return `<optgroup label="${label}">${opts}</optgroup>`;
  }).join('');
}
