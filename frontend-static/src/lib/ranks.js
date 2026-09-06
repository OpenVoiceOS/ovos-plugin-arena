// Rank/rating lookup against the published ELO boards.
//
// competitors.json carries no rank and no rating — those live in the
// per-league, per-language `leaderboard-<league>-<lang>.json` files the
// leaderboard page renders. Any view that wants to show "how good is this
// fighter" reads them from here so the numbers can never drift apart.

const boards = new Map();

export function fetchBoard(base, league, lang) {
  const key = `${league}|${lang}`;
  if (!boards.has(key)) {
    boards.set(key, fetch(`${base}data/leaderboard-${league}-${encodeURIComponent(lang)}.json`)
      .then(r => (r.ok ? r.json() : null))
      .catch(() => null));
  }
  return boards.get(key);
}

const primary = tag => String(tag).split(/[-_]/)[0].toLowerCase();

// Which board to read for a league: the wanted tag when a board exists for
// it, then any board sharing its primary subtag, then en-US, then whichever
// board has the most fighters on it. Returns null for a league with no
// boards at all.
export function pickBoardLang(leaderboards, league, wanted) {
  const forLeague = leaderboards.filter(b => b.modality === league);
  if (!forLeague.length) return null;
  const exact = wanted && forLeague.find(b => b.lang === wanted);
  if (exact) return exact.lang;
  const sibling = wanted && forLeague.find(b => primary(b.lang) === primary(wanted));
  if (sibling) return sibling.lang;
  const english = forLeague.find(b => b.lang === 'en-US');
  if (english) return english.lang;
  return forLeague.reduce((a, b) => ((b.count || 0) > (a.count || 0) ? b : a)).lang;
}

// competitor_id -> { rank, rating, battles } for one board. Empty map when
// the board is missing, so callers can render "unranked" without a second
// code path.
//
// Ranks are recomputed from the ratings with standard competition ranking
// (1, 1, 1, 4): a whole league can still sit at the seed rating with no
// battles behind it, and numbering those 1..23 would read as an order the
// data does not support.
export async function rankIndex(base, league, lang) {
  const index = new Map();
  if (!lang) return index;
  const board = await fetchBoard(base, league, lang);
  const entries = (board?.entries || [])
    .map(e => ({
      id: e.competitor_id,
      rating: Math.round(e.bt_rating ?? e.elo),
      battles: e.battles ?? 0,
    }))
    .sort((a, b) => b.rating - a.rating);
  let rank = 0;
  let previous = null;
  entries.forEach((e, i) => {
    if (e.rating !== previous) {
      rank = i + 1;
      previous = e.rating;
    }
    index.set(e.id, { rank, rating: e.rating, battles: e.battles });
  });
  return index;
}
