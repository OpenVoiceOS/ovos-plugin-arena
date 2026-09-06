// Cost/quality Pareto frontier shared between the /pareto/ page and its
// cross-check script (scripts/pareto-tolerance-check.mjs). Kept independent
// of leaderboard/index.astro's own paretoSectionHtml (which only ever
// plots RTF) — this one takes the cost metric as a parameter.

// Perf is keyed by hardware tier (arena.metrics.perf_metrics_by_tier,
// arena/models.py PredictionRow.hw.host_class), never blended. Pick the
// tier with the most samples, same rule the leaderboard board uses, so a
// reader never mistakes a cpu-x86 number for a gpu one.
export function primaryPerfTier(perf) {
  if (!perf) return null;
  let best = null, bestN = -1;
  for (const [tier, v] of Object.entries(perf)) {
    const n = v.rtf_n ?? v.peak_rss_mb_n ?? 0;
    if (n > bestN) { best = tier; bestN = n; }
  }
  return best;
}

// Cost value for one board entry: peak_rss_mb (max RSS across the tier's
// rows) or rtf (median). Both are lower-is-better by construction (a
// cheaper fighter costs less memory or less compute time), unlike the
// quality axis whose direction varies per metric.
export function costValue(entry, costKey) {
  const tier = primaryPerfTier(entry.perf);
  if (!tier) return { value: null, tier: null };
  const v = entry.perf[tier][costKey];
  return { value: v ?? null, tier };
}

// Build {id, cost, quality, tier} points for every entry that has BOTH a
// cost measurement and a quality score; entries missing either are the
// caller's "unmeasured" bucket, never plotted at cost 0 (that would read
// as free, when it just was never benched). Every point carries its own
// hardware tier — the caller MUST partition by tier (see pointsByTier)
// before feeding these into paretoFrontier/cheapestWithinTolerance, since
// a cpu-x86 memory figure and a gpu one are not the same measurement.
export function costQualityPoints(entries, costKey, qualityKey) {
  return entries
    .map(e => {
      const { value: cost, tier } = costValue(e, costKey);
      const quality = e.metrics?.[qualityKey];
      if (cost == null || quality == null) return null;
      return { id: e.competitor_id, cost, quality, tier };
    })
    .filter(Boolean);
}

// Split a flat points array into one array per hardware tier, in stable
// tier-name order. A frontier or a "within X%" answer computed on the
// unpartitioned array would silently rank a cpu-x86 fighter against a gpu
// one on the same cost axis, which is not a valid comparison (see
// docs/methodology.md, "Hardware tiers are never blended").
export function pointsByTier(points) {
  const byTier = new Map();
  for (const p of points) {
    if (!byTier.has(p.tier)) byTier.set(p.tier, []);
    byTier.get(p.tier).push(p);
  }
  return [...byTier.entries()].sort(([a], [b]) => a.localeCompare(b))
    .map(([tier, pts]) => ({ tier, points: pts }));
}

// Point p is dominated when some other point is at least as cheap AND at
// least as good on quality, and strictly better on one of the two — there
// is then no reason to ever pick p over it. `higherIsBetter` says which
// direction wins on the quality axis; cost is always lower-is-better.
// Callers pass ONE tier's points at a time (see pointsByTier).
export function paretoFrontier(points, higherIsBetter) {
  const dominates = (a, b) => {
    const qAtLeast = higherIsBetter ? a.quality >= b.quality : a.quality <= b.quality;
    const strictly = (higherIsBetter ? a.quality > b.quality : a.quality < b.quality) || a.cost < b.cost;
    return qAtLeast && a.cost <= b.cost && strictly;
  };
  const frontier = points.filter(p => !points.some(o => o !== p && dominates(o, p)));
  return frontier.slice().sort((a, b) => a.cost - b.cost);
}

// The cheapest frontier point whose quality is within `tolerancePct` of the
// best quality on the frontier — "the cheapest fighter that still scores
// within X% of the best, or X% of this tier's spread when the best score
// is zero". A pure relative margin (`bestQuality * tolerancePct/100`)
// collapses to zero whenever bestQuality is exactly 0 — a real case (a WER
// leader with 0.0 mean error) — which would make the slider inert. Instead
// the margin is the LARGER of that relative share and `tolerancePct`% of
// `qualityRange` (max-min quality across the same tier's points), so a
// zero-best board still has a meaningful, slider-responsive band.
export function cheapestWithinTolerance(frontier, higherIsBetter, tolerancePct, qualityRange) {
  if (!frontier.length) return null;
  const bestQuality = higherIsBetter
    ? Math.max(...frontier.map(p => p.quality))
    : Math.min(...frontier.map(p => p.quality));
  const margin = Math.max(
    Math.abs(bestQuality) * tolerancePct / 100,
    (qualityRange || 0) * tolerancePct / 100,
  );
  const within = frontier.filter(p => higherIsBetter
    ? p.quality >= bestQuality - margin
    : p.quality <= bestQuality + margin);
  if (!within.length) return null;
  return within.reduce((a, b) => (a.cost <= b.cost ? a : b));
}
