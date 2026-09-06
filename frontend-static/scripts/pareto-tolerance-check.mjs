// Regression check for cheapestWithinTolerance's zero-best-score case
// (a lower-is-better metric whose leader scores exactly 0, e.g. a WER of
// 0.0) — a pure relative margin collapses to zero there, making the
// tolerance slider a no-op. Not part of the build; run manually or wire
// into CI like a11y.mjs: `node scripts/pareto-tolerance-check.mjs`.
import { paretoFrontier, cheapestWithinTolerance } from '../src/lib/pareto.js';

// A forced zero-best board: the leader has a perfect (0.0) score at high
// cost, a near-tied fighter scores slightly worse for a fraction of the
// cost, and a third fighter is clearly worse and clearly cheaper too (so
// the frontier itself stays interesting, not just two points).
const points = [
  { id: 'leader-perfect', cost: 6457.4, quality: 0.0 },
  { id: 'near-tied-cheap', cost: 100, quality: 0.02 },
  { id: 'worse-and-pricier', cost: 4000, quality: 0.10 },
];
const higher = false; // lower-is-better, like wer_mean
const frontier = paretoFrontier(points, higher);
const qualityRange = Math.max(...points.map(p => p.quality)) - Math.min(...points.map(p => p.quality));

function assert(cond, msg) {
  if (!cond) throw new Error(`FAILED: ${msg}`);
  console.log(`ok: ${msg}`);
}

// At 0% tolerance only the exact leader qualifies.
const at0 = cheapestWithinTolerance(frontier, higher, 0, qualityRange);
assert(at0?.id === 'leader-perfect', 'tolerance=0 picks the exact leader');

// As the slider opens up, the cheap near-tied fighter must eventually win —
// this is the control the review found dead (stuck on the leader at every
// tolerance up to 30 because bestQuality*pct stayed 0 forever).
const at30 = cheapestWithinTolerance(frontier, higher, 30, qualityRange);
assert(at30?.id === 'near-tied-cheap', 'tolerance=30 picks the cheap near-tied fighter, not stuck on the zero-quality leader');

// Monotonic sanity: the chosen cost never goes up as tolerance increases.
let prevCost = Infinity;
for (let t = 0; t <= 30; t += 5) {
  const pick = cheapestWithinTolerance(frontier, higher, t, qualityRange);
  assert(pick.cost <= prevCost, `chosen cost is non-increasing as tolerance rises (t=${t})`);
  prevCost = pick.cost;
}

console.log('\npareto-tolerance-check PASSED');
