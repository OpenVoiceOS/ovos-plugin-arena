// Dex entries for fighter pages — species, rarity, stat hexagons, flavor
// text, moves, evolutions and rivals.
//
// Everything here is derived from the committed boards under `public/data`
// and from the competitor registry, and everything is deterministic: the
// only randomness is a hash of `competitor_id`, so a rebuild of the same
// data produces byte-identical pages.

import { hashString } from './sprite.js';

export const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

// --- species, rarity, habitat ------------------------------------------

const RARITY = {
  micro: 'Common', small: 'Uncommon', base: 'Rare',
  medium: 'Very Rare', large: 'Epic', 'x-large': 'Legendary',
};

export const rarityOf = size => RARITY[size] || 'Unclassified';

const TYPE_NOUN = {
  'gofai': 'rule-carved', 'fuzzy-match': 'blur-tolerant', 'neural-net': 'gradient-fed',
  'template-match': 'lattice-woven', 'keyword-match': 'word-bagging', 'embedding': 'vector-drifting',
  'llm': 'many-tongued', 'ensemble': 'many-headed', 'transformer': 'attention-hungry',
  'statistical': 'frequency-counting', 'classical-ml': 'textbook-trained', 'cloud': 'far-signalling',
};

// "Padatious-species · fuzzy-match / neural-net type"
export function speciesLine(fighter) {
  const parts = [];
  const species = fighter.species || fighter.plugin;
  if (species) parts.push(`${species}-species`);
  const types = fighter.types || [];
  if (types.length) parts.push(`${types.join(' / ')} type`);
  return parts.join(' · ');
}

// --- metric plumbing ----------------------------------------------------

const num = v => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const clamp01 = v => Math.min(1, Math.max(0, v));
const pctText = v => `${(v * 100).toFixed(1)}%`;

const share = (v, raw, flavor = null) => (v == null ? null : { score: Math.round(clamp01(v) * 100), raw, flavor });
const inverted = (v, raw, flavor = null) => (v == null ? null : { score: Math.round(clamp01(1 - v) * 100), raw, flavor });

// A flavor clause is only ever earned by an absolute bar on the measurement
// itself. Relative rank among the six axes decides nothing: a fighter that
// covers twelve languages must never be described as staying home just
// because its other stats happen to score higher.
const grade = (v, isGood, isBad) => (v == null ? null : isGood(v) ? 'good' : isBad(v) ? 'bad' : null);

// Lower-is-better quantities (latency, RTF, memory) span orders of
// magnitude, so they are placed on a log scale spanning the league rather
// than a linear one where every fighter but the slowest scores ~100.
function invLog(v, scale, raw, flavor = null) {
  if (v == null || v <= 0 || !scale || scale.min <= 0) return null;
  if (scale.max <= scale.min) return { score: 50, raw, flavor };
  const t = (Math.log(v) - Math.log(scale.min)) / (Math.log(scale.max) - Math.log(scale.min));
  return { score: Math.round(clamp01(1 - t) * 100), raw, flavor };
}

const rowRss = r => num(r.perf?.gpu?.peak_rss_mb) ?? num(r.perf?.cpu?.peak_rss_mb) ?? num(r.model_mb);
const rowRtf = r => num(r.perf?.gpu?.rtf) ?? num(r.perf?.cpu?.rtf);

const footprintAxis = {
  key: 'footprint', label: 'Footprint', help: 'peak resident memory, inverted (log scale over the league)',
  get: (r, ctx) => {
    const mb = rowRss(r);
    return invLog(mb, ctx.rss, mb == null ? '' : `${mb.toFixed(0)} MB peak RSS`,
      grade(mb, v => v <= 200, v => v >= 2000));
  },
};
const rangeAxis = {
  key: 'range', label: 'Range', help: 'languages covered, against the widest in the league',
  get: (r, ctx) => share(ctx.maxLangs ? r.langCount / ctx.maxLangs : null,
    `${r.langCount} language${r.langCount === 1 ? '' : 's'}`,
    grade(r.langCount,
      v => ctx.polyglotLangs > 1 && v >= ctx.polyglotLangs,
      v => v === 1)),
};
const latencyAxis = {
  key: 'speed', label: 'Speed', help: 'median latency, inverted (log scale over the league)',
  get: (r, ctx) => {
    const ms = num(r.metrics.latency_ms_median);
    return invLog(ms, ctx.latency, ms == null ? '' : `${ms.toFixed(0)} ms median`,
      grade(ms, v => v <= 50, v => v >= 1000));
  },
};

const INTENT_AXES = [
  {
    key: 'accuracy', label: 'Accuracy', help: 'accuracy on held-out phrasings',
    get: r => {
      const v = num(r.metrics.generalization_accuracy) ?? num(r.metrics.accuracy);
      return share(v, v == null ? '' : pctText(v), grade(v, x => x >= 0.8, x => x <= 0.5));
    },
  },
  {
    key: 'robustness', label: 'Robustness', help: 'mean of typo and ASR-noise accuracy',
    get: r => {
      const parts = [num(r.metrics.acc_typos), num(r.metrics.acc_asr_noise)].filter(v => v != null);
      if (!parts.length) return null;
      const v = parts.reduce((a, b) => a + b, 0) / parts.length;
      return share(v, parts.map(pctText).join(' / '), grade(v, x => x >= 0.75, x => x <= 0.5));
    },
  },
  {
    key: 'judgement', label: 'Judgement', help: 'out-of-domain rejection (false-positive rate, inverted)',
    get: r => {
      const v = num(r.metrics.ood_fpr);
      return inverted(v, v == null ? '' : `${pctText(v)} OOD false positives`,
        grade(v, x => x <= 0.2, x => x >= 0.5));
    },
  },
  latencyAxis, footprintAxis, rangeAxis,
];

const ERROR_AXES = [
  {
    key: 'precision', label: 'Precision', help: 'false-accept rate, inverted',
    get: r => {
      const v = num(r.metrics.false_accept_rate);
      return inverted(v, v == null ? '' : `${pctText(v)} false accepts`,
        grade(v, x => x <= 0.05, x => x >= 0.3));
    },
  },
  {
    key: 'recall', label: 'Recall', help: 'false-reject rate, inverted',
    get: r => {
      const v = num(r.metrics.false_reject_rate);
      return inverted(v, v == null ? '' : `${pctText(v)} false rejects`,
        grade(v, x => x <= 0.05, x => x >= 0.3));
    },
  },
  {
    key: 'accuracy', label: 'Accuracy', help: 'overall error rate, inverted',
    get: r => {
      const v = num(r.metrics.error_rate);
      return inverted(v, v == null ? '' : `${pctText(v)} error rate`,
        grade(v, x => x <= 0.05, x => x >= 0.3));
    },
  },
  latencyAxis, footprintAxis, rangeAxis,
];

const AXES = {
  intent: INTENT_AXES,
  intent_template: INTENT_AXES,
  intent_keyword: INTENT_AXES,
  wake_word: ERROR_AXES,
  vad: ERROR_AXES,
  stt: [
    {
      key: 'accuracy', label: 'Accuracy', help: 'mean word error rate, inverted',
      get: r => {
        const v = num(r.metrics.wer_mean);
        return inverted(v, v == null ? '' : `${pctText(v)} mean WER`,
          grade(v, x => x <= 0.15, x => x >= 0.5));
      },
    },
    {
      key: 'consistency', label: 'Consistency', help: 'median word error rate, inverted',
      get: r => {
        const v = num(r.metrics.wer_median);
        return inverted(v, v == null ? '' : `${pctText(v)} median WER`,
          grade(v, x => x <= 0.15, x => x >= 0.5));
      },
    },
    {
      key: 'speed', label: 'Speed', help: 'real-time factor, inverted (log scale over the league)',
      get: (r, ctx) => {
        const rtf = rowRtf(r);
        return invLog(rtf, ctx.rtf, rtf == null ? '' : `${rtf.toFixed(3)} RTF`,
          grade(rtf, x => x <= 0.1, x => x >= 1));
      },
    },
    { ...latencyAxis, key: 'latency', label: 'Latency' },
    footprintAxis, rangeAxis,
  ],
  tts: [
    {
      key: 'naturalness', label: 'Naturalness', help: 'UTMOS mean opinion score (1–5)',
      get: r => {
        const v = num(r.metrics.utmos);
        return v == null ? null : {
          score: Math.round(clamp01((v - 1) / 4) * 100), raw: `${v.toFixed(2)} UTMOS`,
          flavor: grade(v, x => x >= 4, x => x <= 2.5),
        };
      },
    },
    {
      key: 'intelligibility', label: 'Intelligibility', help: 're-transcription word error rate, inverted',
      get: r => {
        const v = num(r.metrics.intelligibility_wer);
        return inverted(v, v == null ? '' : `${pctText(v)} WER when transcribed back`,
          grade(v, x => x <= 0.15, x => x >= 0.5));
      },
    },
    {
      key: 'speed', label: 'Speed', help: 'real-time factor, inverted (median latency when RTF is unmeasured)',
      get: (r, ctx) => {
        const rtf = rowRtf(r);
        if (rtf != null) return invLog(rtf, ctx.rtf, `${rtf.toFixed(3)} RTF`, grade(rtf, x => x <= 0.1, x => x >= 1));
        const ms = num(r.metrics.latency_ms_median);
        return invLog(ms, ctx.latency, ms == null ? '' : `${ms.toFixed(0)} ms median`,
          grade(ms, x => x <= 50, x => x >= 1000));
      },
    },
    {
      key: 'acclaim', label: 'Acclaim', help: 'human vote rating, against the league spread',
      get: (r, ctx) => {
        const v = num(r.rating);
        if (v == null || !ctx.rating || ctx.rating.max <= ctx.rating.min) return null;
        const t = (v - ctx.rating.min) / (ctx.rating.max - ctx.rating.min);
        return {
          score: Math.round(clamp01(t) * 100), raw: `${Math.round(v)} rating`,
          flavor: grade(v, x => x >= 1400, x => x <= 1100),
        };
      },
    },
    footprintAxis, rangeAxis,
  ],
};

export const axesFor = modality => AXES[modality] || INTENT_AXES;

// --- flavor text --------------------------------------------------------

const OPENERS = {
  'fuzzy-match': ['A fuzzy-match hunter', 'A near-enough forager', 'A blur-tolerant scavenger'],
  'template-match': ['A template stalker', 'A lattice-weaving pattern beast', 'A slot-shaped ambusher'],
  'keyword-match': ['A keyword pouncer', 'A tireless word-bag rummager', 'A trigger-word terrier'],
  'neural-net': ['A gradient-fed predator', 'A well-drilled neural grazer', 'A backprop-raised beast'],
  'embedding': ['A vector-space drifter', 'A creature that smells meaning at a distance', 'A cosine-guided wanderer'],
  'transformer': ['An attention-hungry leviathan', 'A context-swallowing giant', 'A head-splitting behemoth'],
  'llm': ['A many-tongued oracle', 'A verbose cave-dweller', 'A prompt-fed familiar'],
  'ensemble': ['A many-headed pack beast', 'A committee in a trench coat', 'A relay team wearing one skin'],
  'gofai': ['A rule-carved automaton', 'An old-school grammar beast', 'A hand-written clockwork thing'],
  'statistical': ['A frequency-counting scholar', 'A tally-keeping creature', 'A count-and-compare specialist'],
  'classical-ml': ['A textbook-trained specimen', 'A feature-fed workhorse', 'A well-behaved classifier'],
  'cloud': ['A far-signalling migrant that hunts over the wire', 'A distant thing that answers by radio',
            'A creature that keeps its brain in another country'],
};
const DEFAULT_OPENER = 'An unclassified arena dweller';

export const STRENGTHS = {
  accuracy: ['rarely misses a paraphrase', 'lands the right label with unsettling calm', 'reads phrasings it has never met'],
  robustness: ['shrugs off typos and microphone hiss', 'hears through a bad connection', 'survives the worst spelling you have'],
  judgement: ['knows when a request was never meant for it', 'lets the wrong question pass in silence', 'keeps its mouth shut off-topic'],
  precision: ['almost never wakes for the wrong sound', 'ignores everything that is not its name', 'holds still through the noise'],
  recall: ['catches nearly everything aimed at it', 'misses almost nothing', 'answers every call'],
  consistency: ['transcribes the easy half of a corpus flawlessly', 'holds its accuracy sample after sample', 'rarely has an off take'],
  naturalness: ['sounds close enough to a person to be unnerving', 'carries a voice worth listening to', 'breathes in the right places'],
  intelligibility: ['speaks clearly enough to be transcribed straight back', 'is understood on the first pass', 'never has to repeat itself'],
  acclaim: ['keeps winning the listening booth', 'wins the votes it is put up for', 'is the one listeners keep picking'],
  speed: ['answers before you finish blinking', 'replies faster than you can look up', 'is gone before the meter moves'],
  latency: ['replies with barely a pause', 'answers on the beat', 'keeps nobody waiting'],
  footprint: ['lives happily on a small board', 'fits where nothing else will', 'travels light'],
  range: ['roams more languages than most of its league', 'is at home in a great many tongues', 'crosses borders its rivals cannot'],
};

export const WEAKNESSES = {
  accuracy: ['guesses wildly once the wording drifts', 'loses the thread on unfamiliar phrasings', 'reaches for the wrong label often'],
  robustness: ['panics when words are misspelled', 'falls apart under microphone noise', 'cannot cope with a slipped keystroke'],
  judgement: ['answers questions nobody asked it', 'cannot tell an off-topic request from a real one', 'volunteers answers it should withhold'],
  precision: ['startles at passing noise', 'wakes for sounds that were never its name', 'cannot be trusted to stay asleep'],
  recall: ['sleeps through half the calls', 'misses more than it catches', 'has to be asked twice'],
  consistency: ['falls apart on the harder recordings', 'swings wildly between takes', 'cannot be relied on twice running'],
  naturalness: ['has a voice like a hinge', 'sounds like a fax machine clearing its throat', 'grates after a sentence or two'],
  intelligibility: ['mumbles when the sentence gets long', 'has to be replayed to be understood', 'garbles the words it was given'],
  acclaim: ['has yet to charm a single listener', 'loses the listening booth every time', 'cannot win a vote'],
  speed: ['takes its time about everything', 'is slower than the patience of anyone waiting', 'thinks it over for far too long'],
  latency: ['keeps you waiting', 'answers long after the question', 'leaves a silence you can walk through'],
  footprint: ['eats memory like a titan', 'needs a machine most people do not own', 'will not fit on a small board'],
  range: ['never leaves its home language', 'speaks exactly one tongue and no other', 'has only ever been measured in one language'],
};

const pick = (list, seedStr) => list[hashString(seedStr) % list.length];

// One or two sentences, stable for a given competitor id. The opener comes
// from the fighter's own type, the second sentence from the axes whose raw
// measurement cleared an absolute bar — a stat is never praised or mocked
// for merely being the highest or lowest of the six.
export function flavorText(fighter, axes) {
  const id = fighter.competitor_id;
  const pool = (fighter.types || []).map(t => OPENERS[t]).find(Boolean);
  const opener = pool ? pick(pool, `${id}:opener`) : DEFAULT_OPENER;
  const tier = fighter.size ? ` of the ${fighter.size} tier` : '';
  const rarity = RARITY[fighter.size] ? `, ${RARITY[fighter.size].toLowerCase()} in its league` : '';
  const first = `${opener}${tier}${rarity}.`;

  const scored = (axes || []).filter(a => a.score != null);
  const good = scored.filter(a => a.flavor === 'good').sort((a, b) => b.score - a.score)[0];
  const bad = scored.filter(a => a.flavor === 'bad').sort((a, b) => a.score - b.score)[0];
  const strong = good && pick(STRENGTHS[good.key] || ['holds its own'], `${id}:s:${good.key}`);
  const weak = bad && pick(WEAKNESSES[bad.key] || ['has a soft spot somewhere'], `${id}:w:${bad.key}`);

  if (strong && weak) return `${first} It ${strong}, but ${weak}.`;
  if (strong) return `${first} It ${strong}.`;
  if (weak) return `${first} It ${weak}.`;
  return first;
}

// --- stat hexagon -------------------------------------------------------

const R = 78, CX = 108, CY = 100;
const point = (i, r) => {
  const a = (Math.PI / 3) * i - Math.PI / 2;
  return [(CX + r * Math.cos(a)).toFixed(1), (CY + r * Math.sin(a)).toFixed(1)];
};
const ring = r => [0, 1, 2, 3, 4, 5].map(i => point(i, r).join(',')).join(' ');

/**
 * Inline radar chart for up to two fighters.
 *
 * @param series  [{ label, color, axes }] — axes carry {label, score|null}
 * @param titleId  id of the element describing the chart, for aria-labelledby
 */
export function hexSvg(series, titleId) {
  const axes = series[0].axes;
  const grid = [0.25, 0.5, 0.75, 1]
    .map(f => `<polygon points="${ring(R * f)}" fill="none" stroke="#3a3238" stroke-width="1"/>`)
    .join('');
  const spokes = axes.map((_, i) =>
    `<line x1="${CX}" y1="${CY}" x2="${point(i, R)[0]}" y2="${point(i, R)[1]}" stroke="#3a3238" stroke-width="1"/>`).join('');

  const hatch = axes.map((a, i) => {
    if (a.score != null) return '';
    const [x1, y1] = point(i, R);
    const wedge = [[CX, CY], point(i - 0.28, R), point(i, R), point(i + 0.28, R)]
      .map(p => p.join(',')).join(' ');
    return `<polygon points="${wedge}" fill="url(#dex-hatch)" stroke="none"/>`
      + `<line x1="${CX}" y1="${CY}" x2="${x1}" y2="${y1}" stroke="#9a8d94" stroke-width="1" stroke-dasharray="3 3"/>`;
  }).join('');

  const shapes = series.map((s) => {
    const pts = s.axes.map((a, i) => point(i, R * ((a.score || 0) / 100)).join(',')).join(' ');
    const dots = s.axes.map((a, i) => {
      const [x, y] = point(i, R * ((a.score || 0) / 100));
      const raw = a.score == null ? 'unmeasured' : `${a.score}/100${a.raw ? ` — ${a.raw}` : ''}`;
      return `<circle cx="${x}" cy="${y}" r="2.6" fill="${s.color}"><title>${esc(s.label)} · ${esc(a.label)}: ${esc(raw)}</title></circle>`;
    }).join('');
    return `<polygon points="${pts}" fill="${s.color}" fill-opacity="0.18" stroke="${s.color}" stroke-width="2"/>${dots}`;
  }).join('');

  const labels = axes.map((a, i) => {
    const [x, y] = point(i, R + 16);
    const anchor = i === 0 || i === 3 ? 'middle' : (i < 3 ? 'start' : 'end');
    return `<text x="${x}" y="${y}" text-anchor="${anchor}" dominant-baseline="middle"
        font-size="9" fill="#ab9a9c" font-family="inherit">${esc(a.label)}</text>`;
  }).join('');

  return `<svg viewBox="0 0 216 200" width="100%" style="max-width:16rem" role="img" aria-labelledby="${titleId}">
    <defs><pattern id="dex-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="#9a8d94" stroke-width="1.5"/></pattern></defs>
    ${grid}${spokes}${hatch}${shapes}${labels}</svg>`;
}

// --- build-time index ---------------------------------------------------

const median = xs => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const spread = xs => (xs.length ? { min: Math.min(...xs), max: Math.max(...xs) } : null);

// Same exact-match replay the matchups page uses: benchmark-seeded battle
// pools ship the reference and both predictions, freeform pools do not and
// contribute no signal.
function battleOutcome(b) {
  if (b.reference == null || b.prediction_a == null || b.prediction_b == null) return null;
  const norm = v => {
    if (v && typeof v === 'object') v = v.intent ?? v.text ?? JSON.stringify(v);
    return String(v ?? '').trim().toLowerCase();
  };
  const ref = norm(b.reference);
  const a = norm(b.prediction_a) === ref;
  const c = norm(b.prediction_b) === ref;
  if (a && !c) return 'a';
  if (c && !a) return 'b';
  return 'tie';
}

const pipelineOf = c => c.config?.intents?.pipeline || [];
const stageBase = stage => String(stage).replace(/-(high|medium|low)$/, '');

/**
 * Load every committed board once and derive everything the fighter pages
 * need. `readJson(name)` resolves a file under `public/data`.
 */
export function buildDexIndex(readJson) {
  const index = readJson('index.json');
  const roster = readJson('competitors.json').competitors || [];

  const byModality = new Map();
  for (const c of roster) {
    if (!byModality.has(c.modality)) byModality.set(c.modality, []);
    byModality.get(c.modality).push(c);
  }

  const dexNumbers = new Map();
  for (const [, list] of byModality) {
    list.map(c => c.competitor_id).sort()
      .forEach((id, i) => dexNumbers.set(id, i + 1));
  }

  // Leaderboard rank and rating, per fighter and per language board.
  const standings = new Map();
  for (const meta of index.leaderboards || []) {
    const board = readJson(meta.file);
    for (const e of board.entries || []) {
      if (!standings.has(e.competitor_id)) standings.set(e.competitor_id, []);
      standings.get(e.competitor_id).push({
        lang: meta.lang, modality: meta.modality, rank: e.rank,
        rating: Math.round(e.bt_rating ?? e.elo ?? 0),
      });
    }
  }

  // Benchmark rows, plus the league-wide spreads the log-scaled axes need.
  const rows = new Map();
  const scales = new Map();
  const raw = new Map();
  for (const meta of index.benchmarks || []) {
    const board = readJson(meta.file);
    if (!raw.has(meta.modality)) raw.set(meta.modality, { latency: [], rtf: [], rss: [] });
    const bucket = raw.get(meta.modality);
    for (const e of board.entries || []) {
      const row = {
        lang: meta.lang, dataset_id: meta.dataset_id, samples: e.samples || 0,
        metrics: e.metrics || {}, perf: e.perf, model_mb: e.model_mb, rank: e.rank,
      };
      if (!rows.has(e.competitor_id)) rows.set(e.competitor_id, []);
      rows.get(e.competitor_id).push(row);
      const lat = num(row.metrics.latency_ms_median);
      if (lat != null && lat > 0) bucket.latency.push(lat);
      const rtf = rowRtf(row);
      if (rtf != null && rtf > 0) bucket.rtf.push(rtf);
      const rss = rowRss(row);
      if (rss != null && rss > 0) bucket.rss.push(rss);
    }
  }
  // A fighter that declares no languages in the registry (an engine that
  // runs anywhere) is credited with the boards it actually competes on.
  const langCounts = new Map();
  for (const c of roster) {
    const declared = (c.langs || []).length;
    const measured = new Set((rows.get(c.competitor_id) || []).map(r => r.lang)).size;
    langCounts.set(c.competitor_id, declared || measured);
  }

  for (const [modality, bucket] of raw) {
    const list = byModality.get(modality) || [];
    const ratings = list.flatMap(c => (standings.get(c.competitor_id) || []).map(s => s.rating));
    scales.set(modality, {
      latency: spread(bucket.latency), rtf: spread(bucket.rtf), rss: spread(bucket.rss),
      rating: spread(ratings),
      maxLangs: Math.max(1, ...list.map(c => langCounts.get(c.competitor_id) || 0)),
      medianRss: median(bucket.rss),
      polyglotLangs: (() => {
        const counts = list.map(c => langCounts.get(c.competitor_id) || 0).sort((a, b) => b - a);
        return counts.length ? counts[Math.floor(counts.length * 0.1)] : 0;
      })(),
    });
  }

  // Head-to-head records replayed from the battle pools.
  const h2h = new Map();
  const bump = (a, b, field) => {
    if (!h2h.has(a)) h2h.set(a, new Map());
    const m = h2h.get(a);
    if (!m.has(b)) m.set(b, { wins: 0, losses: 0, ties: 0, total: 0 });
    const rec = m.get(b);
    rec[field]++; rec.total++;
  };
  for (const meta of index.battles_pools || []) {
    const pool = readJson(meta.file);
    for (const b of pool.battles || []) {
      const outcome = battleOutcome(b);
      if (outcome == null) continue;
      if (outcome === 'tie') { bump(b.competitor_a, b.competitor_b, 'ties'); bump(b.competitor_b, b.competitor_a, 'ties'); }
      else if (outcome === 'a') { bump(b.competitor_a, b.competitor_b, 'wins'); bump(b.competitor_b, b.competitor_a, 'losses'); }
      else { bump(b.competitor_b, b.competitor_a, 'wins'); bump(b.competitor_a, b.competitor_b, 'losses'); }
    }
  }

  // Fusions (multi-stage pipelines) against the single-engine fighters that
  // run each of their stages.
  const singlesByPlugin = new Map();
  const fusions = [];
  for (const c of roster) {
    const pipeline = pipelineOf(c);
    if (pipeline.length === 1) {
      const base = stageBase(pipeline[0]);
      if (!singlesByPlugin.has(base)) singlesByPlugin.set(base, []);
      singlesByPlugin.get(base).push(c);
    } else if (pipeline.length > 1) {
      fusions.push(c);
    }
  }
  for (const list of singlesByPlugin.values()) list.sort((a, b) => a.competitor_id.localeCompare(b.competitor_id));

  const nameOf = new Map(roster.map(c => [c.competitor_id, c.display_name || c.competitor_id]));

  return { index, roster, byModality, dexNumbers, standings, rows, scales, h2h,
           singlesByPlugin, fusions, nameOf, langCounts };
}

// The board a fighter's stats come from for a given language: its widest
// dataset there, ties broken by dataset id so the pick never drifts.
function bestRow(rowsForFighter, lang) {
  const candidates = rowsForFighter.filter(r => r.lang === lang);
  if (!candidates.length) return null;
  return [...candidates].sort((a, b) =>
    b.samples - a.samples || String(a.dataset_id).localeCompare(String(b.dataset_id)))[0];
}

/** Per-language axis values for one fighter, ready to render. */
export function dexStats(fighter, dex) {
  const rowsForFighter = dex.rows.get(fighter.competitor_id) || [];
  const ctx = dex.scales.get(fighter.modality) || {};
  const standing = dex.standings.get(fighter.competitor_id) || [];
  const specs = axesFor(fighter.modality);
  const langCount = dex.langCounts.get(fighter.competitor_id) || (fighter.langs || []).length;
  const langs = [...new Set(rowsForFighter.map(r => r.lang))].sort();

  const byLang = {};
  for (const lang of langs) {
    const row = { ...bestRow(rowsForFighter, lang), langCount };
    row.rating = standing.find(s => s.lang === lang)?.rating ?? null;
    byLang[lang] = {
      dataset_id: row.dataset_id,
      samples: row.samples,
      axes: specs.map(spec => {
        const got = spec.get(row, ctx);
        return {
          key: spec.key, label: spec.label, help: spec.help,
          score: got?.score ?? null, raw: got?.raw || '', flavor: got?.flavor ?? null,
        };
      }),
    };
  }
  return { langs, byLang };
}

/** The language whose board a visitor should see first. */
export function preferredLang(langs, wanted) {
  for (const w of wanted) {
    if (langs.includes(w)) return w;
    const near = langs.find(l => l.split('-')[0] === String(w).split('-')[0]);
    if (near) return near;
  }
  return langs[0] || null;
}

/** Titles a fighter earned on the boards, computed from the same data. */
export function dexBadges(fighter, dex) {
  const standing = dex.standings.get(fighter.competitor_id) || [];
  const ctx = dex.scales.get(fighter.modality) || {};
  const badges = [];
  for (const s of standing.filter(s => s.rank === 1).sort((a, b) => a.lang.localeCompare(b.lang))) {
    badges.push({ title: `Champion of ${s.lang}`, why: `Rank 1 on the ${s.lang} board.` });
  }
  const rss = (dex.rows.get(fighter.competitor_id) || []).map(rowRss).filter(v => v != null);
  const light = rss.length ? Math.min(...rss) : null;
  if (light != null && ctx.medianRss != null && light < ctx.medianRss && standing.some(s => s.rank <= 3)) {
    badges.push({
      title: 'Featherweight',
      why: `Top-three finish on ${Math.round(light)} MB, under the league median of ${Math.round(ctx.medianRss)} MB.`,
    });
  }
  const langCount = dex.langCounts.get(fighter.competitor_id) || 0;
  if (ctx.polyglotLangs && langCount >= ctx.polyglotLangs && langCount > 1) {
    badges.push({ title: 'Polyglot', why: `Covers ${langCount} languages — top tenth of its league.` });
  }
  return badges;
}

/** Nemesis and prey, from head-to-head records with at least 5 decided battles. */
export function dexRivals(fighter, dex) {
  const recs = [...(dex.h2h.get(fighter.competitor_id) || new Map()).entries()]
    .filter(([, r]) => r.total >= 5)
    .map(([id, r]) => ({ id, name: dex.nameOf.get(id) || id, ...r, winRate: (r.wins + r.ties * 0.5) / r.total }))
    .sort((a, b) => a.winRate - b.winRate || a.id.localeCompare(b.id));
  if (!recs.length) return { nemesis: null, prey: null };
  const nemesis = recs[0];
  const prey = recs[recs.length - 1];
  return {
    nemesis: nemesis.winRate < 0.5 ? nemesis : null,
    prey: prey.winRate > 0.5 && prey.id !== nemesis.id ? prey : null,
  };
}

/** Pipeline stages as "moves": tier, engine and the thresholds it fires at. */
export function dexMoves(fighter) {
  const conf = fighter.config?.intents || {};
  return pipelineOf(fighter).map(stage => {
    const tier = (stage.match(/-(high|medium|low)$/) || [, '—'])[1];
    const base = stageBase(stage);
    const settings = conf[base] || {};
    const thresholds = Object.entries(settings)
      .filter(([k]) => k.startsWith('conf_'))
      .map(([k, v]) => `${k.replace('conf_', '')} ${v}`)
      .join(' · ');
    return { stage, base, tier, thresholds };
  });
}

const slim = c => ({ competitor_id: c.competitor_id, display_name: c.display_name || c.competitor_id });

/** Fusion parents, or the fusions a single engine is part of. */
export function dexEvolutions(fighter, dex) {
  const pipeline = pipelineOf(fighter);
  if (pipeline.length > 1) {
    const bases = [...new Set(pipeline.map(stageBase))];
    return {
      kind: 'fusion',
      parents: bases.map(base => {
        const all = dex.singlesByPlugin.get(base) || [];
        return { base, fighters: all.slice(0, 3).map(slim), more: Math.max(0, all.length - 3) };
      }),
    };
  }
  if (pipeline.length === 1) {
    const base = stageBase(pipeline[0]);
    const into = dex.fusions
      .filter(f => pipelineOf(f).some(s => stageBase(s) === base))
      .sort((a, b) => a.competitor_id.localeCompare(b.competitor_id))
      .map(slim);
    return { kind: 'engine', base, into };
  }
  return { kind: 'none' };
}
