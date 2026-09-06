#!/usr/bin/env node
// Every flavor sentence on a fighter page must be earned by the fighter's own
// measurements: a stat is praised or mocked only when its raw value clears an
// absolute bar, never because it happened to be the highest or lowest of the
// six axes. This checks that for every competitor, with the thresholds
// restated here rather than imported, so a wrong predicate in dex.js cannot
// pass by agreeing with itself.
//
// Run: `npm run dex-audit`
import fs from 'node:fs';
import path from 'node:path';
import { buildDexIndex, dexStats, flavorText, preferredLang, STRENGTHS, WEAKNESSES } from '../src/lib/dex.js';

const DATA = path.join(process.cwd(), 'public', 'data');
const readJson = f => JSON.parse(fs.readFileSync(path.join(DATA, f), 'utf-8'));
const dex = buildDexIndex(readJson);

const pct = raw => parseFloat(raw) / 100;
const first = raw => parseFloat(raw);

// (modality, axis key) -> how to read the raw string and what the sentence claims.
const RULES = {
  'range': { v: r => first(r), good: (v, ctx) => ctx.decile > 1 && v >= ctx.decile, bad: v => v === 1 },
  'footprint': { v: r => first(r), good: v => v <= 200, bad: v => v >= 2000 },
  'speed': { v: r => first(r), good: (v, c, raw) => (raw.includes('RTF') ? v <= 0.1 : v <= 50),
             bad: (v, c, raw) => (raw.includes('RTF') ? v >= 1 : v >= 1000) },
  'latency': { v: r => first(r), good: v => v <= 50, bad: v => v >= 1000 },
  'robustness': { v: r => r.split('/').map(pct).reduce((a, b) => a + b, 0) / r.split('/').length,
                  good: v => v >= 0.75, bad: v => v <= 0.5 },
  'judgement': { v: r => 1 - pct(r), good: v => v >= 0.8, bad: v => v <= 0.5 },
  'precision': { v: r => 1 - pct(r), good: v => v >= 0.95, bad: v => v <= 0.7 },
  'recall': { v: r => 1 - pct(r), good: v => v >= 0.95, bad: v => v <= 0.7 },
  'consistency': { v: r => 1 - pct(r), good: v => v >= 0.85, bad: v => v <= 0.5 },
  'naturalness': { v: r => first(r), good: v => v >= 4, bad: v => v <= 2.5 },
  'intelligibility': { v: r => pct(r), good: v => v <= 0.15, bad: v => v >= 0.5 },
  'acclaim': { v: r => first(r), good: v => v >= 1400, bad: v => v <= 1100 },
  'accuracy': {
    v: (r) => (r.includes('WER') || r.includes('error rate') ? 1 - pct(r) : pct(r)),
    good: (v, c, raw) => (raw.includes('error rate') ? v >= 0.95 : raw.includes('WER') ? v >= 0.85 : v >= 0.8),
    bad: (v, c, raw) => (raw.includes('error rate') ? v <= 0.7 : raw.includes('WER') ? v <= 0.5 : v <= 0.5),
  },
};

const phraseKey = new Map();
for (const [key, list] of Object.entries(STRENGTHS)) for (const p of list) phraseKey.set(p, [key, 'good']);
for (const [key, list] of Object.entries(WEAKNESSES)) for (const p of list) phraseKey.set(p, [key, 'bad']);

let checked = 0, clauses = 0, failures = 0;
const texts = new Set();
for (const c of dex.roster) {
  const stats = dexStats(c, dex);
  const lang = preferredLang(stats.langs, ['en-US']);
  const axes = lang ? stats.byLang[lang].axes : [];
  const text = flavorText(c, axes);
  texts.add(text);
  checked++;
  const ctx = { decile: dex.scales.get(c.modality)?.polyglotLangs ?? 0 };
  for (const [phrase, [key, polarity]] of phraseKey) {
    if (!text.includes(phrase)) continue;
    clauses++;
    const axis = axes.find(a => a.key === key);
    const rule = RULES[key];
    if (!axis || axis.score == null || !axis.raw) {
      console.log(`FAIL ${c.competitor_id}: claims "${phrase}" with no measured ${key} axis`);
      failures++;
      continue;
    }
    const value = rule.v(axis.raw);
    const holds = polarity === 'good' ? rule.good(value, ctx, axis.raw) : rule.bad(value, ctx, axis.raw);
    if (!holds) {
      console.log(`FAIL ${c.competitor_id} (${c.modality}): "${phrase}" but ${key} raw = "${axis.raw}" (value ${value})`);
      failures++;
    }
  }
}
console.log(`flavor audit: ${checked} fighters checked, ${clauses} clauses verified against raw measurements, ${failures} contradictions`);
console.log(`distinct flavor texts: ${texts.size} across ${checked} fighters`);
process.exit(failures ? 1 : 0);
