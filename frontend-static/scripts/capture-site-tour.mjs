#!/usr/bin/env node
// Captures every screenshot that docs/site-tour.md shows, at the same file
// names under docs/img/.
//
// The site comes from the local build (dist/). The data comes from a copy of
// public/data that this script refreshes from the registry first, with the
// same offline steps the assemble workflow runs before it publishes:
// prune-data, tally, export-bestiary, export-index and export-evidence. Thus
// the screenshots show the roster of the checked-out registry, not the
// roster of the last data refresh. Set TOUR_RAW_DATA=1 to skip the refresh
// and serve public/data exactly as committed.
//
// Run from frontend-static/:
//   npm run build && ARENA_PYTHON=../.venv/bin/python node scripts/capture-site-tour.mjs
// ARENA_PYTHON is any interpreter that can import the `arena` package.
import { spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { cp, mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.join(__dirname, '..');
const REPO = path.join(FRONTEND, '..');
const DIST = path.join(FRONTEND, 'dist');
const OUT = path.join(REPO, 'docs', 'img');
const PORT = Number(process.env.TOUR_PORT || 4322);
const BASE = (process.env.ASTRO_BASE || '/ovos-plugin-arena').replace(/\/+$/, '');
const ROOT = `http://127.0.0.1:${PORT}${BASE}`;
const PYTHON = process.env.ARENA_PYTHON?.includes('/') ? path.resolve(process.env.ARENA_PYTHON) : (process.env.ARENA_PYTHON || 'python3');
const DESKTOP = { width: 1440, height: 900 };
const MOBILE = { width: 390, height: 844 };
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.jsonl': 'text/plain', '.svg': 'image/svg+xml', '.png': 'image/png' };

function arena(...args) {
  const run = spawnSync(PYTHON, ['-m', 'arena.cli', ...args], { cwd: REPO, stdio: ['ignore', 'ignore', 'inherit'] });
  if (run.status !== 0) throw new Error(`arena.cli ${args[0]} failed: ${run.error?.message || `exit ${run.status}`}`);
}

async function stageData(tmp) {
  const data = path.join(tmp, 'data');
  await cp(path.join(FRONTEND, 'public', 'data'), data, { recursive: true });
  arena('prune-data', '--registry', 'registry', '--data-dir', data);
  // An empty --repo keeps tally offline: with GITHUB_REPOSITORY set it would
  // fetch and close the real vote issues.
  arena('tally', '--data-dir', data, '--output', data, '--repo', '', '--keep-issues-open');
  arena('export-bestiary', '--registry', 'registry', '--data-dir', data, '--output', path.join(data, 'competitors.json'));
  arena('export-index', '--data-dir', data, '--output', path.join(data, 'index.json'));
  arena('export-evidence', '--registry', 'registry', '--data-dir', data,
    '--spec', 'docs/SPECIFICATION.md', '--output', path.join(data, 'evidence.json'));
  return data;
}

function startServer(dataDir) {
  const server = createServer(async (req, res) => {
    let reqPath = decodeURIComponent(req.url.split('?')[0]);
    if (BASE && reqPath.startsWith(BASE)) reqPath = reqPath.slice(BASE.length) || '/';
    const root = reqPath.startsWith('/data/') ? dataDir : DIST;
    let file = path.join(root, reqPath.startsWith('/data/') ? reqPath.slice('/data/'.length) : reqPath);
    if (!file.startsWith(root + path.sep) && file !== root) { res.writeHead(403); res.end('forbidden'); return; }
    if ((await stat(file).catch(() => null))?.isDirectory()) file = path.join(file, 'index.html');
    if (!(await stat(file).catch(() => null))) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  });
  return new Promise(resolve => server.listen(PORT, '127.0.0.1', () => resolve(server)));
}

// Each shot loads a page, prepares it, and saves either the viewport or one
// element. `clip` names the element; `scrollTo` puts an element at the top
// of the viewport before a viewport shot.
const shots = [
  { file: 'home.png', url: '/', ready: '.stat, main' },
  { file: 'leaderboard-overview.png', url: '/leaderboard/?league=stt&lang=de-DE', ready: '.podium' },
  {
    file: 'leaderboard-group-expanded.png', url: '/leaderboard/?league=intent_online&lang=en-US', ready: '.lrow-group',
    async prepare(page) {
      const group = page.locator('details.lrow-group').first();
      await group.evaluate(el => { el.open = true; });
      return group;
    },
  },
  { file: 'leaderboard-unranked.png', url: '/leaderboard/?league=tts&lang=de-DE', ready: '.lb-section-head h2:has-text("Unranked")', scrollTo: '.lb-section-head' },
  {
    file: 'leaderboard-benchmark-utmos.png', url: '/leaderboard/?league=tts&lang=en-US&dataset=intents-for-eval-prompts',
    ready: '.bench-tbl', scrollTo: '.bench:has(.bench-tbl)',
  },
  { file: 'battle.png', url: '/battle/?league=stt&lang=en-US', ready: '#cand-a' },
  {
    file: 'matchups.png', url: '/matchups/?league=intent_online&lang=en-US&mode=bench', ready: '#mu-panels table, #mu-panels svg',
  },
  { file: 'fighters-dex.png', url: '/fighters/?league=intent_online', ready: '#dex-grid > *', scrollTo: '#dex-grid' },
  {
    file: 'fighters-dex-expanded.png', url: '/fighters/?league=intent_online', ready: '#dex-grid details.group-details',
    async prepare(page) {
      const group = page.locator('#dex-grid details.group-details').first();
      await group.locator('summary').click();
      await group.locator('.group-members > *').first().waitFor();
      return group;
    },
  },
  {
    file: 'fighter-detail.png', url: '/fighter/phoonnx-piper-piper-en-us-lessac-high/', ready: 'main',
    scrollTo: 'pre',
  },
  { file: 'evidence.png', url: '/evidence/', ready: '#ev-tbody tr', scrollTo: '#ev-app h2' },
  {
    file: 'evidence-fighter-coverage.png', url: '/evidence/', ready: '#ev-fighters-tbody tr',
    prepare: page => page.locator('#ev-fighters-table'),
  },
  { file: 'leaderboard-mobile.png', url: '/leaderboard/?league=intent_online&lang=en-US', ready: '#league-tabs .lb-tab', viewport: MOBILE },
];

async function capture(browser, shot) {
  const page = await browser.newPage({ viewport: shot.viewport || DESKTOP });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(ROOT + shot.url, { waitUntil: 'networkidle' });
  await page.waitForSelector(shot.ready, { timeout: 15000 });
  await page.waitForTimeout(500);
  const target = shot.prepare ? await shot.prepare(page) : null;
  if (shot.scrollTo) {
    await page.locator(shot.scrollTo).first().evaluate(el => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -80));
  }
  await page.waitForTimeout(300);
  const file = path.join(OUT, shot.file);
  // The sticky site header would cover the top of an element shot.
  if (target) await page.addStyleTag({ content: 'header { position: static !important; }' });
  if (target) await target.screenshot({ path: file });
  else await page.screenshot({ path: file });
  await page.close();
  if (errors.length) throw new Error(`${shot.file}: page errors: ${errors.join('; ')}`);
  console.log(`wrote ${path.relative(REPO, file)}`);
}

const tmp = await mkdtemp(path.join(os.tmpdir(), 'arena-site-tour-'));
let server;
let browser;
let failed = 0;
try {
  const dataDir = process.env.TOUR_RAW_DATA ? path.join(FRONTEND, 'public', 'data') : await stageData(tmp);
  server = await startServer(dataDir);
  browser = await chromium.launch();
  for (const shot of shots) {
    try { await capture(browser, shot); } catch (e) { failed++; console.error(`FAIL ${shot.file}: ${e.message}`); }
  }
  console.log(`${shots.length - failed} of ${shots.length} screenshots written`);
} finally {
  await browser?.close();
  server?.close();
  await rm(tmp, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
