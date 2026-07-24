#!/usr/bin/env node
// Automated accessibility gate for the built Astro site.
//
// Serves dist/ over plain HTTP (base path stripped — the built HTML works
// fine from the filesystem root), runs axe-core against a representative
// set of pages via Playwright + Chromium, and fails the process with a
// non-zero exit code if any "critical" or "serious" violation is found.
//
// Run: `npm run build && npm run a11y`
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(__dirname, '..', 'dist');
const PORT = 4319;
// astro.config.mjs sets a non-root `base` (GitHub Pages subpath) that gets
// baked into every client-side fetch()/asset URL. The built HTML files
// still live at the filesystem root of dist/, so the local server strips
// the base prefix off incoming requests before resolving them on disk —
// that way the axe scan below exercises the real, data-hydrated pages
// (leaderboards, battle pool, vote pool) instead of their empty/loading
// skeletons.
const BASE = (process.env.ASTRO_BASE || '/ovos-plugin-arena').replace(/\/+$/, '');

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml',
  '.xml': 'application/xml', '.txt': 'text/plain', '.png': 'image/png',
  '.wav': 'audio/wav', '.ico': 'image/x-icon',
};

function startServer() {
  const server = createServer(async (req, res) => {
    try {
      let reqPath = decodeURIComponent(req.url.split('?')[0]);
      if (BASE && reqPath.startsWith(BASE)) reqPath = reqPath.slice(BASE.length) || '/';
      let filePath = path.join(DIST, reqPath);
      let st = await stat(filePath).catch(() => null);
      if (st?.isDirectory()) filePath = path.join(filePath, 'index.html');
      st = await stat(filePath).catch(() => null);
      if (!st) {
        res.writeHead(404);
        res.end('not found');
        return;
      }
      const body = await readFile(filePath);
      res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
      res.end(body);
    } catch (e) {
      res.writeHead(500);
      res.end(String(e));
    }
  });
  return new Promise((resolve) => server.listen(PORT, () => resolve(server)));
}

// Representative page set: index, a leaderboard, one battle page, one
// fighter page, methodology, patch-notes, and the free-vote page.
const PAGES = [
  ['/', 'home'],
  ['/leaderboard/', 'leaderboard'],
  ['/battle/', 'battle (blind vote flow)'],
  ['/fighter/azure-stt-en/', 'fighter detail'],
  ['/fighters/', 'fighters bestiary'],
  ['/methodology/', 'methodology'],
  ['/patch-notes/', 'patch notes'],
  ['/vote/', 'free vote flow'],
].map(([route, label]) => [route === '/' ? `${BASE}/` : `${BASE}${route}`, label]);

const FAIL_IMPACTS = new Set(['critical', 'serious']);

async function main() {
  const server = await startServer();
  const browser = await chromium.launch();
  const context = await browser.newContext();
  let hadFailures = false;
  const summary = [];

  try {
    for (const [route, label] of PAGES) {
      const page = await context.newPage();
      await page.goto(`http://127.0.0.1:${PORT}${route}`, { waitUntil: 'networkidle' });
      // let client-side data hydration (fetch of dist/data/*.json) settle
      await page.waitForTimeout(600);

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze();

      const bad = results.violations.filter(v => FAIL_IMPACTS.has(v.impact));
      summary.push({ route, label, total: results.violations.length, bad: bad.length });

      if (bad.length) {
        hadFailures = true;
        console.error(`\n✗ ${label} (${route}) — ${bad.length} critical/serious violation(s):`);
        for (const v of bad) {
          console.error(`  [${v.impact}] ${v.id}: ${v.help} (${v.helpUrl})`);
          for (const node of v.nodes.slice(0, 5)) {
            console.error(`      target: ${node.target.join(' ')}`);
            console.error(`      html: ${node.html.slice(0, 160)}`);
          }
        }
      } else {
        console.log(`✓ ${label} (${route}) — clean (${results.violations.length} non-blocking notice(s))`);
      }
      await page.close();
    }
  } finally {
    await context.close();
    await browser.close();
    server.close();
  }

  console.log('\n--- a11y gate summary ---');
  for (const s of summary) {
    console.log(`${s.bad ? '✗' : '✓'} ${s.label.padEnd(28)} critical/serious: ${s.bad}  (other: ${s.total - s.bad})`);
  }

  if (hadFailures) {
    console.error('\na11y gate FAILED — critical/serious violations found above.');
    process.exit(1);
  }
  console.log('\na11y gate PASSED — no critical/serious violations on the representative page set.');
}

main().catch((e) => {
  console.error('a11y gate crashed:', e);
  process.exit(1);
});
