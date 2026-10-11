#!/usr/bin/env node
// Behaviour checks for the built site: the per-wake-word leagues, the vote
// flows and the filter controls, driven in headless Chromium against the
// fixed arena in ui-fixtures.mjs (every request for data/ is answered from
// it, so a data refresh never changes the outcome).
//
// Run: `npm run build && npm run ui-check`. Exits non-zero on any failure.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import fixtures from './ui-fixtures.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(__dirname, '..', 'dist');
const PORT = Number(process.env.UI_CHECK_PORT || 4321);
const BASE = (process.env.ASTRO_BASE || '/ovos-plugin-arena').replace(/\/+$/, '');
const ROOT = `http://localhost:${PORT}${BASE}`;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };

function startServer() {
  const server = createServer(async (req, res) => {
    let reqPath = decodeURIComponent(req.url.split('?')[0]);
    if (BASE && reqPath.startsWith(BASE)) reqPath = reqPath.slice(BASE.length) || '/';
    let file = path.join(DIST, reqPath);
    if ((await stat(file).catch(() => null))?.isDirectory()) file = path.join(file, 'index.html');
    if (!(await stat(file).catch(() => null))) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  });
  return new Promise(resolve => server.listen(PORT, () => resolve(server)));
}

async function newPage(browser, { blockPopups = false } = {}) {
  const context = await browser.newContext();
  await context.route(/\/data\//, route => {
    const name = new URL(route.request().url()).pathname.split('/data/')[1];
    const body = fixtures[name];
    if (body === undefined) return route.fulfill({ status: 404, body: 'not found' });
    return route.fulfill({ status: 200, contentType: name.endsWith('.json') ? 'application/json' : 'text/plain',
      body: typeof body === 'string' ? body : JSON.stringify(body) });
  });
  // The vote issue page: it reports whether it can still reach the arena tab.
  await context.route(/^https:\/\/github\.com\//, route => route.fulfill({
    status: 200, contentType: 'text/html',
    body: '<script>document.title = window.opener === null ? "opener-null" : "opener-live";</script>',
  }));
  if (blockPopups) await context.addInitScript(() => { window.open = () => null; });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  return { context, page, errors };
}

function expect(cond, message) {
  if (!cond) throw new Error(message);
}

// The vote issue tab must open and must not hold a live `window.opener`,
// or the issue page could navigate the arena tab.
async function voteTab(page, click) {
  const [popup] = await Promise.all([page.context().waitForEvent('page'), page.click(click)]);
  await popup.waitForLoadState();
  await popup.waitForFunction(() => document.title.startsWith('opener-'));
  return popup.title();
}

const text = async (page, sel) => (await page.textContent(sel) || '').replace(/\s+/g, ' ').trim();
const rowsOf = (page, sel) => page.$$eval(sel, rows => rows.map(r => r.textContent.replace(/\s+/g, ' ').trim()));

const checks = {
  async 'wake word overview lists one league per phrase'(page) {
    await page.goto(`${ROOT}/leaderboard/?league=wake_word&lang=en-US`);
    await page.waitForSelector('.ww-overview', { timeout: 5000 });
    const rows = await rowsOf(page, '.ww-overview tbody tr');
    const words = rows.map(r => r.match(/“([^”]+)”/)?.[1]);
    expect(JSON.stringify([...words].sort()) === JSON.stringify(['hey jarvis', 'hey mycroft', 'jarvis', 'sam']),
      `wake words ${JSON.stringify(words)}`);
    const mycroft = rows.find(r => r.startsWith('“hey mycroft”'));
    expect(/^“hey mycroft” 2 2 myc-precise/.test(mycroft), `hey mycroft row: ${mycroft}`);
    expect(!(await page.locator('.bench').count()), 'overview still renders benchmark boards');
  },

  async 'wake word page shows only that phrase, clip and stream boards'(page) {
    await page.goto(`${ROOT}/leaderboard/?league=wake_word&lang=en-US&word=hey_mycroft`);
    await page.waitForSelector('.ds-intro', { timeout: 5000 });
    const boards = await page.$$eval('.ds-intro .ds-id', e => e.map(x => x.textContent));
    expect(JSON.stringify(boards) === JSON.stringify(['synthetic-hey_mycroft', 'ww_stream_hey_mycroft']),
      `boards ${JSON.stringify(boards)}`);
    const names = await page.$$eval('#lb-panels .nm a', a => [...new Set(a.map(x => x.textContent))].sort());
    expect(JSON.stringify(names) === JSON.stringify(['myc-oww', 'myc-precise']), `fighters ${JSON.stringify(names)}`);
    expect(await page.inputValue('#sel-word') === 'hey_mycroft', 'word select does not show the phrase');
  },

  async 'wake word choice survives reload and resets to the overview'(page) {
    await page.goto(`${ROOT}/leaderboard/?league=wake_word&lang=en-US`);
    await page.waitForSelector('.ww-overview', { timeout: 5000 });
    await page.selectOption('#sel-word', 'sam');
    await page.waitForSelector('.ds-intro');
    expect(new URL(page.url()).searchParams.get('word') === 'sam', `url ${page.url()}`);
    await page.reload();
    await page.waitForSelector('.ds-intro');
    expect(await text(page, '.ds-intro .ds-id') === 'sam-wake-word', 'sam board missing after reload');
    await page.click('a[data-word=""]');
    await page.waitForSelector('.ww-overview');
    expect(!new URL(page.url()).searchParams.has('word'), `url keeps word: ${page.url()}`);
    await page.click('.lb-tab[data-league="stt"]');
    await page.waitForSelector('.ds-intro');
    expect(!(await page.isVisible('#sel-word')), 'word select shown outside the wake-word league');
  },

  async 'several boards fold into one card per dataset'(page) {
    await page.goto(`${ROOT}/leaderboard/?league=stt&lang=en-US`);
    await page.waitForSelector('.ds-intro', { timeout: 5000 });
    const folds = await page.$$eval('.bench-fold', d => d.map(x => x.open));
    expect(JSON.stringify(folds) === JSON.stringify([true, false]), `folds ${JSON.stringify(folds)}`);
  },

  async 'blind vote does not report a pop-up that opened as blocked'(page) {
    await page.goto(`${ROOT}/battle/?league=stt&lang=en-US`);
    await page.waitForSelector('#vote-a', { state: 'visible', timeout: 5000 });
    const opener = await voteTab(page, '#vote-a');
    expect(opener === 'opener-null', `vote tab reports ${opener}`);
    await page.waitForSelector('#vote-waiting', { state: 'visible' });
    expect(!(await page.isVisible('#popup-blocked-note')), 'blocked note shown for a pop-up that opened');
  },

  async 'free vote opens the issue tab without calling it blocked'(page) {
    await page.goto(`${ROOT}/vote/`);
    await page.waitForSelector('#fv-app', { state: 'visible', timeout: 5000 });
    const opener = await voteTab(page, '#fv-win-a');
    expect(opener === 'opener-null', `vote tab reports ${opener}`);
    const hint = await text(page, '#fv-hint');
    expect(hint.startsWith('Opening a GitHub issue') && !/blocked/i.test(hint), `hint "${hint}"`);
  },

  async 'blind vote still offers the issue link when the pop-up is blocked'(page) {
    await page.goto(`${ROOT}/battle/?league=stt&lang=en-US`);
    await page.waitForSelector('#vote-a', { state: 'visible', timeout: 5000 });
    await page.click('#vote-a');
    expect(await page.isVisible('#popup-blocked-note'), 'blocked note hidden');
    expect((await page.getAttribute('#popup-link', 'href') || '').includes('issues/new'), 'no issue link');
  },

  async 'a confirmed blind vote never comes back in the same session'(page) {
    await page.goto(`${ROOT}/battle/?league=stt&lang=en-US`);
    await page.waitForSelector('#vote-a', { state: 'visible', timeout: 5000 });
    const voted = await text(page, '#battle-id');
    await page.click('#vote-a');
    await page.click('#btn-submitted');
    await page.click('#btn-continue');
    for (let i = 0; i < 5; i++) {
      expect(await text(page, '#battle-id') !== voted, `voted battle ${voted} offered again`);
      await page.click('#btn-skip');
    }
  },

  async 'voting on every battle keeps the language and modality chips'(page) {
    await page.goto(`${ROOT}/battle/?league=stt&lang=en-US`);
    await page.waitForSelector('#vote-a', { state: 'visible', timeout: 5000 });
    for (let i = 0; i < 3; i++) {
      await page.click('#vote-a');
      await page.click('#btn-submitted');
      await page.click('#btn-continue');
    }
    expect(/voted on every battle/.test(await page.textContent('body')), 'no "voted on every battle" message');
    expect(await page.isVisible('#lang-chips'), 'language chips hidden — no way to pick another selection');
    expect(await page.isVisible('#modality-chips'), 'modality chips hidden');
  },

  async 'wake-word blind battles pair detectors for one phrase and name it'(page) {
    await page.goto(`${ROOT}/battle/?league=wake_word&lang=en-US`);
    await page.waitForSelector('#vote-a', { state: 'visible', timeout: 5000 });
    expect(await text(page, '#battle-count') === 'battle 1 / 1', `count ${await text(page, '#battle-count')}`);
    expect(await text(page, '#battle-id') === '#w-same', `battle ${await text(page, '#battle-id')}`);
    const kicker = await text(page, '#input-kicker');
    expect(kicker.includes('“hey mycroft”'), `kicker "${kicker}" does not name the phrase`);
  },

  async 'free vote names the pooled intent league and survives a blocked pop-up'(page) {
    await page.goto(`${ROOT}/vote/`);
    await page.waitForSelector('#fv-app', { state: 'visible', timeout: 5000 });
    const leagues = await page.$$eval('#fv-league option', o => o.map(x => x.textContent));
    expect(leagues.includes('Intent') && !leagues.includes('intent'), `league labels ${JSON.stringify(leagues)}`);
    await page.click('#fv-win-a');
    const href = await page.getAttribute('#fv-hint a', 'href').catch(() => null);
    expect((href || '').includes('issues/new'), 'blocked pop-up leaves no way to reach the vote issue');
  },

  async 'matchup vote grid counts cast human votes'(page) {
    await page.goto(`${ROOT}/matchups/?league=intent_online&lang=en-US`);
    const cell = page.locator('td.mu-cell[data-row="int-a"][data-col="int-b"]');
    await cell.waitFor({ timeout: 5000 });
    expect(await cell.getAttribute('data-n') === '1', `n=${await cell.getAttribute('data-n')}`);
    expect(await cell.getAttribute('data-wr') === '100', `win rate ${await cell.getAttribute('data-wr')}`);
  },

  async 'matchup grid is scoped to one wake word'(page) {
    await page.goto(`${ROOT}/matchups/?league=wake_word&lang=en-US&all=1`);
    await page.waitForSelector('.mu-grid', { timeout: 5000 });
    const rows = await page.$$eval('.mu-row-head', r => r.map(x => x.textContent.trim()).sort());
    expect(JSON.stringify(rows) === JSON.stringify(['myc-oww', 'myc-precise']), `rows ${JSON.stringify(rows)}`);
    expect(await page.inputValue('#sel-word') === 'hey_mycroft', 'word select missing');
  },

  async 'intent fighters carry their rank from the pooled ladder'(page) {
    await page.goto(`${ROOT}/fighters/?league=intent_online`);
    await page.waitForSelector('#dex-grid .fighter', { timeout: 5000 });
    await page.waitForFunction(() => document.querySelector('.rank-line')?.textContent.includes('Rank'));
    const lines = await rowsOf(page, '#dex-grid .rank-line');
    expect(lines[0].startsWith('Rank #1'), `rank lines ${JSON.stringify(lines)}`);
    const tile = page.locator('.tile-top[data-top="intent_online"]');
    await page.waitForFunction(() =>
      document.querySelector('.tile-top[data-top="intent_online"]')?.textContent.includes('Top in')).catch(() => {});
    const top = (await tile.textContent() || '').trim();
    expect(top.startsWith('Top in en-US: int-a'), `intent tile top line "${top}"`);
  },

  async 'a filter left with no matches stays visible in its select'(page) {
    await page.goto(`${ROOT}/fighters/?league=stt&type=transformer&lang=ca-ES`);
    await page.waitForSelector('#dex-filtered-empty', { state: 'visible', timeout: 5000 });
    expect(await page.inputValue('#sel-type') === 'transformer', `type select reads "${await page.inputValue('#sel-type')}"`);
    expect(await page.inputValue('#sel-flang') === 'ca-ES', `language select reads "${await page.inputValue('#sel-flang')}"`);
    await page.click('#btn-reset-empty');
    await page.waitForSelector('#dex-grid .fighter');
    expect(await page.inputValue('#sel-type') === '', 'reset left the type filter');
  },

  async 'a size chip left with no matches stays visible and pressed'(page) {
    await page.goto(`${ROOT}/fighters/?league=stt&size=small&lang=ca-ES`);
    await page.waitForSelector('#dex-filtered-empty', { state: 'visible', timeout: 5000 });
    const chip = page.locator('.size-chip[data-size="small"]');
    expect(await chip.count() === 1, 'the active size chip is gone');
    expect(await chip.getAttribute('aria-pressed') === 'true', 'the size chip is not pressed');
    expect((await chip.textContent()).includes('(0)'), `chip "${await chip.textContent()}"`);
  },

  async 'pareto cost axis and tolerance survive a reload'(page) {
    await page.goto(`${ROOT}/pareto/?league=stt&lang=en-US`);
    await page.waitForSelector('#pt-panels .bench', { timeout: 5000 });
    await page.selectOption('#sel-cost', 'rtf');
    await page.fill('#rng-tolerance', '12');
    await page.reload();
    await page.waitForSelector('#pt-panels .bench');
    expect(await page.inputValue('#sel-cost') === 'rtf', `cost axis "${await page.inputValue('#sel-cost')}"`);
    expect(await page.inputValue('#rng-tolerance') === '12', `tolerance "${await page.inputValue('#rng-tolerance')}"`);
  },

  async 'pareto hardware tier survives a reload'(page) {
    await page.goto(`${ROOT}/pareto/?league=stt&lang=en-US&dataset=ds-a`);
    await page.waitForSelector('#sel-tier', { state: 'visible', timeout: 5000 });
    await page.selectOption('#sel-tier', 'cpu-x86');
    expect(new URL(page.url()).searchParams.get('tier') === 'cpu-x86', `url ${page.url()}`);
    await page.reload();
    await page.waitForSelector('#pt-panels .bench');
    expect(await page.inputValue('#sel-tier') === 'cpu-x86', `tier "${await page.inputValue('#sel-tier')}"`);
    const charts = await page.$$eval('.pt-chart-wrap h3 code', c => c.map(x => x.textContent));
    expect(JSON.stringify(charts) === '["cpu-x86"]', `charts ${JSON.stringify(charts)}`);
  },

  async 'an old wake-word link naming a dataset opens its phrase'(page) {
    await page.goto(`${ROOT}/leaderboard/?league=wake_word&lang=en-US&dataset=picovoice-jarvis`);
    await page.waitForSelector('.ds-intro', { timeout: 5000 });
    expect(await text(page, '.ds-intro .ds-id') === 'picovoice-jarvis', 'linked board not shown');
    expect(new URL(page.url()).searchParams.get('word') === 'jarvis', `url ${page.url()}`);
    await page.goto(`${ROOT}/leaderboard/?league=wake_word&lang=en-US&dataset=no-such-set`);
    await page.waitForSelector('.ww-overview');
    expect(!new URL(page.url()).searchParams.has('dataset'), `stale dataset kept: ${page.url()}`);
  },
};

const POPUP_BLOCKED = new Set([
  'blind vote still offers the issue link when the pop-up is blocked',
  'free vote names the pooled intent league and survives a blocked pop-up',
]);

async function main() {
  const only = process.argv.slice(2);
  const server = await startServer();
  const browser = await chromium.launch();
  let failed = 0;
  try {
    for (const [name, check] of Object.entries(checks)) {
      if (only.length && !only.some(o => name.includes(o))) continue;
      const { context, page, errors } = await newPage(browser, { blockPopups: POPUP_BLOCKED.has(name) });
      page.setDefaultTimeout(5000);
      try {
        await check(page);
        expect(!errors.length, `page error: ${errors[0]}`);
        console.log(`PASS  ${name}`);
      } catch (e) {
        failed++;
        console.log(`FAIL  ${name}\n      ${String(e.message).split('\n')[0]}`);
      } finally {
        await context.close();
      }
    }
  } finally {
    await browser.close();
    server.close();
  }
  console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
  process.exit(failed ? 1 : 0);
}

main();
