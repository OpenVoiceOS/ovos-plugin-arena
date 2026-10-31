# Frontend

The arena site is a static [Astro](https://astro.build) site in
`frontend-static/`. It has no server. Each page is a static HTML file, and a
browser script on the page reads JSON files from `data/`. The `arena.cli`
commands write those files into `frontend-static/public/data/`. Astro copies
that folder into the build as it is.

Nobody edits a data file by hand. To change what the site shows, change the
registry or the prediction rows, run the CLI again, and build the site again.

[Site tour](site-tour.md) has a screenshot of every page. This page tells
what each page reads and how to build the site.

## Pages

The pages are in `frontend-static/src/pages/`. The shared layout
`src/layouts/Base.astro` holds the navigation bar, the footer and the theme.
Most page content is built in the browser from the data files. Astro
renders only the page frame at build time.

| Path | Source | What it shows |
|---|---|---|
| `/` | `index.astro` | The home page. It counts benchmark boards, languages and open battles from `index.json`, and shows the human and auto-battle split of all ratings. |
| `/leaderboard/` | `leaderboard/index.astro` | The standings. One tab per league in `index.json` `leagues`. Each tab shows the ELO ladder of the battle group and language, and the benchmark boards of the league, with the unranked entries hidden behind a count. The wake word tabs list the phrases first. |
| `/pareto/` | `pareto/index.astro` | Cost against quality. For a benchmark board, it plots the primary metric against a cost such as `peak_rss_mb`, and marks the Pareto frontier. A tolerance slider picks the cheapest fighter within a margin of the best score. |
| `/matchups/` | `matchups/index.astro` | Head-to-head heatmaps per league: win rates from human votes, or score deltas from the benchmark boards. |
| `/battle/` | `battle/index.astro` | The blind A/B battle. It shows two outputs for one input without the fighter names. A vote opens a pre-filled GitHub issue. |
| `/vote/` | `vote/index.astro` | The free vote. The voter picks two fighters by name and says which one they prefer. On the wake word league, the second fighter must listen for the same phrase as the first. |
| `/fighters/` | `fighters/index.astro` | The fighter roster, per league, with rank, types and configuration. |
| `/fighter/<id>/` | `fighter/[id].astro` | One page per fighter: stats, rank on each language board, rivals, configuration and dataset provenance. |
| `/evidence/` | `evidence/index.astro` | Completeness counts per league: fighters, registered datasets, datasets with published predictions, boards. The footer links it for grant reviewers. |
| `/methodology/` | `methodology/index.astro` | A static summary of the rating method. [Methodology](methodology.md) is the full text. |
| `/patch-notes/` | `patch-notes/index.astro` | Rank changes, new fighters and upsets from the last tally. |

`fighter/[id].astro` is the only page that reads files at build time. Its
`getStaticPaths` reads every JSON file in `registry/competitors/` and makes
one page per `competitor_id`. It also reads the committed boards in
`public/data/` once, to compute the stats on each fighter page. A fighter
that has no rows still gets a page.

## Data files

A page that shows boards or battles reads `data/index.json` first. The
index lists the data files the site may load, so the page does not guess a
file name. Most of these pages add `?v=<generated_at>` of the index to each
later fetch, so a cached old board does not hide a new one.

| File | Written by | Read by |
|---|---|---|
| `index.json` | `arena.cli export-index` | Every page except evidence, methodology and patch notes. |
| `metrics-legend.json` | `export-index` (`arena/metrics_legend.py`) | Leaderboard, cost against quality. |
| `competitors.json` | `arena.cli export-bestiary` | Leaderboard, cost against quality, matchups, battle, vote, fighters, fighter. |
| `benchmark-<league>-<dataset>-<lang>.json` | `arena.cli assemble` | Leaderboard, cost against quality, matchups, fighter. |
| `battles-<group>-<dataset>-<lang>.json` | `assemble` | Battle, matchups. |
| `battles-<group>-freeform-<lang>.json` | `assemble` | Free vote, matchups. |
| `elo-seed-<group>-<lang>.json` | `assemble` | `tally` and `verify-replay`. No page reads it. |
| `leaderboard-<group>-<lang>.json` | `arena.cli tally` | Leaderboard, fighters, fighter. |
| `votes.jsonl`, `vote-audit.json` | `tally` | Battle, free vote and matchups, for the human head-to-head records (`src/lib/headToHead.js`). |
| `voter-age-cache.json` | `tally` | `tally` and `verify-replay`. No page reads it. |
| `patch-notes.json` | `tally` (`arena/patch_notes.py`) | Patch notes. |
| `badges/<group>/<competitor_id>.svg` | `tally` (`arena/badges.py`) | Plugin READMEs that embed the badge, at `<site>/data/badges/...`. |
| `evidence.json` | `arena.cli export-evidence` | Evidence. |
| `assemble-summary.json` | `assemble` | The alarm step of `assemble.yml`. No page reads it. |

`<league>` is a league id such as `stt` or `intent_online`. `<group>` is a
battle group: the league id, or `intent` for the four intent leagues.

`index.json` holds these keys:

| Key | Content |
|---|---|
| `leaderboards` | One entry per `leaderboard-*.json`, with the human and total vote counts. |
| `benchmarks` | One entry per `benchmark-*.json` that the site lists. |
| `battles_pools` | One entry per blind battle pool that the site lists. |
| `freeform_pools` | One entry per free vote pool. |
| `leagues` | The league tabs, in tab order: `id`, `label`, `battle_group`, `order`, `voteless`. |
| `predictions_revisions` | The commit of each prediction repo that the boards were built from. |
| `auto_vote_weight` | The weight of one auto-battle against one human vote (`arena/elo.py` `BT_AUTO_WEIGHT`). |
| `has_bestiary` | `true` when `competitors.json` exists. |

Each list entry carries `file`, `modality`, `dataset_id`, `lang`,
`generated_at` and `count`. `count` is the number of entries or battles in
the file.

`export-index` leaves out some files that are on disk:

- A wake word clip board that `arena/ww_listing.py` hides, and the streaming
  board and the battle pool of the same set. The file stays on disk. Only
  the index leaves it out.
- A false wake-up board (`ww_stream` on a negatives set) with no entries.

`leagues` holds only the leagues that have a listed benchmark board with at
least one entry (`arena/models.py` `leagues`). A league with no such board
gets no tab.

## How a vote leaves the site

The battle page and the free vote page do not send a vote anywhere. A vote
button opens this URL in a new tab:

```text
https://github.com/<repo>/issues/new?template=vote.yml&title=vote|<battle_id>|<choice>
```

`<choice>` is `a`, `b`, `tie` or `both_wrong`. The issue form
`.github/ISSUE_TEMPLATE/vote.yml` adds the `vote` label. The voter submits
the issue under their own GitHub account. The `tally` workflow then reads
it. [Rating](rating.md) and [Operations](operations.md) tell what happens
next.

`<repo>` comes from `VITE_GITHUB_REPO` at build time, and falls back to
`OpenVoiceOS/ovos-plugin-arena` (`astro.config.mjs`).

## Build and preview locally

The site needs Node.js 22.12 or later (`engines` in `package.json`). CI
uses Node 24.

1. Install the dependencies from the lock file:

   ```bash
   cd frontend-static
   npm ci
   ```

2. Build the site into `dist/`:

   ```bash
   npm run build
   ```

   The build reads the data files that are committed in `public/data/`.
   To show your own data, run the CLI commands with `--output` or
   `--data-dir` set to `public/data` first (see
   [Local testing](local-testing.md)).

3. Serve the build:

   ```bash
   npm run preview
   ```

   Open `http://localhost:4321/ovos-plugin-arena/`. The path starts with the
   base path, because `astro.config.mjs` sets `base` to
   `/ovos-plugin-arena` by default. The preview server runs in the
   foreground. Stop it with Ctrl-C.

`npm run dev` starts the development server, which rebuilds a page when you
change its source.

Two variables change the base URL. Set them before `npm run build`:

| Variable | Default | Use |
|---|---|---|
| `ASTRO_SITE` | `https://openvoiceos.github.io` | The site origin, for canonical links and the sitemap. |
| `ASTRO_BASE` | `/ovos-plugin-arena` | The path prefix of every page and data file. Use `/` to serve from the root. |
| `VITE_GITHUB_REPO` | `OpenVoiceOS/ovos-plugin-arena` | The repository that receives vote issues. A fork sets its own. |

## Checks

`package.json` has these scripts beside `dev`, `build` and `preview`:

| Command | What it checks | Needs |
|---|---|---|
| `npm run a11y` | Runs axe-core on a set of pages of `dist/`. It fails on any critical or serious WCAG violation. | A build, and Chromium from `npx playwright install chromium`. |
| `npm run ui-check` | Drives the built site in headless Chromium. It answers every `data/` request from the fixed arena in `scripts/ui-fixtures.mjs`, so a data refresh does not change the result. It covers the wake word leagues, the two vote flows and the filter controls. | A build and Chromium. |
| `npm run dex-audit` | Checks that every flavour sentence on a fighter page agrees with the raw measurement behind it. | `npm ci` only. |
| `node scripts/pareto-tolerance-check.mjs` | Checks the tolerance slider of the cost page when the best score is 0. It is not in `package.json` and no workflow runs it. | `npm ci` only. |

The `frontend_behaviour` job of `unit_tests.yml` runs `npm run ui-check` on
every pull request to `dev`, and on each push to `dev` that changes the
paths the workflow lists.

## Deploy

`.github/workflows/pages.yml` builds and deploys the site. It runs on:

- a push to `dev` that changes `frontend-static/` or `pages.yml`,
- the end of an "Assemble battles" or "Tally votes" run, because those
  workflows commit data with `[skip ci]` and a push trigger does not fire for
  them,
- a manual dispatch.

The `build` job checks out `dev`, runs `npm ci` and `npm run build` with the
repository variables `ASTRO_SITE`, `ASTRO_BASE` and `VITE_GITHUB_REPO`, and
uploads `frontend-static/dist` as the Pages artifact. It also runs
`npm run a11y`. That step has `continue-on-error: true`, so a violation does
not stop the deploy. The `deploy` job publishes the artifact. It skips itself
when the repository is private. A failed build or deploy opens an alarm
issue, and the next good run closes it.

## Notes for page authors

- A page injects most of its markup with `innerHTML`. A scoped Astro
  `<style>` rule does not apply to injected markup. Put styles for injected
  markup in a `<style is:global>` block.
- The wake word league is one league per phrase (`src/lib/wakeword.js`). The
  phrase of a fighter is its hotword config key. The phrase of a dataset is
  its registry `wakeword` when a fighter listens for it, else the phrase in
  quotes in its display name, else the longest fighter phrase in its id or
  name.
- `src/lib/ranks.js` reads ranks and ratings only from the
  `leaderboard-*.json` files. `competitors.json` carries no rank, so two
  pages cannot show two ranks for one fighter.

---
[← Predictions](predictions.md) · [Home](index.md) · [FAQ →](faq.md)
