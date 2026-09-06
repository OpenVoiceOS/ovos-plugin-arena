# OVOS Plugin Arena — frontend

This is the static Astro site that renders the OVOS Plugin Arena. leaderboards,
head-to-head battles, matchup heatmaps, and the fighter roster all live here.
The site reads its data from the JSON files under `public/data/`. That data
comes from `arena.cli`, which builds it from prediction pools and vote
tallies. Nobody edits those files by hand. Regenerate them through the CLI
and rebuild the site.

To run the site locally, install dependencies with `npm ci`, then start the
dev server with `npm run dev`. For a production build, run `npm run build`.
That writes the static site to `dist/`. `npm run preview` serves that build
locally before it goes out to GitHub Pages.

Pages live under `src/pages/`. `leaderboard/` and `matchups/` render the
ranking tables and the head-to-head grids. `battle/` and `vote/` drive the
blind-vote flow visitors use to cast votes. `fighters/` lists the roster, and
`fighter/[id]/` renders one fighter's own page. `evidence/` and
`methodology/` document how a rank or a matchup number came to be.
`patch-notes/` tracks changes to the arena itself.

The site carries an accessibility gate, `npm run a11y`. It runs axe-core
against the built `dist/` through Playwright and Chromium. It serves the
build locally, scans a representative set of pages, and fails with a
non-zero exit code on any critical or serious WCAG violation. It prints the
rule, the offending selector, and a snippet of the markup. Run `npm run
build` first so the gate scans real output. Install the browser once with
`npx playwright install chromium` before the first run.

Most of the page content here is injected client-side through `innerHTML`
from `<script>` tags rather than rendered by Astro at build time. Styling for
that injected markup, such as links, tables, or heatmap cells, has to live in
a global `<style is:global>` block rather than a scoped one. A scoped rule
never applies to injected content, and that can hide a contrast regression
that only the gate, or a real browser, will catch.
