# Astro Starter Kit: Minimal

```sh
npm create astro@latest -- --template minimal
```

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
/
├── public/
├── src/
│   └── pages/
│       └── index.astro
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

## ♿ Accessibility gate

`npm run a11y` runs [axe-core](https://github.com/dequelabs/axe-core) against
the **built** site (`dist/`) via Playwright + Chromium. It serves `dist/`
locally, scans a representative page set — home, leaderboard, battle (blind
vote flow), a fighter detail page, the fighters bestiary, methodology, patch
notes, and the free-vote flow — and fails (non-zero exit) if any
**critical** or **serious** violation is found, printing the rule id, the
offending selector, and a snippet of the offending HTML.

```sh
npm run build      # generates dist/
npm run a11y       # scans it
```

The first run needs a Chromium binary:

```sh
npx playwright install chromium   # or `--with-deps chromium` if you have sudo
```

No system-level browser dependencies are required to just *run* Chromium
headless on a typical Linux desktop; `--with-deps` is only needed in minimal
containers that are missing shared libraries.

The gate is wired into `.github/workflows/pages.yml` as a **non-blocking**
step (`continue-on-error: true`) right after the build step, so violations
show up in CI logs without blocking deploys. Once the team is happy with the
signal, drop `continue-on-error` there to make it a hard gate.

Notes for fixing violations found by the gate:
- Prefer native semantic elements (`<button>`, `<a>`, `<table>`/`<th scope>`,
  `<label>`) over sprinkling `aria-*` attributes onto generic `div`/`span`.
- Most of this site's content is injected client-side via `innerHTML` from
  page `<script>` tags, and injected markup **does not** receive Astro's
  scoped-style attribute — so anything meant to style dynamically-injected
  content (links, tables, badges…) belongs in `Base.astro`'s global
  `<style is:global>` block, or in a per-page `<style is:global>` block, not
  a scoped one. A scoped rule there will silently never apply and can hide
  contrast/visual regressions that only axe (or a real browser) will catch.
- Don't add `aria-live` to whole leaderboards/battle grids — it causes
  screen readers to announce every re-render. Scope `aria-live` narrowly
  (a single status line), if used at all.

## 👀 Want to learn more?

Feel free to check [our documentation](https://docs.astro.build) or jump into our [Discord server](https://astro.build/chat).
