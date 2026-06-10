// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // Set site + base for GitHub Pages deployment.
  // Override via ASTRO_SITE / ASTRO_BASE env vars in CI.
  site: process.env.ASTRO_SITE || 'https://openvoiceos.github.io',
  base: process.env.ASTRO_BASE || '/ovos-plugin-arena',
  output: 'static',
  // No trackers, no external scripts injected
  vite: {
    define: {
      // Runtime config: set VITE_BACKEND_URL to enable Mode A (POST votes to API).
      // Leave unset for Mode C (GitHub issue voting).
      'import.meta.env.VITE_BACKEND_URL': JSON.stringify(process.env.VITE_BACKEND_URL || ''),
      'import.meta.env.VITE_GITHUB_REPO': JSON.stringify(
        process.env.VITE_GITHUB_REPO || 'OpenVoiceOS/ovos-plugin-arena'
      ),
    },
  },
});
