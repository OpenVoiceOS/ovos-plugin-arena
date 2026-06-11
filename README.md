# OVOS Plugin Arena

*Which plugin should I use?* — answered with data.

A fully GitHub-native benchmarking and voting arena for
[OpenVoiceOS](https://github.com/OpenVoiceOS) plugins. Reproducible benchmark
scripts rank plugins on labelled datasets; blind A/B battles let humans refine
those rankings. **No servers, no databases, no accounts** — the repository is
the arena:

- every **fighter** (competitor) is a JSON file in `registry/competitors/`
- every **dataset** is a JSON file in `registry/datasets/`
- every **benchmark** is one reproducible Python script in `benchmarks/`
- benchmark **predictions** are published to HuggingFace datasets
- **votes** are GitHub issues, tallied by a scheduled Action
- **leaderboards** are JSON files committed by CI and served by GitHub Pages

## Status

**Alpha — intent leagues only.** The intent benchmark (7 fighters × 12
languages over
[`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval))
is live across three independent leagues — `intent_template` (Padatious,
Padacioso, Nebulento), `intent_keyword` (Adapt, Palavreado) and the open
`intent` league for mixed-paradigm fusions (Padapt, Nebulapt) — each with
its own benchmark boards, battle pools and ELO. STT, TTS and wake-word
follow the same contracts but have no published arena benchmarks yet.
Pages deployment activates when the repository goes public.

## Transparency

- **AI usage:** this project is developed with AI coding agents (Claude) under
  human direction and review. Benchmark numbers are *not* AI-generated: every
  prediction row comes from actually running the real OVOS plugins over the
  published datasets via the scripts in `benchmarks/`, and every run is
  reproducible from a pinned dataset revision recorded in each row.
- **Votes:** the vote log is the public GitHub issue history — auditable and
  deterministically replayable at any time.

## How it works

```
registry/*.json ──┐
                  ▼
benchmarks/<bench>.py        one script per benchmark: trains each fighter,
   │                         predicts the test split, publishes JSONL rows
   ▼
HuggingFace dataset          predictions/<competitor_id>.jsonl  (§3.2 contract)
   │
   ▼  assemble.yml (daily)
frontend-static/public/data/ battles-*.json     blind A/B pools
                             benchmark-*.json   auto-metric boards
                             elo-seed-*.json    benchmark-derived initial ELO
                             leaderboard-*.json ELO boards
                             competitors.json   fighter bestiary
   │
   ▼  Astro build → GitHub Pages
voter picks A/B  →  prefilled GitHub issue  →  tally.yml (hourly)
                                                parses, dedupes, replays ELO,
                                                commits boards, closes issues
```

**ELO seeding:** before any human vote exists, the ELO board is seeded by
deterministic auto-battles derived from the benchmark metrics (one auto-battle
per sample where exactly one fighter is correct, at ¼ K-factor). Human votes
then move ratings at full weight on top of the seed.

**Determinism (§P5):** battle ids are content hashes — re-running `assemble`
never invalidates open votes — and both the seed and the vote replay are fully
deterministic, so the standings are reproducible from public data alone.

## The fighters

Each fighter is a *shippable configuration*: its `config` is a valid
`mycroft.conf` fragment — an `intents` section with a tier-suffixed
`pipeline` plus per-plugin config blocks. Single-stage pipelines benchmark
one engine in its paradigm league; multi-stage pipelines are **fusion**
fighters competing in the open intent league under portmanteau names —
**Padapt** (Padatious × Adapt, the stock OVOS cascade) and **Nebulapt**
(Nebulento × Adapt). Fighters carry a **species** (the plugin class they
instantiate), architecture **types** (GOFAI, fuzzy-match, neural-net,
ensemble, …), a **size** class (micro → titan), and a procedurally
generated sprite derived from their id hash. Browse the bestiary on the
**Fighters** page or in `registry/competitors/`. All fighters are evaluated
end-to-end through the real OPM pipeline plugins — the plugin owns its
confidence thresholds; the arena owns none.

## Running a benchmark

```bash
pip install ".[hf]"
python benchmarks/intent_intents_for_eval.py                  # full run (CPU, ~15 min)
python benchmarks/intent_intents_for_eval.py \
    --competitors padatious-medium --langs en-US --max-samples 50   # smoke run
python benchmarks/intent_intents_for_eval.py --upload         # publish to HF
```

Runs are resumable; rows carry the pinned dataset revision, plugin version and
runner version. Uploading requires HF write access to the results repo.

## Assembling the arena locally

```bash
pip install ".[hf]"
python -m arena.cli assemble --predictions OpenVoiceOS/ovos-intent-bench-intents-for-eval
python -m arena.cli export-bestiary
python -m arena.cli export-index
python -m arena.cli tally --keep-issues-open   # dry-run vote tally

cd frontend-static
npm install
npm run dev      # http://localhost:4321/ovos-plugin-arena
```

`assemble` also accepts a local predictions directory instead of an HF repo id.

## Tests

```bash
pip install ".[test]"
python -m pytest tests/ -q
```

## Fork your own arena

1. Fork this repo; enable Actions, set Settings → Pages → Source to
   "GitHub Actions" (public repo required for Pages).
2. Set the `HF_PREDICTIONS` repo variable to your prediction dataset(s)
   (comma-separated), and `ASTRO_SITE` / `ASTRO_BASE` to your Pages URL.
3. Drop your fighters and datasets as JSON files in `registry/`, add a
   benchmark script in `benchmarks/`, publish predictions to HF.
4. Run the `Assemble battles` workflow once; voters take it from there.

## Key files

| Path | Purpose |
|---|---|
| `registry/` | Declarative fighters + datasets (JSON) and their loaders |
| `benchmarks/` | One reproducible prediction script per benchmark |
| `runner/` | Plugin adapters used by bench scripts (intent pipeline, STT runner) |
| `arena/` | Core library: prediction loading, metrics, battles, ELO, CLI |
| `frontend-static/` | Astro static site (leaderboards, battles, bestiary) |
| `.github/workflows/assemble.yml` | Daily data refresh from HF predictions |
| `.github/workflows/tally.yml` | Hourly vote tally → leaderboards |
| `.github/workflows/pages.yml` | Astro build + Pages deploy |
| `.github/ISSUE_TEMPLATE/vote.yml` | The vote issue form (applies the `vote` label) |
| `docs/SPECIFICATION.md` | Full specification |

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme.
