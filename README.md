# OVOS Plugin Arena

*Which plugin should I use?* This arena answers with data.

**Live site:** [openvoiceos.github.io/ovos-plugin-arena](https://openvoiceos.github.io/ovos-plugin-arena)

![The arena's Standings page: an ELO ladder ranking OVOS plugins](docs/img/leaderboard-overview.png)

*See every page in the [visual site tour](docs/site-tour.md).*

This is a GitHub-native benchmarking and voting arena for
[OpenVoiceOS](https://github.com/OpenVoiceOS) plugins. Reproducible benchmark
scripts rank plugins on labelled datasets. Blind A/B battles let humans refine
those rankings. **No servers, no databases, no accounts.** The repository is
the arena:

- every **fighter** (competitor) is a JSON file in `registry/competitors/`
- every **dataset** is a JSON file in `registry/datasets/`
- every **benchmark** is one reproducible Python script in `benchmarks/`
- benchmark **predictions** are published to HuggingFace datasets
- **votes** are GitHub issues, tallied by a scheduled Action
- **leaderboards** are JSON files committed by CI and served by GitHub Pages

## Status

> [!WARNING]
> **Work in progress.** Development is focused on the platform itself. The tasks,
> datasets and fighter configurations are still being defined, so every
> leaderboard, rating and vote on the site and in the prediction datasets is
> placeholder data. At launch the results will be regenerated and the votes reset.

**Alpha: all modalities wired.** Every league has a reproducible benchmark
script and registry fighters:

| League | Benchmark | Board ranks on |
|---|---|---|
| `intent_zero_shot` · `intent_online` · `intent_offline` · `intent_keyword` | `benchmarks/intent_*.py` over [`intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval), `ovos-intents-v5`, `massive-templates` and public intent sets | `generalization_accuracy` |
| `stt` | `benchmarks/stt_*.py` over MInDS-14, FLEURS, MLS and other corpora | `wer_mean` |
| `wake_word` | `benchmarks/ww_*.py` over clip test sets such as [`synthetic-wakewords`](https://huggingface.co/datasets/OpenVoiceOS/synthetic-wakewords) | `error_rate` |
| `ww_stream` | `benchmarks/ww_stream.py` over the `ww-negatives-<lang>` sets of [`ww-stream-bench`](https://huggingface.co/datasets/OpenVoiceOS/ww-stream-bench), hours of audio with no wake word | `fa_per_hour` |
| `tts` | `benchmarks/tts_intents_prompts.py` | mean UTMOS, with STT round-trip intelligibility WER beside it |
| `vad` | `benchmarks/vad_speech.py` over `speech-vs-nonspeech-<lang>` | `error_rate` |

Each board's metric also seeds the ELO ladder of its league, except `ww_stream`, which has no wake word to battle on. The
[docs](docs/index.md) define every metric.

The intent leagues are fully populated with published predictions. STT, TTS
and wake-word fighters and datasets are registered and runnable, and
prediction sweeps run on an ongoing basis: boards render as swept rows land.
Pages deployment is live at https://openvoiceos.github.io/ovos-plugin-arena/.

## Transparency

- **AI usage:** this project is developed with AI coding agents (Claude) under
  human direction and review. Benchmark numbers are *not* AI-generated: every
  prediction row comes from actually running the real OVOS plugins over the
  published datasets via the scripts in `benchmarks/`, and every run is
  reproducible from a pinned dataset revision recorded in each row.
- **Votes:** the vote log is the public GitHub issue history. It is auditable
  and deterministically replayable at any time.

## How it works

```
registry/*.json ──┐
                  ▼
benchmarks/<bench>.py        one script per benchmark: trains each fighter,
   │                         predicts the test split, publishes JSONL rows
   ▼
HF repos (1 per dataset)     predictions/<lang>/<competitor>.jsonl, split per lang
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
                                                records new votes in votes.jsonl,
                                                replays the record, commits
                                                boards, closes issues
```

**ELO seeding:** before any human vote exists, deterministic auto-battles
derived from the benchmark rows seed the ELO ladder of each battle group.
The benchmark picks the winner of each sample: the correct answer for intent,
wake word and VAD, the lower WER for STT, the higher composite score for
TTS. An auto-battle carries a quarter of the weight of a human vote. A pair
of fighters whose confidence intervals on the primary metric overlap gets no
auto-battles, in every league. The auto-battle weight of one pair is capped
at 5 human votes. Human votes then move ratings at full weight on top of the
seed. [docs/rating.md](docs/rating.md) gives the maths.

**Determinism (§P5):** battle ids are content hashes, so re-running `assemble`
never invalidates open votes. Both the seed and the vote replay are fully
deterministic, so the standings are reproducible from public data alone.

## The fighters

Each fighter is a *shippable configuration*. Its `config` is a valid
`mycroft.conf` fragment: an `intents` section with a tier-suffixed
`pipeline` plus per-plugin config blocks. Single-stage pipelines benchmark
one engine in the league its training regime implies. Multi-stage pipelines
are **fusion** fighters, filed by their heaviest stage, under portmanteau
names:
**Padapt** (Padatious × Adapt, the stock OVOS cascade) and **Nebulapt**
(Nebulento × Adapt). Fighters carry a **species** (the plugin class they
instantiate), architecture **types** (GOFAI, fuzzy-match, neural-net,
ensemble, and others), a **size** class (micro to titan), and a
procedurally generated sprite derived from their id hash. Browse the
bestiary on the **Fighters** page or in `registry/competitors/`. An intent
fighter runs end-to-end through the real OVOS pipeline plugin classes,
which the runner imports from `runner.intent_pipeline.ENGINE_REGISTRY`.
Audio fighters load through OPM. The plugin owns its confidence thresholds,
and the arena owns none.

## Running a benchmark

```bash
pip install ".[hf,audio]"   # audio extra (numpy/soundfile/pyarrow) for stt/ww/tts
# intent (shared engine in runner/intent_bench.py)
python benchmarks/intent_intents_for_eval.py                  # full run (CPU, ~15 min)
python benchmarks/intent_intents_for_eval.py \
    --competitors padatious-medium --langs en-US --max-samples 50   # smoke run
# audio modalities (shared engine in runner/media_bench.py)
python benchmarks/stt_minds14.py --dataset minds14-en-US --max-samples 50
python benchmarks/ww_hey_mycroft.py --competitors openwakeword-hey-mycroft
python benchmarks/tts_intents_prompts.py --langs en-US --max-samples 30
python benchmarks/stt_minds14.py --upload                    # publish to HF
```

Every benchmark script takes `--competitors`, `--langs`, `--max-samples`,
`--output-dir` and `--upload`. The audio scripts also take `--dataset`. An
intent script runs one fixed dataset and has no `--dataset` flag; it takes
`--train-timeout-secs`, `--train-max-rss-mb` and `--transcript-cache-dir`.
[`docs/benchmarks.md`](docs/benchmarks.md) lists the flags per script. Runs are resumable, and
rows carry the pinned dataset revision, plugin version, and runner version.
Audio benchmarks instantiate the real OVOS STT/TTS/wake-word plugins
offline. Uploading requires HF write access to the prediction repo of the
dataset, `OpenVoiceOS/ovos-<modality>-bench-<dataset_id>`.

## Assembling the arena locally

```bash
pip install ".[hf]"
python -m arena.cli assemble --predictions OpenVoiceOS/ovos-intent-bench-intents-for-eval
python -m arena.cli export-bestiary
python -m arena.cli export-index
python -m arena.cli tally   # no --repo: replays the committed votes.jsonl and rewrites the boards

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

1. Fork this repo. Enable Actions, and set Settings → Pages → Source to
   "GitHub Actions" (public repo required for Pages).
2. Point each dataset file in `registry/datasets/` at your prediction repo
   with its `predictions_hf` field. `assemble` reads those repos. Pass
   `--hf-owner` to a benchmark script to upload to your own owner. Set the
   `ASTRO_SITE` and `ASTRO_BASE` repo variables to your Pages URL.
3. Drop your fighters and datasets as JSON files in `registry/`, add a
   benchmark script in `benchmarks/`, and publish predictions to HF.
4. Run the `Assemble battles` workflow once. Voters take it from there.

## Documentation

Full docs live in [`docs/`](docs/index.md). The front page tells what the
arena measures, how to read a board, how to reproduce a number and how to
add a plugin. These pages give the details:

- [Architecture](docs/architecture.md): the data flow from the registry to
  the site, and every workflow.
- [Rating](docs/rating.md): the board intervals, the auto-battles, the vote
  rules and the Bradley-Terry ladder, with every parameter.
- [Registry](docs/registry.md) and [Predictions](docs/predictions.md): the
  fighter and dataset JSON fields, and the prediction row format.
- [Add a fighter](docs/add-a-fighter.md), [FAQ](docs/faq.md) and
  [Glossary](docs/glossary.md).

## Key files

| Path | Purpose |
|---|---|
| `registry/` | Declarative fighters + datasets (JSON) and their loaders |
| `benchmarks/` | One reproducible prediction script per benchmark |
| `runner/` | Plugin adapters + shared bench engines: `intent_bench` (intent), `media_bench` + `stt_bench`/`ww_bench`/`tts_bench` (audio) |
| `arena/` | Core library: prediction loading, metrics, battles, ELO, CLI |
| `frontend-static/` | Astro static site (leaderboards, battles, bestiary) |
| `.github/workflows/assemble.yml` | Daily data refresh from HF predictions |
| `.github/workflows/tally.yml` | Hourly vote tally → leaderboards |
| `.github/workflows/pages.yml` | Astro build + Pages deploy |
| `.github/ISSUE_TEMPLATE/vote.yml` | The vote issue form (applies the `vote` label) |
| `docs/SPECIFICATION.md` | Full specification |
| `docs/add-a-fighter.md` | How to add your plugin as a fighter |
| `docs/operations.md` | Maintainer runbook for the vote loop (verify, audit, replay, troubleshoot) |
| `docs/reproduce-a-row.md` | Reproduce one leaderboard row end to end in about five minutes |

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
