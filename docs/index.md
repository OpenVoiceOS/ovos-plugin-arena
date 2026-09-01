# Docs index

Start with the [README](../README.md) for what the arena is and the
quickest path to running something. This directory covers everything the
README doesn't have room for.

| Doc | What it covers |
|---|---|
| [`SPECIFICATION.md`](SPECIFICATION.md) | The full spec: design principles, leagues, the registry and prediction-row contracts, matchmaking rules, the rating system, the vote flow. Start here for *what the system is defined to do*. |
| [`local-testing.md`](local-testing.md) | Guide to running everything locally: environment setup, unit tests, a benchmark smoke run, the assembler, replay verification, and previewing the Astro frontend. Start here to *try it on your own machine*. |
| [`add-a-fighter.md`](add-a-fighter.md) | Field reference for a competitor JSON file: required/optional fields, the VAD dual-key gotcha, the validation command. |
| [`adding-a-fighter.md`](adding-a-fighter.md) | Worked walkthrough of one real fighter file end to end, plus how a merged fighter actually gets benched and reaches the boards. |
| [`benchmarks.md`](benchmarks.md) | Per-modality benchmark scripts: shared engines, common flags, what each league's benchmark measures. |
| [`runner.md`](runner.md) | The always-on STT prediction runner (`runner/queue.yaml`, the `ser9` deployment, `queue_tools.py` sweep generation). |
| [`reproduce-a-row.md`](reproduce-a-row.md) | Reproduce one leaderboard row end to end: install, run a benchmark script, assemble a board, and compare against the published number. |
| [`leagues.md`](leagues.md) | Canonical definition of what each league scores and how — the metric formulas, in one place. |
| [`methodology.md`](methodology.md) | Why the rating system works the way it does: Bradley-Terry vs. sequential ELO, confidence intervals, vote-fraud resistance, UTMOS/intelligibility scoring. |
| [`operations.md`](operations.md) | Maintainer runbook for the vote loop: verifying a vote landed, auditing discards, replaying from public logs, troubleshooting. |
| [`registry-audit.md`](registry-audit.md) | Audit notes on registry data quality and modality-specific conventions. |
| [`dataset-review.md`](dataset-review.md) | Review notes on the eval datasets themselves. |
| [`dataset-gapfill.md`](dataset-gapfill.md) | Notes on filling coverage gaps in the eval datasets. |
| [`site-tour.md`](site-tour.md) | Guided walkthrough of the live site with a screenshot of every page and state: leaderboards, battle voting, matchups, the fighter bestiary, and the evidence rollup. |
| [`ensemble-rationale.md`](ensemble-rationale.md) | Why the fusion/ensemble fighters (Padapt, Nebulapt, ...) are built the way they are. |
