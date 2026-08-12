# Operations runbook, the vote loop

This is the maintainer's guide to the running arena: how a vote gets from
the live site into a leaderboard, when each scheduled job fires, how to
confirm a vote actually landed, how to audit what the fraud rules rejected,
and how to replay the whole arena from the public log when something looks
wrong.

See [`methodology.md`](methodology.md) for *why* the rules exist and
[`SPECIFICATION.md`](SPECIFICATION.md) for the full spec. This page is about
*operating* the loop day to day.

## The loop, end to end

1. **Casting a vote.** A voter opens the deployed Pages site, picks a blind
   battle, and clicks A / B / Tie / Both wrong. The UI opens a pre-filled
   GitHub issue using the `Arena Vote` template
   (`.github/ISSUE_TEMPLATE/vote.yml`): it carries the `vote` label and a
   title of the exact form `vote|<battle_id>|<a|b|tie|both_wrong>`. The
   voter submits the issue as-is, the title is machine-parsed and must not
   be edited. An optional comment field lets them note anything about the
   battle. It is never parsed.
2. **Hourly tally.** `.github/workflows/tally.yml` runs at :17 past every
   hour (`cron: '17 * * * *'`, plus manual `workflow_dispatch`). It lists
   every `vote`-labelled issue, open *and* closed, full history every time, parses titles, deduplicates, applies the anti-fraud rules
   (`methodology.md` → "Vote fraud / dedup resistance"), replays the
   surviving votes on top of the ELO seed, and writes
   `leaderboard-<modality>-<lang>.json`, `vote-audit.json` and
   `patch-notes.json` under `frontend-static/public/data/`. It then commits
   and pushes straight to `dev` (`[skip ci]`) and closes each processed
   issue with a comment (counted, discarded, or duplicate) plus a
   `processed` label.
3. **Daily assemble.** `.github/workflows/assemble.yml` runs once a day at
   03:00 UTC (`cron: '0 3 * * *'`). It pulls the latest published benchmark
   predictions from HuggingFace, rebuilds the blind battle pools and the
   benchmark-derived ELO seed, and commits the refreshed
   `battles-*.json` / `benchmark-*.json` / `elo-seed-*.json` artifacts.
   Battle ids are content hashes of `(modality, dataset, lang, sample,
   competitor pair)`, so re-running assemble never invalidates an open
   vote, a vote issue from last week still resolves to the same battle
   today.
4. **Pages deploy.** `.github/workflows/pages.yml` is triggered by
   `workflow_run` once `assemble` or `tally` finishes (not on a schedule of
   its own). It builds the Astro site over whatever is currently committed
   under `frontend-static/public/data/` and publishes it to GitHub Pages.
5. **Rating moves.** The commit tally pushes to `dev` is the rating change, there is no separate "publish" step. Once the Pages deploy after that
   commit finishes, the live site reflects the new standings.

## Verifying a vote landed

Work backwards from whichever of these you can see:

- **The issue itself.** A processed vote issue is *closed* and carries the
  `processed` label, with a bot comment saying either the vote counted,
  was discarded (with the specific reason), or was a duplicate of an
  earlier vote on the same battle. An issue still *open* with only the
  `vote` label has not been picked up by a tally run yet, wait for the
  next `:17` run, or check whether `tally.yml`'s last run actually
  succeeded (Actions tab).
- **The tally commit.** `git log --oneline -- frontend-static/public/data/`
  on `dev` shows one `chore(data): update leaderboards from vote tally`
  commit per run that had at least one counted vote. A run with zero
  counted votes intentionally makes **no commit** (the workflow's
  empty-diff guard skips it), that is expected behavior, not a failure.
- **The leaderboard's vote count.** Open
  `leaderboard-<modality>-<lang>.json` for the battle's league and check
  `human_vote_count`, it should have gone up by exactly the number of
  *counted* votes since the last commit (discarded and down-weighted votes
  still move other counters but are called out separately in the audit
  file, see below).
- **`patch-notes.json`.** Written alongside the leaderboard on every run
  with counted votes. It diffs the board against what was on disk before
  the run, so it's the fastest way to see *what changed* without diffing
  full leaderboard JSON by hand.

## Auditing discards

Every tally run overwrites `vote-audit.json` with the complete, current
picture, it is not an append-only log, it is regenerated fresh from the
full vote history every time (consistent with `tally` always replaying
every issue, open and closed):

- `discarded`: every vote whose weight is zero, with the exact
  `discarded_reason` (`daily_vote_cap_exceeded` or `account_too_new`).
  A discarded vote's issue is still closed and commented, the voter is
  told why, but the vote is not deleted from GitHub, only excluded from
  the rating.
- `downweighted`: every vote whose weight was reduced but not zeroed (the
  one-sided-voter rule, weight `0.5`), these *do* count, just less.
- `counted`: the total number of votes that moved the rating at whatever
  weight they carry.

Nothing is ever deleted: a discarded vote's issue stays in the repository
forever, closed and labelled, and its outcome is fully derivable by
re-reading the issue history, that is what makes the vote log auditable.

## Replaying the arena from public logs

Anyone can reproduce the current standings from nothing but the public
repository. Push access is not required:

```bash
pip install ".[hf]"
python -m arena.cli assemble --predictions <HF predictions repo id(s)>
python -m arena.cli tally --data-dir frontend-static/public/data \
                           --output frontend-static/public/data \
                           --repo <owner>/<repo>
```

Running `tally` without `--repo` (or with an empty one) skips fetching
issues entirely and only replays whatever `battles-*.json` / vote data is
already on disk locally, useful for a dry run over a hand-built or
previously-fetched vote log, but note that with no `--repo` the
account-age gate has nothing to check against and does not discard
anything (there is no cached age to gate on). For a full, honest replay
always pass `--repo`.

The account-age cache (`voter-age-cache.json`, committed alongside the
leaderboards) is fetched from the GitHub API once per author the first
time they're seen and never re-fetched afterwards, this is what keeps
replay itself fully offline and deterministic: the same committed data
directory, tallied twice, produces byte-identical leaderboards (aside from
each run's own `generated_at` timestamp).

## Replay proof

`verify-replay` (`.github/workflows/replay-proof.yml`, on every push to
`dev` and daily) is the automated version of the manual replay above: it
re-runs the exact same pure `dedupe_votes` → `resolve_vote_weights` →
`build_elo_board` path `tally` uses, then diffs the freshly-replayed
standings against the committed `leaderboard-<league>-<lang>.json` and
`vote-audit.json` files field-by-field (ratings, ranks, vote counts, `generated_at` is ignored). Exit 0 means every published board is exactly
reproducible from the public vote log. Any other exit code means the
published data has drifted from what the log actually supports, and the
CI job fails loudly with a JSON diff of exactly which fields moved.

```bash
python -m arena.cli verify-replay --data-dir frontend-static/public/data \
                                   --repo <owner>/<repo>
# or, offline against a saved vote-issue snapshot:
python -m arena.cli verify-replay --data-dir frontend-static/public/data \
                                   --votes-file vote-log-snapshot.json
```

The check is strict: every committed `leaderboard-*.json` (and
`vote-audit.json`) MUST be exactly what replaying the current vote log
against the current battles pools produces, including a league that has
no published board at all, a league with counted votes and no
`leaderboard-<league>-<lang>.json` is a mismatch, not a tolerated gap.
Published artifacts are derived data: if a code change (e.g. a league
split, a new rating field) makes the committed boards no longer
reproducible, the fix is to delete the stale artifacts and regenerate
them with `assemble` + `tally`, not to grandfather the old shape into the
proof.

## Troubleshooting

- **A vote issue closes immediately with "does not match the vote title
  format".** The issue title was edited, or the voting UI's pre-fill
  broke, the title must be exactly `vote|<battle_id>|<a|b|tie|both_wrong>`
  with no extra characters. Check the deployed site's issue-template
  pre-fill logic and the `battle_id` values it's reading from the current
  `battles-*.json` pool. A stale cached page pointing at a battle id from
  a much older `assemble` run will also trigger this if the id no longer
  parses (it should still parse, only unrecognized battle ids get a
  separate "not in the current battles pool" message, so title-format
  drift specifically points at the pre-fill, not the data).
- **A vote issue closes with "Battle `<id>` is not in the current battles
  pool".** The battle pool was regenerated by an `assemble` run before the
  vote was cast, and this particular battle didn't survive into the new
  pool (e.g. the source dataset entry rotated out of the sampled subset).
  This is not itself a bug, the voter's browser tab was open against a
  now-superseded pool. Ask them to reload the site and vote again.
- **`tally.yml` runs green but the leaderboard didn't change.** Check
  `vote-audit.json`'s `discarded` list first, every open vote issue may
  have been legitimately discarded (new account, over the daily cap). If
  `discarded` is empty too, there may simply have been no new vote issues
  since the last run. A "green but skipped" tally with no commit is the
  normal outcome of an hour with zero qualifying votes, not a failure.
- **Two consecutive tally runs produce different leaderboards for the same
  vote log.** This would mean a fraud rule or the ELO replay stopped being
  a pure function of the vote log, treat it as a real bug, not an
  operational issue: pin down which artifact differs (`vote-audit.json` vs
  `leaderboard-*.json`), diff both runs' inputs (`battles-*.json`,
  `voter-age-cache.json`) to rule out the data dir itself changing between
  runs, and only then look at `arena/fraud.py` / `arena/cli.py`'s replay
  path.
- **The daily `assemble` run invalidated votes that were about to be
  tallied.** It shouldn't, battle ids are content hashes of
  `(modality, dataset, lang, sample, competitor pair)`, not of anything
  `assemble` regenerates per run, so the same underlying battle keeps the
  same id across runs. If a batch of votes really did all fail with "not
  in the current battles pool" right after an `assemble` run, look at
  whether the *dataset sample* or the *competitor pair* actually changed
  (e.g. a prediction file was replaced with a different sample selection)
  rather than assuming the hashing itself is unstable.

---
[← Methodology](methodology.md) · [Home](index.md) · [Registry audit →](registry-audit.md)
