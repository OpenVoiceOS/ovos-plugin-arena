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
   its own). It builds the Astro site over whatever is committed
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

Running `tally` without `--repo` (or with an empty one) records nothing
and replays the committed vote record as it stands, useful for a dry run
that rebuilds every board from data already on disk. `--keep-issues-open`
fetches and records as usual but writes nothing back to GitHub, which is
the read-only way to refresh the record against the live repository. The
voter is still owed an answer, and the next run without the flag gives
it: every run comments on and closes every recorded issue that is still
open, not only the ones it recorded itself, so a dry run or a run that
died mid-way leaves nothing hanging.

The account-age cache (`voter-age-cache.json`, committed alongside the
leaderboards) is fetched from the GitHub API once per author the first
time they're seen and never re-fetched afterwards, this is what keeps
replay itself fully offline and deterministic: the same committed data
directory, tallied twice, produces byte-identical leaderboards (aside from
each run's own `generated_at` timestamp).

## Replay proof

`verify-replay` (`.github/workflows/replay-proof.yml`, on every push to
`dev` and daily) is the automated version of the manual replay above: it
re-runs `replay_boards`, the exact same pure path `tally` and `assemble`
build their boards with, over the committed `votes.jsonl`, then diffs the
freshly-replayed
standings against the committed `leaderboard-<league>-<lang>.json` and
`vote-audit.json` files field-by-field (ratings, ranks, vote counts, `generated_at` is ignored). Exit 0 means every published board is exactly
reproducible from the public vote log. Any other exit code means the
published data has drifted from what the log actually supports, and the
CI job fails loudly with a JSON diff of exactly which fields moved.

```bash
python -m arena.cli verify-replay --data-dir frontend-static/public/data
# or against a snapshot of the record kept elsewhere:
python -m arena.cli verify-replay --data-dir frontend-static/public/data \
                                   --votes-file vote-record-snapshot.jsonl
```

Nothing here touches the network: the vote record, the seeds and the
account-age cache are all committed. `assemble` and `tally` both publish
the boards *and* `vote-audit.json` from that one replay, so a roster
change that flips a recorded vote between counted and discarded can never
leave the two disagreeing.

A record line that does not parse stops both commands with the file and
line number named and a non-zero exit. Public evidence that cannot be
read is never skipped past: fix the line (its content is in the run that
wrote it) and re-run.

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

## Alarms

Six workflows write or verify the committed data:
`.github/workflows/assemble.yml`, `tally.yml`, `replay-proof.yml`,
`pages.yml`, `hourly-predictions.yml` and `publish-sample-sets.yml`. Each opens a tracking issue the
first time a run fails, and closes it again on the next run that succeeds
(`.github/actions/notify-failure`, a composite action every one of them
calls). The issue carries the `ci-failure` label and a title of the form
`CI: <workflow name> failing`, so exactly one open issue ever tracks a
given workflow's red streak, whether it lasts one run or two days. Seeing
one of these titles open means: check the linked run's log, fix whatever
broke, and either re-run the workflow or wait for its next schedule — the
issue closes itself once a run of that same workflow goes green again.

Two more alarms catch a run that stays green while the data it produces
goes quietly wrong, rather than failing outright:

- **`replay-proof.yml` emits `::warning::` when the replay counted zero
  human votes** across every board. `verify-replay` always reproduces the
  published boards from a seed with zero human votes on it (there is
  nothing to disagree with), so a pass under these conditions proves only
  that the deterministic replay logic works, not that a real vote actually
  reaches a leaderboard. Treat this warning as a sign to check
  `tally.yml`'s recent runs and `vote-audit.json`'s `counted` count
  directly, the replay proof alone cannot tell the two states apart.
- **`assemble.yml` emits one `::warning::` per board named in
  `assemble-summary.json`'s `boards_without_ranked_fighters`**, plus one
  for `rows_dropped_off_revision` when it is non-zero. Every fighter on an
  unranked board scored zero samples, board-wide, and a dataset revision
  mismatch means rows were swept against a corpus version other than the
  one the registry pins. Both are legitimate, temporary states
  while a re-sweep against the right revision is in flight, so the run
  still publishes the board rather than withholding it; the warning exists
  so a re-sweep that never actually happens does not go unnoticed for
  weeks.
- **`assemble.yml` emits one `::warning::` per board named in
  `assemble-summary.json`'s `boards_without_negatives`**. This applies to
  wake-word boards only: a board whose rows are all labelled positive
  carries no negative (silence/other-speech) clips at all, so
  `false_accept_rate`/`fa_per_hour` cannot be computed for any fighter on
  it. The board still publishes with whatever it can measure (error rate,
  false rejects); the warning flags that a negatives corpus still needs to
  be swept and pooled onto that board via the dataset's own
  `negatives_dataset_ids` (see `runner/audio_io.py`'s pooling of negative
  clips into a fighter's own prediction stream).
- **`assemble.yml` emits one `::warning::` per board named in
  `assemble-summary.json`'s `boards_unmanaged_on_error`**. A dataset with a
  `sample_policy` whose manifest simply hasn't been published yet never
  reaches this list — that is the ordinary, silent-to-WARNING-only
  "unmanaged" state `arena.cli._load_sample_set` returns `None` for. This
  alarm fires only when the fetch or parse itself failed (an import error,
  a network transport error, a malformed manifest) — the board still
  publishes unfiltered, but the failure needs fixing, not a manifest
  republish.

## Sample-set manifests

A dataset's registry `sample_policy` fixes how many rows (and which ones,
deterministically) a sweep draws, but the sweep itself never records which
rows those were. `.github/workflows/publish-sample-sets.yml` runs weekly
(and on demand via `workflow_dispatch`, with an optional dataset id glob)
and publishes a `sample_sets/<lang>.json` manifest — the exact set of
sample ids a policy-capped dataset selects — to every `role: eval` dataset
that declares a `sample_policy` and doesn't already have a manifest for
the current policy (`runner/publish_sample_set.py --skip-existing`).

`arena assemble` downloads this manifest alongside a dataset's predictions
and filters every fighter's rows to it before scoring a board. A board
whose dataset has no published manifest yet — either because the workflow
hasn't run since the dataset gained a `sample_policy`, or because
publication for that one dataset failed or timed out — falls back to
scoring every row a fighter submitted, unfiltered, and reports
`sample_set: "unmanaged"` with `coverage: null` on that board. This is a
degrade path, not a build failure: the board is still comparable across
fighters that were swept from the same unrestricted pool, it just cannot
guarantee every fighter was scored over the identical sample subset. Seeing
`"unmanaged"` on a board that has a `sample_policy` in the registry means
the manifest hasn't landed yet — check the latest
`publish-sample-sets.yml` run for that dataset's id in the step summary.

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
  superseded pool. Ask them to reload the site and vote again.
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
