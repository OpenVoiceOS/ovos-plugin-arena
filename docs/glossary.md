# Glossary

The terms of the arena, in alphabetical order. Each entry names the file or
field that defines the term.

**auto-battle** (auto vote). A battle that the benchmark rows decide, with no
voter. `arena.assembler.seed_elo` makes one for each sample where the
primary metric picks a winner. It carries a quarter of the weight of a
human vote, and the total weight of one fighter pair is capped. See
[Rating](rating.md#auto-battles-and-the-seed).

**badge.** An SVG file at `data/badges/<group>/<competitor_id>.svg` that shows the
rank and rating of a fighter on its ladder. `tally` writes it
(`arena/badges.py`). A plugin README can embed it.

**battle.** Two outputs of two fighters for the same sample, shown to a
voter without the fighter names. A battle lives in a battle pool,
`battles-<group>-<dataset>-<lang>.json`. Its `battle_id` is a hash of the
group, dataset, language, sample and the sorted fighter pair.

**battle group.** The pool in which fighters battle and share one ladder.
`arena.models.battle_group()` maps the four intent leagues to `intent` and
every other league to itself. Nine leagues thus make six battle groups.

**bestiary.** `competitors.json`, the registry as one flat file for the
site. `ovos-arena export-bestiary` writes it.

**board** (benchmark board). The ranking of the fighters of one league on
one dataset in one language, on the primary metric:
`benchmark-<league>-<dataset>-<lang>.json`. Votes do not change it.

**bucket.** A part of an intent test set, named in the `bucket` field of a
row, for example `paraphrase` or `typos`. The dataset field `bucket_roles`
marks a bucket as `in_distribution` or `generalization`.

**datashape.** The kind of training data an intent engine needs: phrase
templates (`template`) or vocabulary rules (`keyword`). An intent training
set declares it in `paradigm`. A test set names one training set per
datashape in `train_datasets`.

**dataset.** One JSON file in `registry/datasets/<league>/`. It names the
source corpus on Hugging Face, the pinned commit, the languages, the
licence and the role.

**dataset role.** The `role` field of a dataset. `eval` is a held-out test
set that a board scores. `train` is a training corpus, for example the
templates an intent engine trains on. `unrestricted` is open data that a
run can draw on, for example the spoken-word negatives of the wake word
sets.

**ELO seed.** `elo-seed-<group>-<lang>.json`. The ladder state that the
auto-battles of all datasets of one group and language make, before any
human vote.

**false wake-up board.** A `ww_stream` board over a `ww-negatives-<lang>`
set: hours of audio with no wake word. It ranks fighters on `fa_per_hour`,
the false wake-ups per hour.

**featured phrase.** A wake phrase for which a listed clip board ranks a
trained model whose plugin is not `ovos-ww-plugin-wakeforge` (WakeHuBERT).
`arena/ww_listing.py` lists a clip board only when its phrase is featured.

**fighter** (competitor). One plugin with one config that a user can ship:
one JSON file in `registry/competitors/<league>/`. The same plugin with
another model or threshold is another fighter. Boards, battles and ladders
key on `competitor_id`.

**free-form battle.** A battle with no sample. A voter who tested two
plugins outside the arena says which one they prefer. The pool is
`battles-<group>-freeform-<lang>.json`. In the `wake_word` group, only
fighters of the same phrase make a pair.

**generic fighter.** A wake word fighter that works for any phrase without
training on it. Its registry `types` holds `zero-shot`, `few-shot` or
`keyword-match`. Every other wake word fighter is a trained model.

**hidden board.** A wake word board that `arena/ww_listing.py` leaves out of
`index.json`. The file stays on disk. The site does not list it.

**label set.** The `label_set` field of an `intent_offline` fighter: the
test sets whose labels its pretrained model can emit. The runner skips the
fighter on every other test set.

**ladder** (leaderboard, ELO ladder). The ranking of the fighters of one
battle group in one language on the Bradley-Terry rating:
`leaderboard-<group>-<lang>.json`. Auto-battles seed it and human votes move
it.

**league** (modality). One competition with its own task, datasets and
boards. The leagues are `intent_zero_shot`, `intent_online`,
`intent_offline`, `intent_keyword`, `wake_word`, `ww_stream`, `stt`, `tts`
and `vad`. `intent` and `intent_template` are dataset folders, not leagues.

**metric ladder.** A ladder inside `leaderboard-<group>-<lang>.json` for one
metric. The primary metric ladder holds the human votes. A secondary metric
ladder holds only auto-battles.

**OOD** (out of scope). An intent test row whose `reference_intent` is
`null`. The correct answer is no intent.

**pinned revision.** A 40-character commit in the `source.revision` field of
a dataset. A board drops rows made against another commit and counts them in
`rows_other_revision`. A branch name does not pin.

**prediction repo.** The Hugging Face dataset repo that holds the rows of
one dataset: `OpenVoiceOS/ovos-<league>-bench-<dataset_id>`, named in the
`predictions_hf` field. See [Predictions](predictions.md).

**prediction row** (row). One line of a prediction file: the output of one
fighter for one sample, with its reference and timing. `runner/schema.py`
and `arena.models.PredictionRow` define it.

**primary metric.** The metric that sorts a board and decides the
auto-battles. `arena.metrics.PRIMARY_METRIC` holds the default per league,
and a dataset can name its own in `primary_metric`.

**provisional.** A ladder with fewer than 10 counted human votes
(`PROVISIONAL_MIN_HUMAN_VOTES`). Its order comes mostly from the seed.

**replay proof.** The check that the published ladders follow from the
committed seeds, vote record and age cache. `ovos-arena verify-replay` runs
it, and `replay-proof.yml` runs that command on each data change.

**sample.** One test item: an utterance, a clip, a prompt or a recording.
Its `sample_id` is the same for every fighter, so two rows with the same
`sample_id` answer the same input.

**sample set** (manifest). `sample_sets/<ll_RR>.json` in a prediction repo:
the sample ids that the `sample_policy` of a dataset selects. The file name
is the language tag with `_` in place of `-`, for example `en_US.json`.
`python -m runner.publish_sample_set` writes it. A board then scores every
fighter on those ids only, and marks an entry that covers less than 90% of
them with `sample_set_partial`.

**sample policy.** The `sample_policy` field of a dataset: the cap on
samples per run (`max_samples`) and the seed of the selection.

**stream capability.** A wake word fighter with `"stream"` in its
`capabilities` field. Only such a fighter runs in the `ww_stream` league.

**tied with leader.** `tied_with_leader: true` on a board entry: its
interval overlaps the interval of rank 1.

**trained model.** A wake word fighter that is not generic. Its model was
trained on the phrase.

**trained_on.** The `trained_on` field of a fighter: the datasets whose data
was in its training data. The runner skips those pairs, the fleet runner
never schedules them, and `assemble` drops published rows for them. Such
rows never reach a board.

**unranked.** A board entry with `unranked: true` and rank 0. The
`unranked_reason` tells why. [The front page](index.md#how-to-read-a-board)
lists every reason.

**version blended.** `version_blended: true` on a board entry: its rows come
from more than one plugin version. `plugin_versions` lists them.

**vote.** A GitHub issue with the title `vote|<battle_id>|<choice>` and the
label `vote`. The choice is `a`, `b`, `tie` or `both_wrong`.

**vote audit.** `vote-audit.json`: the count of counted votes, and every
discarded or down-weighted vote with its reason.

**vote record.** `votes.jsonl`: every vote issue as it was first seen, one
line per issue. `tally` appends to it and never rewrites a line.

**wake phrase.** The phrase a wake word fighter detects: the key of its
`hotwords` config. A clip set names its phrase in `wakeword`.
`registry.loaders.is_same_phrase` compares the two. The fleet runner never
schedules a fighter on a set of another phrase, and `assemble` drops such
rows, so a fighter is scored and battles only on the sets of its own
phrase.

---
[← FAQ](faq.md) · [Home](index.md)
