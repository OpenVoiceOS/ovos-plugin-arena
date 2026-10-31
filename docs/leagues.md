# Leagues

A league is one competition. It has its own datasets, its own prediction rows
and its own benchmark board for each dataset and language. Each league has a
page that defines its task and its metrics exactly as
[`arena/metrics.py`](../arena/metrics.py) computes them.

| League | Task | Primary metric | Page |
|---|---|---|---|
| `intent_zero_shot`, `intent_online`, `intent_offline`, `intent_keyword` | Name the intent of a written utterance, or none when it is out of scope. | `generalization_accuracy`, higher is better | [Intent](leagues/intent.md) |
| `wake_word` | Decide per clip whether the wake phrase is present. | `error_rate`, lower is better | [Wake word](leagues/wake_word.md) |
| `ww_stream` | Stream hours of audio with no wake word, and do not wake up. | `fa_per_hour`, lower is better | [False wake-ups](leagues/ww_stream.md) |
| `stt` | Transcribe a spoken clip. | `wer_mean`, lower is better | [STT](leagues/stt.md) |
| `tts` | Speak a prompt. | `utmos`, higher is better; human votes rank the ladder | [TTS](leagues/tts.md) |
| `vad` | Decide per clip whether it holds speech. | `error_rate`, lower is better | [VAD](leagues/vad.md) |

The arena ranks fighters with two signals (see
[SPECIFICATION.md](SPECIFICATION.md)):

- A **benchmark board** ranks on the primary metric, computed from the
  published prediction rows.
- An **ELO ladder** starts from auto-battles derived from the same rows, and
  human blind A/B votes then refine it. It ranks on `bt_rating`, a batch
  Bradley-Terry fit with bootstrap confidence intervals (see
  [methodology.md](methodology.md)).

The primary metric of a league sorts its benchmark board and decides the
auto-battles that seed its ladder (§4 R5).

Battles and ladders are pooled per battle group. The battle group is the
league itself, except for the four intent leagues. They share one `intent`
pool, because a blind vote judges two answers to the same utterance and
cannot see how each engine was prepared.

A board ranks an entry only when the primary metric rests on 30 or more
rows (`arena.metrics.MIN_BOARD_SAMPLES`). Under that, the entry is unranked
with the reason `too_few_samples`.

Battles have their own floor of the same size, except in the wake word
leagues. In the `intent`, `stt`, `tts` and `vad` groups, `assemble` makes
battles and an ELO seed for a `(league, dataset, lang)` only when the rows of
all its fighters together hold 30 or more distinct samples. The floor counts
the samples of the whole set, not of one entry, so a set can give battles
while some of its entries are unranked. In `wake_word` and `ww_stream`
there is no sample floor. Only the fighters that the board ranks take part
in battles and the seed, and a board with no ranked fighter gives neither.

A league appears on the site once `data/index.json` lists a benchmark board
of it with at least one entry, ranked or not. `export-index` lists every
`benchmark-*.json` file with its entry count, and the site builds its tabs
from that list. It leaves out two kinds of board: a wake word board that
`arena/ww_listing.py` hides, and a false wake-up board with no entry. The
files stay on disk.

The [front page](index.md#how-to-read-a-board) tells how to read a board and
lists every reason an entry can be unranked.

## ELO seeding

`assemble` derives an auto-battle for every `(sample, fighter pair)` where the
primary metric separates the two fighters. The better one wins. The battles
replay in a fixed order at a quarter of the K-factor and seed the ladder
before any human vote (§4 R5).

| Battle group | The winner of a sample |
|---|---|
| `intent` | The correct answer, OOD rejection included. Rows in an in-distribution bucket give no battle. |
| `wake_word`, `vad` | The correct decision. |
| `stt` | The lower WER. |
| `tts` | The higher UTMOS × intelligibility × judge agreement composite. When both rows have no intelligibility score, the higher UTMOS wins. When one row has it and the other does not, no battle. |
| `ww_stream` | No auto-battle. The recordings hold no wake word, so no fighter catches an onset that the other misses. |

In every battle group, a pair of fighters whose confidence intervals on the
primary metric overlap gets no auto-battles on that dataset (R5a). The
weight of the auto-battles of one pair is capped at 5 human votes (R5b).
Human votes stay the primary signal of the TTS ladder, and the composite
seed is an objective cross-check beside them.

[Rating](rating.md) gives every rule and parameter.
[Methodology](methodology.md#how-auto-battles-are-decided) tells why.

Media-classification and agent-plugin leagues are planned for later.

## Rank badges

Every `tally` run writes an SVG badge per fighter to
`data/badges/<group>/<competitor_id>.svg` on the Pages site. `<group>` is the
battle group, so the four intent leagues share the `intent` folder. The
badge shows the rank and rating of the fighter on its ladder. It stays
byte-identical between runs when nothing changed. Embed it in a plugin
README:

```markdown
[![OVOS Arena](https://openvoiceos.github.io/ovos-plugin-arena/data/badges/stt/<competitor_id>.svg)](https://openvoiceos.github.io/ovos-plugin-arena/leaderboard/)
```

---
[← Front page](index.md) · [Home](index.md) · [Intent →](leagues/intent.md)
