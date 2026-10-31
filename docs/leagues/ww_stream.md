# False wake-up league (`ww_stream`)

The `ww_stream` league measures false wake-ups. A wake word fighter streams
hours of recordings that hold no wake word, as a live microphone feeds it.
The board counts how often per hour the fighter woke up. A false wake-up
rate needs hours of audio, and a clip of a few seconds cannot give it. This
league is therefore separate from the [clip league](wake_word.md), which
measures detection.

The league has one dataset per language, `ww-negatives-<lang>`. The
publish script writes it only for a language that has a featured phrase.
[Which boards the site lists](wake_word.md#which-boards-the-site-lists)
defines a featured phrase. [Datasets](../datasets.md) lists the sets.

## How a fighter runs

`runner/ww_bench.py` (`WakeWordStreamBench`) uses the same engine stack as
the clip league. Only fighters that declare `"stream"` in `capabilities` run
here. Every wake word fighter in the registry declares it, because every OVOS
wake word plugin runs on the live microphone stream.

The runner resets the engine once, then feeds the whole recording frame by
frame, with no leading silence. A frame engine gets 1280 samples (80 ms at
16 kHz) per call. A windowed engine, one with a `time_between_checks`
attribute such as the vosk plugin, gets the 2048-sample listener chunk and a
fresh engine per recording, as in the [clip league](wake_word.md#how-a-fighter-runs).
The runner does not reset the engine after an activation. The plugin's own
latch and cooldown decide when it can fire again. Each activation becomes an
event `(time_s, score)`. The score is the engine's numeric `confidence` or
`score` after it fires, or 1.0 when the engine has no such number. A
configured VAD and verifier judge the 2 s of audio before the activation
(`GATE_WINDOW_S`). The activation counts only when the VAD hears speech in
that window and the verifier confirms it.

A row has `prediction: "WW_STREAM"`. Its `extras` hold `events`,
`duration_s`, the `category` of the recording, and empty `truth_onsets` and
`truth_ends`. The runner caches the events of each recording per fighter
configuration in `ww-stream-events` under `~/.cache/ovos-plugin-arena` (or
`$ARENA_CACHE`), so a second run reads them back.

`benchmarks/ww_stream.py --dataset ww-negatives-<lang> --competitors <ids>`
runs the bench. Every fighter streams every recording, so name the fighters.

## Metrics

`arena.metrics.score_ww_stream` computes these metrics for each
`(dataset, lang)`. The recordings hold no wake word, so every event at or
above the threshold is a false wake-up.

| Metric | Definition | Better |
|---|---|---|
| **`fa_per_hour`** (primary) | False wake-ups per hour of streamed audio, at threshold 0.5. | lower |
| `negative_hours` | The hours of streamed audio. This is the denominator of `fa_per_hour`. | n/a |
| `n_false_accepts` | False wake-ups at threshold 0.5. | lower |
| `fa_per_hour@<category>`, `negative_hours@<category>` | The same per kind of audio: `read_speech`, `conversation`, `broadcast`, `music`, `noise`, `foreign_speech`. | lower, n/a |
| `det_fa_per_hour@<thr>` | False wake-ups per hour at thresholds 0.1 to 0.9. | lower |

An engine with no score reports 1.0 for every event, so every activation
counts at each threshold. The scorer also matches events to wake word
onsets when a recording holds some, but no registered set has onsets.

**Ranking.** The board holds only the fighters of featured phrases.
`assemble` drops the rows of every other fighter before it builds the board.
An entry backed by less than 4 h of audio (`MIN_NEGATIVES_BOARD_HOURS`) is
unranked with the reason `too_few_samples`. The entries carry no confidence
interval.

## Ladder

The league gives no auto-battles and no blind battles. An auto-battle needs
a wake word onset that one fighter catches and the other misses, and the
recordings hold none. Every row has the prediction `WW_STREAM`, and
`assemble_battles` skips a pair of rows with equal predictions. The board is
the only ranking.

## The site

The leaderboard shows each listed false wake-up board in its own section,
"False wake-ups per hour", on the wake word overview. It ranks the fighters
by `fa_per_hour`, lowest first, with a column for each kind of audio. The
site lists a board only when a listed clip board has the same primary
language. A board that is not listed stays on disk.

## Datasets

All sets come from one corpus,
[`OpenVoiceOS/ww-stream-bench`](https://huggingface.co/datasets/OpenVoiceOS/ww-stream-bench),
pinned to one commit. A `ww-negatives-<lang>` entry reads the
`manifest-negatives-<lang>.jsonl` file of the corpus. Every row of it names
its `category`.

For English, the recordings are about 20 hours: read speech, meeting
conversation, radio, parliament and talk recordings, music, household and
outdoor sounds, and speech in other languages. The `notes` of the entry give
the hours of each kind. The dataset card of the corpus lists every source
with its revision, its licence and its hours.

The corpus also holds negatives for other languages, and a recording of the
clips of each clip test set. These get no registry entry.

The recordings use none of the TigreGotico negative corpora of the clip
boards (`not-wake-words-speech-en`, `ESC-50`, `public_domain_sounds_3secs`,
`FMA_3secs`, `ambient_noises`). The corpus card lists them as sources a
fighter trained on, and leaves them out.

## Building the corpus

`scripts/publish_ww_stream_bench.py` builds the corpus from pinned source
revisions with a fixed seed and uploads it. It writes a `ww-negatives-<lang>`
registry entry for each language with a featured phrase
(`registry_featured_phrases` in `arena/ww_listing.py`):

```
usage: publish_ww_stream_bench.py [-h] --work WORK [--seed SEED]
                                  [--only [ONLY ...]] [--upload]
```

`scripts/build_ww_stream_corpus.py` builds the recordings from local audio.
It writes 16 kHz mono recordings and `manifest.jsonl`. The output depends
only on the inputs and `--seed`.

## Samples

The sets have no `sample_policy` and no sample-set manifest, so the board
marks every entry `sample_set: "unmanaged"`. `--max-samples` on
`benchmarks/ww_stream.py` caps the number of recordings, for a smoke run.

## Known limits

- **One operating point per engine.** `fa_per_hour` counts the events at
  threshold 0.5. An engine with no score fires at its own configured
  threshold, so its rate is the rate at that threshold.
- **No detection.** This board does not tell whether a fighter catches the
  wake word. Read the clip board of the phrase beside it.

---
[← Wake word](wake_word.md) · [Home](../index.md) · [STT →](stt.md)
