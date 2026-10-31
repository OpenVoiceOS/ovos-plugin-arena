# VAD league (`vad`)

The VAD league measures one decision per clip. A clip either holds speech
(a positive) or does not (a negative). The fighter must report speech on the
positives and stay silent on music, noise and environmental sound. Both
errors cost something in a voice assistant. A false accept wakes the
pipeline for nothing. A false reject drops the words of the user.

VAD is the same binary detection task as the [wake word league](wake_word.md).
The two leagues share the scorer, the confidence interval and the ranking
rules.

## How a fighter runs

`runner/vad_bench.py` (`VADBench`) loads the fighter's real OVOS `VADEngine`
through OPM (`ovos_plugin_manager.vad.load_vad_plugin`). The config block
comes from `config.VAD`, or from `config.listener.VAD` when the first is
absent. The `module` key names the plugin. When it is absent, the runner uses
the fighter's `plugin`.

For each clip the runner does these steps:

1. It calls `reset()` on the engine, when the engine has one.
2. It converts the clip to 16-bit little-endian PCM at 16 kHz.
3. It feeds the clip in frames of 480 samples (30 ms) through
   `is_silence(chunk)`. This is the call the OVOS listener makes to find the
   end of speech.
4. It stops at the first frame that `is_silence` reports as not silent. The
   clip is then `speech`. When no frame is voiced, the clip is `silence`.

A trailing part shorter than one frame is not fed. A frame that raises an
error counts as silent, and the runner logs it at debug level. The engine
uses its own threshold. The arena records only the decision and the latency.

The league registers each engine once, at its own reference threshold. A
threshold variant is one more point on the same engine's curve. The
silero-vad and webrtcvad variants agreed with their default on 91 to 98
percent of clips.
`noise-vad` runs the plugin's `threshold` method. Under the shipped `all`
method, the plugin compares each frame with the loudest frame since
`reset()`, and the runner resets the engine before every clip. The first
frame of every clip would then count as speech. A fighter with an empty
`langs` list runs on every language, and the VAD fighters leave the list
empty.

A row looks like this. It is the first row of a 4-clip smoke run of
`silero-vad` on `speech-vs-nonspeech-en-US`, with the `hw` block and
`created_at` left out:

```json
{"competitor_id": "silero-vad", "sample_id": "speech/602ba55abb1e6d0fbce92065.wav",
 "dataset_id": "speech-vs-nonspeech-en-US",
 "dataset_revision": "40ce77cb32a384e4d50a568e1ec39ac804019d33",
 "lang": "en-US", "modality": "vad", "plugin_id": "ovos-vad-plugin-silero",
 "plugin_version": "ovos-vad-plugin-silero==0.1.5a1",
 "runner_version": "ovos-plugin-arena==0.1.0a1",
 "label": "speech", "prediction": "speech", "audio_url": null,
 "latency_ms": 17.55, "elapsed_ms": 17.59, "peak_rss_mb": null}
```

`label` is `speech` or `non_speech`. `prediction` is `speech` or `silence`.
A clip from a parquet corpus has no stable URL, so its `audio_url` is `null`.
A clip from an audio folder carries its Hugging Face resolve URL.

## Run it

`benchmarks/vad_speech.py` runs the league. Its default dataset is
`speech-vs-nonspeech-en-US`. This smoke run writes 4 clips of each class:

```
python benchmarks/vad_speech.py --competitors silero-vad --max-samples 4 \
    --output-dir predictions
```

The rows go to
`predictions/speech-vs-nonspeech-en-US/vad/predictions/en-US/silero-vad.jsonl`.
When the plugin is not installed, the script logs
`plugin ovos-vad-plugin-silero is not installed — skipping` and exits 0.
`--upload` publishes the rows to `OpenVoiceOS/ovos-vad-bench-<dataset_id>`.
[Benchmarks](../benchmarks.md) gives every flag.

## Metrics

`arena.metrics.score_vad` is `score_wake_word`. It computes these metrics
for each `(dataset, lang)`. The tokens `speech`, `voice`, `positive`,
`detected` and the other present tokens of the wake word scorer mean speech.
Every other non-empty token, `silence` and `non_speech` included, means no
speech.

Let FA be the negatives that the fighter called speech, and FR the positives
that it called silence. Let N be the negatives, P the positives and
n = N + P.

| Metric | Formula | Better |
|---|---|---|
| **`error_rate`** (primary) | (FA + FR) / n | lower |
| `accuracy` | 1 − `error_rate` | higher |
| `false_accept_rate` | FA / N | lower |
| `false_reject_rate` | FR / P | lower |
| `latency_ms_median` | The median time to decide one clip. | lower |

`error_rate` weights the two error types by the number of clips in each
class. Read `false_accept_rate` and `false_reject_rate` beside it.

VAD rows carry no `audio_secs`. The `fa_per_hour` metric of the wake word
scorer is therefore absent on VAD boards.

**Confidence interval.** The interval is a seeded bootstrap of the 0/1 error
of each scored clip: 1000 rounds (`BOOTSTRAP_ROUNDS`), seed 0
(`BOOTSTRAP_SEED`), 2.5th to 97.5th percentile. `tied_with_leader` is true
when the interval of an entry overlaps the interval of rank 1.

**Ranking.** `build_benchmark_board` ranks an entry only when both
`false_accept_rate` and `false_reject_rate` exist. An entry that scored one
class only is unranked with the reason `one_class`. An entry with fewer than
30 scored clips (`MIN_BOARD_SAMPLES`) is unranked with the reason
`too_few_samples`. Entries with equal `error_rate` share a rank.

## Battles and ELO

The VAD battle group is the league itself. `assemble` builds a blind battle
pool per `(dataset, lang)`, an ELO seed per language
(`elo-seed-vad-<lang>.json`) and a ladder (`leaderboard-vad-<lang>.json`).
A voter hears the source clip and sees two decisions.

`assemble_battles` skips a pair of rows whose decisions are equal, because
that vote carries no signal. It puts first the clips where both fighters are
wrong, then the clips where one is wrong. A pool holds at most 200 battles
(`DEFAULT_MAX_BATTLES`), taken in turn from each pair.

For the ELO seed, a clip where exactly one fighter is right is an
auto-battle won by that fighter. A pair whose confidence intervals overlap
gets no auto-battles on that dataset. The auto-battle weight of one pair is
capped at 5 human votes (`MAX_AUTO_WEIGHT_PER_PAIR`). A `(dataset, lang)`
gives battles and a seed only when its rows hold 30 or more distinct clips.
[Rating](../rating.md#auto-battles-and-the-seed) gives the seed rules and
[the ladder](../rating.md#the-ladder) the fit.

## Datasets

[Datasets](../datasets.md#voice-activity-detection-vad) lists every VAD set
with its source, revision and licence. There are two families.

| Family | Positives | Negatives |
|---|---|---|
| `speech-vs-nonspeech-<lang>` | The clips of one language of a speech corpus: `PolyAI/minds14` telephone speech, or `FBK-MT/Speech-MASSIVE-test` for the languages that MInDS-14 lacks. | The shared non-speech pool (`negatives_sources`). |
| `ava-speech-conditions` | Windows of film audio inside a human speech segment. | Windows of the same films outside any speech segment. |

The shared non-speech pool lists four TigreGotico corpora:
`public_domain_sounds_3secs/pd_3sec`, `ESC-50`, `FMA_3secs` and
`ambient_noises`. `_pool_negatives` gives each corpus an even share of the
cap, so one large corpus does not fill the pool. Inside a corpus it picks
files at even steps through the sorted listing. The positives and negatives
alternate, so a run that stops early still holds both classes.

`ava-speech-conditions` reads `nccratliri/vad-human-ava-speech`, a mirror of
AVA-Speech. Each clip is a `<name>.wav` with a `<name>.json` that gives the
onset and offset of each human speech segment. `_stream_ava_speech` cuts
3 s windows (`seg_s`) from each speech segment and from each gap between
segments. A segment or gap shorter than 3 s gives no window, so no window
crosses a label boundary. A window id is `<wav>#<label>_<start>`.

## Samples

No VAD set declares a `sample_policy`, and the VAD adapter does not read
one. `VADBench.iter_samples` caps each class with the `--max-samples` value
of the run. With no cap, a run reads every positive and every file of the
four negative corpora. No sample-set manifest exists for VAD, so every board
entry is marked `sample_set: "unmanaged"`.

## Fairness rules

**The same audio for every fighter.** Every fighter of a `(dataset, lang)`
reads the same clip selection under the same cap. The selection depends only
on the corpus order and the cap.

**Rows from another corpus revision are dropped.** When the registry pins a
dataset to a 40-character commit, the board drops rows made against any
other revision and counts them in `rows_other_revision`. The VAD sets pin
the branch `main`, so this rule does not apply to them.

**No fighter is scored on its own training data.** The rule of
[the wake word league](wake_word.md#fairness-rules) applies. A fighter that
lists a dataset in `trained_on` does not run on it, and the assembler drops
its rows for that dataset.

## Known limits

- **A clip is one decision.** The league does not score where speech starts
  or ends. A fighter that marks one voiced frame anywhere in a clip gets the
  same score as a fighter that tracks the whole utterance.
- **No unit of time.** `false_accept_rate` is a share of clips. It does not
  tell how often a VAD fires per hour on a live microphone.
- **Branch pins.** The VAD sets pin `main`, so a change upstream can change
  the clips of a later run.

---
[← TTS](tts.md) · [Home](../index.md) · [Methodology →](../methodology.md)
