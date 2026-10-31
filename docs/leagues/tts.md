# TTS league (`tts`)

The TTS league measures how a fighter speaks a written prompt. A prompt has
no single correct waveform, so the league has no gold reference. It uses two
signals:

- **Human votes.** A voter reads the prompt, hears two clips without the
  fighter names, and picks the better one. The votes move the ELO ladder.
- **Objective judges.** Pinned models score every clip: a naturalness score
  (UTMOS), perceptual quality scores (SIGMOS, DNSMOS, NISQA), and an
  intelligibility score from a panel of speech recognisers. The benchmark
  board ranks on mean UTMOS. A composite of UTMOS and intelligibility seeds
  the ladder before any human vote.

The board is an objective cross-check. The ladder, with its human votes, is
the main ranking of the league.

## How a fighter runs

`runner/tts_bench.py` (`TTSBench`) loads the fighter's real OVOS TTS plugin
through OPM (`ovos_plugin_manager.tts.load_tts_plugin`). The fighter's
`config.tts` block names the plugin in `module`, or the runner uses the
fighter's `plugin`. The runner builds the plugin with this config:

```python
{"lang": lang, "module": module, **config["tts"][module]}
```

For each prompt the runner does these steps:

1. It calls `engine.get_tts(text, wav_path, lang=lang)` directly, with no
   audio player. It times this call alone. `latency_ms` and `elapsed_ms`
   hold this synthesis time, and the judges below are not part of it.
2. It writes the clip to `audio/<lang>/<competitor_id>/<hash>.wav`, where
   `<hash>` is the first 16 hex digits of the SHA-1 of the prompt text.
3. It scores the clip with the UTMOS judge, then with SIGMOS, DNSMOS and
   NISQA.
4. It scores intelligibility with the ASR judge panel of the language (see
   below).
5. It records the Hugging Face resolve URL of the clip as both `prediction`
   and `audio_url`. The URL points into the prediction repo, at
   `audio/<lang>/<competitor_id>/<hash>.wav`.

The judge scores go into the `extras` object of the row, because the row
schema has no fixed field for them. `audio_secs` is the length of the clip.

**A failed synthesis still gives a row.** When `get_tts` raises, the row has
`prediction: null`, `audio_url: null` and `extras.synthesis_error`. It carries
no UTMOS score. When the language has an ASR judge, the row records
`intelligibility_wer` and `intelligibility_cer` of 1.0, the worst value.
A judge that fails on a rendered clip also gives 1.0, with
`extras.intelligibility_error`. A failed SIGMOS, DNSMOS or NISQA judge only
leaves its own keys out.

## Run it

`benchmarks/tts_intents_prompts.py` runs the league. Its default dataset is
`intents-for-eval-prompts`:

```
usage: tts_intents_prompts.py [-h] [--dataset DATASET]
                              [--competitors COMPETITORS] [--langs LANGS]
                              [--max-samples MAX_SAMPLES]
                              [--output-dir OUTPUT_DIR] [--upload]
                              [--hf-owner HF_OWNER]
```

The rows go to
`<output-dir>/<dataset_id>/tts/predictions/<lang>/<competitor_id>.jsonl`, and
the clips to `<output-dir>/<dataset_id>/tts/audio/`. `--upload` publishes both
to `OpenVoiceOS/ovos-tts-bench-<dataset_id>`.

`python -m runner.rescore_tts --dataset-id <id>` adds missing SIGMOS, DNSMOS
and NISQA scores to published rows from the stored clips, with no new
synthesis. With `--rejudge-intelligibility` it also runs the full judge panel
on rows that a single judge scored. It skips a row with no `audio_url`.

## The judges

Every judge is pinned. The row records the judge and its revision beside each
score.

| Judge | Model repo | What it scores | Row keys |
|---|---|---|---|
| UTMOS | `TigreGotico/utmos-onnx` | Naturalness, 1 to 5. | `utmos`, `utmos_judge`, `utmos_judge_revision` |
| SIGMOS (P.804) | `TigreGotico/sigmos-onnx` | `col`, `disc`, `loud`, `noise`, `reverb`, `sig`, `ovrl`. | `sigmos.<dim>` |
| DNSMOS (P.835) | `TigreGotico/dnsmos-onnx` | `sig`, `bak`, `ovrl`. | `dnsmos.<dim>` |
| NISQA-v2 | `TigreGotico/nisqa-onnx` | `mos`, `noi`, `dis`, `col`, `loud`. | `nisqa.<dim>` |

`runner/tts_bench.py` holds the commit of each judge (`UTMOS_JUDGE_REVISION`
and its siblings). Every SIGMOS, DNSMOS and NISQA dimension is on a 1 to 5
scale where higher is better, including `noise` and `disc`.

### Intelligibility

The intelligibility check asks whether a listener can understand the clip.
The runner transcribes the clip with a panel of offline speech recognisers
and compares the result with the prompt.

`runner.asr_judges.resolve_judge_panel(lang)` builds the panel from these
sources, without duplicate models:

1. The primary judge, `resolve_judge_model(lang)`. It is the offline STT
   model that `ovos-config` recommends for the language (`_RECOMMENDS`).
   When there is none, it is a dedicated model pinned for the primary
   subtag (`_FALLBACK`). When there is none of those, it is the model of the
   `ovos-stt-plugin-onnx-asr` language registry.
2. The `ovos-config` recommend and the `_FALLBACK` model, when they differ
   from the primary judge.
3. The multilingual `whisper-base` model of `onnx-asr`, which is always a
   member.

Every judge loads through `onnx-asr`. The runner decodes the clip to 16 kHz
mono, and each judge transcribes it. `arena.rover` combines the transcripts
into one consensus with word-level ROVER. It aligns the transcripts
word by word and takes the plurality word in each slot. A gap counts as a
candidate, and a tie goes to the primary judge. The agreement is the mean
share of the panel that voted for the winner of each slot.

The scorer then computes WER and CER of the consensus against the prompt,
with the same `normalize_transcript` as the [STT league](stt.md). CER is a
character-level Levenshtein distance over the normalised text without
spaces. The row stores:

| Key | Value |
|---|---|
| `intelligibility_wer`, `intelligibility_cer` | WER and CER of the consensus. |
| `intelligibility_judges` | Each judge's model, revision and raw transcript. |
| `intelligibility_consensus` | The ROVER consensus. |
| `intelligibility_agreement` | The panel agreement, 0 to 1. |
| `intelligibility_judge`, `intelligibility_judge_revision` | The primary judge. |
| `intelligibility_rover` | `true` for a row the full panel scored. |

`arena.rover` is a pure function of the stored transcripts. Anybody can
recompute the consensus of a row from `intelligibility_judges` with no ASR
run.

**A language with no judge.** `judge_available(lang)` is false when no
recommend, no `_FALLBACK` entry and no registry entry covers the language.
The row then records `intelligibility: "not_available"`, with `null` WER and
CER and `intelligibility_judge: "none"`. A number from a recogniser that was
never trained on the language would measure the recogniser, not the voice.

## Metrics

`arena.metrics.score_tts` computes these metrics for each
`(dataset, lang)`. A row without a finite `utmos` value is left out of the
UTMOS mean.

| Metric | Formula | Better |
|---|---|---|
| **`utmos`** (primary) | The mean `utmos` over the rows that have one. | higher |
| `n_scored` | The number of rows behind `utmos`. | n/a |
| `intelligibility_wer` | The mean `intelligibility_wer` over the rows that have one. Failed renders count as 1.0. | lower |
| `intelligibility_wer_ci_lower`, `intelligibility_wer_ci_upper` | A seeded bootstrap interval of that mean. | n/a |
| `intelligibility_n_scored` | The number of rows behind `intelligibility_wer`. | n/a |
| `sigmos.<dim>`, `dnsmos.<dim>`, `nisqa.<dim>` | The mean of each dimension over the rows that have it. | higher |
| `latency_ms_median` | The median synthesis time of one prompt. | lower |

**Confidence interval.** The interval of the primary metric is a seeded
bootstrap of the mean UTMOS: 1000 rounds (`BOOTSTRAP_ROUNDS`), seed 0
(`BOOTSTRAP_SEED`), 2.5th to 97.5th percentile.

**Ranking.** The floor of 30 rows (`MIN_BOARD_SAMPLES`) applies to
`n_scored`, not to the row count. An entry with fewer than 30 scored clips
is unranked with the reason `too_few_samples`. An entry with no scored clip
at all is unranked with the reason `run failed`.

**Judge agreement.** Each TTS board carries a `judge_agreement` block. It
takes the mean score of each fighter under five judges: UTMOS, SIGMOS
`ovrl`, DNSMOS `ovrl`, NISQA `mos`, and intelligibility (WER, negated so
higher is better). It gives the Spearman rank correlation of each pair of
judges over the fighters they share, and the top 5 fighters of each judge.
When the judges disagree, the board says so.

**Judge audit.** The board lists the primary intelligibility judges of its
rows in `intelligibility_judges`. When a board mixes judges, it adds a
warning and counts the same-sample pairs whose judges differ in
`intelligibility_judge_mismatched_pairs`.

## Battles and ELO

The TTS battle group is the league itself. `assemble` builds a blind battle
pool per `(dataset, lang)` (`battles-tts-<dataset>-<lang>.json`), an ELO seed
per language (`elo-seed-tts-<lang>.json`) and a ladder
(`leaderboard-tts-<lang>.json`). A voter reads the prompt and hears the two
clips. The pool has no notion of a wrong answer, so its battles come in
`battle_id` order, taken in turn from each pair, up to 200 per pool.

### How an auto-battle is decided

`arena.metrics.tts_seed_score` gives each row one score:

    score = (min(max(utmos, 0), 5) / 5) × (1 − min(max(cer, 0), 1)) × agreement

`cer` is `intelligibility_cer`. `agreement` is `intelligibility_agreement`,
or 1.0 on a row that a single judge scored. On a row with
`intelligibility: "not_available"`, the score is `utmos / 5` alone. A row
with no UTMOS, or with no CER on a judged language, has no score.

The product makes each factor count. A natural voice that nobody can
understand scores low, and so does a clear voice that sounds robotic. Low
panel agreement lowers the score again, because judges that cannot agree on
the words are evidence of unclear speech.

`arena.assembler.auto_outcome` compares two rows of the same prompt:

| Rows | Outcome |
|---|---|
| Both carry a CER, from the same primary judge. | The higher `tts_seed_score` wins. Equal scores give no battle. |
| Both carry a CER, from different primary judges. | No battle, because the two WERs are on different scales. |
| Neither carries a CER. | The higher `utmos` wins. Equal scores give no battle. |
| One carries a CER and the other does not. | No battle. |

The seed then applies the gates of every league. A pair whose UTMOS
intervals overlap on the dataset gives no auto-battles there. The
auto-battle weight of one pair is capped at 5 human votes
(`MAX_AUTO_WEIGHT_PER_PAIR`). A `(dataset, lang)` gives battles and a seed
only when its rows hold 30 or more distinct prompts.
[Rating](../rating.md#auto-battles-and-the-seed) gives the seed rules and
[the ladder](../rating.md#the-ladder) the fit.

## Datasets

[Datasets](../datasets.md#text-to-speech-tts) lists the prompt sets.

| Dataset | Prompts |
|---|---|
| `intents-for-eval-prompts` | The test utterances of `OpenVoiceOS/intents-for-eval`, one `{lang}/test.jsonl` file per language. |
| `massive-prompts` | The test utterances of `OpenVoiceOS/massive-templates`, one `{lang}/test.jsonl` file per language. |

The runner reads the `utterance` column (`reference_fields.text`). A prompt
is an assistant command or question, as a user would say it. An eval file
whose rows still hold template markup, such as `(a|b)`, `[word]` or `{slot}`,
makes the run fail with `UnexpandedUtterances`.

Most TTS fighters are voices of `phoonnx`. `scripts/gen_phoonnx_fighters.py`
writes one fighter per voice of the phoonnx voice index.

## Samples

The prompt sets declare no `sample_policy`, and the TTS adapter does not
read one. A run reads every prompt of the language. `--max-samples N` takes
the first N prompts. The sample id is `<lang>/<index>`, with a 5-digit index,
so the same prompt keeps its id across runs. No sample-set manifest exists
for TTS, so every board entry is marked `sample_set: "unmanaged"`.

## Fairness rules

**The same prompts for every fighter.** Every fighter of a language reads
the same prompt file, in the same order.

**Pinned judges.** Each judge is a fixed model commit, and every row records
it. A score from another judge revision is visible as such.

**One recogniser per language.** A board that mixes primary intelligibility
judges warns, and pairs with different judges give no intelligibility
auto-battle.

**No invented intelligibility.** A language that no recogniser covers seeds
on naturalness alone, and the row says so.

**Synthesis time only.** `latency_ms` and the performance data cover the
`get_tts` call, not the judges.

## Known limits

- **Failed renders leave the UTMOS mean.** A row with no clip has no UTMOS,
  so `utmos` is the mean over the clips the fighter did render. Read
  `n_scored` beside it. The intelligibility mean counts each failure as 1.0.
- **UTMOS is a model.** It was trained on human ratings with their own
  language mix. A gap in UTMOS between voices of different languages says
  little. Boards are per language, so the board never compares them.
  [Methodology](../methodology.md#objective-tts-scoring-utmos-4-r14) lists
  the known biases.
- **Judge revisions are recorded, not enforced.** `onnx_asr.load_model` has
  no revision argument. It loads the current model of the Hub repo. The
  revision on the row is the commit pinned when the judge was chosen.
- **A panel member can be weak.** `whisper-base` is in every panel. On a
  language it transcribes poorly, it lowers the agreement of every clip
  alike.
- **Branch pins.** The prompt sets pin `main`, so a change upstream can
  change the prompts of a later run.
- **No pronunciation reference.** The intelligibility check tests whether
  recognisers recover the words. It does not test stress, prosody or the
  pronunciation of names.

---
[← STT](stt.md) · [Home](../index.md) · [VAD →](vad.md)
