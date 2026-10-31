# STT league (`stt`)

The STT league measures how well a fighter transcribes a spoken clip. Each
row is one clip with a gold transcript. The scorer compares the fighter's
text with the gold text word by word. The board ranks on the mean word error
rate (WER) of the rows.

## How a fighter runs

`runner/stt_bench.py` (`STTBench`) loads the fighter's real OVOS STT plugin
through OPM (`ovos_plugin_manager.stt.load_stt_plugin`). The fighter's
`config.stt` block names the plugin in `module`, or the runner uses the
fighter's `plugin`. The runner builds the plugin with this config:

```python
{"lang": lang, "module": module, **config["stt"][module]}
```

`lang` is the language of the dataset split that the run reads.

For each clip the runner does these steps:

1. It decodes the clip to mono float32 and resamples it to 16 kHz
   (`runner.audio_io.decode_audio_bytes`).
2. It wraps the samples in `AudioData` with a sample width of 2 bytes.
3. It calls `engine.transcribe(audio, lang=lang)` and times the call.
4. It keeps the first hypothesis. A plugin can return a string, a
   `(text, confidence)` tuple, or a list of either. When the plugin gives no
   confidence, the row records 1.0.

The plugin owns its model, its decoding and any text post-processing. The
arena never changes the text the plugin returns.

A fighter runs on the languages that it shares with the dataset.
`competitor_langs` matches on the primary subtag, so a fighter that lists
`pt` runs on a `pt-PT` split. A fighter with an empty `langs` list runs on
every language of the dataset.

A row holds `reference_text` (the gold transcript), `prediction` (the
hypothesis), `confidence`, `latency_ms`, `audio_secs` (the clip length) and
`audio_url`. A clip from a parquet corpus has no stable URL, so its
`audio_url` is `null`. A clip from a corpus with a `metadata.csv`,
`.tsv` or `.jsonl` manifest carries its Hugging Face resolve URL.

**Rows the runner skips.** Some corpora mark a segment as "do not score"
with the STM sentinel `IGNORE_TIME_SEGMENT_IN_SCORING` in place of a
transcript. `_SENTINEL_REFERENCES` lists that value, and the runner writes
no row for such a clip. A run with `--max-samples N` can thus write fewer
than N rows.

## Run it

Each STT corpus family has its own script in `benchmarks/`, for example
`stt_minds14.py`, `stt_fleurs.py`, `stt_voxpopuli.py` and `stt_mls.py`.
They share one set of flags:

```
usage: stt_minds14.py [-h] [--dataset DATASET] [--competitors COMPETITORS]
                      [--langs LANGS] [--max-samples MAX_SAMPLES]
                      [--output-dir OUTPUT_DIR] [--upload]
                      [--hf-owner HF_OWNER]
```

The run writes rows to
`<output-dir>/<dataset_id>/stt/predictions/<lang>/<competitor_id>.jsonl`.
`--upload` publishes them to `OpenVoiceOS/ovos-stt-bench-<dataset_id>`.
[Benchmarks](../benchmarks.md) gives the scripts and the flags, and
[Reproduce a row](../reproduce-a-row.md) compares a local row with a
published one.

## Metrics

### Text normalisation

`arena.metrics.normalize_transcript` (`WER_NORMALIZER_VERSION = 1`) runs on
the reference and on the hypothesis before the scorer splits them into
words. It does these steps in order:

1. Unicode NFKC normalisation.
2. Case folding.
3. Each run of digits becomes its digits spelled out one by one, in
   English: `"42"` becomes `"four two"`.
4. Each underscore becomes a space.
5. Each character that is not a word character or a space is deleted.
6. Runs of white space collapse to one space, and the ends are trimmed.

The rows on Hugging Face keep the raw text. The arena normalises only when
it scores. Every STT board records `wer_normalizer_version`, so a reader can
tell which convention made its numbers.

### Formulas

For one row, let R be the normalised reference words and H the normalised
hypothesis words. Let d(R, H) be the word-level Levenshtein distance:
substitutions plus deletions plus insertions, each of cost 1.

    row WER = d(R, H) / |R|

`row_wer` computes this from `reference_text` and `prediction`. It reads a
stored `wer` field only when the row has no raw text. A row whose reference
normalises to no words has a WER of 0.0.

| Metric | Formula | Better |
|---|---|---|
| **`wer_mean`** (primary) | The mean of row WER over the scored rows. | lower |
| `wer_median` | The median of row WER over the scored rows. | lower |
| `latency_ms_median` | The median time of one `transcribe` call. | lower |

`wer_mean` gives every clip the same weight, whatever its length. A row WER
can exceed 1.0 when the hypothesis has many inserted words.

**Confidence interval.** The interval is a seeded bootstrap with 1000 rounds
(`BOOTSTRAP_ROUNDS`) and seed 0 (`BOOTSTRAP_SEED`), 2.5th to 97.5th
percentile. Each round draws the rows again and computes
`sum(word errors) / sum(reference words)` over the draw
(`bootstrap_ratio_ci`). The interval thus describes the pooled WER of the
corpus, in which a long clip has more weight than a short one. It is not an
interval of `wer_mean` itself. See the known limits below.

**Ranking.** An entry with fewer than 30 scored rows (`MIN_BOARD_SAMPLES`)
is unranked with the reason `too_few_samples`. Entries with an equal
`wer_mean` share a rank. `tied_with_leader` is true when the interval of an
entry overlaps the interval of rank 1. The [front page](../index.md#how-to-read-a-board)
lists every unranked reason.

The board entry also carries a `perf` block with the real-time factor and
the peak memory per hardware tier.
[Methodology](../methodology.md#performance-boards-rtf-peak-memory-model-size-per-hardware-tier-m2)
explains it.

## Battles and ELO

The STT battle group is the league itself. `assemble` builds a blind battle
pool per `(dataset, lang)` (`battles-stt-<dataset>-<lang>.json`), an ELO seed
per language (`elo-seed-stt-<lang>.json`) and a ladder
(`leaderboard-stt-<lang>.json`). A voter hears the source clip, reads the
gold transcript and sees two hypotheses.

`assemble_battles` skips a pair whose two hypotheses are the same text. It
also skips a pair whose two rows carry different `reference_text` for the
same `sample_id`, and counts it in `skipped_reference_mismatches`. It puts
first the clips where both fighters have a WER above 0, then the clips where
exactly one has a WER of 0. A pool holds at most 200 battles
(`DEFAULT_MAX_BATTLES`), taken in turn from each pair.

For the ELO seed, each clip and each pair of fighters with different row
WER gives one auto-battle. The fighter with the lower row WER wins it. The
seed applies these gates:

- A pair whose confidence intervals overlap on the dataset gives no
  auto-battles there.
- The auto-battle weight of one pair is capped at 5 human votes
  (`MAX_AUTO_WEIGHT_PER_PAIR`).
- A `(dataset, lang)` gives battles and a seed only when its rows hold 30
  or more distinct clips.

[Rating](../rating.md#auto-battles-and-the-seed) gives the seed rules, and
[the ladder](../rating.md#the-ladder) the fit.

## Datasets

[Datasets](../datasets.md#speech-to-text-stt) lists every STT set with its
source, revision, licence and languages. The families are:

| Family | Speech |
|---|---|
| `minds14-<lang>` | Telephone calls to a banking service, from `PolyAI/minds14`. |
| `fleurs-<lang>` | Read sentences, from `google/fleurs`. |
| `voxpopuli-<lang>` | European Parliament speeches, from `facebook/voxpopuli`. |
| `mls-<lang>` | Read audiobooks, from `facebook/multilingual_librispeech`. |
| `speech-massive-<lang>` | Spoken assistant commands, from `FBK-MT/Speech-MASSIVE-test`. |
| `ami-en-GB`, `edacc-en-GB` | Meeting speech and accented conversational English, from `edinburghcstr`. |
| `festcat-ca-ES`, `lafrescat-ca-ES` | Catalan speech, from `projecte-aina`. |

A multilingual entry (`lang: "multi"`) puts `{lang}` in its `source.subset`
or `source.file_pattern`. The runner fills in the language of the run, so
one entry gives one split per language.

On a parquet corpus, the runner reads only the shards whose path names the
split, under the subset when the entry gives one: `<subset>/<split>-*`,
`<subset>/<split>/`, or the same under `data/`. When no shard matches, the
run raises an error. It never falls back to every parquet file of the repo,
so a wrong subset cannot mix the clips of other languages into a board. Each
`fleurs-<lang>` set names its subset `parquet-data/<code>`.

## Samples and manifests

Some families declare `sample_policy: {"max_samples": 2000, "seed": 1337}`:
`voxpopuli`, `mls` and `speech-massive`. The cap is per language.
On a parquet corpus, `select_sample_positions` shuffles the row positions
with the seed and keeps the first 2000. Every fighter and every run thus
draws the same rows. `--max-samples` can make a run smaller than the policy
cap, never larger.

A set without a policy reads every row. A `--max-samples` cap on such a set
takes the first rows of the stream.

`python -m runner.publish_sample_set` publishes the selected sample ids of a
policy set as `sample_sets/<ll_RR>.json` in its prediction repo. `assemble`
then scores every fighter on exactly those ids, and the board marks the
entries `sample_set: "manifest"`. An entry that covers less than 90% of the
manifest is unranked with the reason `sample_set_partial`. A set without a
manifest marks its entries `sample_set: "unmanaged"`, and each fighter is
scored on the rows it has.

## Fairness rules

**One normaliser for every fighter.** The scorer normalises both sides the
same way for every fighter, and recomputes WER from the raw text. A fighter
that writes numbers as digits and a fighter that writes them as words lose
the same amount for the same choice. A stored `wer` field cannot change the
board when the row holds its raw text.

**The same clips for every fighter.** On a policy set, the manifest fixes
the clips. A fighter that covers less than 90% of them is not ranked.

**The plugin keeps its own defaults.** The arena passes only the config of
the registry entry and the language. It adds no prompt, no beam setting and
no vocabulary.

**Rows from another corpus revision are dropped.** When the registry pins a
dataset to a 40-character commit, the board drops rows made against any
other revision and counts them in `rows_other_revision`. The STT sets pin
the branch `main`, so this rule does not apply to them.

**No fighter is scored on its own training data.** A fighter that lists a
dataset in `trained_on` does not run on it, and the assembler drops its rows
for that dataset.

## Known limits

- **The interval and the ranked value differ.** The board ranks on
  `wer_mean`, the mean of row WER. The interval is a bootstrap of the pooled
  WER, `sum(errors) / sum(words)`. On a corpus with clips of mixed length the
  two can differ, and `wer_mean` can fall outside the interval. The tie and
  seed gates compare these intervals.
- **Digits are spelled in English.** Step 3 of the normaliser writes digits
  as English words in every language. A hypothesis that spells a number in
  its own language does not match a gold transcript with digits, and the
  reverse.
- **No WER on an empty reference.** A row whose reference normalises to no
  words scores 0.0, whatever the hypothesis says.
- **Text normalisation is minimal.** The normaliser does not expand
  abbreviations, does not unify spelling variants, and does not map number
  words to digits. A fighter whose style differs from the gold transcripts in
  these ways pays for it.
- **Branch pins.** The STT sets pin `main`, so a change upstream can change
  the clips of a later run.

---
[← False wake-ups](ww_stream.md) · [Home](../index.md) · [TTS →](tts.md)
