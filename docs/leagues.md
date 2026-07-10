# Leagues — tasks & metrics

A **league** is one `(modality)` competition with its own benchmark datasets,
prediction rows, benchmark board, battle pool and ELO standings (separate per
language). This page is the canonical definition of *what task each league
scores and how* — the metric formulas here are exactly what
[`arena/metrics.py`](../arena/metrics.py) computes and what
[`arena/assembler.py`](../arena/assembler.py) turns into the ELO seed.

The arena ranks competitors with two signals (see [SPECIFICATION.md](SPECIFICATION.md)):
a **benchmark board** (objective metric straight from predictions) and an **ELO
ladder** (benchmark-seeded, then refined by human blind-A/B votes; ranked by
`bt_rating`, a batch Bradley-Terry fit with bootstrap confidence intervals —
see [methodology.md](methodology.md)). The
*primary metric* of a league is the number that (a) sorts its benchmark board
and (b) decides the auto-vote outcome that seeds ELO (§4 R5).

## Intent leagues

All three intent leagues share one **task**: map a written utterance to an
intent id (and, where annotated, fill its slots), trained only from the
dataset's own training corpus. The arena owns no confidence numbers — each
engine fires through its own `match_high/medium/low` gate, first stage to fire
wins (exactly as `ovos-core` dispatches its pipeline). A sample whose
`reference_intent` is `null` is **out-of-scope (OOD)**: the correct behaviour
is to predict nothing.

Leagues differ only in *who may compete*, because keyword- and template-paradigm
engines consume different supervision and must not be ranked against each other:

| League | Eligible engines | Paradigm / supervision |
|---|---|---|
| `intent_template` | Padatious, Padacioso, Nebulento, Jurebes, Linha Fina, Markov, … | template — `{slot}` phrase templates + example values |
| `intent_keyword` | Adapt, Palavreado, … | keyword — Adapt-style `required_vocab`/`optional_vocab` rules |
| `intent` (open) | mixed-paradigm pipeline **fusions** (ensembles) | any mix; multi-stage cascades (e.g. Padapt = Padatious × Adapt) |

Paradigm leagues are pure — `runner/intent_bench.py:check_league` rejects a
fighter that carries a stage from the wrong paradigm. The open `intent` league
accepts any mix and is where ensemble cascades compete.

**Metrics** (`score_intent`), per `(league, dataset, lang)`:

| Metric | Meaning | Direction |
|---|---|---|
| **`accuracy`** *(primary → ELO seed)* | share of **all** samples answered correctly, **including correct OOD rejections** | higher better |
| `macro_f1` | unweighted mean per-intent F1 (OOD false-positives hurt the wrongly-fired intent's precision) | higher better |
| `ood_fpr` | false-positive rate on OOD samples — how often the engine hallucinates an intent on out-of-scope input | lower better |
| `slot_exact_match` | exact match over the whole gold slot dict, on rows where the intent was correct and gold slots exist | higher better |
| `latency_ms_median` | median per-utterance match latency | lower better |
| `acc_<bucket>` | accuracy within each test bucket (`template` / `paraphrase` / `near_ood` / `far_ood` / `asr_noise` / `typos`) | higher better |

Per-row scoring lives in the §3.2 `exact_match` field (`reference_intent is
None → prediction is None`); `accuracy` counts those correct rejections.

**Datasets**: `intents-for-eval` (12 langs, 50 intents, 6 buckets) and
`massive-templates` (52 langs, template-only). Each eval corpus links its
paradigm-specific `role: train` sets via `train_datasets`.

## Wake-word league (`wake_word`)

**Task**: per-clip detection. Each fighter's real OVOS `HotWordEngine` is fed
one labelled clip — wake word present (`positive`) or absent (`negative`) —
streamed frame-by-frame (1280 samples = 80 ms @ 16 kHz) through `update()` /
`found_wake_word()`, wrapped in leading + trailing silence so streaming feature
buffers warm exactly as a live mic would drive them
(`runner/ww_bench.py`). The engine owns its own threshold; the arena records
only the binary decision and latency.

**Metrics** (`score_wake_word`), per `(dataset, lang)`:

| Metric | Meaning | Direction |
|---|---|---|
| **`error_rate`** *(primary → ELO seed)* | share of all scored clips decided wrong (`false_accepts + false_rejects` over scored) | lower better |
| `accuracy` | `1 − error_rate` | higher better |
| `false_accept_rate` | fires on a negative — noise/other speech wrongly wakes the assistant (over all negatives) | lower better |
| `false_reject_rate` | misses a positive — user says the wake word and nothing happens (over all positives) | lower better |
| `latency_ms_median` | median per-clip detection latency | lower better |

**Datasets**: per wake phrase — `synthetic-wakewords-hey_mycroft`,
`synthetic-wakewords-hey_jarvis` (TTS positives) and `community-computer` (real
community recordings), each drawing negatives from a shared not-wake-word pool
(speech, ESC-50, FMA music, ambient noise, public-domain sounds) so the
false-accept rate spans realistic scenarios.

**Stacked fighters.** A wake-word competitor is not only a bare engine — it is
the engine *as the listener actually stacks it*. A fighter config MAY add a
**pre-wake VAD** gate (`config.VAD`: the detector only runs on clips the VAD
calls speech, suppressing false-accepts on non-speech) and/or a **verifier**
(`config.hotword_verifier`: an activation only counts when the verifier, e.g. a
speaker check, confirms it). **Each distinct `(engine, VAD, verifier)`
combination is its own competitor** with its own false-accept / false-reject
trade-off — `openwakeword-hey-mycroft`, `openwakeword-hey-mycroft-silero`,
`openwakeword-hey-mycroft-speaker` and `openwakeword-hey-mycroft-silero-speaker`
are four different fighters in the same league.

**Threshold / config variants are distinct fighters too.** The same engine at a
different activation threshold lands a different point on the false-accept /
false-reject curve, so each threshold is its own competitor and its own battle
entry — `openwakeword-hey-mycroft-thr03` (sensitive) and `-thr07` (strict)
compete alongside the 0.5 default. The same holds for VAD thresholds
(`silero-vad-thr03` / `-thr07`) and webrtcvad aggressiveness (`webrtcvad-mode1`).

## VAD league (`vad`)

**Task**: per-clip speech / non-speech detection — the same binary-detection
task as wake word, so it shares the scorer. Each fighter's real OVOS
`VADEngine` is fed a clip frame by frame through `is_silence()`; the clip counts
as **speech** if any frame is voiced (`runner/vad_bench.py`).

**Metrics** (`score_vad`), per `(dataset, lang)` — identical shape to wake word,
and **both error directions matter**:

| Metric | Meaning | Direction |
|---|---|---|
| **`error_rate`** *(primary → ELO seed)* | share of clips decided wrong | lower better |
| `false_accept_rate` | fires **speech** on non-speech (music/noise) | lower better |
| `false_reject_rate` | misses real **speech** | lower better |
| `accuracy`, `latency_ms_median` | — | — |

**Datasets**: `speech-vs-nonspeech` (English speech) plus a per-language
`speech-vs-nonspeech-<lang>` for each MInDS-14 language (de, fr, it, es, pt, nl,
pl, ru, cs, ko, zh, en-US/GB/AU) — MInDS-14 telephone speech as positives vs the
same non-speech pool, so VAD is benchmarked across many languages. VAD fighters
are language-agnostic (run on every language's set). The same VAD plugins also
appear as pre-wake gates in the wake-word league above.

## ELO seeding (all leagues with an objective metric)

`assemble` derives an auto-vote for every `(sample, competitor-pair)` where the
primary metric separates the two fighters: the correct one "wins" that battle,
replayed in deterministic order at **K/4** to seed the ladder before any human
vote (§4 R5, §5). Intent uses per-row correctness (incl. OOD rejection);
wake-word uses per-clip correctness; both reduce to "exactly one fighter right
on this sample → it wins". TTS has no objective metric and therefore no seed —
its ladder accrues purely from blind-A/B human votes.

## Out of scope here

`stt` (WER) and `tts` (human-vote only) are defined in
[SPECIFICATION.md §7](SPECIFICATION.md); media-classification and agent-plugin
leagues are tracked for later absorption in
[NGI0-Commons-Fund#14](https://github.com/OpenVoiceOS/NGI0-Commons-Fund/issues/14).
