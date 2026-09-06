"""Plain-language explanations of every league and every board metric.

A leaderboard column header is a code name (``ood_fpr``, ``sigmos.disc``)
that means nothing to a visitor who did not write the scorer. This module is
the single source of truth for what each number means, which direction is
better, what scale it is on, and the caveat that changes how to read it.
``arena.cli export-index`` writes it to ``data/metrics-legend.json`` and the
site renders it beside the boards; the prose here matches docs/leagues.md.

Every metric key that appears in a published ``benchmark-*.json`` board MUST
have an entry — a number nobody can explain has no business being ranked on.
"""

from __future__ import annotations

from typing import Any, Literal

Direction = Literal["higher", "lower", "neutral"]

# One paragraph per league: the task, what the fighter is handed, what it
# must produce, and what a human voter is asked to judge.
LEAGUE_INTROS: dict[str, str] = {
    "intent_template": (
        "Template-paradigm intent engines compete here. A fighter is trained "
        "only from phrase templates with {slot} placeholders and example slot "
        "values, then handed one written utterance at a time and must answer "
        "with the intent it belongs to — or with nothing at all, when the "
        "utterance is out of scope. Voters are shown two engines' answers to "
        "the same utterance and pick the one that understood it."
    ),
    "intent_keyword": (
        "Keyword-paradigm intent engines compete here. A fighter is trained "
        "from Adapt-style required and optional vocabulary rules rather than "
        "phrase templates, then handed one written utterance at a time and "
        "must answer with the intent it belongs to, or with nothing when the "
        "utterance is out of scope. Voters are shown two engines' answers to "
        "the same utterance and pick the one that understood it."
    ),
    "intent": (
        "The open intent league is for pipeline fusions: multi-stage cascades "
        "that mix paradigms, such as a template matcher backed by a keyword "
        "matcher. Supervision may come from any of the training corpora. The "
        "task is the same one the pure leagues run — read an utterance, name "
        "the intent, stay silent on out-of-scope input — and voters judge the "
        "two answers side by side."
    ),
    "stt": (
        "Speech-to-text engines transcribe a spoken clip. A fighter receives "
        "the audio and returns the words it heard; the transcript is compared "
        "with the human reference after both sides are normalised the same "
        "way, so punctuation and formatting differences do not count as "
        "errors. Voters listen to the clip and pick the transcript that is "
        "closer to what was said."
    ),
    "tts": (
        "Text-to-speech voices read a sentence aloud. A fighter receives the "
        "text and returns synthesised audio, which is scored two ways: how "
        "natural it sounds and how accurately a panel of listeners-by-machine "
        "can transcribe it back. Human ears are the primary signal here — "
        "voters hear two clips of the same sentence and pick the better one."
    ),
    "wake_word": (
        "Wake-word detectors listen for the phrase that starts a "
        "conversation. A fighter is fed one labelled clip at a time, frame by "
        "frame exactly as a live microphone would drive it, and must decide "
        "whether the phrase was spoken. Missing the phrase and waking up on "
        "the wrong sound are both failures, and the engine owns its own "
        "threshold. Voters compare how two detectors behaved on the same clip."
    ),
    "vad": (
        "Voice-activity detectors decide whether a clip contains speech at "
        "all. A fighter is fed the clip frame by frame and the clip counts as "
        "speech when any frame is voiced. Both mistakes matter: calling music "
        "or noise speech wakes the pipeline for nothing, and missing real "
        "speech drops the user's words. Voters compare the two decisions on "
        "the same clip."
    ),
    "ww_stream": (
        "Streaming wake-word detectors run continuously over hours of audio "
        "instead of isolated clips, the way they run against a live "
        "microphone. A fighter reports every activation with its timestamp, "
        "and an activation counts only when it lands close enough to a real "
        "spoken onset. Hours of continuous non-wake audio are what make the "
        "false-alarm rate meaningful."
    ),
}


def _metric(
    label: str,
    meaning: str,
    direction: Direction,
    unit: str,
    caveat: str | None = None,
) -> dict[str, Any]:
    entry = {"label": label, "meaning": meaning, "direction": direction, "unit": unit}
    if caveat:
        entry["caveat"] = caveat
    return entry


_TRAINED_ON_CAVEAT = (
    "Includes the test buckets whose phrasings resemble the training "
    "templates; generalization_accuracy excludes them and is what the board "
    "is ranked on."
)
_NEAR_TRAINING_CAVEAT = (
    "Phrasing close to the training material, so this bucket is excluded "
    "from generalization_accuracy and counts only towards overall accuracy."
)
_MOS_CAVEAT = (
    "A machine listener's opinion score, not a human panel, and the judge "
    "models are trained mostly on English-adjacent audio — compare voices "
    "within one language, never across languages."
)

METRICS: dict[str, dict[str, Any]] = {
    # ---- intent ----
    "generalization_accuracy": _metric(
        "Generalization",
        "Share of correct answers on the phrasings least like the training "
        "data — free-form paraphrases, out-of-scope input, typos and "
        "transcription noise.",
        "higher", "share of rows, 0–100%",
        "The ranked number for every intent league: it is the one that says "
        "how the engine behaves on words it was never shown.",
    ),
    "accuracy": _metric(
        "Overall accuracy",
        "Share of all scored rows the fighter got right, counting a correct "
        "silence on out-of-scope input as right.",
        "higher", "share of rows, 0–100%", _TRAINED_ON_CAVEAT,
    ),
    "macro_f1": _metric(
        "Macro F1",
        "Average of the per-intent F1 scores, weighting a rare intent as "
        "heavily as a common one.",
        "higher", "score, 0–100%",
        "Firing the wrong intent on out-of-scope input hurts that intent's "
        "precision, so hallucinations show up here.",
    ),
    "ood_fpr": _metric(
        "Out-of-scope false fires",
        "How often the engine answers with some intent on an utterance that "
        "has no intent at all.",
        "lower", "share of out-of-scope rows, 0–100%",
        "Only defined on corpora that carry out-of-scope rows.",
    ),
    "slot_exact_match": _metric(
        "Slot exact match",
        "Share of rows where every extracted slot value matched the "
        "annotation exactly.",
        "higher", "share of rows, 0–100%",
        "Scored only on rows that have gold slots and where the intent was "
        "already correct, so it measures extraction, not recognition.",
    ),
    "ece": _metric(
        "Calibration error",
        "How far the confidence an engine reports drifts from how often it is "
        "actually right at that confidence.",
        "lower", "error magnitude, 0–1",
        "A dataset-wide aggregate with no per-row value, so it never gets its "
        "own ranking ladder.",
    ),
    # Per-bucket accuracy on the intents-for-eval test set, which tags every
    # row with the kind of challenge it poses. ``near_ood`` never appears as
    # its own column: arena/metrics.py's BUCKET_METRIC_NAMES publishes it
    # under acc_in_distribution.
    "acc_test": _metric(
        "Test bucket",
        "Accuracy on rows the corpus left unlabelled by bucket — the default "
        "held-out test rows.",
        "higher", "share of rows, 0–100%",
    ),
    "acc_template": _metric(
        "Held-out templates",
        "Accuracy on utterances built from phrase templates held back from "
        "training, so the wording follows a pattern the engine was taught but "
        "this exact template was not.",
        "higher", "share of rows, 0–100%", _NEAR_TRAINING_CAVEAT,
    ),
    "acc_in_distribution": _metric(
        "In-distribution",
        "Accuracy on in-domain utterances phrased close to the training "
        "material.",
        "higher", "share of rows, 0–100%", _NEAR_TRAINING_CAVEAT,
    ),
    "acc_paraphrase": _metric(
        "Paraphrases",
        "Accuracy on free-form rewordings of an intent that keep the meaning "
        "but not the training phrasing.",
        "higher", "share of rows, 0–100%",
    ),
    "acc_typos": _metric(
        "Typos",
        "Accuracy on utterances with spelling mistakes of the kind a person "
        "makes typing.",
        "higher", "share of rows, 0–100%",
    ),
    "acc_asr_noise": _metric(
        "Speech-recognition noise",
        "Accuracy on utterances corrupted the way a speech recogniser "
        "mishears them, so it shows how an engine copes downstream of an "
        "imperfect transcript.",
        "higher", "share of rows, 0–100%",
    ),
    "acc_far_ood": _metric(
        "Far out-of-scope",
        "Accuracy on utterances from a completely different domain, where "
        "answering nothing is the correct behaviour.",
        "higher", "share of rows, 0–100%",
    ),
    # ---- stt ----
    "wer_mean": _metric(
        "Word error rate, mean",
        "Average share of words a transcript got wrong — words inserted, "
        "deleted or substituted against the human reference.",
        "lower", "share of words, 0–100%",
        "Both sides are lowercased, stripped of punctuation and spelled out "
        "digit by digit first, so formatting differences do not count as "
        "errors. A single disastrous clip can pull the mean far above the "
        "median.",
    ),
    "wer_median": _metric(
        "Word error rate, median",
        "The middle clip's word error rate, unaffected by a handful of "
        "catastrophic transcripts.",
        "lower", "share of words, 0–100%",
    ),
    # ---- detection: wake word and VAD ----
    "error_rate": _metric(
        "Error rate",
        "Share of clips decided wrong in either direction — missed "
        "detections plus false alarms.",
        "lower", "share of clips, 0–100%",
        "One number over two very different failures; read it together with "
        "the false-accept and false-reject columns, which trade off against "
        "each other as an engine's threshold moves.",
    ),
    "false_accept_rate": _metric(
        "False alarms",
        "How often the detector fires on audio that should not have "
        "triggered it — other speech, music or noise.",
        "lower", "share of negative clips, 0–100%",
        "Measured over clips, not over hours, so it does not say how often a "
        "device would wake up on its own in a day.",
    ),
    "false_reject_rate": _metric(
        "Misses",
        "How often the user says the phrase, or speaks, and the detector "
        "stays silent.",
        "lower", "share of positive clips, 0–100%",
    ),
    "fa_per_hour": _metric(
        "False alarms per hour",
        "How many times an hour of non-wake audio wrongly triggers the "
        "detector — the annoyance a user actually feels.",
        "lower", "count per hour",
        "Omitted for a fighter backed by too little negative audio to give a "
        "stable estimate; fa_per_hour_hours says how much audio there was.",
    ),
    "fa_per_hour_hours": _metric(
        "Negative audio covered",
        "Hours of non-wake audio behind the false-alarms-per-hour figure.",
        "neutral", "hours",
        "Coverage, not performance — it is here so a board never implies more "
        "evidence than it has.",
    ),
    # ---- tts ----
    "utmos": _metric(
        "Naturalness",
        "A machine estimate of the mean opinion score listeners would give "
        "the clip for how natural it sounds.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "intelligibility_wer": _metric(
        "Intelligibility",
        "Word error rate when a panel of speech recognisers transcribes the "
        "synthesised clip back — how easy the voice is to understand.",
        "lower", "share of words, 0–100%",
        "Measured by machine listeners, so a voice that is hard for models "
        "and easy for people scores worse than it deserves.",
    ),
    "intelligibility_wer_ci_lower": _metric(
        "Intelligibility, lower bound",
        "Lower end of the 95% confidence interval around the intelligibility "
        "word error rate.",
        "neutral", "share of words, 0–100%",
        "Two voices whose intervals overlap are not separated by the data.",
    ),
    "intelligibility_wer_ci_upper": _metric(
        "Intelligibility, upper bound",
        "Upper end of the 95% confidence interval around the intelligibility "
        "word error rate.",
        "neutral", "share of words, 0–100%",
        "Two voices whose intervals overlap are not separated by the data.",
    ),
    "intelligibility_n_scored": _metric(
        "Clips scored for intelligibility",
        "How many synthesised clips the recogniser panel transcribed back.",
        "neutral", "count",
    ),
    "n_scored": _metric(
        "Rows scored",
        "How many rows this fighter actually produced a scorable answer for.",
        "neutral", "count",
        "A fighter with far fewer scored rows than its rivals was measured on "
        "less evidence, however good its score looks.",
    ),
    "latency_ms_median": _metric(
        "Latency",
        "The middle row's processing time, measured on the machine that ran "
        "the sweep.",
        "lower", "milliseconds",
        "Hardware-dependent: comparable between fighters on one board, not "
        "against numbers from anywhere else.",
    ),
    "sigmos.sig": _metric(
        "Speech quality (SIGMOS)",
        "How clean and undistorted the speech itself sounds, judged apart "
        "from anything in the background.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "sigmos.noise": _metric(
        "Background cleanliness (SIGMOS)",
        "How free the clip is of hiss and background noise; a clean clip "
        "scores high.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "sigmos.col": _metric(
        "Coloration (SIGMOS)",
        "How free the voice is of muffling and tinny or boxy timbre.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "sigmos.disc": _metric(
        "Continuity (SIGMOS)",
        "How free the clip is of clicks, gaps and dropouts; smooth audio "
        "scores high.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "sigmos.loud": _metric(
        "Loudness (SIGMOS)",
        "How comfortable the playback level is, neither faint nor blaring.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "sigmos.reverb": _metric(
        "Room echo (SIGMOS)",
        "How free the clip is of reverberation, the hollow sound of a large "
        "room.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "sigmos.ovrl": _metric(
        "Overall quality (SIGMOS)",
        "The SIGMOS judge's single summary score across all its dimensions.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "dnsmos.sig": _metric(
        "Speech quality (DNSMOS)",
        "A second judge's view of how clean the speech itself sounds.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "dnsmos.bak": _metric(
        "Background cleanliness (DNSMOS)",
        "A second judge's view of how free the clip is of background noise.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "dnsmos.ovrl": _metric(
        "Overall quality (DNSMOS)",
        "The DNSMOS judge's single summary score.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "nisqa.mos": _metric(
        "Overall quality (NISQA)",
        "A third judge's summary opinion score for the clip.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "nisqa.noi": _metric(
        "Background cleanliness (NISQA)",
        "A third judge's view of how free the clip is of background noise.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "nisqa.col": _metric(
        "Coloration (NISQA)",
        "A third judge's view of how free the voice is of muffled or tinny "
        "timbre.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "nisqa.dis": _metric(
        "Continuity (NISQA)",
        "A third judge's view of how free the clip is of clicks and dropouts.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
    "nisqa.loud": _metric(
        "Loudness (NISQA)",
        "A third judge's view of how comfortable the playback level is.",
        "higher", "MOS, 1–5", _MOS_CAVEAT,
    ),
}


def metrics_legend() -> dict[str, Any]:
    """The legend payload published as ``data/metrics-legend.json``."""
    return {"metrics": METRICS, "leagues": LEAGUE_INTROS}
