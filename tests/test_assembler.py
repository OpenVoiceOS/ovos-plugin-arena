"""Unit tests for arena.assembler — battles and ELO seeding."""
from __future__ import annotations

from arena.assembler import assemble_battles, auto_outcome, seed_elo
from arena.elo import INITIAL_ELO
from arena.models import PredictionRow, VoteOutcome


def _row(competitor, prediction, reference="media:play_song", **over):
    base = dict(
        competitor_id=competitor,
        sample_id="en-US/00001",
        dataset_id="intents-for-eval",
        lang="en-US",
        plugin_id=f"plugin-{competitor}",
        utterance="play a song",
        reference_intent=reference,
        prediction=prediction,
    )
    base.update(over)
    return PredictionRow(**base)


def _samples(*rows_per_sample):
    """Build the samples mapping from lists of rows (one list per sample)."""
    samples = {}
    for i, rows in enumerate(rows_per_sample):
        sample_id = f"en-US/{i:05d}"
        samples[sample_id] = {
            r.competitor_id: r.model_copy(update={"sample_id": sample_id})
            for r in rows
        }
    return samples


class TestAutoOutcome:
    def test_one_correct_wins(self):
        a = _row("x", "media:play_song")
        b = _row("y", "media:stop")
        assert auto_outcome(a, b, "intent") == VoteOutcome.CANDIDATE_A
        assert auto_outcome(b, a, "intent") == VoteOutcome.CANDIDATE_B

    def test_both_correct_no_signal(self):
        a = _row("x", "media:play_song")
        b = _row("y", "media:play_song")
        assert auto_outcome(a, b, "intent") is None

    def test_both_wrong_no_signal(self):
        a = _row("x", "media:stop")
        b = _row("y", "media:next")
        assert auto_outcome(a, b, "intent") is None

    def test_ood_rejection_wins(self):
        a = _row("x", None, reference=None)
        b = _row("y", "media:play_song", reference=None)
        assert auto_outcome(a, b, "intent") == VoteOutcome.CANDIDATE_A

    def test_stt_lower_wer_wins(self):
        a = _row("x", "ola mundo", wer=0.0)
        b = _row("y", "ola munto", wer=0.5)
        assert auto_outcome(a, b, "stt") == VoteOutcome.CANDIDATE_A

    def test_wake_word_correct_wins(self):
        a = _row("x", "detected", reference=None, label="positive")
        b = _row("y", "not_detected", reference=None, label="positive")
        assert auto_outcome(a, b, "wake_word") == VoteOutcome.CANDIDATE_A
        assert auto_outcome(b, a, "wake_word") == VoteOutcome.CANDIDATE_B

    def test_wake_word_both_correct_no_signal(self):
        a = _row("x", "detected", reference=None, label="positive")
        b = _row("y", "detected", reference=None, label="positive")
        assert auto_outcome(a, b, "wake_word") is None

    def test_tts_has_no_auto_signal(self):
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi")
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi")
        assert auto_outcome(a, b, "tts") is None


class TestAssembleBattles:
    def test_identical_predictions_skipped(self):
        samples = _samples([_row("x", "same"), _row("y", "same")])
        assert assemble_battles("intent", "d", "en-US", samples) == []

    def test_disagreement_battled(self):
        samples = _samples([_row("x", "media:play_song"), _row("y", "media:stop")])
        battles = assemble_battles("intent", "d", "en-US", samples)
        assert len(battles) == 1
        battle = battles[0]
        assert {battle.competitor_a, battle.competitor_b} == {"x", "y"}
        assert battle.input_text == "play a song"
        assert battle.reference == "media:play_song"

    def test_deterministic(self):
        samples = _samples(
            [_row("x", "a"), _row("y", "b"), _row("z", "c")],
            [_row("x", "a"), _row("y", "d")],
        )
        runs = [assemble_battles("intent", "d", "en-US", samples) for _ in range(2)]
        assert [b.battle_id for b in runs[0]] == [b.battle_id for b in runs[1]]

    def test_battle_ids_stable_across_pool_sizes(self):
        samples = _samples([_row("x", "a"), _row("y", "b")])
        full = assemble_battles("intent", "d", "en-US", samples)
        capped = assemble_battles("intent", "d", "en-US", samples, max_battles=1)
        assert full[0].battle_id == capped[0].battle_id

    def test_max_battles_cap(self):
        samples = _samples(*[
            [_row("x", f"a{i}"), _row("y", f"b{i}")] for i in range(20)
        ])
        battles = assemble_battles("intent", "d", "en-US", samples, max_battles=5)
        assert len(battles) == 5

    def test_pairs_interleaved(self):
        # 3 competitors → 3 pairs; with cap 3 every pair appears once
        samples = _samples(*[
            [_row("x", f"a{i}"), _row("y", f"b{i}"), _row("z", f"c{i}")]
            for i in range(10)
        ])
        battles = assemble_battles("intent", "d", "en-US", samples, max_battles=3)
        pairs = {tuple(sorted((b.competitor_a, b.competitor_b))) for b in battles}
        assert len(pairs) == 3

    def test_both_wrong_prioritised(self):
        samples = _samples(
            [_row("x", "media:play_song"), _row("y", "wrong")],   # one-wrong
            [_row("x", "wrong1"), _row("y", "wrong2")],           # both-wrong
        )
        battles = assemble_battles("intent", "d", "en-US", samples, max_battles=1)
        assert battles[0].sample_id == "en-US/00001"  # the both-wrong sample

    def test_intent_payload_shape(self):
        samples = _samples([
            _row("x", "media:play_song",
                 predicted_slots={"song": "africa"}),
            _row("y", None),
        ])
        battles = assemble_battles("intent", "d", "en-US", samples)
        payloads = {battles[0].competitor_a: battles[0].prediction_a,
                    battles[0].competitor_b: battles[0].prediction_b}
        assert payloads["x"] == {"intent": "media:play_song",
                                 "slots": {"song": "africa"}}
        assert payloads["y"] == {"intent": None}


class TestStimulus:
    def _one(self, modality, rows):
        samples = {"s0": {r.competitor_id: r.model_copy(update={"sample_id": "s0"})
                          for r in rows}}
        battles = assemble_battles(modality, "d", "x", samples)
        assert len(battles) == 1
        return battles[0]

    def test_stt_uses_source_audio_and_transcript(self):
        a = _row("x", "ola mundo", reference=None, reference_text="ola mundo",
                 utterance=None, audio_url="https://hf/a.wav")
        b = _row("y", "ola munto", reference=None, reference_text="ola mundo",
                 utterance=None, audio_url="https://hf/a.wav")
        battle = self._one("stt", [a, b])
        assert battle.input_text is None
        assert battle.audio_url == "https://hf/a.wav"
        assert battle.reference == "ola mundo"
        assert battle.prediction_a in ("ola mundo", "ola munto")

    def test_tts_shows_prompt_and_audio_payload(self):
        a = _row("x", "https://hf/a.wav", reference=None, utterance=None,
                 input_text="hello there")
        b = _row("y", "https://hf/b.wav", reference=None, utterance=None,
                 input_text="hello there")
        battle = self._one("tts", [a, b])
        assert battle.input_text == "hello there"
        assert battle.audio_url is None
        assert battle.prediction_a.endswith(".wav")

    def test_wake_word_uses_label_reference(self):
        a = _row("x", "detected", reference=None, utterance=None,
                 label="positive", audio_url="https://hf/w.wav")
        b = _row("y", "not_detected", reference=None, utterance=None,
                 label="positive", audio_url="https://hf/w.wav")
        battle = self._one("wake_word", [a, b])
        assert battle.audio_url == "https://hf/w.wav"
        assert battle.reference == "positive"
        assert battle.prediction_a in ("detected", "not_detected")


class TestSeedElo:
    def test_correct_competitor_seeds_higher(self):
        samples = _samples(*[
            [_row("good", "media:play_song"), _row("bad", "wrong")]
            for _ in range(10)
        ])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.ratings["good"] > INITIAL_ELO > seed.ratings["bad"]
        assert seed.auto_vote_count == 10
        assert seed.wins["good"] == 10
        assert seed.competitor_plugin["good"] == "plugin-good"

    def test_no_signal_pairs_not_counted(self):
        samples = _samples([_row("x", "media:play_song"),
                            _row("y", "media:play_song")])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        # competitors still listed (known to the board)
        assert set(seed.competitor_plugin) == {"x", "y"}

    def test_human_vote_only_lists_fighters_at_baseline(self):
        # tts has no auto-outcome; fighters must still appear at baseline ELO
        a = _row("voice_a", "https://hf/a.wav", reference=None, input_text="hi")
        b = _row("voice_b", "https://hf/b.wav", reference=None, input_text="hi")
        samples = {"s0": {"voice_a": a, "voice_b": b}}
        seed = seed_elo("tts", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        assert seed.ratings == {"voice_a": INITIAL_ELO, "voice_b": INITIAL_ELO}
        assert seed.battles == {"voice_a": 0, "voice_b": 0}

    def test_deterministic(self):
        samples = _samples(*[
            [_row("x", "media:play_song" if i % 2 else "w1"),
             _row("y", "w2" if i % 3 else "media:play_song")]
            for i in range(9)
        ])
        seeds = [seed_elo("intent", "en-US", {"d": samples}, "t") for _ in range(2)]
        assert seeds[0].ratings == seeds[1].ratings
