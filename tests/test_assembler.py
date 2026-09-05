"""Unit tests for arena.assembler — battles and ELO seeding."""
from __future__ import annotations

import pytest

from arena.assembler import (
    assemble_battles,
    auto_outcome,
    freeform_battles,
    seed_elo,
)
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

    def test_tts_higher_utmos_wins(self):
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi",
                 extras={"utmos": 4.2})
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 2.1})
        assert auto_outcome(a, b, "tts") == VoteOutcome.CANDIDATE_A
        assert auto_outcome(b, a, "tts") == VoteOutcome.CANDIDATE_B

    def test_tts_equal_utmos_no_signal(self):
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi",
                 extras={"utmos": 3.5})
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 3.5})
        assert auto_outcome(a, b, "tts") is None

    def test_tts_missing_utmos_no_signal(self):
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi")
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 3.5})
        assert auto_outcome(a, b, "tts") is None

    def test_tts_composite_score_beats_raw_utmos(self):
        # A carries higher UTMOS but is far less intelligible (high CER) —
        # tts_seed_score = (utmos/5) * (1-cer) must let B win despite B's
        # lower UTMOS: 4.5/5*(1-0.8)=0.18 (A) vs 3.0/5*(1-0.05)=0.57 (B).
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi",
                 extras={"utmos": 4.5, "intelligibility_cer": 0.8})
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 3.0, "intelligibility_cer": 0.05})
        assert auto_outcome(a, b, "tts") == VoteOutcome.CANDIDATE_B
        assert auto_outcome(b, a, "tts") == VoteOutcome.CANDIDATE_A

    def test_tts_low_agreement_discounts_composite_score(self):
        # Same UTMOS/CER on both sides, but A's judges disagreed heavily on
        # what was said (low intelligibility_agreement) — A must lose even
        # though its raw CER matches B's.
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi",
                 extras={"utmos": 4.0, "intelligibility_cer": 0.1,
                         "intelligibility_agreement": 0.3})
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 4.0, "intelligibility_cer": 0.1,
                         "intelligibility_agreement": 1.0})
        assert auto_outcome(a, b, "tts") == VoteOutcome.CANDIDATE_B

    def test_tts_mixed_cer_signal_no_vote(self):
        # A has an intelligibility CER, B is a legacy row without one —
        # mixing a composite score against a raw UTMOS is not a fair
        # comparison, so there is no auto-vote at all.
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi",
                 extras={"utmos": 3.0, "intelligibility_cer": 0.1})
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 4.5})
        assert auto_outcome(a, b, "tts") is None
        assert auto_outcome(b, a, "tts") is None

    def test_tts_legacy_rows_without_cer_fall_back_to_utmos(self):
        # Both rows predate intelligibility judging (no CER at all) — the
        # comparison must still fall back to plain UTMOS, same as before
        # ROVER seeding existed.
        a = _row("x", "https://hf/a.wav", reference=None, input_text="hi",
                 extras={"utmos": 4.2})
        b = _row("y", "https://hf/b.wav", reference=None, input_text="hi",
                 extras={"utmos": 2.1})
        assert auto_outcome(a, b, "tts") == VoteOutcome.CANDIDATE_A


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

    def test_reference_mismatch_pair_skipped_and_counted(self):
        # Legacy colliding sample_id: same sample_id, but the two rows were
        # actually scored against different underlying stimuli (different
        # reference_intent) — a pre-#70 shard collision. Pairing them would
        # silently produce a nonsense battle.
        samples = _samples([
            _row("x", "media:play_song", reference="media:play_song"),
            _row("y", "media:stop", reference="media:different_stimulus"),
        ])
        stats: dict[str, int] = {}
        battles = assemble_battles("intent", "d", "en-US", samples, stats=stats)
        assert battles == []
        assert stats["skipped_reference_mismatches"] == 1

    def test_reference_match_pair_still_battled(self):
        samples = _samples([
            _row("x", "media:play_song", reference="media:play_song"),
            _row("y", "media:stop", reference="media:play_song"),
        ])
        stats: dict[str, int] = {}
        battles = assemble_battles("intent", "d", "en-US", samples, stats=stats)
        assert len(battles) == 1
        assert stats["skipped_reference_mismatches"] == 0

    def test_reference_mismatch_whitespace_normalized_not_flagged(self):
        samples = _samples([
            _row("x", "hello", reference_text="hello  world",
                 reference=None, sample_id="en-US/00000"),
            _row("y", "goodbye", reference_text="hello world",
                 reference=None, sample_id="en-US/00000"),
        ])
        stats: dict[str, int] = {}
        battles = assemble_battles("stt", "d", "en-US", samples, stats=stats)
        assert len(battles) == 1
        assert stats["skipped_reference_mismatches"] == 0

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


class TestFreeformBattles:
    def test_all_pairs_no_self(self):
        cp = {"a": "pa", "b": "pb", "c": "pc"}
        battles = freeform_battles("intent", "en-US", cp)
        pairs = {tuple(sorted((b.competitor_a, b.competitor_b))) for b in battles}
        assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}
        assert all(b.competitor_a != b.competitor_b for b in battles)
        assert all(b.dataset_id == "freeform" and b.sample_id == "freeform"
                   for b in battles)
        assert all(b.modality.value == "intent" for b in battles)

    def test_carries_plugin_ids(self):
        b = freeform_battles("stt", "en-US", {"x": "plug-x", "y": "plug-y"})[0]
        plugs = {b.plugin_a, b.plugin_b}
        assert plugs == {"plug-x", "plug-y"}

    def test_ids_stable_and_match_vote_format(self):
        from arena.models import battle_id_for
        cp = {"adapt": "p1", "padatious": "p2"}
        battles = freeform_battles("intent", "pt-PT", cp)
        assert len(battles) == 1
        expected = battle_id_for("intent", "freeform", "pt-PT", "freeform",
                                 "adapt", "padatious")
        assert battles[0].battle_id == expected

    def test_single_competitor_no_battles(self):
        assert freeform_battles("tts", "en-US", {"solo": "p"}) == []


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
        # Bradley-Terry sufficient statistics (arena/rating.py) are captured
        # too, at the reduced auto-vote weight — not just the legacy ELO
        # ledger's ratings/wins/losses.
        assert seed.pairwise_wins["good"]["bad"] == pytest.approx(2.5)  # 10 * 0.25
        assert seed.pairwise_games["good"]["bad"] == pytest.approx(2.5)
        assert seed.pairwise_games["bad"]["good"] == pytest.approx(2.5)

    def test_no_signal_pairs_not_counted(self):
        samples = _samples([_row("x", "media:play_song"),
                            _row("y", "media:play_song")])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        # competitors still listed (known to the board)
        assert set(seed.competitor_plugin) == {"x", "y"}

    def test_tts_unscored_rows_list_fighters_at_baseline(self):
        # rows with no utmos score at all give no auto-signal; fighters must
        # still appear on the board at baseline ELO.
        a = _row("voice_a", "https://hf/a.wav", reference=None, input_text="hi")
        b = _row("voice_b", "https://hf/b.wav", reference=None, input_text="hi")
        samples = {"s0": {"voice_a": a, "voice_b": b}}
        seed = seed_elo("tts", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        assert seed.ratings == {"voice_a": INITIAL_ELO, "voice_b": INITIAL_ELO}
        assert seed.battles == {"voice_a": 0, "voice_b": 0}

    def test_tts_significant_utmos_gap_seeds_higher(self):
        samples = _samples(*[
            [_row("good", "https://hf/g.wav", reference=None, input_text="hi",
                  extras={"utmos": 4.5}),
             _row("bad", "https://hf/b.wav", reference=None, input_text="hi",
                  extras={"utmos": 2.0})]
            for _ in range(10)
        ])
        seed = seed_elo("tts", "en-US", {"d": samples}, "t")
        assert seed.ratings["good"] > INITIAL_ELO > seed.ratings["bad"]
        assert seed.auto_vote_count == 10

    def test_tts_overlapping_utmos_cis_no_seed(self):
        # near-identical UTMOS scores across the pair — CIs overlap, no
        # significant gap to seed a rating from.
        samples = _samples(*[
            [_row("x", "https://hf/x.wav", reference=None, input_text="hi",
                  extras={"utmos": 3.50 + (0.01 if i % 2 else -0.01)}),
             _row("y", "https://hf/y.wav", reference=None, input_text="hi",
                  extras={"utmos": 3.50 + (-0.01 if i % 2 else 0.01)})]
            for i in range(20)
        ])
        seed = seed_elo("tts", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        assert seed.ratings == {"x": INITIAL_ELO, "y": INITIAL_ELO}

    def test_tts_auto_weight_capped_per_pair(self):
        samples = _samples(*[
            [_row("good", "https://hf/g.wav", reference=None, input_text="hi",
                  extras={"utmos": 4.5}),
             _row("bad", "https://hf/b.wav", reference=None, input_text="hi",
                  extras={"utmos": 2.0})]
            for _ in range(200)
        ])
        seed = seed_elo("tts", "en-US", {"d": samples}, "t")
        assert seed.pairwise_games["good"]["bad"] == pytest.approx(5.0)
        assert seed.pairwise_games["bad"]["good"] == pytest.approx(5.0)

    def test_deterministic(self):
        samples = _samples(*[
            [_row("x", "media:play_song" if i % 2 else "w1"),
             _row("y", "w2" if i % 3 else "media:play_song")]
            for i in range(9)
        ])
        seeds = [seed_elo("intent", "en-US", {"d": samples}, "t") for _ in range(2)]
        assert seeds[0].ratings == seeds[1].ratings


class TestSeedEloInDistribution:
    """The seeded ladder is the board's primary-metric ranking, so it has to
    battle over the same rows ``generalization_accuracy`` is computed from —
    otherwise a rating labelled "generalization" is still in-distribution
    phrasing."""

    @staticmethod
    def _mem_vs_gen():
        """20 samples: the memorizer wins every in-distribution row, the
        generalizer wins every out-of-distribution one."""
        samples = {}
        for i in range(10):
            samples[f"template-{i}"] = {
                "mem": _row("mem", "media:play_song", bucket="template"),
                "gen": _row("gen", "wrong", bucket="template"),
            }
            samples[f"paraphrase-{i}"] = {
                "mem": _row("mem", "wrong", bucket="paraphrase"),
                "gen": _row("gen", "media:play_song", bucket="paraphrase"),
            }
        for sample_id, rows in samples.items():
            samples[sample_id] = {
                c: r.model_copy(update={"sample_id": sample_id})
                for c, r in rows.items()
            }
        return samples

    def test_template_rows_seed_no_battles(self):
        seed = seed_elo("intent", "en-US", {"d": self._mem_vs_gen()}, "t")
        assert seed.auto_vote_count == 10
        assert seed.wins["gen"] == 10
        assert seed.wins["mem"] == 0
        assert seed.ratings["gen"] > INITIAL_ELO > seed.ratings["mem"]

    def test_in_distribution_rows_seed_no_battles(self):
        samples = _samples(*[
            [_row("mem", "media:play_song", bucket="in_distribution"),
             _row("gen", "wrong", bucket="in_distribution")]
            for _ in range(10)
        ])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        assert seed.ratings == {"mem": INITIAL_ELO, "gen": INITIAL_ELO}

    def test_in_distribution_gap_alone_does_not_open_the_significance_gate(self):
        # Out-of-distribution rows: the pair trades wins evenly, aggregate
        # CIs overlap, no real signal. In-distribution rows: a big one-sided
        # gap. Gating on overall accuracy would call the pair significant and
        # seed the out-of-distribution coin-flips as if they meant something.
        samples = {}
        for i in range(20):
            samples[f"paraphrase-{i}"] = {
                "x": _row("x", "media:play_song" if i % 2 == 0 else "wrong",
                          bucket="paraphrase"),
                "y": _row("y", "media:play_song" if i % 2 else "wrong",
                          bucket="paraphrase"),
            }
            samples[f"template-{i}"] = {
                "x": _row("x", "media:play_song", bucket="template"),
                "y": _row("y", "wrong", bucket="template"),
            }
        for sample_id, rows in samples.items():
            samples[sample_id] = {
                c: r.model_copy(update={"sample_id": sample_id})
                for c, r in rows.items()
            }
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        assert seed.pairwise_games == {}

    def test_bucket_filter_is_intent_only(self):
        # A wake-word or STT row carrying an intent-corpus bucket label is
        # still a perfectly good battle — nothing was memorized there.
        ww = _samples(*[
            [_row("good", "detected", reference=None, label="positive",
                  bucket="template"),
             _row("bad", "not_detected", reference=None, label="positive",
                  bucket="template")]
            for _ in range(10)
        ])
        seed = seed_elo("wake_word", "en-US", {"d": ww}, "t")
        assert seed.auto_vote_count == 10
        assert seed.ratings["good"] > INITIAL_ELO > seed.ratings["bad"]

        stt = _samples(*[
            [_row("good", "play a song", wer=0.0, bucket="template"),
             _row("bad", "clay a wrong", wer=0.5, bucket="template")]
            for _ in range(10)
        ])
        seed = seed_elo("stt", "en-US", {"d": stt}, "t")
        assert seed.auto_vote_count == 10


class TestSeedEloBiasAudit:
    """§4 seed-battle bias audit (A1.3): significance gate + weight cap."""

    def test_no_signal_pair_contributes_no_auto_battles(self):
        # x and y each get it right ~50% of the time, on the *same* samples
        # in a way that keeps their aggregate accuracy statistically
        # indistinguishable — individual per-sample disagreements exist,
        # but the pair as a whole has no real signal.
        samples = _samples(*[
            [_row("x", "media:play_song" if i % 2 == 0 else "wrong"),
             _row("y", "media:play_song" if i % 2 == 1 else "wrong")]
            for i in range(20)
        ])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 0
        assert seed.pairwise_games == {}
        # still listed on the board at baseline
        assert seed.ratings == {"x": INITIAL_ELO, "y": INITIAL_ELO}

    def test_clear_signal_pair_still_seeds(self):
        # x is correct on every sample, y never is — an unambiguous,
        # statistically significant gap that must still seed the rating.
        samples = _samples(*[
            [_row("x", "media:play_song"), _row("y", "wrong")]
            for _ in range(20)
        ])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 20
        assert seed.ratings["x"] > INITIAL_ELO > seed.ratings["y"]

    def test_auto_weight_capped_per_pair(self):
        # A large dataset (200 samples, clear signal) must not accumulate
        # unbounded Bradley-Terry weight for one pair.
        samples = _samples(*[
            [_row("x", "media:play_song"), _row("y", "wrong")]
            for _ in range(200)
        ])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        assert seed.auto_vote_count == 200  # legacy sequential-ELO count is uncapped
        assert seed.pairwise_games["x"]["y"] == pytest.approx(5.0)  # BT weight is capped
        assert seed.pairwise_games["y"]["x"] == pytest.approx(5.0)
        # win rate is preserved by the cap (x won every capped "game")
        assert seed.pairwise_wins["x"]["y"] == pytest.approx(5.0)
        assert seed.pairwise_wins["y"]["x"] == pytest.approx(0.0)

    def test_small_signal_pair_under_cap_is_unaffected(self):
        samples = _samples(*[
            [_row("x", "media:play_song"), _row("y", "wrong")]
            for _ in range(4)
        ])
        seed = seed_elo("intent", "en-US", {"d": samples}, "t")
        # 4 auto votes * BT_AUTO_WEIGHT (0.25) = 1.0, well under the 5.0 cap
        assert seed.pairwise_games["x"]["y"] == pytest.approx(1.0)
