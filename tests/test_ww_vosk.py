"""The vosk wake-word fighters and the frame size the runner feeds them."""
from __future__ import annotations

import numpy as np
import pytest

from registry.loaders import list_competitors, load_competitor
from runner import ww_bench
from runner.ww_bench import SAMPLE_RATE, WWStack, _detect_stack, _detect_stream


class _CallCountingEngine:
    """Counts calls instead of audio, like ovos-ww-plugin-vosk.

    Every ``found_wake_word`` call stands for ``SEC_BETWEEN_WW_CHECKS`` seconds.
    Each ``time_between_checks`` seconds it takes the audio buffered since its
    last check, the way the plugin's recogniser does, and fires when that audio
    holds the whole loud burst. A burst cut by the window edge never fires.
    """

    SEC_BETWEEN_WW_CHECKS = 0.2
    time_between_checks = 1.0

    def __init__(self, burst_samples: int):
        self.burst_samples = burst_samples
        self.counter = 0.0
        self.pending = b""
        self.chunk_sizes: list[int] = []

    def reset(self):
        """Like the plugin's: nothing is reset."""

    def update(self, chunk: bytes):
        self.chunk_sizes.append(len(chunk))
        self.pending += chunk

    def found_wake_word(self) -> bool:
        self.counter += self.SEC_BETWEEN_WW_CHECKS
        if self.counter < self.time_between_checks:
            return False
        self.counter = 0.0
        window, self.pending = self.pending, b""
        samples = np.frombuffer(window, dtype="<i2")
        return int(np.count_nonzero(np.abs(samples) > 1000)) >= self.burst_samples


class _AudioCountingEngine(_CallCountingEngine):
    """Counts seconds of audio, like ovos-ww-plugin-vosk with an audio clock."""

    @property
    def SEC_BETWEEN_WW_CHECKS(self):  # noqa: N802 - hides the inherited constant
        raise AttributeError("SEC_BETWEEN_WW_CHECKS")

    def update(self, chunk: bytes):
        super().update(chunk)
        self.counter += len(chunk) / (SAMPLE_RATE * 2)

    def found_wake_word(self) -> bool:
        if self.counter < self.time_between_checks:
            return False
        self.counter = 0.0
        window, self.pending = self.pending, b""
        samples = np.frombuffer(window, dtype="<i2")
        return int(np.count_nonzero(np.abs(samples) > 1000)) >= self.burst_samples


class _RollingWindowEngine(_AudioCountingEngine):
    """Transcribes the last ``window_seconds`` of audio every ``time_between_checks``."""

    window_seconds = 2.0

    def update(self, chunk: bytes):
        super().update(chunk)
        self.pending = self.pending[-int(self.window_seconds * SAMPLE_RATE * 2):]

    def found_wake_word(self) -> bool:
        if self.counter < self.time_between_checks:
            return False
        self.counter = 0.0
        samples = np.frombuffer(self.pending, dtype="<i2")
        found = int(np.count_nonzero(np.abs(samples) > 1000)) >= self.burst_samples
        if found:
            self.pending = b""
        return found


CHUNK_BYTES = 2048 * 2  # OPM Microphone.chunk_size, as ovos-dinkum-listener reads it
WINDOW_S = 5 * 2048 / SAMPLE_RATE  # time_between_checks 1.0 / 0.2 s per call, 5 calls of one chunk
BURST_S = 0.5


def _burst_clip(seconds: float = BURST_S) -> tuple[np.ndarray, int]:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype("float32"), n


class _Competitor:
    plugin = "stub-call-counting"
    config = {"hotwords": {"hey_stub": {"module": "stub-call-counting"}}}


def _stub_stack(monkeypatch, burst_samples: int, built=None, base=_CallCountingEngine) -> WWStack:
    class Stub(base):
        def __init__(self, key_phrase, config):
            super().__init__(burst_samples)
            if built is not None:
                built.append(self)

    monkeypatch.setattr(ww_bench, "load_plugin_class", lambda loader, name: Stub)
    return ww_bench.WakeWordBench().load_engine(_Competitor(), "en-US")


def test_call_counting_engine_gets_the_listener_chunk():
    clip, n = _burst_clip()
    engine = _CallCountingEngine(burst_samples=int(n * 0.9))
    _detect_stack(WWStack(engine), clip)
    assert set(engine.chunk_sizes[:-1]) == {CHUNK_BYTES}


def test_other_engines_keep_their_frame_size():
    class Plain:
        def update(self, chunk):
            self.sizes.append(len(chunk))

        def found_wake_word(self):
            return False

        sizes: list = []

    engine = Plain()
    engine.sizes = []
    _detect_stack(WWStack(engine), np.zeros(SAMPLE_RATE, dtype="float32"))
    assert set(engine.sizes[:-1]) == {1280 * 2}


def test_check_phase_is_spread_over_one_check_window():
    engine = _CallCountingEngine(burst_samples=1)
    phases = [ww_bench._check_phase(engine, f"clip-{i}.wav") for i in range(400)]
    assert all(0.0 <= p < WINDOW_S for p in phases)
    assert ww_bench._check_phase(engine, "clip-0.wav") == phases[0]
    assert len({round(p, 3) for p in phases}) > 300
    assert np.mean(phases) == pytest.approx(WINDOW_S / 2, abs=0.04)


@pytest.mark.parametrize("base,chunks", [
    (_CallCountingEngine, 5),  # 1.0 s / 0.2 s per call
    (_AudioCountingEngine, 8),  # 1.0 s of 2048-sample chunks, rounded up
    (_RollingWindowEngine, 8),
])
def test_windowed_engines_get_chunk_fresh_engine_and_phase_over_their_window(
        monkeypatch, base, chunks):
    window = chunks * 2048 / SAMPLE_RATE
    stack = _stub_stack(monkeypatch, 1, base=base)
    assert stack.rebuild is not None
    assert ww_bench._frame_samples(stack.ww) == 2048
    phases = [ww_bench._check_phase(stack.ww, f"clip-{i}.wav") for i in range(400)]
    assert all(0.0 <= p < window for p in phases)
    assert np.mean(phases) == pytest.approx(window / 2, abs=0.04)
    assert max(phases) > window * 0.9
    probe = base(1)
    _detect_stack(WWStack(probe), np.zeros(SAMPLE_RATE, dtype="float32"))
    assert set(probe.chunk_sizes[:-1]) == {CHUNK_BYTES}


@pytest.mark.parametrize("base", [_AudioCountingEngine, _RollingWindowEngine])
def test_audio_counting_engine_hears_clips_at_every_phase_of_its_window(monkeypatch, base):
    clip, n = _burst_clip()
    stack = _stub_stack(monkeypatch, int(n * 0.9), base=base)
    hits = [_detect_stack(stack, clip, f"clip-{i}.wav") for i in range(60)]
    # a 0.5 s burst fits one 1.024 s window for some phases only; a rolling
    # window of 2 s keeps it until the next check, so most phases hear it
    if base is _RollingWindowEngine:
        assert sum(hits) > len(hits) * 0.5
    else:
        assert 0 < sum(hits) < len(hits)


def test_clips_without_a_url_get_a_phase_key_of_their_own():
    one = np.sin(np.arange(8000) / 7.0).astype("float32")
    two = np.sin(np.arange(8000) / 11.0).astype("float32")
    assert ww_bench._phase_key({"array": one}) == ww_bench._phase_key({"array": one.copy()})
    assert ww_bench._phase_key({"array": one}) != ww_bench._phase_key({"array": two})
    assert ww_bench._phase_key({"array": one, "audio_url": None}) != ""
    assert ww_bench._phase_key({"array": one, "audio_url": "https://x/y.wav"}) == "https://x/y.wav"


def test_other_engines_have_no_check_phase():
    assert ww_bench._check_phase(object(), "clip.wav") == 0.0


def test_detection_follows_where_the_clip_falls_in_the_check_window(monkeypatch):
    clip, n = _burst_clip()
    stack = _stub_stack(monkeypatch, int(n * 0.9))
    outcomes = []
    for i in range(60):
        key = f"clip-{i}.wav"
        start = 3.0 + ww_bench._check_phase(stack.ww, key)
        edge = (int(start // WINDOW_S) + 1) * WINDOW_S
        inside = min(BURST_S, edge - start)  # burst seconds in the window it starts in
        overlap = max(inside, BURST_S - inside)
        if 0.42 < overlap < 0.48:
            continue  # too close to the stub's threshold to call
        assert _detect_stack(stack, clip, key) is (overlap >= 0.48), key
        outcomes.append(overlap >= 0.48)
    # a burst is cut by a window edge for some phases and not for others
    assert 0 < sum(outcomes) < len(outcomes)


def test_call_counting_engine_stream_timestamps_follow_its_chunk(monkeypatch):
    clip, n = _burst_clip()
    stack = _stub_stack(monkeypatch, int(n * 0.9))
    lead = np.zeros(int(WINDOW_S * SAMPLE_RATE), dtype="float32")
    tail = np.zeros(SAMPLE_RATE // 2, dtype="float32")
    events = _detect_stream(stack, np.concatenate([lead, clip, tail]))
    # the burst fills the second window, which closes on the tenth chunk
    assert [t for t, _ in events] == [pytest.approx(9 * 2048 / SAMPLE_RATE)]


def test_each_stream_recording_starts_on_a_fresh_engine(monkeypatch):
    clip, n = _burst_clip()
    first = np.concatenate([np.zeros(int(0.3 * SAMPLE_RATE), dtype="float32"), clip])
    second = np.concatenate([clip, np.zeros(3 * SAMPLE_RATE, dtype="float32")])
    alone = _detect_stream(_stub_stack(monkeypatch, int(n * 0.9)), second)
    after_first = _stub_stack(monkeypatch, int(n * 0.9))
    _detect_stream(after_first, first)
    assert _detect_stream(after_first, second) == alone
    assert alone  # the comparison is not between two empty lists


def test_call_counting_engine_starts_each_clip_with_a_fresh_check_clock(monkeypatch):
    clip, n = _burst_clip()
    stack = _stub_stack(monkeypatch, int(n * 0.9))
    off_phase = np.zeros(int(0.7 * SAMPLE_RATE), dtype="float32")
    key = next(k for k in (f"c{i}" for i in range(50))
               if (3 + ww_bench._check_phase(stack.ww, k)) % WINDOW_S < 0.05)
    assert _detect_stack(stack, clip, key) is True
    assert _detect_stack(stack, off_phase, "other") is False
    assert _detect_stack(stack, clip, key) is True


def test_the_engine_is_built_before_the_timed_call_not_inside_it(monkeypatch):
    clip, n = _burst_clip()
    built: list = []
    stack = _stub_stack(monkeypatch, int(n * 0.9), built)
    bench = ww_bench.WakeWordBench()
    bench.prepare(stack, {"array": clip})
    assert len(built) == 2  # the loaded engine and the spare
    monkeypatch.setattr(stack, "rebuild", lambda: pytest.fail("built inside predict"))
    ctx = type("Ctx", (), {"competitor": _Competitor()})()
    row = bench.predict(stack, {"array": clip, "label": "positive", "audio_url": "u"}, ctx)
    assert row["prediction"] in {"detected", "not_detected"}
    assert stack.spare is None


def test_run_competitor_lang_prepares_outside_the_timed_call(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from runner import media_bench

    events: list[str] = []

    class Adapter(media_bench.MediaBenchAdapter):
        modality = "wake_word"

        def iter_samples(self, dataset_def, lang, revision, max_samples):
            yield "s1", {"array": None}

        def load_engine(self, competitor, lang):
            return object()

        def prepare(self, engine, sample):
            events.append("prepare")

        def predict(self, engine, sample, ctx):
            events.append("predict")
            return {"prediction": "detected", "label": "positive"}

    def timed(fn):
        events.append("timer start")
        return fn(), 1.0, None

    monkeypatch.setattr(media_bench, "measure_call", timed)
    competitor = SimpleNamespace(competitor_id="c", plugin="p", model_revision=None,
                                 training_regime=None, modality="wake_word")
    eval_def = SimpleNamespace(source=SimpleNamespace(revision="r"))
    media_bench.run_competitor_lang(Adapter(), competitor, "d", "en-US", eval_def, "r",
                                    tmp_path / "o.jsonl", tmp_path, "repo")
    assert events == ["prepare", "timer start", "predict"]


def _vosk_fighters():
    return [c for c in list_competitors("wake_word") if c.plugin == "ovos-ww-plugin-vosk"]


def test_hey_mycroft_uses_the_shipped_ovos_default():
    hotword = load_competitor("wake_word", "vosk-ww-hey-mycroft").config["hotwords"]["hey_mycroft"]
    assert hotword["samples"] == ["hey mycroft", "hey microsoft", "hey mike roft", "hey minecraft"]
    assert hotword["rule"] == "fuzzy"


def test_every_vosk_fighter_names_where_its_samples_come_from():
    fighters = _vosk_fighters()
    assert len(fighters) >= 70
    for fighter in fighters:
        hotword = next(iter(fighter.config["hotwords"].values()))
        assert hotword["samples"], fighter.competitor_id
        assert str(hotword["samples"]) in fighter.notes, fighter.competitor_id
        assert hotword["rule"] in {"equals", "fuzzy"}, fighter.competitor_id
        assert hotword["rule"] in fighter.notes, fighter.competitor_id


def test_the_check_after_the_phrase_runs_even_when_it_falls_past_the_clip(monkeypatch):
    clip, n = _burst_clip()
    stack = _stub_stack(monkeypatch, int(n * 0.9), base=_RollingWindowEngine)
    window = 8 * 2048 / SAMPLE_RATE

    def last_check_before_end(key):
        end = 3.0 + ww_bench._check_phase(stack.ww, key) + BURST_S
        return end % window

    # the burst ends 0.08 to 0.2 s after a check, so the next check comes
    # nearly a whole window later: more than the probe's 0.5 s tail
    keys = [k for k in (f"clip-{i}.wav" for i in range(400))
            if 0.08 < last_check_before_end(k) < 0.2]
    assert len(keys) >= 5
    assert all(_detect_stack(stack, clip, k) for k in keys)


def test_silence_stays_undetected_with_the_longer_tail(monkeypatch):
    stack = _stub_stack(monkeypatch, 1, base=_RollingWindowEngine)
    silence = np.zeros(3 * SAMPLE_RATE, dtype="float32")
    assert not any(_detect_stack(stack, silence, f"clip-{i}.wav") for i in range(40))


def test_predict_gives_clips_without_a_url_different_phase_keys(monkeypatch):
    keys = []
    monkeypatch.setattr(ww_bench, "_detect_stack",
                        lambda stack, array, key="": keys.append(key) or False)
    ctx = type("Ctx", (), {"competitor": _Competitor()})()
    for freq in (7.0, 11.0):
        clip = np.sin(np.arange(8000) / freq).astype("float32")
        ww_bench.WakeWordBench().predict(None, {"array": clip, "label": "negative"}, ctx)
    assert len(set(keys)) == 2
    assert "" not in keys
