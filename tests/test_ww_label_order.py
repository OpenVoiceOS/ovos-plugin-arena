"""A binary-detection stream must alternate its two classes.

The runner stops a sweep at a batch cap or a deadline. When the sampler
yielded every positive before the first negative, a capped sweep held only
positives, and a board ranked on false rejects alone: a detector that fires
on every clip scored a perfect error rate.
"""
from itertools import islice

import huggingface_hub
import pytest

from runner import audio_io


def _clips(prefix, n):
    return [("org/corpus", f"{prefix}/{i}.wav", "main") for i in range(n)]


def _labels(monkeypatch, pos, neg, cap=0):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: __file__)
    monkeypatch.setattr(audio_io, "decode_audio_bytes", lambda raw: ([0.0], 16000))
    return [s["label"] for _, s in audio_io._emit_ww(pos, neg, cap)]


def test_the_first_two_clips_hold_both_classes(monkeypatch):
    labels = _labels(monkeypatch, _clips("pos", 1000), _clips("neg", 5))
    assert labels[:2] == ["positive", "negative"]


def test_a_cut_at_any_even_length_is_balanced_while_both_classes_last(monkeypatch):
    labels = _labels(monkeypatch, _clips("pos", 6), _clips("neg", 6))
    for cut in range(2, 13, 2):
        assert labels[:cut].count("positive") == labels[:cut].count("negative") == cut // 2


def test_every_clip_is_still_yielded_once(monkeypatch):
    labels = _labels(monkeypatch, _clips("pos", 7), _clips("neg", 3))
    assert labels.count("positive") == 7 and labels.count("negative") == 3


def test_the_per_class_cap_still_applies(monkeypatch):
    labels = _labels(monkeypatch, _clips("pos", 50), _clips("neg", 50), cap=4)
    assert labels == ["positive", "negative"] * 4


class _Def:
    def __init__(self, **kw):
        self.__dict__.update(dict(reference_fields={}, positives_dir=None, negative_dirs=None,
                                  negatives_sources=None, negatives_hf=None, negatives_dir=None,
                                  negatives_dataset_ids=None,
                                  wakeword="ww"), **kw)


class _Src:
    hf_id, subset, file_pattern, split, revision = "org/corpus", None, None, None, "main"

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_clip(label):
    return lambda *a, **k: iter([(f"{label}{i}", {"array": [0.0], "sr": 16000, "label": label})
                                 for i in range(20)])


def test_pooled_dataset_negatives_reach_a_short_wake_word_sweep(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: __file__)
    monkeypatch.setattr(audio_io, "decode_audio_bytes", lambda raw: ([0.0], 16000))
    monkeypatch.setattr(audio_io, "_repo_audio", lambda hf, rev: [f"ww/{i}.wav" for i in range(20)])
    monkeypatch.setattr(audio_io, "_pooled_dataset_negatives",
                        lambda ids, cap, failures: _fake_clip("pooled")())
    ds = _Def(source=_Src(), negatives_dataset_ids=["mlsw"])
    head = [s["label"] for _, s in islice(audio_io.stream_ww(ds, "main"), 4)]
    assert head[0] == "positive" and "pooled" in head


def test_a_short_vad_sweep_over_parquet_speech_holds_both_classes(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: __file__)
    monkeypatch.setattr(audio_io, "decode_audio_bytes", lambda raw: ([0.0], 16000))
    monkeypatch.setattr(audio_io, "stream_audio_dataset", _fake_clip("speech"))
    monkeypatch.setattr(audio_io, "_pool_negatives", lambda sources, cap: _clips("noise", 20))
    ds = _Def(source=_Src(split="test"), negatives_sources=["org/noise"])
    head = [s["label"] for _, s in islice(audio_io.stream_vad(ds, "main", max_per_class=20), 2)]
    assert head == ["speech", "non_speech"]


def test_a_failing_pooled_source_yields_the_rest_then_raises(monkeypatch, caplog):
    import registry.loaders

    def broken_stream(*a, **k):
        raise OSError("no parquet files")
        yield

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: __file__)
    monkeypatch.setattr(audio_io, "decode_audio_bytes", lambda raw: ([0.0], 16000))
    monkeypatch.setattr(audio_io, "_repo_audio", lambda hf, rev: [f"ww/{i}.wav" for i in range(3)]
                        + [f"other/{i}.wav" for i in range(3)])
    monkeypatch.setattr(audio_io, "stream_audio_dataset", broken_stream)
    monkeypatch.setattr(registry.loaders, "load_dataset", lambda modality, did: _Def(source=_Src()))
    ds = _Def(source=_Src(), negatives_dataset_ids=["mlsw-negatives-en-US"])
    labels = []
    with pytest.raises(OSError, match="no parquet files"):
        labels.extend(s["label"] for _, s in audio_io.stream_ww(ds, "main"))
    assert labels.count("positive") == 3 and labels.count("negative") == 3
    assert "mlsw-negatives-en-US" in caplog.text
