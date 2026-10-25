"""A row's plugin_version must name the installed distribution and its version,
also when the plugin id is an entry point of a distribution with another name."""
import importlib.metadata
from types import SimpleNamespace

from runner.media_bench import _plugin_version


class _Dist:
    def __init__(self, name, version, entry_points):
        self.metadata = {"Name": name}
        self.version = version
        self.entry_points = [SimpleNamespace(name=n, group="opm.wake_word") for n in entry_points]


def _no_such_package(name):
    raise importlib.metadata.PackageNotFoundError(name)


def test_entry_point_of_another_distribution(monkeypatch):
    dists = [_Dist("ovos-vad-plugin-silero", "0.1.3a2", ["ovos-vad-plugin-silero"]),
             _Dist("ovos-ww-plugin-wakeforge", "0.18.0a1",
                   ["ovos-ww-plugin-wakeforge", "ovos-ww-plugin-wakeforge-zeroshot"])]
    monkeypatch.setattr(importlib.metadata, "version", _no_such_package)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: dists)
    competitor = SimpleNamespace(plugin="ovos-ww-plugin-wakeforge-zeroshot")
    assert _plugin_version(competitor) == "ovos-ww-plugin-wakeforge==0.18.0a1"


def test_plugin_that_is_its_own_distribution(monkeypatch):
    versions = {"ovos-ww-plugin-vosk": "0.1.11a3"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    competitor = SimpleNamespace(plugin="ovos-ww-plugin-vosk")
    assert _plugin_version(competitor) == "ovos-ww-plugin-vosk==0.1.11a3"


def test_plugin_that_is_not_installed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _no_such_package)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])
    competitor = SimpleNamespace(plugin="ovos-ww-plugin-missing")
    assert _plugin_version(competitor) == "ovos-ww-plugin-missing"


def test_stream_events_are_not_reused_across_versions_of_the_providing_distribution(
        monkeypatch, tmp_path):
    from runner import ww_bench

    monkeypatch.setattr(ww_bench, "STREAM_EVENT_CACHE", tmp_path)
    calls = []
    monkeypatch.setattr(ww_bench, "_detect_stream", lambda stack, array: calls.append(1) or [])
    monkeypatch.setattr(importlib.metadata, "version", _no_such_package)
    stack = SimpleNamespace(vad=None, verifier=None, config={})
    competitor = SimpleNamespace(competitor_id="wakeforge-zeroshot-hey-mycroft",
                                 plugin="ovos-ww-plugin-wakeforge-zeroshot", config={})
    ctx = SimpleNamespace(competitor=competitor)
    sample = {"audio_url": "https://example.invalid/rec.flac", "array": []}
    for version in ("0.18.0a1", "0.18.0a1", "0.19.0a1"):
        dists = [_Dist("ovos-ww-plugin-wakeforge", version, ["ovos-ww-plugin-wakeforge-zeroshot"])]
        monkeypatch.setattr(importlib.metadata, "distributions", lambda dists=dists: dists)
        ww_bench._cached_events(stack, sample, ctx)
    assert len(calls) == 2
