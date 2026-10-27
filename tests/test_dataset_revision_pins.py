"""Wake-word and intent datasets name one immutable commit of every corpus they read."""
from __future__ import annotations

import json
import re

import pytest

from arena.metrics import is_pinned_revision
from registry.loaders import REGISTRY_ROOT, load_dataset
from runner import audio_io

PINNED_MODALITIES = ("wake_word", "intent", "intent_template", "intent_keyword")
DEFINITIONS = {
    path: json.loads(path.read_text())
    for modality in PINNED_MODALITIES
    for path in sorted((REGISTRY_ROOT / "datasets" / modality).glob("*.json"))
}
# ``org/name@<revision>`` with an optional ``/subdir``
NEGATIVES_ENTRY = re.compile(
    r"(?P<hf_id>[^/@]+/[^/@]+)(?:@(?P<revision>[^/]+))?(?:/(?P<subdir>.+))?")


def _negatives_entries():
    for path, definition in DEFINITIONS.items():
        for spec in definition.get("negatives_sources") or []:
            yield path, spec
        if definition.get("negatives_hf"):
            yield path, definition["negatives_hf"]


def test_every_source_revision_is_a_commit_sha():
    hf_sources = {path: d["source"] for path, d in DEFINITIONS.items()
                  if d["source"].get("type") == "huggingface"}
    unpinned = sorted(f"{path.parent.name}/{path.name}: {source.get('revision', 'main')}"
                      for path, source in hf_sources.items()
                      if not is_pinned_revision(source.get("revision", "main")))
    assert len(hf_sources) > 100
    assert unpinned == []


def test_every_negatives_source_is_pinned_to_a_commit_sha():
    entries = list(_negatives_entries())
    unpinned = sorted(f"{path.name}: {spec}" for path, spec in entries
                      if not is_pinned_revision(NEGATIVES_ENTRY.fullmatch(spec)["revision"]))
    assert len(entries) > 100
    assert unpinned == []


@pytest.mark.parametrize("spec, expected", [
    ("org/noise", ("org/noise", "main", None)),
    ("org/noise/sub/dir", ("org/noise", "main", "sub/dir")),
    ("org/noise@" + "a" * 40, ("org/noise", "a" * 40, None)),
    ("org/noise@" + "b" * 40 + "/sub", ("org/noise", "b" * 40, "sub")),
])
def test_a_negatives_entry_names_repo_revision_and_subdir(spec, expected):
    assert audio_io.negatives_source(spec) == expected


def _record_listings(monkeypatch):
    calls = []

    def all_audio(hf_id, revision, subdir=None):
        calls.append((hf_id, revision, subdir))
        return [f"{subdir or 'clips'}/{n}.wav" for n in range(3)]

    monkeypatch.setattr(audio_io, "_all_audio", all_audio)
    monkeypatch.setattr(audio_io, "_ww_split", lambda dataset_def, revision: ([], []))
    return calls


def test_negatives_sources_are_listed_and_fetched_at_their_pinned_revision(monkeypatch):
    calls = _record_listings(monkeypatch)
    path = REGISTRY_ROOT / "datasets" / "wake_word" / "community-amelia.json"
    expected = {(m["hf_id"], m["revision"], m["subdir"])
                for m in map(NEGATIVES_ENTRY.fullmatch, DEFINITIONS[path]["negatives_sources"])}

    _, negatives = audio_io.ww_clip_selection(
        load_dataset("wake_word", "community-amelia"), "unused")

    assert set(calls) == expected
    revision_of = {hf_id: revision for hf_id, revision, _ in expected}
    assert {(hf_id, rev) for hf_id, _, rev in negatives} == set(revision_of.items())


def test_negatives_hf_is_listed_and_fetched_at_its_pinned_revision(monkeypatch):
    calls = _record_listings(monkeypatch)
    sha = "c" * 40
    definition = load_dataset("wake_word", "community-amelia").model_copy(update={
        "negatives_sources": None, "negatives_hf": f"org/noise@{sha}", "negatives_dir": "neg"})

    _, negatives = audio_io.ww_clip_selection(definition, "unused")

    assert calls == [("org/noise", sha, "neg")]
    assert {(hf_id, rev) for hf_id, _, rev in negatives} == {("org/noise", sha)}
