"""The site lists only the wake-word boards that compare real models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import arena.cli as arena_cli
from arena.metrics import build_benchmark_board
from arena.models import PredictionRow
from arena.ww_listing import build_listing, is_trained_model, registry_featured_phrases
from registry.schemas import CompetitorDef, DatasetDef

WAKEHUBERT = "ovos-ww-plugin-wakeforge"


def _fighter(cid: str, phrase: str, plugin: str, types: list[str]) -> CompetitorDef:
    return CompetitorDef(
        competitor_id=cid, modality="wake_word", plugin=plugin, langs=["en-US"],
        config={"hotwords": {phrase: {"module": plugin}}}, types=types,
    )


FIGHTERS = [
    # hey_jarvis: two WakeHuBERT models and one microWakeWord model, three generic fighters
    _fighter("wf-jarvis-a", "hey_jarvis", WAKEHUBERT, ["neural-net"]),
    _fighter("wf-jarvis-b", "hey_jarvis", WAKEHUBERT, ["neural-net"]),
    _fighter("mww-jarvis", "hey_jarvis", "ovos-ww-plugin-microwakeword", ["neural-net"]),
    _fighter("vosk-jarvis", "hey_jarvis", "ovos-ww-plugin-vosk", ["neural-net", "keyword-match"]),
    _fighter("zs-jarvis", "hey_jarvis", "ovos-ww-plugin-wakeforge-zeroshot",
             ["neural-net", "zero-shot"]),
    _fighter("lw-jarvis", "hey jarvis", "ovos-ww-plugin-local-wake",
             ["template-matching", "few-shot"]),
    # hey_k9: only WakeHuBERT models
    _fighter("wf-k9-a", "hey_k9", WAKEHUBERT, ["neural-net"]),
    _fighter("wf-k9-b", "hey_k9", WAKEHUBERT, ["neural-net"]),
    # computer: one trained model and the generic fighters
    _fighter("oww-computer", "computer", "ovos-ww-plugin-openwakeword", ["neural-net"]),
    _fighter("ew-computer", "computer", "ovos-ww-plugin-efficientword-net",
             ["neural-net", "few-shot"]),
    # alexa: two trained models, one of them unranked on the board below
    _fighter("oww-alexa", "alexa", "ovos-ww-plugin-openwakeword", ["neural-net"]),
    _fighter("mww-alexa", "alexa", "ovos-ww-plugin-microwakeword", ["neural-net"]),
    # a phrase no board covers
    _fighter("oww-stop", "stop", "ovos-ww-plugin-openwakeword", ["neural-net"]),
]


def _board(dataset_id: str, positives: int | None, ranked: list[str], unranked=()) -> dict:
    entries = [{"competitor_id": c, "unranked": False} for c in ranked]
    entries += [{"competitor_id": c, "unranked": True} for c in unranked]
    return {"modality": "wake_word", "dataset_id": dataset_id, "lang": "en-US",
            "positives": positives, "entries": entries}


BOARDS = [
    # 50 positives, 2 trained models (one microWakeWord): listed, hey_jarvis featured
    _board("jarvis-50", 50, ["wf-jarvis-a", "mww-jarvis", "vosk-jarvis", "zs-jarvis"]),
    # one positive short of 50
    _board("jarvis-49", 49, ["wf-jarvis-a", "mww-jarvis", "vosk-jarvis"]),
    # 100 positives but one trained model beside generic fighters
    _board("jarvis-one-trained", 100, ["wf-jarvis-a", "vosk-jarvis", "zs-jarvis", "lw-jarvis"]),
    # 100 positives, two trained models, both WakeHuBERT: hey_k9 not featured, unlisted
    _board("k9-100", 100, ["wf-k9-a", "wf-k9-b"]),
    # 100 positives, one trained model ranked and one unranked
    _board("alexa-100", 100, ["oww-alexa"], unranked=["mww-alexa"]),
    # no positive count recorded
    _board("computer-legacy", None, ["oww-computer", "ew-computer"]),
]


@pytest.fixture
def listing():
    return build_listing(BOARDS, FIGHTERS)


def test_generic_fighters_are_told_apart_from_trained_models():
    assert {f.competitor_id for f in FIGHTERS if is_trained_model(f)} == {
        "wf-jarvis-a", "wf-jarvis-b", "mww-jarvis", "wf-k9-a", "wf-k9-b",
        "oww-computer", "oww-alexa", "mww-alexa", "oww-stop"}


def test_a_listed_board_needs_fifty_positives_two_trained_models_and_a_featured_phrase(listing):
    assert listing.clip_boards == {("jarvis-50", "en-US")}
    assert listing.unlisted_clip_boards == {
        ("jarvis-49", "en-US"), ("jarvis-one-trained", "en-US"), ("k9-100", "en-US"),
        ("alexa-100", "en-US"), ("computer-legacy", "en-US")}


def test_a_phrase_is_featured_when_a_listed_board_ranks_a_non_wakehubert_model(listing):
    assert listing.featured_phrases == {"hey_jarvis"}


def test_only_an_unlisted_clip_board_is_hidden(listing):
    hidden = {
        (modality, dataset_id): listing.is_hidden(modality, dataset_id, "en-US")
        for modality, dataset_id in [
            ("wake_word", "jarvis-50"), ("wake_word", "jarvis-49"),
            ("wake_word", "freeform"), ("stt", "jarvis-49"),
        ]
    }
    assert hidden == {
        ("wake_word", "jarvis-50"): False, ("wake_word", "jarvis-49"): True,
        ("wake_word", "freeform"): False, ("stt", "jarvis-49"): False,
    }


def _row(cid: str, sample: int, label: str, prediction: str) -> PredictionRow:
    return PredictionRow(
        competitor_id=cid, sample_id=f"s{sample}", dataset_id="d", lang="en-US",
        plugin_id="p", modality="wake_word", label=label, prediction=prediction)


def test_a_board_counts_the_positive_clips_of_its_test_set():
    # a: 3 positives + 2 negatives; b: 4 positives + 1 negative; the test set holds the larger
    a = [_row("a", i, "positive", "positive") for i in range(3)] + [
        _row("a", i, "negative", "negative") for i in range(3, 5)]
    b = [_row("b", i, "positive", "negative") for i in range(4)] + [
        _row("b", 4, "negative", "negative")]
    board = build_benchmark_board(
        "wake_word", "d", "en-US", {"a": a, "b": b}, "2026-01-01T00:00:00+00:00")
    assert board.positives == 4
    assert build_benchmark_board(
        "stt", "d", "en-US", {}, "2026-01-01T00:00:00+00:00").positives is None


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_export_index_leaves_hidden_boards_out_and_keeps_their_files(tmp_path, monkeypatch):
    import registry.loaders as loaders_mod

    monkeypatch.setattr(loaders_mod, "list_competitors", lambda modality=None: FIGHTERS)
    monkeypatch.setattr(loaders_mod, "list_datasets", lambda modality=None: [])
    data = tmp_path / "data"
    data.mkdir()
    for board in BOARDS:
        _write(data / f"benchmark-wake_word-{board['dataset_id']}-en-US.json", board)
    for clip in ("jarvis-50", "jarvis-49"):
        _write(data / f"battles-wake_word-{clip}-en-US.json", {
            "modality": "wake_word", "dataset_id": clip, "lang": "en-US", "battles": []})
    _write(data / "battles-wake_word-freeform-en-US.json", {
        "modality": "wake_word", "dataset_id": "freeform", "lang": "en-US", "battles": []})
    out = tmp_path / "index.json"
    with pytest.raises(SystemExit) as exit_info:
        arena_cli.main(["export-index", "--data-dir", str(data), "--output", str(out)])
    assert exit_info.value.code == 0
    index = json.loads(out.read_text())
    assert sorted((b["modality"], b["dataset_id"]) for b in index["benchmarks"]) == [
        ("wake_word", "jarvis-50")]
    assert sorted(p["dataset_id"] for p in index["battles_pools"]) == ["jarvis-50"]
    assert [p["dataset_id"] for p in index["freeform_pools"]] == ["freeform"]
    assert len(list(data.glob("benchmark-*.json"))) == len(BOARDS)


class _Stub:
    """Registry stand-in for the assemble path: ``group_rows`` reads these attributes."""

    def __init__(self, fighter: CompetitorDef) -> None:
        self.competitor_id = fighter.competitor_id
        self.plugin = fighter.plugin
        self.types = fighter.types
        self.config = fighter.config
        self.label_set = None
        self.trained_on: list[str] = []


def test_assemble_ranks_only_fighters_of_featured_phrases_on_a_negatives_board(
    tmp_path, monkeypatch,
):
    import registry.loaders as loaders_mod
    from registry.loaders import load_dataset

    dataset_id = "ww-negatives-en-US"
    revision = load_dataset("ww_stream", dataset_id).source.revision
    stubs = [_Stub(f) for f in FIGHTERS]
    monkeypatch.setattr(
        loaders_mod, "list_competitors",
        lambda modality=None: stubs if modality in (None, "wake_word") else [])
    monkeypatch.setattr(arena_cli, "_load_sample_set", lambda *args: None)
    data = tmp_path / "data"
    data.mkdir()
    for board in BOARDS:
        _write(data / f"benchmark-wake_word-{board['dataset_id']}-en-US.json", board)
    preds = tmp_path / "predictions"
    preds.mkdir()
    for fighter in ("wf-jarvis-a", "vosk-jarvis", "wf-k9-a", "oww-computer"):
        rows = [
            PredictionRow(
                competitor_id=fighter, sample_id=f"neg{h}", dataset_id=dataset_id,
                dataset_revision=revision, lang="en-US", plugin_id="p",
                modality="ww_stream", prediction="WW_STREAM",
                extras={"events": [], "truth_onsets": [], "duration_s": 3600.0})
            for h in range(5)
        ]
        (preds / f"{fighter}.jsonl").write_text(
            "\n".join(r.model_dump_json() for r in rows) + "\n")
    with pytest.raises(SystemExit) as exit_info:
        arena_cli.main(["assemble", "--predictions", str(preds), "--output", str(data)])
    assert exit_info.value.code == 0
    board = json.loads((data / f"benchmark-ww_stream-{dataset_id}-en-US.json").read_text())
    # hey_jarvis is the one featured phrase of BOARDS; hey_k9 and computer are hidden
    assert sorted(e["competitor_id"] for e in board["entries"]) == ["vosk-jarvis", "wf-jarvis-a"]


NEGATIVES = {"ww-negatives-en-US", "ww-negatives-de-DE"}


def test_a_negatives_board_is_hidden_when_its_language_has_no_listed_clip_board():
    listing = build_listing(BOARDS, FIGHTERS, NEGATIVES)
    hidden = {
        (dataset_id, lang): listing.is_hidden("ww_stream", dataset_id, lang)
        for dataset_id, lang in [
            ("ww-negatives-en-US", "en-US"),
            ("ww-negatives-de-DE", "de-DE"),
        ]
    }
    assert hidden == {("ww-negatives-en-US", "en-US"): False, ("ww-negatives-de-DE", "de-DE"): True}
    # with no listed clip board at all, every negatives board is hidden
    empty = build_listing([], FIGHTERS, NEGATIVES)
    assert empty.is_hidden("ww_stream", "ww-negatives-en-US", "en-US")


def test_a_stream_board_of_an_unregistered_set_is_hidden():
    # a phrase stream left on disk, even one whose clip board is listed
    listing = build_listing(BOARDS, FIGHTERS, NEGATIVES)
    assert ("jarvis-50", "en-US") in listing.clip_boards
    assert listing.is_hidden("ww_stream", "ww_stream_jarvis-50", "en-US")
    assert listing.is_hidden("ww_stream", "ww_stream_never-benched", "en-US")


def test_a_negatives_clip_pool_is_never_listed():
    # mlsw-negatives sets hold no positives: a pool of negatives, not a board
    pool = _board("mlsw-negatives-en-US", None, ["oww-alexa", "mww-alexa"])
    listing = build_listing([*BOARDS, pool], FIGHTERS, NEGATIVES)
    assert listing.is_hidden("wake_word", "mlsw-negatives-en-US", "en-US")
    assert listing.featured_phrases == {"hey_jarvis"}


def test_a_negatives_board_ranks_only_fighters_of_featured_phrases():
    listing = build_listing(BOARDS, FIGHTERS, NEGATIVES)
    assert {f.competitor_id for f in FIGHTERS if listing.ranks_on_false_wake_set(f)} == {
        "wf-jarvis-a", "wf-jarvis-b", "mww-jarvis", "vosk-jarvis", "zs-jarvis", "lw-jarvis"}


def test_export_index_lists_only_the_negatives_board_of_a_listed_language(
    tmp_path, monkeypatch,
):
    import registry.loaders as loaders_mod

    datasets = [
        DatasetDef(dataset_id=f"ww-negatives-{lang}", modality="ww_stream", lang=lang,
                   source={"type": "path", "path": "x"}, primary_metric="fa_per_hour")
        for lang in ("en-US", "de-DE")
    ]
    monkeypatch.setattr(loaders_mod, "list_competitors", lambda modality=None: FIGHTERS)
    monkeypatch.setattr(
        loaders_mod, "list_datasets",
        lambda modality=None: datasets if modality == "ww_stream" else [])
    data = tmp_path / "data"
    data.mkdir()
    for board in BOARDS:
        _write(data / f"benchmark-wake_word-{board['dataset_id']}-en-US.json", board)
    _write(data / "benchmark-wake_word-mlsw-negatives-en-US-en-US.json",
           _board("mlsw-negatives-en-US", None, ["oww-alexa", "mww-alexa"]))
    stream_boards = [("ww-negatives-en-US", "en-US"), ("ww-negatives-de-DE", "de-DE"),
                     ("ww_stream_jarvis-50", "en-US")]
    for dataset_id, lang in stream_boards:
        _write(data / f"benchmark-ww_stream-{dataset_id}-{lang}.json", {
            "modality": "ww_stream", "dataset_id": dataset_id, "lang": lang,
            "entries": [{"competitor_id": "wf-jarvis-a"}]})
    out = tmp_path / "index.json"
    with pytest.raises(SystemExit) as exit_info:
        arena_cli.main(["export-index", "--data-dir", str(data), "--output", str(out)])
    assert exit_info.value.code == 0
    index = json.loads(out.read_text())
    assert sorted((b["modality"], b["dataset_id"]) for b in index["benchmarks"]) == [
        ("wake_word", "jarvis-50"), ("ww_stream", "ww-negatives-en-US")]


def _set(dataset_id: str, phrase: str | None, lang: str = "en-US", cap: int = 100) -> DatasetDef:
    return DatasetDef(
        dataset_id=dataset_id, modality="wake_word", lang=lang, role="eval",
        source={"type": "path", "path": "x"}, wakeword=phrase,
        sample_policy={"max_samples": cap})


def _trained(cid: str, phrase: str, plugin: str, trained_on=(), langs=("en-US",)):
    return CompetitorDef(
        competitor_id=cid, modality="wake_word", plugin=plugin, langs=list(langs),
        config={"hotwords": {phrase: {"module": plugin}}}, types=["neural-net"],
        trained_on=list(trained_on))


def test_the_registry_predicts_the_featured_phrases_before_any_board_exists():
    oww, mww = "ovos-ww-plugin-openwakeword", "ovos-ww-plugin-microwakeword"
    sets = [
        _set("alexa-set", "alexa"),
        _set("k9-set", "hey k9"),
        _set("nabu-set", "okay_nabu"),
        _set("small-set", "computer", cap=30),
        _set("de-set", "hallo", lang="de-DE"),
        _set("negatives", None),
    ]
    fighters = [
        # alexa: two trained models, one not WakeHuBERT: featured
        _trained("oww-alexa", "alexa", oww),
        _trained("wf-alexa", "alexa", WAKEHUBERT),
        # hey_k9: two WakeHuBERT models only
        _trained("wf-k9-a", "hey_k9", WAKEHUBERT),
        _trained("wf-k9-b", "hey_k9", WAKEHUBERT),
        # okay_nabu: the second model is trained on the only set
        _trained("mww-nabu", "okay_nabu", mww),
        _trained("wf-nabu", "okay_nabu", WAKEHUBERT, trained_on=["nabu-set"]),
        # computer: two models, but the set caps at 30 clips
        _trained("oww-computer", "computer", oww),
        _trained("mww-computer", "computer", mww),
        # hallo: two models of another language than the set's
        _trained("oww-hallo", "hallo", oww, langs=["nl-NL"]),
        _trained("mww-hallo", "hallo", mww, langs=["nl-NL"]),
        *[f for f in FIGHTERS if not is_trained_model(f)],
    ]
    assert registry_featured_phrases(sets, fighters, {}) == {"alexa"}
    # a board's own positive count overrides the sample cap
    assert registry_featured_phrases(
        sets, fighters, {("small-set", "en-US"): 60, ("alexa-set", "en-US"): 49}) == {"computer"}


def test_the_registry_holds_generic_fighters_of_featured_phrases_only():
    from arena.ww_listing import board_positives, phrase_of
    from registry.loaders import list_competitors, list_datasets

    fighters = list_competitors("wake_word")
    featured = registry_featured_phrases(
        list_datasets("wake_word"), fighters,
        board_positives(Path(__file__).parent.parent / "frontend-static/public/data"))
    generic = [c for c in fighters if not is_trained_model(c)]
    assert featured and generic
    assert sorted(c.competitor_id for c in generic if phrase_of(c) not in featured) == []


def test_every_wake_word_fighter_has_a_test_set_of_its_phrase():
    from arena.ww_listing import phrase_of
    from registry.loaders import list_competitors, list_datasets, wakeword_key

    tested = {wakeword_key(d.wakeword) for d in list_datasets("wake_word")
              if d.wakeword and d.role == "eval"}
    assert sorted(c.competitor_id for c in list_competitors("wake_word")
                  if phrase_of(c) not in tested) == []
