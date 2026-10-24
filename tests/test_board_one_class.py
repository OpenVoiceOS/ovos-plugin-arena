"""A detection board must not rank an entry scored on one class only.

With positives alone, error_rate is the false-reject rate, so a detector
that fires on every clip scores 0.0 and ranks first. With negatives alone,
a detector that never fires wins the same way.
"""
from arena.metrics import build_benchmark_board
from arena.models import PredictionRow


def _rows(cid, pos, neg, fire):
    rows = [("positive", i) for i in range(pos)] + [("negative", pos + i) for i in range(neg)]
    return [PredictionRow(competitor_id=cid, sample_id=f"s{i}", dataset_id="d", lang="en-US",
                          plugin_id="p", modality="wake_word", label=label,
                          prediction="detected" if fire else "not_detected")
            for label, i in rows]


def _board(by_competitor, modality="wake_word"):
    for rows in by_competitor.values():
        for r in rows:
            r.modality = modality
    return build_benchmark_board(modality, "d", "en-US", by_competitor, "t", min_samples=1)


def test_an_always_firing_detector_on_positives_only_is_not_ranked():
    board = _board({"always": _rows("always", 40, 0, fire=True),
                    "honest": _rows("honest", 40, 40, fire=False)})
    entry = {e.competitor_id: e for e in board.entries}
    assert entry["always"].unranked and entry["always"].rank == 0
    assert entry["always"].unranked_reason.startswith("one_class — no negatives")
    assert entry["honest"].rank == 1


def test_a_never_firing_detector_on_negatives_only_is_not_ranked():
    board = _board({"silent": _rows("silent", 0, 40, fire=False)})
    assert board.entries[0].unranked
    assert board.entries[0].unranked_reason.startswith("one_class — no positives")


def test_both_classes_rank_as_before():
    board = _board({"a": _rows("a", 40, 40, fire=True)})
    assert board.entries[0].rank == 1 and not board.entries[0].unranked


def test_vad_boards_get_the_same_guard():
    board = _board({"v": _rows("v", 40, 0, fire=True)}, modality="vad")
    assert board.entries[0].unranked


def test_an_entry_that_scored_no_label_keeps_its_own_reason():
    rows = [PredictionRow(competitor_id="x", sample_id=f"s{i}", dataset_id="d", lang="en-US",
                          plugin_id="p", modality="wake_word", label=None, prediction="detected",
                          latency_ms=5.0) for i in range(40)]
    board = build_benchmark_board("wake_word", "d", "en-US", {"x": rows}, "t", min_samples=1)
    entry = board.entries[0]
    assert entry.unranked and not entry.unranked_reason.startswith("one_class")


def test_the_board_logic_version_moves_so_cached_boards_are_rebuilt():
    from arena import metrics
    assert metrics.BOARD_LOGIC_VERSION >= 4
