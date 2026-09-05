"""Rows are scored against the dataset revision the registry pins.

A corpus can change its row population, its bucket names and its train/test
split from one revision to the next. Scoring a shard swept on an older
revision under the current rules produces a number that describes neither
revision — on ``intents-for-eval`` it moved padacioso's
``generalization_accuracy`` from 0.0953 to 0.364 on identical input, because
the old shard's ``near_ood`` rows entered a metric the new bucket layout
excludes.
"""
from __future__ import annotations

import json

from arena.metrics import (
    build_benchmark_board,
    drop_rows_off_pinned_revision,
    is_pinned_revision,
    score_intent,
)
from arena.models import PredictionRow
from runner.intent_bench import stratified_sample

PIN = "d6f497f21c844c400d74bfda13773ccbaf902645"
OLD = "63b9ab14b8b1c2e4f6a70d5c3e9182f4b5c6d7e8"

#: Bucket layout and file order of one ``intents-for-eval`` language on the
#: pinned revision (en-US, measured on the corpus itself).
EN_US_BUCKETS = [
    ("template", 500),
    ("in_distribution", 45),
    ("paraphrase", 696),
    ("far_ood", 49),
    ("asr_noise", 42),
    ("typos", 49),
]


def _row(index: int, bucket: str, correct: bool, *, revision: str,
         competitor: str = "padacioso-medium") -> PredictionRow:
    """One real-shape intent prediction row."""
    return PredictionRow(
        competitor_id=competitor,
        sample_id=f"en-US/{index:05d}",
        dataset_id="intents-for-eval",
        lang="en-US",
        modality="intent",
        plugin_id="ovos-padacioso-pipeline-plugin",
        plugin_version="1.0.0",
        dataset_revision=revision,
        utterance=f"utterance number {index}",
        reference_intent="media:play_song",
        prediction="media:play_song" if correct else "wrong",
        exact_match=correct,
        bucket=bucket,
    )


def _shard(revision: str, *, buckets=EN_US_BUCKETS, correct_in=frozenset(),
           competitor: str = "padacioso-medium") -> list[PredictionRow]:
    """A whole language's rows, laid out bucket by bucket like the corpus."""
    rows = []
    index = 0
    for bucket, count in buckets:
        for _ in range(count):
            rows.append(_row(index, bucket, bucket in correct_in,
                             revision=revision, competitor=competitor))
            index += 1
    return rows


class TestPinnedRevisionDetection:
    def test_a_commit_sha_is_a_pin(self):
        assert is_pinned_revision(PIN)

    def test_a_branch_name_is_not_a_pin(self):
        assert not is_pinned_revision("main")
        assert not is_pinned_revision("refs/pr/3")
        assert not is_pinned_revision(None)
        assert not is_pinned_revision("")


class TestRowsFromAnotherRevisionAreDropped:
    def test_whole_stale_shard_leaves_the_competitor_unscored(self):
        """A fighter swept only on the old revision has not been measured
        against the pinned corpus at all."""
        board = build_benchmark_board(
            "intent", "intents-for-eval", "en-US",
            {"padacioso-medium": _shard(OLD)}, "t", dataset_revision=PIN,
        )
        entry = board.entries[0]
        assert entry.samples == 0
        assert entry.rows_other_revision == 1381
        assert entry.unranked is True
        assert entry.rank == 0
        assert "no rows on the pinned dataset revision" in entry.unranked_reason
        assert "generalization_accuracy" not in entry.metrics
        # The board still names the plugin the stale rows were swept with,
        # so an operator can see WHICH fighter needs the re-sweep.
        assert entry.plugin_id == "ovos-padacioso-pipeline-plugin"
        assert board.dataset_revision == PIN

    def test_a_stale_fighter_never_takes_rank_one_from_a_current_one(self):
        by_competitor = {
            "stale": _shard(OLD, correct_in={b for b, _ in EN_US_BUCKETS},
                            competitor="stale"),
            "current": _shard(PIN, correct_in={"paraphrase"},
                              competitor="current"),
        }
        board = build_benchmark_board(
            "intent", "intents-for-eval", "en-US", by_competitor, "t",
            dataset_revision=PIN,
        )
        ranked = [e for e in board.entries if not e.unranked]
        assert [e.competitor_id for e in ranked] == ["current"]
        assert ranked[0].rank == 1

    def test_mixed_revision_competitor_scores_only_the_pinned_rows(self):
        """Half the shard swept before the re-pin, half after: only the
        pinned half may reach the metric."""
        pinned_rows = _shard(PIN, correct_in={"paraphrase", "typos"})
        stale_rows = _shard(OLD, correct_in={b for b, _ in EN_US_BUCKETS})
        board = build_benchmark_board(
            "intent", "intents-for-eval", "en-US",
            {"padacioso-medium": pinned_rows + stale_rows}, "t",
            dataset_revision=PIN,
        )
        entry = board.entries[0]
        assert entry.samples == len(pinned_rows)
        assert entry.rows_other_revision == len(stale_rows)
        # Identical to scoring the pinned rows alone — the stale rows moved
        # nothing.
        assert entry.metrics["generalization_accuracy"] == (
            score_intent(pinned_rows)["generalization_accuracy"]
        )
        # 696 paraphrase + 49 typos correct out of 696+49+42+49 generalizing
        # rows; template/in_distribution never enter.
        assert entry.metrics["generalization_accuracy"] == round(
            (696 + 49) / (696 + 49 + 42 + 49), 4
        )

    def test_a_branch_pinned_dataset_drops_nothing(self):
        rows = _shard(OLD, correct_in={"paraphrase"})
        board = build_benchmark_board(
            "intent", "intents-for-eval", "en-US", {"padacioso-medium": rows},
            "t", dataset_revision="main",
        )
        entry = board.entries[0]
        assert entry.samples == len(rows)
        assert entry.rows_other_revision == 0
        assert board.dataset_revision is None

    def test_helper_reports_the_drop_per_competitor(self):
        kept, dropped = drop_rows_off_pinned_revision(
            {
                "a": _shard(PIN)[:10] + _shard(OLD)[:3],
                "b": _shard(OLD)[:7],
                "c": _shard(PIN)[:5],
            },
            PIN,
        )
        assert dropped == {"a": 3, "b": 7, "c": 0}
        assert [len(v) for v in (kept["a"], kept["b"], kept["c"])] == [10, 0, 5]


class TestInDistributionStaysExcluded:
    """Mutation guard. ``near_ood`` is the old spelling of the same bucket;
    it must stay out of ``generalization_accuracy`` and keep publishing to
    ``acc_in_distribution``. Adding it back to the ranked population is
    exactly the regression that moved padacioso from 0.0953 to 0.364."""

    @staticmethod
    def _legacy_shard():
        """The old revision's layout: 1,750 rows, ``near_ood`` in place of
        ``in_distribution``. The engine memorises its in-distribution rows
        and gets the generalizing ones wrong."""
        return _shard(
            OLD,
            buckets=[("template", 500), ("near_ood", 400),
                     ("paraphrase", 700), ("far_ood", 50),
                     ("asr_noise", 50), ("typos", 50)],
            correct_in={"template", "near_ood"},
        )

    def test_near_ood_never_enters_generalization_accuracy(self):
        metrics = score_intent(self._legacy_shard())
        # Every generalizing row is wrong, so the metric is 0.0 — not the
        # 900/1750 the memorised buckets would inflate it to.
        assert metrics["generalization_accuracy"] == 0.0
        assert metrics["accuracy"] == round(900 / 1750, 4)

    def test_near_ood_publishes_as_acc_in_distribution(self):
        metrics = score_intent(self._legacy_shard())
        assert metrics["acc_in_distribution"] == 1.0
        assert "acc_near_ood" not in metrics

    def test_a_branch_pinned_legacy_shard_scores_the_same(self):
        """The exclusion is the scorer's, not the revision filter's: rows
        that survive on a branch-pinned dataset are excluded identically."""
        board = build_benchmark_board(
            "intent", "intents-for-eval", "en-US",
            {"padacioso-medium": self._legacy_shard()}, "t",
            dataset_revision="main",
        )
        entry = board.entries[0]
        assert entry.samples == 1750
        assert entry.metrics["acc_in_distribution"] == 1.0
        assert entry.metrics["generalization_accuracy"] == 0.0
        # Scoring zero on the generalizing rows is a result, so the entry is
        # ranked. Only having no rows on the pinned revision is not.
        assert entry.unranked is False
        assert entry.rows_other_revision == 0


class TestStratifiedMaxSamples:
    """``--max-samples`` on a bucket-ordered corpus. The pinned test file
    stores whole buckets back to back, so a head slice of 1,000 takes all
    500 template rows, all 45 in_distribution rows and 455 paraphrase rows
    — and nothing at all from far_ood, asr_noise or typos, which is three
    quarters of what generalization_accuracy is supposed to average."""

    @staticmethod
    def _corpus():
        return [
            (i, {"utterance": f"u{i}", "split": bucket})
            for i, bucket in enumerate(
                b for b, n in EN_US_BUCKETS for _ in range(n)
            )
        ]

    @staticmethod
    def _counts(sample):
        counts = {}
        for _, row in sample:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        return counts

    def test_every_bucket_survives_the_cap(self):
        sample = stratified_sample(self._corpus(), 1000)
        assert len(sample) == 1000
        counts = self._counts(sample)
        assert set(counts) == {b for b, _ in EN_US_BUCKETS}

    def test_bucket_counts_are_proportional(self):
        sample = stratified_sample(self._corpus(), 1000)
        counts = self._counts(sample)
        for bucket, total in EN_US_BUCKETS:
            expected = 1000 * total / 1381
            assert abs(counts[bucket] - expected) <= 1, bucket

    def test_a_head_slice_would_have_lost_three_buckets(self):
        """The defect this replaces, stated as a fact about the corpus."""
        head = self._counts(self._corpus()[:1000])
        assert set(head) == {"template", "in_distribution", "paraphrase"}

    def test_the_sample_is_stable_across_runs(self):
        """Resume depends on it: a second run must ask for the same rows,
        or it appends a different subset to the same shard."""
        first = [i for i, _ in stratified_sample(self._corpus(), 1000)]
        second = [i for i, _ in stratified_sample(self._corpus(), 1000)]
        assert first == second

    def test_rows_keep_their_original_index(self):
        """``sample_id`` is built from the index, so a capped sweep and a
        full sweep must name the same row identically."""
        corpus = self._corpus()
        for index, row in stratified_sample(corpus, 1000):
            assert corpus[index] == (index, row)

    def test_a_tiny_bucket_still_gets_a_row(self):
        corpus = [(i, {"split": "big" if i else "tiny"}) for i in range(1000)]
        counts = self._counts(stratified_sample(corpus, 10))
        assert counts == {"tiny": 1, "big": 9}

    def test_an_unbucketed_corpus_keeps_the_head_slice(self):
        """Nothing to stratify — caps on corpora without buckets draw
        exactly the rows they always drew."""
        corpus = [(i, {"utterance": f"u{i}"}) for i in range(100)]
        assert stratified_sample(corpus, 10) == corpus[:10]

    def test_a_cap_above_the_corpus_size_is_a_no_op(self):
        corpus = self._corpus()
        assert stratified_sample(corpus, 5000) == corpus

    def test_a_cap_smaller_than_the_bucket_count_spends_it_on_breadth(self):
        sample = stratified_sample(self._corpus(), 3)
        assert len(sample) == 3
        # The three widest buckets, ties broken by name (far_ood and typos
        # both hold 49 rows).
        assert set(self._counts(sample)) == {
            "paraphrase", "template", "far_ood"
        }


class TestResumeRegeneratesAcrossRevisions:
    """A ``sample_id`` is an index into whichever revision produced it, so
    resume that keys on the id alone never notices a re-pinned dataset."""

    @staticmethod
    def _seed_shard(path, revision, count):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(
            json.dumps({
                "competitor_id": "padacioso-medium",
                "sample_id": f"en-US/{i:05d}",
                "dataset_id": "intents-for-eval",
                "dataset_revision": revision,
                "lang": "en-US",
                "prediction": "media:play_song",
            }) + "\n"
            for i in range(count)
        ), encoding="utf-8")

    def test_a_stale_shard_is_regenerated_not_resumed(self, tmp_path):
        from runner.intent_bench import done_samples, prune_other_revisions

        out = tmp_path / "en-US" / "padacioso-medium.jsonl"
        self._seed_shard(out, OLD, 1750)

        # Keyed on sample_id alone, all 1,750 ids read as done and a 3-row
        # run at the new revision would write nothing.
        assert len(done_samples(out)) == 1750

        assert prune_other_revisions(out, PIN) == 1750
        assert done_samples(out, PIN) == set()
        todo = [i for i in range(3) if f"en-US/{i:05d}" not in done_samples(out, PIN)]
        assert todo == [0, 1, 2]

    def test_the_file_never_holds_two_revisions(self, tmp_path):
        out = tmp_path / "en-US" / "padacioso-medium.jsonl"
        self._seed_shard(out, OLD, 1750)
        from runner.intent_bench import prune_other_revisions

        prune_other_revisions(out, PIN)
        with out.open("a", encoding="utf-8") as fh:
            for i in range(3):
                fh.write(json.dumps({
                    "sample_id": f"en-US/{i:05d}", "dataset_revision": PIN,
                }) + "\n")
        revisions = {
            json.loads(line)["dataset_revision"]
            for line in out.read_text(encoding="utf-8").splitlines()
        }
        assert revisions == {PIN}

    def test_rows_on_the_pinned_revision_still_resume(self, tmp_path):
        from runner.intent_bench import done_samples, prune_other_revisions

        out = tmp_path / "en-US" / "padacioso-medium.jsonl"
        self._seed_shard(out, PIN, 400)
        assert prune_other_revisions(out, PIN) == 0
        assert len(done_samples(out, PIN)) == 400

    def test_done_samples_ignores_other_revisions_on_a_sha_pin(self, tmp_path):
        """Mutation guard: ``done_samples`` must actually read
        ``dataset_revision``, not just return every id in the file."""
        from runner.intent_bench import done_samples

        out = tmp_path / "mixed.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(
            json.dumps({"sample_id": f"en-US/{i:05d}",
                        "dataset_revision": PIN if i % 2 else OLD}) + "\n"
            for i in range(10)
        ), encoding="utf-8")
        assert done_samples(out, PIN) == {
            f"en-US/{i:05d}" for i in (1, 3, 5, 7, 9)
        }
        assert len(done_samples(out)) == 10

    def test_seeding_from_hf_leaves_stale_rows_behind(self, tmp_path):
        """A published shard predating the re-pin must not seed a local
        file, or the pair reads as complete and never re-runs."""
        from runner.autorun import _rows_on_revision

        payload = b"".join(
            json.dumps({"sample_id": f"en-US/{i:05d}", "dataset_revision": OLD}
                       ).encode() + b"\n"
            for i in range(1750)
        )
        kept, skipped = _rows_on_revision(payload, PIN)
        assert skipped == 1750
        assert kept == b""


class TestBranchPinnedDatasetsAreNeverPruned:
    """164 of the registry's 167 datasets pin a branch, not a commit. Their
    rows carry whichever sha the branch held when they were swept, and many
    predate the ``dataset_revision`` column entirely (§3.2: loaders MUST NOT
    require it). Comparing them against the sha the branch resolves to today
    would empty every one of those shards on the first run after any upstream
    commit — and autorun then publishes the emptied local shard back over the
    Hub, because the local file is the source of truth an upload snapshots
    from.
    """

    TIP = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"   # today's branch tip
    SWEPT = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"  # the sha rows carry

    @staticmethod
    def _seed(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                        encoding="utf-8")

    def _older_sha_shard(self, path, n=563):
        self._seed(path, [
            {"competitor_id": "chromium-stt-en", "sample_id": f"en-US/{i:05d}",
             "dataset_id": "minds14-en-US", "lang": "en-US",
             "dataset_revision": self.SWEPT, "prediction": "hello"}
            for i in range(n)
        ])

    def _no_revision_shard(self, path, n=60):
        """Rows published before ``dataset_revision`` existed as a column."""
        self._seed(path, [
            {"competitor_id": "microwakeword", "sample_id": f"en-US/{i:05d}",
             "dataset_id": "synthetic-wakewords-hey_jarvis", "lang": "en-US",
             "prediction": "hey jarvis"}
            for i in range(n)
        ])

    def test_prune_leaves_an_older_sha_shard_alone(self, tmp_path):
        from runner.intent_bench import prune_other_revisions

        out = tmp_path / "chromium-stt-en.jsonl"
        self._older_sha_shard(out)
        assert prune_other_revisions(out, "main") == 0
        assert len(out.read_text().splitlines()) == 563

    def test_prune_leaves_a_column_less_shard_alone(self, tmp_path):
        from runner.intent_bench import prune_other_revisions

        out = tmp_path / "microwakeword.jsonl"
        self._no_revision_shard(out)
        assert prune_other_revisions(out, "main") == 0
        assert len(out.read_text().splitlines()) == 60

    def test_every_row_counts_as_done_on_a_branch_pin(self, tmp_path):
        from runner.intent_bench import done_samples

        older = tmp_path / "a.jsonl"
        self._older_sha_shard(older, 563)
        assert len(done_samples(older, "main")) == 563

        columnless = tmp_path / "b.jsonl"
        self._no_revision_shard(columnless, 60)
        assert len(done_samples(columnless, "main")) == 60

    def test_media_bench_resumes_a_branch_pinned_shard(self, tmp_path):
        """Drives the real ``media_bench.run_competitor_lang``: a shard swept
        at an older branch sha must resume, not be wiped and re-run."""
        from tests.test_media_bench import StubAdapter, _competitor, _eval_def

        from runner import media_bench as mb

        out = tmp_path / "out.jsonl"
        self._seed(out, [
            {"competitor_id": "whispercpp-base", "sample_id": f"pt-PT/{i:05d}",
             "dataset_id": "minds14-pt-PT", "lang": "pt-PT",
             "dataset_revision": self.SWEPT, "prediction": f"hyp{i}"}
            for i in range(3)
        ])
        adapter = StubAdapter(n=3)
        result = mb.run_competitor_lang(
            adapter, _competitor(), "minds14-pt-PT", "pt-PT", _eval_def(),
            self.TIP, out, tmp_path / "audio", "owner/repo",
        )
        assert result.written == 0          # everything already done
        assert adapter.loaded == 0          # no model load, nothing to do
        assert len(out.read_text().splitlines()) == 3

    def test_media_bench_resumes_a_shard_with_no_revision_column(self, tmp_path):
        from tests.test_media_bench import StubAdapter, _competitor, _eval_def

        from runner import media_bench as mb

        out = tmp_path / "out.jsonl"
        self._seed(out, [
            {"competitor_id": "whispercpp-base", "sample_id": f"pt-PT/{i:05d}",
             "dataset_id": "minds14-pt-PT", "lang": "pt-PT",
             "prediction": f"hyp{i}"}
            for i in range(3)
        ])
        adapter = StubAdapter(n=3)
        result = mb.run_competitor_lang(
            adapter, _competitor(), "minds14-pt-PT", "pt-PT", _eval_def(),
            self.TIP, out, tmp_path / "audio", "owner/repo",
        )
        assert result.written == 0
        assert len(out.read_text().splitlines()) == 3

    def test_media_bench_still_prunes_a_sha_pinned_dataset(self, tmp_path):
        """The intents-for-eval case still works: a declared sha pin does
        prune rows from another revision."""
        from types import SimpleNamespace

        from tests.test_media_bench import StubAdapter, _competitor

        from runner import media_bench as mb

        pinned_def = SimpleNamespace(
            source=SimpleNamespace(hf_id="PolyAI/minds14", revision=PIN))
        out = tmp_path / "out.jsonl"
        self._seed(out, [
            {"competitor_id": "whispercpp-base", "sample_id": f"pt-PT/{i:05d}",
             "dataset_id": "minds14-pt-PT", "lang": "pt-PT",
             "dataset_revision": self.SWEPT, "prediction": f"hyp{i}"}
            for i in range(3)
        ])
        result = mb.run_competitor_lang(
            StubAdapter(n=3), _competitor(), "minds14-pt-PT", "pt-PT",
            pinned_def, PIN, out, tmp_path / "audio", "owner/repo",
        )
        assert result.written == 3
        revisions = {
            json.loads(line)["dataset_revision"]
            for line in out.read_text().splitlines()
        }
        assert revisions == {PIN}

    def test_intent_bench_resumes_a_branch_pinned_shard(self, tmp_path):
        """Drives the real ``intent_bench.run_competitor_lang``. The shard was
        swept at an older sha of ``main``; nothing may be pruned and the
        language must read as already complete."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from runner import intent_bench

        eval_def = SimpleNamespace(
            source=SimpleNamespace(hf_id="org/eval-repo", revision="main",
                                   file_pattern="{lang}/test.jsonl",
                                   subset=None, split="test"),
            train_datasets={"template": "train-ds"},
            input="text",
            reference_granularity="intent",
        )
        out = tmp_path / "out.jsonl"
        self._seed(out, [
            {"competitor_id": "x", "sample_id": f"ca-ES/{i:05d}",
             "dataset_id": "meteocat", "lang": "ca-ES",
             "dataset_revision": self.SWEPT, "prediction": "weather"}
            for i in range(3)
        ])
        rows = [{"utterance": f"quin temps fa {i}", "expected_intent": "weather"}
                for i in range(3)]
        with patch.object(intent_bench, "fetch_rows", return_value=rows):
            try:
                written = intent_bench.run_competitor_lang(
                    SimpleNamespace(competitor_id="x"), "meteocat", "ca-ES",
                    eval_def, {}, self.TIP, out,
                )
            except Exception:
                # The stub competitor cannot build a real pipeline. Getting
                # that far means the shard was pruned and every row queued
                # for regeneration, which the row assertions below catch.
                written = None
        assert len(out.read_text().splitlines()) == 3
        assert {json.loads(line)["dataset_revision"]
                for line in out.read_text().splitlines()} == {self.SWEPT}
        # Nothing to do, so the run returned before touching a pipeline.
        assert written == 0

    def test_intent_bench_resumes_a_shard_with_no_revision_column(self, tmp_path):
        from types import SimpleNamespace
        from unittest.mock import patch

        from runner import intent_bench

        eval_def = SimpleNamespace(
            source=SimpleNamespace(hf_id="org/eval-repo", revision="main",
                                   file_pattern="{lang}/test.jsonl",
                                   subset=None, split="test"),
            train_datasets={}, input="text", reference_granularity="intent",
        )
        out = tmp_path / "out.jsonl"
        self._seed(out, [
            {"competitor_id": "x", "sample_id": f"ca-ES/{i:05d}",
             "dataset_id": "meteocat", "lang": "ca-ES", "prediction": "weather"}
            for i in range(3)
        ])
        rows = [{"utterance": f"quin temps fa {i}", "expected_intent": "weather"}
                for i in range(3)]
        with patch.object(intent_bench, "fetch_rows", return_value=rows):
            try:
                written = intent_bench.run_competitor_lang(
                    SimpleNamespace(competitor_id="x"), "meteocat", "ca-ES",
                    eval_def, {}, self.TIP, out,
                )
            except Exception:
                # The stub competitor cannot build a real pipeline. Getting
                # that far means the shard was pruned and every row queued
                # for regeneration, which the row assertions below catch.
                written = None
        assert len(out.read_text().splitlines()) == 3
        assert written == 0

    def test_seed_from_hf_keeps_a_branch_pinned_shard_whole(
        self, tmp_path, monkeypatch,
    ):
        """A published shard of a branch-pinned dataset seeds verbatim."""
        from runner.autorun import seed_from_hf

        remote = tmp_path / "remote.jsonl"
        remote.write_text("".join(
            json.dumps({"sample_id": f"en-US/{i:05d}"}) + "\n"
            for i in range(60)
        ), encoding="utf-8")
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            lambda *a, **k: str(remote))

        class _Lister:
            def list_files(self, repo):
                return {"predictions/en-US/microwakeword.jsonl":
                        remote.stat().st_size}

        out = tmp_path / "local.jsonl"
        seed_from_hf(out, "owner/repo", "en-US", "microwakeword",
                     _Lister(), "main")
        assert len(out.read_text().splitlines()) == 60

    def test_seed_from_hf_filters_a_sha_pinned_shard(self, tmp_path, monkeypatch):
        """The intents-for-eval case: a declared sha pin still filters."""
        from runner.autorun import seed_from_hf

        remote = tmp_path / "remote.jsonl"
        remote.write_text("".join(
            json.dumps({"sample_id": f"en-US/{i:05d}",
                        "dataset_revision": PIN if i < 4 else OLD}) + "\n"
            for i in range(10)
        ), encoding="utf-8")
        monkeypatch.setattr("huggingface_hub.hf_hub_download",
                            lambda *a, **k: str(remote))

        class _Lister:
            def list_files(self, repo):
                return {"predictions/en-US/padacioso-medium.jsonl":
                        remote.stat().st_size}

        out = tmp_path / "local.jsonl"
        seed_from_hf(out, "owner/repo", "en-US", "padacioso-medium",
                     _Lister(), PIN)
        assert len(out.read_text().splitlines()) == 4


class TestMaxSamplesWiring:
    """The stratified draw has to be wired into the runner, not merely
    correct in isolation: the defect was ``test_rows[:max_samples]`` inside
    ``run_competitor_lang``."""

    def test_a_capped_run_writes_every_bucket(self, tmp_path):
        """1,000 of a real 1,381-row en-US test split, through the actual
        ``run_competitor_lang``."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from runner import intent_bench

        rows = [
            {"utterance": f"utterance {i}", "expected_intent": "media:play_song",
             "split": bucket}
            for i, bucket in enumerate(
                b for b, n in EN_US_BUCKETS for _ in range(n)
            )
        ]
        assert len(rows) == 1381

        class _Pipeline:
            stage_names = ["stub"]

            def __init__(self, *a, **kw):
                pass

            def train(self, train_data):
                pass

            def predict(self, utterance):
                return "media:play_song", {}, 1.0, 0.5, "stub"

        competitor = SimpleNamespace(
            competitor_id="padacioso-medium",
            modality=SimpleNamespace(value="intent_template"),
            plugin="ovos-padacioso-pipeline-plugin",
            pipeline=["padacioso-medium"],
            pipeline_plugins=[],
            config={"intents": {"pipeline": ["padacioso-medium"]}},
        )
        eval_def = SimpleNamespace(
            source=SimpleNamespace(hf_id="OpenVoiceOS/intents-for-eval",
                                   revision=PIN, file_pattern="{lang}/test.jsonl",
                                   subset=None, split="test"),
            train_datasets={}, input="text", reference_granularity="intent",
        )
        out = tmp_path / "padacioso-medium.jsonl"
        with patch.object(intent_bench, "fetch_rows", return_value=rows), \
             patch.object(intent_bench, "needed_paradigms", return_value=set()), \
             patch.object(intent_bench, "IntentPipeline", _Pipeline):
            written = intent_bench.run_competitor_lang(
                competitor, "intents-for-eval", "en-US", eval_def, {},
                PIN, out, max_samples=1000,
            )

        assert written == 1000
        written_rows = [json.loads(line)
                        for line in out.read_text().splitlines()]
        counts = {}
        for row in written_rows:
            counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1
        assert set(counts) == {b for b, _ in EN_US_BUCKETS}
        for bucket, total in EN_US_BUCKETS:
            assert abs(counts[bucket] - 1000 * total / 1381) <= 1, bucket
        # sample_id keeps the index into the FULL corpus, not into the sample.
        assert max(int(r["sample_id"].split("/")[1]) for r in written_rows) > 1000
