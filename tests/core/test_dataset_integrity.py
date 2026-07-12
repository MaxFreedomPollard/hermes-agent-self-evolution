"""Tests for holdout integrity: deduplication and seeded splits."""

import json

import pytest

from evolution.core.dataset_builder import (
    EvalExample,
    GoldenDatasetLoader,
    dedupe_examples,
    split_examples,
)


def _example(task: str, rubric: str = "does the thing") -> EvalExample:
    return EvalExample(task_input=task, expected_behavior=rubric)


class TestDedupeExamples:
    def test_exact_duplicates_removed_first_kept(self):
        examples = [
            _example("Find papers about transformers", rubric="first"),
            _example("Find papers about transformers", rubric="second"),
        ]
        kept = dedupe_examples(examples)
        assert len(kept) == 1
        assert kept[0].expected_behavior == "first"

    def test_case_and_punctuation_variants_are_duplicates(self):
        examples = [
            _example("Find papers about transformers."),
            _example("find papers about Transformers"),
        ]
        assert len(dedupe_examples(examples)) == 1

    def test_near_duplicates_removed(self):
        examples = [
            _example("Search the arxiv api for recent papers about diffusion models"),
            _example("Search the arxiv api for recent papers about diffusion models today"),
        ]
        assert len(dedupe_examples(examples)) == 1

    def test_distinct_tasks_kept(self):
        examples = [
            _example("Find the arXiv ID of the BERT paper"),
            _example("Generate a BibTeX entry for the ResNet paper"),
            _example("List recent papers on protein folding"),
        ]
        assert len(dedupe_examples(examples)) == 3

    def test_threshold_is_respected(self):
        examples = [
            _example("alpha beta gamma delta"),
            _example("alpha beta gamma epsilon"),  # Jaccard 3/5 = 0.6
        ]
        assert len(dedupe_examples(examples, jaccard_threshold=0.9)) == 2
        assert len(dedupe_examples(examples, jaccard_threshold=0.5)) == 1

    def test_empty_input(self):
        assert dedupe_examples([]) == []


class TestSplitExamples:
    def _pool(self, n: int) -> list[EvalExample]:
        return [_example(f"task number {i} about topic {i}") for i in range(n)]

    def test_ratios(self):
        dataset = split_examples(self._pool(20))
        assert len(dataset.train) == 10
        assert len(dataset.val) == 5
        assert len(dataset.holdout) == 5

    def test_same_seed_same_split(self):
        pool = self._pool(20)
        first = split_examples(pool, seed=13)
        second = split_examples(pool, seed=13)
        assert [e.task_input for e in first.holdout] == [e.task_input for e in second.holdout]

    def test_different_seed_different_split(self):
        pool = self._pool(40)
        first = split_examples(pool, seed=13)
        second = split_examples(pool, seed=14)
        assert [e.task_input for e in first.holdout] != [e.task_input for e in second.holdout]

    def test_splits_are_disjoint_and_complete(self):
        pool = self._pool(21)
        dataset = split_examples(pool)
        train = {e.task_input for e in dataset.train}
        val = {e.task_input for e in dataset.val}
        holdout = {e.task_input for e in dataset.holdout}
        assert not (train & val) and not (train & holdout) and not (val & holdout)
        assert len(train | val | holdout) == 21

    def test_input_order_is_not_mutated(self):
        pool = self._pool(10)
        original = [e.task_input for e in pool]
        split_examples(pool)
        assert [e.task_input for e in pool] == original

    def test_empty_pool(self):
        dataset = split_examples([])
        assert dataset.all_examples == []


class TestGoldenLoaderIntegrity:
    def _write_golden(self, path, tasks):
        golden = path / "golden.jsonl"
        with open(golden, "w") as f:
            for task in tasks:
                f.write(json.dumps({"task_input": task, "expected_behavior": "rubric"}) + "\n")
        return path

    def test_load_is_deterministic_across_calls(self, tmp_path):
        path = self._write_golden(tmp_path, [f"task number {i} about topic {i}" for i in range(12)])
        first = GoldenDatasetLoader.load(path)
        second = GoldenDatasetLoader.load(path)
        assert [e.task_input for e in first.holdout] == [e.task_input for e in second.holdout]

    def test_load_dedupes(self, tmp_path):
        path = self._write_golden(
            tmp_path,
            ["do the thing", "Do the thing!", "a completely different task entirely"],
        )
        dataset = GoldenDatasetLoader.load(path)
        assert len(dataset.all_examples) == 2
