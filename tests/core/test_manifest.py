"""Tests for run manifests: usage tracking, fingerprints, serialization."""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.manifest import (
    DatasetFingerprint,
    LMUsageSummary,
    RunManifest,
    UsageTracker,
    config_snapshot,
    text_digest,
)


def _example(task: str) -> EvalExample:
    return EvalExample(task_input=task, expected_behavior="rubric")


def _dataset() -> EvalDataset:
    return EvalDataset(
        train=[_example("task a"), _example("task b")],
        val=[_example("task c")],
        holdout=[_example("task d")],
    )


def _entry(cost=0.001, prompt=100, completion=50, timestamp=None) -> dict:
    return {
        "timestamp": timestamp or datetime.now().isoformat(),
        "cost": cost,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


class TestTextDigest:
    def test_deterministic(self):
        assert text_digest("hello") == text_digest("hello")

    def test_differs_by_content(self):
        assert text_digest("hello") != text_digest("hello!")


class TestConfigSnapshot:
    def test_json_serializable_with_paths(self):
        config = EvolutionConfig(hermes_agent_path=Path("/tmp/repo"))
        snapshot = config_snapshot(config)
        json.dumps(snapshot)  # Must not raise
        assert snapshot["hermes_agent_path"] == "/tmp/repo"
        assert snapshot["iterations"] == 10


class TestDatasetFingerprint:
    def test_counts(self):
        fp = DatasetFingerprint.from_dataset(_dataset())
        assert (fp.train, fp.val, fp.holdout) == (2, 1, 1)

    def test_order_within_split_does_not_matter(self):
        a = EvalDataset(train=[_example("x"), _example("y")], holdout=[_example("z")])
        b = EvalDataset(train=[_example("y"), _example("x")], holdout=[_example("z")])
        assert DatasetFingerprint.from_dataset(a).sha256 == DatasetFingerprint.from_dataset(b).sha256

    def test_split_membership_matters(self):
        a = EvalDataset(train=[_example("x")], holdout=[_example("z")])
        b = EvalDataset(train=[_example("z")], holdout=[_example("x")])
        assert DatasetFingerprint.from_dataset(a).sha256 != DatasetFingerprint.from_dataset(b).sha256


class TestUsageTracker:
    def test_sums_tokens_cost_and_calls(self):
        history: list = []
        tracker = UsageTracker(history=history).start()
        history.append(_entry(cost=0.002))
        history.append(_entry(cost=0.003))
        summary = tracker.finish()
        assert summary.calls == 2
        assert summary.prompt_tokens == 200
        assert summary.completion_tokens == 100
        assert summary.total_tokens == 300
        assert summary.cost_usd == pytest.approx(0.005)
        assert not summary.possibly_incomplete

    def test_ignores_entries_before_start(self):
        history = [_entry(timestamp="2000-01-01T00:00:00")]
        tracker = UsageTracker(history=history).start()
        history.append(_entry())
        assert tracker.finish().calls == 1

    def test_unknown_cost_counted_separately(self):
        history: list = []
        tracker = UsageTracker(history=history).start()
        history.append(_entry(cost=None))
        history.append(_entry(cost=0.004))
        summary = tracker.finish()
        assert summary.cost_usd == pytest.approx(0.004)
        assert summary.calls_with_unknown_cost == 1

    def test_all_costs_unknown_reports_none(self):
        history: list = []
        tracker = UsageTracker(history=history).start()
        history.append(_entry(cost=None))
        summary = tracker.finish()
        assert summary.cost_usd is None
        assert summary.calls_with_unknown_cost == 1

    def test_no_calls(self):
        summary = UsageTracker(history=[]).start().finish()
        assert summary.calls == 0
        assert summary.cost_usd is None

    def test_finish_without_start(self):
        assert UsageTracker(history=[_entry()]).finish().calls == 0

    def test_eviction_sets_incomplete_flag(self):
        history: list = []
        tracker = UsageTracker(history=history, max_size=2).start()
        history.append(_entry())
        history.append(_entry())
        assert tracker.finish().possibly_incomplete


class TestRunManifest:
    def _build(self, scores=None, deployed=False) -> RunManifest:
        return RunManifest.build(
            run_id="arxiv-20260712_120000",
            skill_name="arxiv",
            skill_path=Path("/repo/skills/research/arxiv/SKILL.md"),
            config=EvolutionConfig(hermes_agent_path=None),
            dataset=_dataset(),
            baseline_text="baseline skill text",
            evolved_text="evolved skill text",
            constraint_results=[
                SimpleNamespace(constraint_name="size_limit", passed=True, message="ok", details=None),
                SimpleNamespace(constraint_name="growth_limit", passed=False, message="too big", details="d"),
            ],
            scores=scores,
            deployed=deployed,
            elapsed_seconds=12.5,
            usage=LMUsageSummary(calls=3, total_tokens=900, cost_usd=0.01),
        )

    def test_round_trip(self, tmp_path):
        manifest = self._build(
            scores={"baseline_holdout": 0.4, "evolved_holdout": 0.6, "improvement": 0.2},
            deployed=True,
        )
        path = manifest.save(tmp_path / "run" / "manifest.json")
        loaded = RunManifest.load(path)
        assert loaded["run_id"] == "arxiv-20260712_120000"
        assert loaded["deployed"] is True
        assert loaded["scores"]["improvement"] == pytest.approx(0.2)
        assert loaded["usage"]["cost_usd"] == pytest.approx(0.01)
        assert loaded["dataset"]["train"] == 2
        assert loaded["schema_version"] == 1

    def test_constraints_recorded_including_failures(self):
        manifest = self._build()
        names = {c["name"]: c["passed"] for c in manifest.to_dict()["constraints"]}
        assert names == {"size_limit": True, "growth_limit": False}

    def test_rejected_run_has_no_scores(self, tmp_path):
        manifest = self._build(scores=None, deployed=False)
        loaded = RunManifest.load(manifest.save(tmp_path / "manifest.json"))
        assert loaded["scores"] is None
        assert loaded["deployed"] is False

    def test_digests_distinguish_baseline_and_evolved(self):
        d = self._build().to_dict()
        assert d["baseline"]["sha256"] != d["evolved"]["sha256"]
        assert d["baseline"]["chars"] == len("baseline skill text")
