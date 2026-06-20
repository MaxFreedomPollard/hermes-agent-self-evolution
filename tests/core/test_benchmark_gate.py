"""Tests for benchmark gate fail-closed behavior and score comparison."""

from evolution.core.benchmark_gate import BenchmarkGate
from evolution.core.config import EvolutionConfig


def test_tblite_gate_fails_closed_without_baseline(tmp_path):
    config = EvolutionConfig(run_tblite=True, tblite_command="echo {}")

    result = BenchmarkGate(config).run_tblite_comparison(tmp_path)

    assert not result.passed
    assert "baseline score" in result.message


def test_tblite_compare_allows_configured_regression_threshold():
    config = EvolutionConfig(tblite_regression_threshold=0.02)

    result = BenchmarkGate(config).compare_tblite_scores(0.90, 0.885)

    assert result.passed
    assert result.baseline_score == 0.90
    assert result.evolved_score == 0.885


def test_tblite_compare_rejects_regression_beyond_threshold():
    config = EvolutionConfig(tblite_regression_threshold=0.02)

    result = BenchmarkGate(config).compare_tblite_scores(0.90, 0.87)

    assert not result.passed
    assert "regressed" in result.message
