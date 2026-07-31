"""Tests for composite code fitness.

The gate runners are injected, so nothing here starts a benchmark or the
hermes-agent test suite. Reproduction scripts are real files run with
``sys.executable`` in tmp_path: that path has to work for real, and it costs
milliseconds. No network, no LM.
"""

import json
import sys

import pytest

from evolution.core.gates import GateResult, GateStatus
from evolution.code.fitness_code import (
    BaselineSnapshot,
    BugReproduction,
    CodeFitnessEvaluator,
    FitnessError,
    FitnessWeights,
    ReproStatus,
)

BEFORE = '''"""Toy tools."""


def read_lines(path, limit=10):
    """Return up to *limit* lines."""
    try:
        with open(path) as handle:
            return handle.read().splitlines()[:limit - 1]
    except OSError:
        return []
'''

# The fix: same shape, one slice corrected.
FIXED = BEFORE.replace("[:limit - 1]", "[:limit]")

# Fixes the bug but changes the signature, which the guardrails refuse.
UNSAFE = FIXED.replace("def read_lines(path, limit=10):", "def read_lines(path, limit=10, encoding='utf-8'):")


def passed(name="pytest", score=None, message="ok"):
    return GateResult(name, GateStatus.PASSED, message, score=score)


def failed(name="pytest", message="1 failed"):
    return GateResult(name, GateStatus.FAILED, message)


def unavailable(name="tblite", message="not found"):
    return GateResult(name, GateStatus.UNAVAILABLE, message)


class Recorder:
    """A stand-in gate runner that records how often it was called."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, repo, *args, **kwargs):
        self.calls += 1
        if len(self.results) == 1:
            return self.results[0]
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class NamedBenchmarks:
    """Benchmark runner that answers per benchmark name."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, repo, name, **kwargs):
        self.calls.append(name)
        return self.mapping[name]


def make_evaluator(tmp_path, **kwargs):
    kwargs.setdefault("pytest_runner", Recorder(passed()))
    kwargs.setdefault("benchmark_runner", Recorder(unavailable()))
    return CodeFitnessEvaluator(repo=tmp_path, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Bug reproduction
# ──────────────────────────────────────────────────────────────────────────


def write_script(tmp_path, name, body, mode=None):
    path = tmp_path / name
    path.write_text(body)
    if mode is not None:
        path.chmod(mode)
    return path


class TestBugReproductionCommand:
    def test_plain_script_runs_as_a_script(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "pass\n"))
        assert repro.command("py") == ["py", str(repro.script)]

    def test_test_prefixed_script_runs_under_pytest(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "test_repro.py", "def test_x():\n    pass\n"))
        assert repro.command("py")[:4] == ["py", "-m", "pytest", str(repro.script)]

    def test_test_suffixed_script_runs_under_pytest(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "issue742_test.py", "def test_x():\n    pass\n"))
        assert "pytest" in repro.command("py")

    def test_executable_non_python_script_runs_directly(self, tmp_path):
        script = write_script(tmp_path, "repro.sh", "#!/bin/sh\nexit 0\n", mode=0o755)
        assert BugReproduction(script=script).command() == [str(script)]

    def test_unrunnable_script_is_an_error(self, tmp_path):
        script = write_script(tmp_path, "repro.txt", "not a script\n", mode=0o644)
        with pytest.raises(FitnessError, match="do not know how to run"):
            BugReproduction(script=script).command()

    def test_explicit_interpreter_wins(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "pass\n"), python="/opt/py"
        )
        assert repro.command("ignored")[0] == "/opt/py"


class TestBugReproductionRun:
    def test_missing_script_is_unavailable_not_fixed(self, tmp_path):
        repro = BugReproduction(script=tmp_path / "ghost.py")
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.UNAVAILABLE
        assert not result.fixed
        assert not result.measured

    def test_exit_zero_means_fixed(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "import sys\nsys.exit(0)\n"))
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.FIXED
        assert result.fixed and result.measured
        assert result.exit_code == 0

    def test_non_zero_exit_means_the_bug_is_present(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "import sys\nsys.exit(3)\n"))
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.PRESENT
        assert not result.fixed
        assert result.exit_code == 3

    def test_present_marker_overrides_a_clean_exit(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "print('BUG_PRESENT: still truncating')\n")
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.PRESENT

    def test_fixed_marker_overrides_a_dirty_exit(self, tmp_path):
        repro = BugReproduction(
            script=write_script(
                tmp_path, "repro.py", "import sys\nprint('BUG_FIXED')\nsys.exit(1)\n"
            )
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.FIXED

    def test_timeout_is_an_error_not_a_fix(self, tmp_path):
        repro = BugReproduction(
            script=write_script(tmp_path, "repro.py", "import time\ntime.sleep(30)\n"),
            timeout=1,
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.ERROR
        assert not result.fixed
        assert "timed out" in result.message

    def test_a_pytest_style_reproduction_works(self, tmp_path):
        repro = BugReproduction(
            script=write_script(
                tmp_path, "test_issue_742.py", "def test_fixed():\n    assert 1 == 1\n"
            )
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.FIXED

    def test_a_failing_pytest_reproduction_reports_the_bug(self, tmp_path):
        repro = BugReproduction(
            script=write_script(
                tmp_path, "test_issue_742.py", "def test_fixed():\n    assert 1 == 2\n"
            )
        )
        result = repro.run(tmp_path, sys.executable)
        assert result.status is ReproStatus.PRESENT
        assert result.details

    def test_result_is_json_serialisable(self, tmp_path):
        repro = BugReproduction(script=write_script(tmp_path, "repro.py", "pass\n"))
        blob = json.loads(json.dumps(repro.run(tmp_path, sys.executable).to_dict()))
        assert blob["status"] == "fixed"


class FakeRepro(BugReproduction):
    """A reproduction whose verdict is fixed in advance."""

    def __init__(self, status):
        super().__init__(script=__file__)
        self._status = status

    def run(self, repo, python=None):
        from evolution.code.fitness_code import ReproResult

        return ReproResult(self._status, f"stubbed {self._status.value}")


# ──────────────────────────────────────────────────────────────────────────
# Composite scoring
# ──────────────────────────────────────────────────────────────────────────


class TestHardGate:
    def test_a_safety_failure_skips_the_expensive_gates(self, tmp_path):
        runner = Recorder(passed())
        evaluator = make_evaluator(tmp_path, pytest_runner=runner)
        fitness = evaluator.evaluate(BEFORE, UNSAFE, label="c01")

        assert not fitness.accepted
        assert fitness.total == 0.0
        assert fitness.rejection_reason.startswith("safety:")
        assert fitness.pytest_result.status is GateStatus.SKIPPED
        assert runner.calls == 0

    def test_an_unchanged_candidate_is_rejected_without_running_anything(self, tmp_path):
        runner = Recorder(passed())
        evaluator = make_evaluator(tmp_path, pytest_runner=runner)
        fitness = evaluator.evaluate(BEFORE, BEFORE, label="c01")

        assert not fitness.accepted
        assert fitness.total == 0.0
        assert fitness.rejection_reason == "no change from the baseline"
        assert runner.calls == 0

    def test_a_failing_test_suite_is_fatal_regardless_of_everything_else(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(failed()),
            repro=FakeRepro(ReproStatus.FIXED),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED, label="c01")

        assert not fitness.accepted
        assert fitness.total == 0.0
        assert "hard gate" in fitness.rejection_reason
        assert fitness.quality.score == pytest.approx(1.0)

    def test_a_failing_test_suite_stops_the_benchmarks_running(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.9)})
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(failed()),
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
        )
        evaluator.evaluate(BEFORE, FIXED)
        assert benchmarks.calls == []

    def test_unavailable_pytest_is_noted_but_not_fatal(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, pytest_runner=Recorder(unavailable("pytest", "no tests/"))
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.accepted
        assert any("hard gate did not actually verify" in n for n in fitness.notes)

    def test_strict_mode_rejects_an_unavailable_pytest(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            pytest_runner=Recorder(unavailable("pytest", "no tests/")),
            strict=True,
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert not fitness.accepted
        assert fitness.total == 0.0


class TestBugFitness:
    def test_fixing_the_bug_scores_full_marks(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert fitness.components["bug_fix"] == 1.0
        assert fitness.total == pytest.approx(1.0)

    def test_not_fixing_the_bug_is_a_rejection_by_default(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.PRESENT))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert not fitness.accepted
        assert "bug not fixed" in fitness.rejection_reason
        assert fitness.total == 0.0

    def test_require_bug_fix_off_scores_it_instead_of_rejecting(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path, repro=FakeRepro(ReproStatus.PRESENT), require_bug_fix=False
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert fitness.components["bug_fix"] == 0.0
        # quality 1.0 at weight 0.2, bug 0.0 at weight 0.5
        assert fitness.total == pytest.approx(0.2 / 0.7)

    def test_a_timed_out_reproduction_is_not_a_fix(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.ERROR))
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert not fitness.accepted

    def test_an_unavailable_reproduction_is_dropped_from_the_score(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.UNAVAILABLE))
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert "bug_fix" not in fitness.components


class TestBenchmarkFitness:
    def test_a_regression_rejects_the_candidate(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": failed("tblite", "regressed -8%")})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert not fitness.accepted
        assert "tblite" in fitness.rejection_reason
        assert fitness.total == 0.0

    def test_a_benchmark_score_enters_the_weighted_total(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.5)})
        evaluator = make_evaluator(
            tmp_path,
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
            repro=FakeRepro(ReproStatus.FIXED),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.components["benchmark"] == 0.5
        assert fitness.total == pytest.approx((1.0 * 0.5 + 0.5 * 0.3 + 1.0 * 0.2))

    def test_two_benchmarks_average(self, tmp_path):
        benchmarks = NamedBenchmarks(
            {
                "tblite": passed("tblite", score=0.6),
                "yc_bench": passed("yc_bench", score=1.0),
            }
        )
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite", "yc_bench"]
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.components["benchmark"] == pytest.approx(0.8)

    def test_an_unavailable_benchmark_is_excluded_not_scored_zero(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": unavailable("tblite")})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)

        assert fitness.accepted
        assert "benchmark" not in fitness.components
        assert fitness.total == pytest.approx(1.0)

    def test_strict_mode_rejects_an_unavailable_benchmark(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": unavailable("tblite")})
        evaluator = make_evaluator(
            tmp_path,
            benchmark_runner=benchmarks,
            benchmarks=["tblite"],
            strict=True,
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert not fitness.accepted
        assert "tblite" in fitness.rejection_reason


class TestWeighting:
    def test_with_nothing_measurable_the_total_is_the_quality_score(self, tmp_path):
        evaluator = make_evaluator(tmp_path)
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.components == {"quality": fitness.quality.score}
        assert fitness.total == pytest.approx(fitness.quality.score)

    def test_quality_regressions_lower_the_score(self, tmp_path):
        sloppy = FIXED.replace(
            "    except OSError:\n        return []\n", "    except:\n        pass\n"
        )
        evaluator = make_evaluator(tmp_path, require_bug_fix=False)
        fitness = evaluator.evaluate(BEFORE, sloppy)
        assert fitness.accepted
        assert fitness.total < 1.0

    def test_custom_weights_are_respected(self, tmp_path):
        evaluator = make_evaluator(
            tmp_path,
            repro=FakeRepro(ReproStatus.PRESENT),
            require_bug_fix=False,
            weights=FitnessWeights(bug_fix=0.9, benchmark=0.0, quality=0.1),
        )
        fitness = evaluator.evaluate(BEFORE, FIXED)
        assert fitness.total == pytest.approx(0.1)
        assert fitness.weights_used["bug_fix"] == 0.9


class TestOnDiskGuard:
    def test_scoring_a_candidate_that_is_not_applied_is_refused(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text(BEFORE)
        evaluator = make_evaluator(tmp_path, target=target)
        with pytest.raises(FitnessError, match="does not contain the candidate"):
            evaluator.evaluate(BEFORE, FIXED)

    def test_scoring_an_applied_candidate_is_allowed(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text(FIXED)
        evaluator = make_evaluator(tmp_path, target=target)
        assert evaluator.evaluate(BEFORE, FIXED).accepted


class TestBaseline:
    def test_snapshot_reports_green_tests_and_a_reproducing_bug(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.PRESENT))
        baseline = evaluator.snapshot_baseline(BEFORE)

        assert isinstance(baseline, BaselineSnapshot)
        assert baseline.tests_green
        assert baseline.bug_reproduces

    def test_snapshot_flags_a_bug_that_does_not_reproduce(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        baseline = evaluator.snapshot_baseline(BEFORE)
        assert not baseline.bug_reproduces

    def test_snapshot_flags_a_red_baseline(self, tmp_path):
        evaluator = make_evaluator(tmp_path, pytest_runner=Recorder(failed()))
        assert not evaluator.snapshot_baseline(BEFORE).tests_green

    def test_snapshot_records_benchmark_baselines_for_later_comparison(self, tmp_path):
        benchmarks = NamedBenchmarks({"tblite": passed("tblite", score=0.77)})
        evaluator = make_evaluator(
            tmp_path, benchmark_runner=benchmarks, benchmarks=["tblite"]
        )
        baseline = evaluator.snapshot_baseline(BEFORE)

        assert baseline.benchmark_baselines() == {"tblite": 0.77}
        assert evaluator.benchmark_baselines["tblite"] == 0.77

    def test_snapshot_is_json_serialisable(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.PRESENT))
        blob = json.loads(json.dumps(evaluator.snapshot_baseline(BEFORE).to_dict()))
        assert blob["tests_green"] is True
        assert blob["bug_reproduces"] is True


class TestFitnessRecord:
    def test_to_dict_is_json_serialisable(self, tmp_path):
        evaluator = make_evaluator(tmp_path, repro=FakeRepro(ReproStatus.FIXED))
        blob = json.loads(json.dumps(evaluator.evaluate(BEFORE, FIXED, "c01").to_dict()))

        assert blob["label"] == "c01"
        assert blob["accepted"] is True
        assert blob["safety"]["passed"] is True
        assert blob["pytest"]["status"] == "passed"
        assert blob["repro"]["status"] == "fixed"

    def test_rejected_is_the_inverse_of_accepted(self, tmp_path):
        evaluator = make_evaluator(tmp_path)
        assert evaluator.evaluate(BEFORE, UNSAFE).rejected
        assert not evaluator.evaluate(BEFORE, FIXED).rejected
