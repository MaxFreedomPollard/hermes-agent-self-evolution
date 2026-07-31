"""Composite fitness for an evolved hermes-agent code candidate.

PLAN.md specifies four signals for Phase 4 and one of them is not like the
others:

    pytest          hard gate - any failure is immediate rejection
    benchmarks      broad capability check, scored
    bug repro       did this mutation fix the bug it was aimed at
    code quality    the heuristics in safety.py, scored

"Hard gate" is taken literally here. A candidate whose test run fails scores
``0.0``, not "0.0 for tests and full marks everywhere else". There is no
weighting that lets a red suite through, because a code change that breaks
tests is not a partially good code change.

Availability is never assumed. hermes-agent ships no benchmark directories
today, so :mod:`evolution.core.gates` reports them ``UNAVAILABLE`` and this
module drops them from the weighted average rather than scoring an absent
benchmark as a zero (which would reject everything) or a pass (which would
certify nothing). Under ``strict=True`` an unavailable gate is a rejection
instead, for a release process that must prove every gate actually ran.

The candidate source is expected to be **on disk already** when
:meth:`CodeFitnessEvaluator.evaluate` is called - pytest and the reproduction
script run against the working tree, not against a string. :class:`CodeOrganism`
is what puts it there.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from evolution.core.gates import (
    GateChain,
    GateResult,
    GateStatus,
    run_benchmark_gate,
    run_pytest_gate,
)
from evolution.code.safety import (
    QualitySignals,
    SafetyReport,
    quality_signals,
    run_safety_checks,
)

__all__ = [
    "FitnessError",
    "ReproStatus",
    "ReproResult",
    "BugReproduction",
    "FitnessWeights",
    "CodeFitness",
    "BaselineSnapshot",
    "CodeFitnessEvaluator",
]


class FitnessError(RuntimeError):
    """Raised when a candidate cannot be scored honestly."""


# ──────────────────────────────────────────────────────────────────────────
# Bug reproduction
# ──────────────────────────────────────────────────────────────────────────


class ReproStatus(str, Enum):
    FIXED = "fixed"
    PRESENT = "present"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


# A repro script can be explicit about its verdict instead of relying on its
# exit code, which matters for scripts that also print a diagnostic.
FIXED_MARKERS = ("BUG_FIXED", "BUG FIXED")
PRESENT_MARKERS = ("BUG_PRESENT", "BUG_REPRODUCED", "BUG PRESENT")


@dataclass
class ReproResult:
    """What one run of a reproduction script said."""

    status: ReproStatus
    message: str
    exit_code: Optional[int] = None
    details: str = ""
    duration_s: float = 0.0

    @property
    def fixed(self) -> bool:
        return self.status is ReproStatus.FIXED

    @property
    def measured(self) -> bool:
        """True when the script ran and returned a verdict."""
        return self.status in (ReproStatus.FIXED, ReproStatus.PRESENT)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "message": self.message,
            "exit_code": self.exit_code,
            "details": self.details,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class BugReproduction:
    """A script that demonstrates one specific bug.

    Contract: the script exits ``0`` when the bug is **fixed** and non-zero
    when it still reproduces, which is what a pytest file expressing the
    desired behaviour does for free. A script can override that by printing
    ``BUG_FIXED`` or ``BUG_PRESENT``.

    ``test_*.py`` files run under pytest; any other ``.py`` file runs as a
    plain script; anything else must be executable and is run directly.
    """

    script: Path
    issue: Optional[str] = None
    timeout: int = 300
    python: Optional[str] = None

    def __post_init__(self) -> None:
        self.script = Path(self.script).expanduser()

    @property
    def name(self) -> str:
        return self.script.name

    def available(self) -> bool:
        return self.script.is_file()

    def command(self, python: Optional[str] = None) -> list[str]:
        """Build the command that runs this reproduction."""
        interpreter = self.python or python or "python"
        if self.script.suffix == ".py":
            stem = self.script.stem
            if stem.startswith("test_") or stem.endswith("_test"):
                return [interpreter, "-m", "pytest", str(self.script), "-q", "--tb=short"]
            return [interpreter, str(self.script)]
        if os.access(self.script, os.X_OK):
            return [str(self.script)]
        raise FitnessError(
            f"do not know how to run reproduction script {self.script} "
            "(use a .py file, or mark the script executable)"
        )

    def run(self, repo: Path, python: Optional[str] = None) -> ReproResult:
        """Run the reproduction against the working tree at *repo*."""
        if not self.available():
            return ReproResult(
                ReproStatus.UNAVAILABLE,
                f"reproduction script not found: {self.script}",
            )

        try:
            cmd = self.command(python)
        except FitnessError as exc:
            return ReproResult(ReproStatus.UNAVAILABLE, str(exc))

        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(repo),
            )
        except subprocess.TimeoutExpired:
            return ReproResult(
                ReproStatus.ERROR,
                f"reproduction timed out after {self.timeout}s",
                duration_s=time.time() - started,
            )
        except OSError as exc:
            return ReproResult(
                ReproStatus.UNAVAILABLE, f"could not run reproduction: {exc}"
            )

        elapsed = time.time() - started
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        tail = "\n".join(output.strip().splitlines()[-20:])

        if any(marker in output for marker in FIXED_MARKERS):
            return ReproResult(
                ReproStatus.FIXED,
                "reproduction reports the bug is fixed",
                proc.returncode,
                tail,
                elapsed,
            )
        if any(marker in output for marker in PRESENT_MARKERS):
            return ReproResult(
                ReproStatus.PRESENT,
                "reproduction reports the bug is still present",
                proc.returncode,
                tail,
                elapsed,
            )

        if proc.returncode == 0:
            return ReproResult(
                ReproStatus.FIXED,
                "reproduction exited 0 - bug appears fixed",
                proc.returncode,
                tail,
                elapsed,
            )
        return ReproResult(
            ReproStatus.PRESENT,
            f"reproduction exited {proc.returncode} - bug still reproduces",
            proc.returncode,
            tail,
            elapsed,
        )


# ──────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FitnessWeights:
    """Relative weight of each *scored* component.

    pytest is absent from this list on purpose: it gates rather than scores.
    Weights are renormalized over whatever components were actually
    measurable, so a run with no benchmark installed still produces a
    meaningful number instead of silently capping at 0.7.
    """

    bug_fix: float = 0.5
    benchmark: float = 0.3
    quality: float = 0.2

    def to_dict(self) -> dict:
        return {
            "bug_fix": self.bug_fix,
            "benchmark": self.benchmark,
            "quality": self.quality,
        }


@dataclass
class CodeFitness:
    """The full verdict on one candidate."""

    label: str
    accepted: bool
    total: float
    safety: SafetyReport
    quality: QualitySignals
    pytest_result: GateResult
    benchmark_results: list[GateResult] = field(default_factory=list)
    repro: Optional[ReproResult] = None
    components: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.accepted

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "accepted": self.accepted,
            "total": round(self.total, 4),
            "rejection_reason": self.rejection_reason,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "safety": self.safety.to_dict(),
            "quality": self.quality.to_dict(),
            "pytest": self.pytest_result.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmark_results],
            "repro": self.repro.to_dict() if self.repro else None,
            "notes": list(self.notes),
        }


@dataclass
class BaselineSnapshot:
    """The state of the repo before any mutation, for comparison and sanity."""

    source: str
    pytest_result: GateResult
    repro: Optional[ReproResult] = None
    benchmark_results: list[GateResult] = field(default_factory=list)

    @property
    def tests_green(self) -> bool:
        return self.pytest_result.status is GateStatus.PASSED

    @property
    def bug_reproduces(self) -> bool:
        """True when the repro script confirms the bug exists to be fixed."""
        return self.repro is not None and self.repro.status is ReproStatus.PRESENT

    def benchmark_baselines(self) -> dict[str, Optional[float]]:
        return {r.name: r.score for r in self.benchmark_results}

    def to_dict(self) -> dict:
        return {
            "pytest": self.pytest_result.to_dict(),
            "repro": self.repro.to_dict() if self.repro else None,
            "benchmarks": [b.to_dict() for b in self.benchmark_results],
            "tests_green": self.tests_green,
            "bug_reproduces": self.bug_reproduces,
        }


PytestRunner = Callable[..., GateResult]
BenchmarkRunner = Callable[..., GateResult]


class CodeFitnessEvaluator:
    """Score candidates against the ladder PLAN.md specifies.

    The runners are injectable so the scoring logic can be tested without a
    hermes-agent checkout, and so an operator can substitute a cheaper test
    subset for the 2550-test suite during a long run.
    """

    def __init__(
        self,
        repo: Path,
        *,
        target: Optional[Path] = None,
        repro: Optional[BugReproduction] = None,
        benchmarks: Sequence[str] = (),
        benchmark_baselines: Optional[dict[str, Optional[float]]] = None,
        weights: Optional[FitnessWeights] = None,
        python: Optional[str] = None,
        pytest_subset: Optional[Sequence[str]] = None,
        pytest_timeout: int = 900,
        regression_threshold: float = 0.02,
        strict: bool = False,
        require_bug_fix: bool = True,
        safety_checks: Optional[Iterable[Callable[[str, str], object]]] = None,
        pytest_runner: PytestRunner = run_pytest_gate,
        benchmark_runner: BenchmarkRunner = run_benchmark_gate,
    ) -> None:
        self.repo = Path(repo)
        self.target = Path(target) if target else None
        self.repro = repro
        self.benchmarks = tuple(benchmarks)
        self.benchmark_baselines = dict(benchmark_baselines or {})
        self.weights = weights or FitnessWeights()
        self.python = python
        self.pytest_subset = list(pytest_subset) if pytest_subset else None
        self.pytest_timeout = pytest_timeout
        self.regression_threshold = regression_threshold
        self.strict = strict
        self.require_bug_fix = require_bug_fix
        self.safety_checks = safety_checks
        self.pytest_runner = pytest_runner
        self.benchmark_runner = benchmark_runner

    # ── gates ───────────────────────────────────────────────────────────

    def _run_pytest(self) -> GateResult:
        return self.pytest_runner(
            self.repo,
            subset=self.pytest_subset,
            timeout=self.pytest_timeout,
            python=self.python,
        )

    def _run_benchmark(self, name: str, fast: bool = True) -> GateResult:
        return self.benchmark_runner(
            self.repo,
            name,
            baseline=self.benchmark_baselines.get(name),
            regression_threshold=self.regression_threshold,
            fast=fast,
        )

    def _run_repro(self) -> Optional[ReproResult]:
        if self.repro is None:
            return None
        return self.repro.run(self.repo, self.python)

    # ── baseline ────────────────────────────────────────────────────────

    def snapshot_baseline(self, source: str) -> BaselineSnapshot:
        """Measure the repo before evolution starts.

        Two things this catches early, both of which invalidate a whole run:
        a baseline test suite that is already red, and a reproduction script
        that already passes (so there is no bug to fix, or the script does not
        actually reproduce it).
        """
        pytest_result = self._run_pytest()
        benchmarks = [self._run_benchmark(name) for name in self.benchmarks]
        for result in benchmarks:
            if result.score is not None:
                self.benchmark_baselines.setdefault(result.name, result.score)
        return BaselineSnapshot(
            source=source,
            pytest_result=pytest_result,
            repro=self._run_repro(),
            benchmark_results=benchmarks,
        )

    # ── candidate scoring ───────────────────────────────────────────────

    def evaluate(
        self,
        before_source: str,
        after_source: str,
        label: str = "candidate",
    ) -> CodeFitness:
        """Score one candidate that is already written to the working tree."""
        if self.target is not None and self.target.is_file():
            on_disk = self.target.read_text(encoding="utf-8")
            if on_disk != after_source:
                raise FitnessError(
                    f"{self.target} does not contain the candidate being scored - "
                    "apply the mutation before evaluating it"
                )

        safety = run_safety_checks(before_source, after_source, self.safety_checks)
        quality = quality_signals(before_source, after_source)

        if after_source == before_source:
            # A candidate identical to the baseline cannot have fixed anything,
            # and running 2550 tests to confirm that would be a waste.
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=GateResult(
                    "pytest",
                    GateStatus.SKIPPED,
                    "not run - candidate is identical to the baseline",
                ),
                rejection_reason="no change from the baseline",
            )

        if not safety.passed:
            failure = safety.first_failure()
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=GateResult(
                    "pytest",
                    GateStatus.SKIPPED,
                    "not run - candidate failed the safety guardrails",
                ),
                rejection_reason=f"safety: {failure.message}" if failure else "safety",
                notes=["expensive gates skipped: guardrails rejected the candidate"],
            )

        chain = GateChain(strict=self.strict).run(
            self._run_pytest,
            *[self._benchmark_thunk(name) for name in self.benchmarks],
        )
        pytest_result = chain.results[0]
        benchmark_results = list(chain.results[1:])

        notes: list[str] = []
        if pytest_result.status is GateStatus.UNAVAILABLE:
            notes.append(
                "pytest could not run - the hard gate did not actually verify anything"
            )

        if not chain.passed:
            blocker = chain.blockers[0]
            reason = (
                "pytest failed - hard gate, no partial credit"
                if blocker.name == "pytest" and blocker.status is GateStatus.FAILED
                else f"{blocker.name}: {blocker.message}"
            )
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=pytest_result,
                benchmark_results=benchmark_results,
                rejection_reason=reason,
                notes=notes,
            )

        repro = self._run_repro()
        if (
            repro is not None
            and self.require_bug_fix
            and repro.status is not ReproStatus.UNAVAILABLE
            and not repro.fixed
        ):
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=pytest_result,
                benchmark_results=benchmark_results,
                repro=repro,
                rejection_reason=f"bug not fixed: {repro.message}",
                notes=notes,
            )

        components, weights_used = self._score_components(
            repro, benchmark_results, quality
        )
        total = self._weighted_total(components, weights_used)

        return CodeFitness(
            label=label,
            accepted=True,
            total=total,
            safety=safety,
            quality=quality,
            pytest_result=pytest_result,
            benchmark_results=benchmark_results,
            repro=repro,
            components=components,
            weights_used=weights_used,
            notes=notes,
        )

    # ── internals ───────────────────────────────────────────────────────

    def _benchmark_thunk(self, name: str) -> Callable[[], GateResult]:
        return lambda: self._run_benchmark(name)

    def _score_components(
        self,
        repro: Optional[ReproResult],
        benchmark_results: Sequence[GateResult],
        quality: QualitySignals,
    ) -> tuple[dict[str, float], dict[str, float]]:
        components: dict[str, float] = {}
        weights: dict[str, float] = {}

        if repro is not None and repro.measured:
            components["bug_fix"] = 1.0 if repro.fixed else 0.0
            weights["bug_fix"] = self.weights.bug_fix

        scored = [r.score for r in benchmark_results if r.score is not None]
        if scored:
            components["benchmark"] = sum(scored) / len(scored)
            weights["benchmark"] = self.weights.benchmark

        components["quality"] = quality.score
        weights["quality"] = self.weights.quality
        return components, weights

    @staticmethod
    def _weighted_total(
        components: dict[str, float], weights: dict[str, float]
    ) -> float:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0
        weighted = sum(components[k] * weights[k] for k in components)
        return max(0.0, min(1.0, weighted / total_weight))
