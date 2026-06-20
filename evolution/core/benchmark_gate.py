"""Benchmark regression gates for evolved artifacts.

The Phase-1 plan treats TBLite/YC-Bench as gates, not fitness functions: a
candidate that improves the task-specific eval but regresses broad benchmarks
must be rejected. This module wires the configured TBLite command into a
fail-closed comparison gate while keeping the expensive benchmark opt-in.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class BenchmarkResult:
    passed: bool
    name: str
    baseline_score: Optional[float] = None
    evolved_score: Optional[float] = None
    message: str = ""
    details: str = ""


class BenchmarkGate:
    """Run and compare benchmark scores for baseline vs evolved artifacts."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def run_tblite_score(self, hermes_repo: Path) -> float:
        """Run the configured TBLite command and return its numeric score.

        ``HERMES_TBLITE_COMMAND``/``tblite_command`` must print JSON containing
        either ``score`` or ``pass_rate``. The command is executed in
        ``hermes_repo``. The caller controls which skill file is present in the
        working tree before invoking this method.
        """

        if not self.config.tblite_command:
            raise ValueError("TBLite gate requested but HERMES_TBLITE_COMMAND is not set")
        return self._run_score_command(hermes_repo)

    def compare_tblite_scores(self, baseline_score: float, evolved_score: float) -> BenchmarkResult:
        """Compare baseline and evolved TBLite scores using the configured threshold."""

        baseline = float(baseline_score)
        evolved = float(evolved_score)
        allowed = baseline - self.config.tblite_regression_threshold
        passed = evolved >= allowed
        delta = evolved - baseline
        return BenchmarkResult(
            passed=passed,
            name="tblite",
            baseline_score=baseline,
            evolved_score=evolved,
            message=(
                f"TBLite {'passed' if passed else 'regressed'}: "
                f"baseline={baseline:.3f}, evolved={evolved:.3f}, delta={delta:+.3f}, "
                f"allowed_floor={allowed:.3f}"
            ),
        )

    def run_tblite_comparison(self, hermes_repo: Path) -> BenchmarkResult:
        """Run the current tree and compare with configured baseline score.

        This convenience method is useful for tests and scripts that already
        measured the baseline externally. The main evolution pipeline usually
        runs baseline and evolved scores itself so the comparison is fully
        self-contained.
        """

        if self.config.tblite_baseline_score is None:
            return BenchmarkResult(
                passed=False,
                name="tblite",
                message=(
                    "TBLite gate needs a baseline score. Provide --tblite-baseline-score, "
                    "set HERMES_TBLITE_BASELINE_SCORE, or let evolve_skill run the "
                    "baseline before temporarily writing the evolved skill."
                ),
            )

        try:
            evolved = self.run_tblite_score(hermes_repo)
        except Exception as exc:
            return BenchmarkResult(
                passed=False,
                name="tblite",
                message=f"TBLite gate failed: {exc}",
            )
        return self.compare_tblite_scores(self.config.tblite_baseline_score, evolved)

    def _run_score_command(self, hermes_repo: Path) -> float:
        command = shlex.split(self.config.tblite_command or "")
        result = subprocess.run(
            command,
            cwd=str(hermes_repo),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"TBLite command failed ({result.returncode}): {result.stderr or result.stdout}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("TBLite command must print JSON with score/pass_rate") from exc

        score = payload.get("score", payload.get("pass_rate"))
        if score is None:
            raise ValueError("TBLite JSON output must include score or pass_rate")
        return float(score)
