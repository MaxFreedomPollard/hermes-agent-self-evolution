"""Stop one tool's description from stealing another tool's selections.

This is the part PLAN.md singles out as the hard part of Phase 2, and the
reason is arithmetic. Suppose ``search_files`` is picked correctly 60% of the
time and ``read_file`` 90%. Rewrite ``search_files`` to sound like the answer to
every question about files and it might climb to 85% while dragging
``read_file`` down to 70%. Aggregate accuracy went up. The agent got worse: it
now grep-scans files a user asked it to read.

Aggregate accuracy cannot see that, so it is not what decides. Every candidate
is evaluated over the whole catalogue at once, per-tool selection rates are
computed side by side, and a candidate is rejected when *any* individual tool
regresses beyond tolerance, however good the average looks. When a tool does
regress the confusion matrix says where its selections went, which is the
difference between "search_files got worse" and "search_files lost eleven
selections to read_file".

Tolerance is configurable because zero is right for a large dataset and cruel
for a small one: with eight examples per tool, one flipped answer is a 12.5%
swing that may be noise. :class:`CrossToolGuard` defaults to zero - no
regression at all, which is what PLAN.md asks for - and lets a caller loosen it
deliberately rather than by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from evolution.core.gates import GateResult, GateStatus
from evolution.tools.selection_eval import (
    NO_TOOL,
    SelectionOutcome,
    SelectionReport,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "ConfusionMatrix",
    "ToolRate",
    "CrossToolReport",
    "ToolRegression",
    "ToolImprovement",
    "CrossToolVerdict",
    "CrossToolGuard",
]

# PLAN.md: "No individual tool's selection rate regresses."
DEFAULT_TOLERANCE = 0.0

# Rates are ratios of small integers; this only guards float representation.
_EPSILON = 1e-9


# ──────────────────────────────────────────────────────────────────────────
# Matrices
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ConfusionMatrix:
    """What got picked, for each tool that should have been picked.

    ``counts[expected][predicted]`` is a tally, so the diagonal is correct
    selections and everything off it is a misroute. :data:`NO_TOOL` is a row and
    a column like any other tool, because "answered directly when it should have
    called something" and its inverse are exactly the failures a description
    rewrite causes.
    """

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, expected: str, predicted: str, n: int = 1) -> None:
        row = self.counts.setdefault(expected, {})
        row[predicted] = row.get(predicted, 0) + n

    def row(self, expected: str) -> dict[str, int]:
        return dict(self.counts.get(expected, {}))

    def column(self, predicted: str) -> dict[str, int]:
        """Which tools lost selections to *predicted*, and how many."""
        return {
            expected: row[predicted]
            for expected, row in self.counts.items()
            if row.get(predicted) and expected != predicted
        }

    def total(self) -> int:
        return sum(sum(row.values()) for row in self.counts.values())

    def correct(self, expected: str) -> int:
        return self.counts.get(expected, {}).get(expected, 0)

    def opportunities(self, expected: str) -> int:
        return sum(self.counts.get(expected, {}).values())

    def misroutes(self, expected: str) -> list[tuple[str, int]]:
        """Where *expected*'s selections went instead, worst first."""
        row = self.row(expected)
        row.pop(expected, None)
        return sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))

    def top_confusions(self, limit: int = 5) -> list[tuple[str, str, int]]:
        """The worst ``(expected, predicted, count)`` mix-ups across the board."""
        pairs = [
            (expected, predicted, count)
            for expected, row in self.counts.items()
            for predicted, count in row.items()
            if predicted != expected and count > 0
        ]
        pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
        return pairs[:limit]

    def to_dict(self) -> dict:
        return {expected: dict(row) for expected, row in sorted(self.counts.items())}


@dataclass(frozen=True)
class ToolRate:
    """How often one tool was chosen when it was the right answer."""

    tool: str
    opportunities: int
    correct: int

    @property
    def rate(self) -> float:
        if self.opportunities <= 0:
            return 0.0
        return self.correct / self.opportunities

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "opportunities": self.opportunities,
            "correct": self.correct,
            "rate": round(self.rate, 4),
        }


@dataclass
class CrossToolReport:
    """Per-tool selection rates plus the confusion matrix behind them."""

    rates: dict[str, ToolRate] = field(default_factory=dict)
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    n: int = 0
    overall_accuracy: float = 0.0
    param_accuracy: float = 0.0
    combined_score: float = 0.0

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[SelectionOutcome],
        tools: Optional[Iterable[str]] = None,
    ) -> "CrossToolReport":
        """Build a report from scored outcomes.

        *tools* seeds the rate table so a tool with zero examples still appears,
        with zero opportunities. A silently absent row is how a regression hides.
        """
        confusion = ConfusionMatrix()
        tallies: dict[str, list[int]] = {}
        for name in tools or []:
            tallies.setdefault(name, [0, 0])
        tallies.setdefault(NO_TOOL, [0, 0])

        for outcome in outcomes:
            expected = outcome.expected_tool
            predicted = outcome.predicted_tool
            confusion.record(expected, predicted)
            counts = tallies.setdefault(expected, [0, 0])
            counts[0] += 1
            if outcome.tool_correct:
                counts[1] += 1

        report = SelectionReport(outcomes=list(outcomes))
        return cls(
            rates={
                name: ToolRate(tool=name, opportunities=counts[0], correct=counts[1])
                for name, counts in sorted(tallies.items())
            },
            confusion=confusion,
            n=len(outcomes),
            overall_accuracy=report.tool_accuracy,
            param_accuracy=report.param_accuracy,
            combined_score=report.score,
        )

    @classmethod
    def from_report(
        cls,
        report: SelectionReport,
        tools: Optional[Iterable[str]] = None,
    ) -> "CrossToolReport":
        return cls.from_outcomes(report.outcomes, tools=tools)

    def rate(self, tool: str) -> float:
        entry = self.rates.get(tool)
        return entry.rate if entry else 0.0

    def opportunities(self, tool: str) -> int:
        entry = self.rates.get(tool)
        return entry.opportunities if entry else 0

    @property
    def measured_tools(self) -> list[str]:
        """Tools that actually had at least one chance to be selected."""
        return [name for name, entry in self.rates.items() if entry.opportunities > 0]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "param_accuracy": round(self.param_accuracy, 4),
            "combined_score": round(self.combined_score, 4),
            "rates": {name: entry.to_dict() for name, entry in self.rates.items()},
            "confusion": self.confusion.to_dict(),
        }


# ──────────────────────────────────────────────────────────────────────────
# The guard
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolRegression:
    """One tool that got worse, and where its selections went."""

    tool: str
    baseline_rate: float
    candidate_rate: float
    opportunities: int
    stolen_by: dict[str, int] = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.candidate_rate - self.baseline_rate

    def describe(self) -> str:
        text = (
            f"{self.tool}: {self.baseline_rate:.1%} -> {self.candidate_rate:.1%} "
            f"({self.delta:+.1%} over {self.opportunities} example(s))"
        )
        if self.stolen_by:
            thief = ", ".join(
                f"{name} +{count}"
                for name, count in sorted(self.stolen_by.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            text += f"; lost to {thief}"
        return text

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "baseline_rate": round(self.baseline_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "delta": round(self.delta, 4),
            "opportunities": self.opportunities,
            "stolen_by": dict(self.stolen_by),
        }


@dataclass(frozen=True)
class ToolImprovement:
    """One tool that got better."""

    tool: str
    baseline_rate: float
    candidate_rate: float
    opportunities: int

    @property
    def delta(self) -> float:
        return self.candidate_rate - self.baseline_rate

    def describe(self) -> str:
        return (
            f"{self.tool}: {self.baseline_rate:.1%} -> {self.candidate_rate:.1%} "
            f"({self.delta:+.1%})"
        )

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "baseline_rate": round(self.baseline_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "delta": round(self.delta, 4),
            "opportunities": self.opportunities,
        }


@dataclass
class CrossToolVerdict:
    """Accept or reject, with the per-tool evidence for the decision."""

    accepted: bool
    baseline_accuracy: float
    candidate_accuracy: float
    regressions: list[ToolRegression] = field(default_factory=list)
    improvements: list[ToolImprovement] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE
    reason: str = ""

    @property
    def overall_delta(self) -> float:
        return self.candidate_accuracy - self.baseline_accuracy

    def summary(self) -> str:
        head = "accepted" if self.accepted else "REJECTED"
        return (
            f"cross-tool {head}: overall {self.baseline_accuracy:.1%} -> "
            f"{self.candidate_accuracy:.1%} ({self.overall_delta:+.1%}); {self.reason}"
        )

    def to_gate_result(self) -> GateResult:
        """Express the verdict as a gate so it can join a GateChain."""
        return GateResult(
            name="cross_tool",
            status=GateStatus.PASSED if self.accepted else GateStatus.FAILED,
            message=self.reason,
            score=self.candidate_accuracy,
            baseline=self.baseline_accuracy,
            details="\n".join(r.describe() for r in self.regressions),
        )

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "candidate_accuracy": round(self.candidate_accuracy, 4),
            "overall_delta": round(self.overall_delta, 4),
            "tolerance": self.tolerance,
            "reason": self.reason,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "ignored": list(self.ignored),
        }


@dataclass
class CrossToolGuard:
    """Compare two cross-tool reports and decide whether to accept the candidate.

    ``tolerance`` is how far a single tool may fall before the candidate is
    rejected, as an absolute rate difference. Zero means no regression at all.

    ``min_opportunities`` skips tools with too few examples to say anything;
    those tools are listed in the verdict's ``ignored`` field rather than being
    silently dropped, so nobody reads a pass as coverage it did not have.

    ``require_overall_improvement`` additionally demands the aggregate move up.
    Off by default: holding every tool steady while shortening 3,896 chars of
    description is a legitimate win.
    """

    tolerance: float = DEFAULT_TOLERANCE
    min_opportunities: int = 1
    require_overall_improvement: bool = False

    def compare(
        self,
        baseline: CrossToolReport,
        candidate: CrossToolReport,
    ) -> CrossToolVerdict:
        regressions: list[ToolRegression] = []
        improvements: list[ToolImprovement] = []
        ignored: list[str] = []

        tools = sorted(set(baseline.rates) | set(candidate.rates))
        for tool in tools:
            opportunities = baseline.opportunities(tool) or candidate.opportunities(tool)
            if opportunities < max(1, self.min_opportunities):
                if opportunities == 0:
                    ignored.append(tool)
                else:
                    ignored.append(f"{tool} ({opportunities} example(s))")
                continue

            before = baseline.rate(tool)
            after = candidate.rate(tool)
            delta = after - before

            if delta < -abs(self.tolerance) - _EPSILON:
                regressions.append(
                    ToolRegression(
                        tool=tool,
                        baseline_rate=before,
                        candidate_rate=after,
                        opportunities=opportunities,
                        stolen_by=self._stolen_by(tool, baseline, candidate),
                    )
                )
            elif delta > _EPSILON:
                improvements.append(
                    ToolImprovement(
                        tool=tool,
                        baseline_rate=before,
                        candidate_rate=after,
                        opportunities=opportunities,
                    )
                )

        overall_delta = candidate.overall_accuracy - baseline.overall_accuracy
        accepted = not regressions
        if accepted and self.require_overall_improvement and overall_delta <= _EPSILON:
            accepted = False
            reason = (
                f"no per-tool regression, but overall accuracy did not improve "
                f"({overall_delta:+.1%})"
            )
        elif regressions:
            worst = min(regressions, key=lambda r: r.delta)
            reason = (
                f"{len(regressions)} tool(s) regressed beyond a "
                f"{self.tolerance:.1%} tolerance, worst {worst.describe()}"
            )
            if overall_delta > 0:
                reason += (
                    f" - rejected despite the aggregate improving {overall_delta:+.1%}"
                )
        else:
            reason = (
                f"no tool regressed beyond a {self.tolerance:.1%} tolerance "
                f"({len(improvements)} improved, {len(ignored)} not measurable)"
            )

        return CrossToolVerdict(
            accepted=accepted,
            baseline_accuracy=baseline.overall_accuracy,
            candidate_accuracy=candidate.overall_accuracy,
            regressions=regressions,
            improvements=improvements,
            ignored=ignored,
            tolerance=self.tolerance,
            reason=reason,
        )

    def gate(
        self,
        baseline: CrossToolReport,
        candidate: CrossToolReport,
    ) -> GateResult:
        """The comparison as a :class:`GateResult`, ready for a GateChain."""
        return self.compare(baseline, candidate).to_gate_result()

    @staticmethod
    def _stolen_by(
        tool: str,
        baseline: CrossToolReport,
        candidate: CrossToolReport,
    ) -> dict[str, int]:
        """Which tools newly absorbed *tool*'s selections in the candidate."""
        before = baseline.confusion.row(tool)
        after = candidate.confusion.row(tool)
        stolen: dict[str, int] = {}
        for predicted, count in after.items():
            if predicted == tool:
                continue
            gained = count - before.get(predicted, 0)
            if gained > 0:
                stolen[predicted] = gained
        return stolen
