"""Tests for the cross-tool regression guard.

The headline case is the one PLAN.md warns about: a candidate whose aggregate
accuracy improves while one tool's selection rate falls. That must be rejected,
and the report must be able to say which tool absorbed the lost selections.
"""

import pytest

from evolution.core.gates import GateChain, GateResult, GateStatus
from evolution.tools.cross_tool import (
    ConfusionMatrix,
    CrossToolGuard,
    CrossToolReport,
)
from evolution.tools.selection_eval import NO_TOOL, ToolSelectionExample, score_selection


def outcomes(pairs):
    """Build scored outcomes from ``(expected, predicted)`` pairs."""
    return [
        score_selection(
            ToolSelectionExample(task=f"task {i}", correct_tool=expected), predicted
        )
        for i, (expected, predicted) in enumerate(pairs)
    ]


def report(pairs, tools=None):
    return CrossToolReport.from_outcomes(outcomes(pairs), tools=tools)


class TestConfusionMatrix:
    def test_records_and_reads_rows(self):
        matrix = ConfusionMatrix()
        matrix.record("read_file", "read_file")
        matrix.record("read_file", "search_files", 2)
        assert matrix.row("read_file") == {"read_file": 1, "search_files": 2}
        assert matrix.correct("read_file") == 1
        assert matrix.opportunities("read_file") == 3
        assert matrix.total() == 3

    def test_column_shows_who_lost_selections(self):
        matrix = ConfusionMatrix()
        matrix.record("read_file", "search_files", 3)
        matrix.record("terminal", "search_files", 1)
        matrix.record("search_files", "search_files", 5)
        assert matrix.column("search_files") == {"read_file": 3, "terminal": 1}

    def test_misroutes_are_ordered_worst_first(self):
        matrix = ConfusionMatrix()
        matrix.record("read_file", "read_file", 4)
        matrix.record("read_file", "terminal", 1)
        matrix.record("read_file", "search_files", 3)
        assert matrix.misroutes("read_file") == [("search_files", 3), ("terminal", 1)]

    def test_top_confusions_excludes_the_diagonal(self):
        matrix = ConfusionMatrix()
        matrix.record("a", "a", 10)
        matrix.record("a", "b", 2)
        matrix.record("b", "c", 3)
        assert matrix.top_confusions() == [("b", "c", 3), ("a", "b", 2)]

    def test_top_confusions_respects_the_limit(self):
        matrix = ConfusionMatrix()
        for i in range(5):
            matrix.record(f"t{i}", "thief", i + 1)
        assert len(matrix.top_confusions(limit=2)) == 2

    def test_unknown_row_is_empty_not_an_error(self):
        assert ConfusionMatrix().row("ghost") == {}
        assert ConfusionMatrix().opportunities("ghost") == 0

    def test_serialises(self):
        matrix = ConfusionMatrix()
        matrix.record("a", "b")
        assert matrix.to_dict() == {"a": {"b": 1}}


class TestCrossToolReport:
    def test_rates_per_tool(self):
        built = report(
            [
                ("read_file", "read_file"),
                ("read_file", "search_files"),
                ("search_files", "search_files"),
                ("search_files", "search_files"),
            ]
        )
        assert built.rate("read_file") == 0.5
        assert built.rate("search_files") == 1.0
        assert built.overall_accuracy == 0.75

    def test_confusion_matrix_is_populated(self):
        built = report([("read_file", "search_files"), ("read_file", "read_file")])
        assert built.confusion.row("read_file") == {"search_files": 1, "read_file": 1}

    def test_unmeasured_tools_are_seeded_at_zero(self):
        built = report([("read_file", "read_file")], tools=["read_file", "terminal"])
        assert built.opportunities("terminal") == 0
        assert built.rate("terminal") == 0.0
        assert built.measured_tools == ["read_file"]

    def test_no_tool_is_always_a_row(self):
        assert NO_TOOL in report([("read_file", "read_file")]).rates

    def test_no_tool_is_scored_like_any_other_tool(self):
        built = report([(NO_TOOL, NO_TOOL), (NO_TOOL, "read_file")])
        assert built.rate(NO_TOOL) == 0.5
        assert built.confusion.row(NO_TOOL)["read_file"] == 1

    def test_unknown_tool_reads_as_zero(self):
        built = report([("read_file", "read_file")])
        assert built.rate("ghost") == 0.0
        assert built.opportunities("ghost") == 0

    def test_empty_report(self):
        built = report([])
        assert built.n == 0 and built.overall_accuracy == 0.0

    def test_param_accuracy_flows_through(self):
        example = ToolSelectionExample(
            task="t", correct_tool="read_file", correct_params={"path": "a.py"}
        )
        built = CrossToolReport.from_outcomes(
            [score_selection(example, "read_file", {"path": "b.py"})]
        )
        assert built.param_accuracy == 0.0
        assert built.overall_accuracy == 1.0

    def test_serialises(self):
        blob = report([("read_file", "read_file")]).to_dict()
        assert blob["rates"]["read_file"]["rate"] == 1.0
        assert blob["confusion"]["read_file"] == {"read_file": 1}


class TestGuardRejectsStealing:
    """The case PLAN.md calls out: aggregate up, one tool down."""

    def baseline(self):
        # read_file 4/4, search_files 1/4. Overall 5/8.
        return report(
            [("read_file", "read_file")] * 4
            + [("search_files", "search_files")]
            + [("search_files", "read_file")] * 3,
            tools=["read_file", "search_files"],
        )

    def candidate(self):
        # search_files climbs to 4/4, read_file drops to 2/4. Overall 6/8.
        return report(
            [("read_file", "read_file")] * 2
            + [("read_file", "search_files")] * 2
            + [("search_files", "search_files")] * 4,
            tools=["read_file", "search_files"],
        )

    def test_the_aggregate_really_did_improve(self):
        assert self.candidate().overall_accuracy > self.baseline().overall_accuracy

    def test_candidate_is_rejected_anyway(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert verdict.accepted is False
        assert [r.tool for r in verdict.regressions] == ["read_file"]

    def test_reason_admits_the_aggregate_improved(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert "despite the aggregate improving" in verdict.reason

    def test_the_thief_is_named(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert verdict.regressions[0].stolen_by == {"search_files": 2}
        assert "lost to search_files +2" in verdict.regressions[0].describe()

    def test_the_improvement_is_still_recorded(self):
        verdict = CrossToolGuard().compare(self.baseline(), self.candidate())
        assert [i.tool for i in verdict.improvements] == ["search_files"]

    def test_verdict_becomes_a_blocking_gate(self):
        gate = CrossToolGuard().gate(self.baseline(), self.candidate())
        assert isinstance(gate, GateResult)
        assert gate.status is GateStatus.FAILED
        assert gate.blocking and not gate.passed
        assert "read_file" in gate.details

    def test_gate_chain_stops_on_the_rejection(self):
        guard = CrossToolGuard()
        later_gate_ran = []

        def later():
            later_gate_ran.append(True)
            return GateResult("later", GateStatus.PASSED, "ok")

        chain = GateChain().run(lambda: guard.gate(self.baseline(), self.candidate()), later)
        assert chain.passed is False
        assert later_gate_ran == []
        assert [r.name for r in chain.blockers] == ["cross_tool"]


class TestGuardAcceptance:
    def test_identical_reports_are_accepted(self):
        built = report([("read_file", "read_file"), ("terminal", "read_file")])
        verdict = CrossToolGuard().compare(built, built)
        assert verdict.accepted
        assert verdict.regressions == []
        assert verdict.overall_delta == 0.0

    def test_a_pure_improvement_is_accepted(self):
        baseline = report([("read_file", "terminal"), ("read_file", "read_file")])
        candidate = report([("read_file", "read_file"), ("read_file", "read_file")])
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert verdict.accepted
        assert verdict.improvements[0].delta == pytest.approx(0.5)

    def test_holding_steady_while_shrinking_text_is_allowed(self):
        built = report([("read_file", "read_file")] * 4)
        assert CrossToolGuard().compare(built, built).accepted

    def test_require_overall_improvement_rejects_a_flat_candidate(self):
        built = report([("read_file", "read_file")] * 4)
        verdict = CrossToolGuard(require_overall_improvement=True).compare(built, built)
        assert verdict.accepted is False
        assert "did not improve" in verdict.reason
        assert verdict.regressions == []


class TestTolerance:
    def baseline(self):
        return report([("read_file", "read_file")] * 4, tools=["read_file"])

    def candidate(self):
        # 3/4: a 25 point drop.
        return report(
            [("read_file", "read_file")] * 3 + [("read_file", "terminal")],
            tools=["read_file"],
        )

    def test_zero_tolerance_is_the_default(self):
        assert CrossToolGuard().tolerance == 0.0
        assert not CrossToolGuard().compare(self.baseline(), self.candidate()).accepted

    def test_a_configured_tolerance_absorbs_a_small_drop(self):
        assert CrossToolGuard(tolerance=0.3).compare(self.baseline(), self.candidate()).accepted

    def test_a_drop_exactly_at_tolerance_is_accepted(self):
        assert CrossToolGuard(tolerance=0.25).compare(self.baseline(), self.candidate()).accepted

    def test_a_drop_just_past_tolerance_is_rejected(self):
        guard = CrossToolGuard(tolerance=0.2499)
        assert not guard.compare(self.baseline(), self.candidate()).accepted

    def test_tolerance_is_reported_in_the_reason(self):
        verdict = CrossToolGuard(tolerance=0.1).compare(self.baseline(), self.baseline())
        assert "10.0% tolerance" in verdict.reason


class TestMinOpportunities:
    def test_thin_tools_are_ignored_and_listed(self):
        baseline = report(
            [("read_file", "read_file")] * 4 + [("terminal", "terminal")],
            tools=["read_file", "terminal"],
        )
        candidate = report(
            [("read_file", "read_file")] * 4 + [("terminal", "read_file")],
            tools=["read_file", "terminal"],
        )
        guard = CrossToolGuard(min_opportunities=3)
        verdict = guard.compare(baseline, candidate)
        assert verdict.accepted
        assert "terminal (1 example(s))" in verdict.ignored

    def test_the_same_drop_is_caught_at_the_default_threshold(self):
        baseline = report(
            [("read_file", "read_file")] * 4 + [("terminal", "terminal")],
            tools=["read_file", "terminal"],
        )
        candidate = report(
            [("read_file", "read_file")] * 4 + [("terminal", "read_file")],
            tools=["read_file", "terminal"],
        )
        assert not CrossToolGuard().compare(baseline, candidate).accepted

    def test_tools_with_no_examples_are_listed_plainly(self):
        built = report([("read_file", "read_file")], tools=["read_file", "ghost"])
        verdict = CrossToolGuard().compare(built, built)
        assert "ghost" in verdict.ignored
        assert NO_TOOL in verdict.ignored


class TestVerdictReporting:
    def test_summary_reads_as_a_sentence(self):
        baseline = report([("read_file", "read_file")] * 2)
        candidate = report([("read_file", "terminal")] * 2)
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert verdict.summary().startswith("cross-tool REJECTED")
        assert "100.0% -> 0.0%" in verdict.summary()

    def test_accepted_summary(self):
        built = report([("read_file", "read_file")])
        assert CrossToolGuard().compare(built, built).summary().startswith("cross-tool accepted")

    def test_serialises_the_evidence(self):
        baseline = report([("read_file", "read_file")] * 2)
        candidate = report([("read_file", "terminal")] * 2)
        blob = CrossToolGuard().compare(baseline, candidate).to_dict()
        assert blob["accepted"] is False
        assert blob["regressions"][0]["tool"] == "read_file"
        assert blob["regressions"][0]["delta"] == -1.0
        assert blob["overall_delta"] == -1.0

    def test_passing_verdict_is_a_passing_gate(self):
        built = report([("read_file", "read_file")])
        gate = CrossToolGuard().compare(built, built).to_gate_result()
        assert gate.passed and gate.name == "cross_tool"
        assert gate.baseline == 1.0 and gate.score == 1.0

    def test_regression_describes_itself(self):
        baseline = report([("read_file", "read_file")] * 4, tools=["read_file"])
        candidate = report(
            [("read_file", "read_file")] * 2 + [("read_file", "terminal")] * 2,
            tools=["read_file"],
        )
        regression = CrossToolGuard().compare(baseline, candidate).regressions[0]
        assert regression.describe().startswith("read_file: 100.0% -> 50.0% (-50.0%")
        assert regression.to_dict()["opportunities"] == 4

    def test_a_tool_that_vanishes_from_the_candidate_counts_as_zero(self):
        baseline = report([("read_file", "read_file")] * 2, tools=["read_file"])
        candidate = CrossToolReport()
        verdict = CrossToolGuard().compare(baseline, candidate)
        assert not verdict.accepted
        assert verdict.regressions[0].candidate_rate == 0.0
