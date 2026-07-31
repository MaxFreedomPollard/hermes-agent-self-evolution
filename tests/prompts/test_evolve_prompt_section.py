"""Tests for the Phase 3 entry point: gate ladder, CLI, and write-back.

Offline throughout. The optimizer, the harness, and the benchmark runners are
all replaced with stubs, which is the only way to exercise the accept/reject
logic that matters here - the decision about whether a prompt change is allowed
to reach disk is the product, and it has to be testable without an LM.
"""

import json

import dspy
import pytest
from click.testing import CliRunner

from evolution.core.gates import GateChain, GateResult, GateStatus
from evolution.prompts import evolve_prompt_section as ep
from evolution.prompts.behavioral_eval import (
    BehavioralOutcome,
    BehavioralReport,
    BehavioralSuite,
)
from evolution.prompts.sections import (
    CACHE_BLOCK_TOKENS,
    ActiveSessionReport,
    EvolvableSection,
    SectionInventory,
    constant_strings,
    load_sections,
)

PROMPT_SOURCE = '''"""Prompt builder."""

CONTEXT_FILE_MAX_CHARS = 20_000

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, created by Nous Research. You are helpful and "
    "direct, and you admit uncertainty rather than guessing."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts with the "
    "memory tool: preferences, environment details, stable conventions.\\n"
    "Do NOT save PR numbers, commit SHAs, or completed-work logs."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references a past conversation, use session_search first."
)

SKILLS_GUIDANCE = (
    "Save a non-trivial workflow as a skill. Patch an outdated skill on sight."
)

KANBAN_GUIDANCE = (
    "# Kanban protocol\\n"
    "Leave me alone."
)

PLATFORM_HINTS = {
    "cli": "You are a CLI AI Agent. Try not to use markdown.",
}
'''


@pytest.fixture(autouse=True)
def restore_dspy_settings():
    """evolve() configures a global default LM. Put the old one back.

    No LM is ever called here, but leaving a configured default behind would
    change how unrelated test modules behave depending on run order.
    """
    previous = dspy.settings.lm
    yield
    dspy.configure(lm=previous)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(PROMPT_SOURCE, encoding="utf-8")
    (tmp_path / "batch_runner.py").write_text("# fake runner\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def builder(repo):
    return repo / "agent" / "prompt_builder.py"


def gate(name, status, message="", score=None, baseline=None):
    return GateResult(name=name, status=status, message=message, score=score, baseline=baseline)


def flat(output: str) -> str:
    """Collapse rich's console wrapping so assertions can span line breaks."""
    return " ".join(output.split())


# ──────────────────────────────────────────────────────────────────────────
# Gate ladder
# ──────────────────────────────────────────────────────────────────────────


class TestGateLadder:
    def test_zero_tolerance_is_the_phase_default(self):
        assert ep.ZERO_REGRESSION_TOLERANCE == 0.0

    def test_unavailable_benchmarks_pass_permissively(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep,
            "run_benchmark_gate",
            lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "not found"),
        )
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "Short guidance."})
        assert chain.passed
        assert [r.status for r in chain.results][1:] == [
            GateStatus.UNAVAILABLE,
            GateStatus.UNAVAILABLE,
        ]

    def test_strict_mode_blocks_on_an_unavailable_benchmark(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep,
            "run_benchmark_gate",
            lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "not found"),
        )
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(
            repo, {"MEMORY_GUIDANCE": "Short guidance."}, strict=True
        )
        assert not chain.passed
        assert chain.blockers[0].status is GateStatus.UNAVAILABLE

    def test_zero_tolerance_rejects_a_small_regression(self, repo, monkeypatch):
        calls = {"n": 0}

        def fake_benchmark(repo_arg, name, baseline=None, regression_threshold=0.02, **kw):
            calls["n"] += 1
            if baseline is None:
                return gate(name, GateStatus.PASSED, "baseline", score=0.80)
            score = 0.79
            if score - baseline < -abs(regression_threshold):
                return gate(name, GateStatus.FAILED, "regressed", score=score, baseline=baseline)
            return gate(name, GateStatus.PASSED, "held", score=score, baseline=baseline)

        monkeypatch.setattr(ep, "run_benchmark_gate", fake_benchmark)
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "Short."})
        assert not chain.passed
        assert chain.blockers[0].name == "tblite"

    def test_a_one_percent_drop_would_pass_at_the_old_two_percent_tolerance(
        self, repo, monkeypatch
    ):
        def fake_benchmark(repo_arg, name, baseline=None, regression_threshold=0.02, **kw):
            if baseline is None:
                return gate(name, GateStatus.PASSED, "baseline", score=0.80)
            score = 0.79
            if score - baseline < -abs(regression_threshold):
                return gate(name, GateStatus.FAILED, "regressed", score=score)
            return gate(name, GateStatus.PASSED, "held", score=score)

        monkeypatch.setattr(ep, "run_benchmark_gate", fake_benchmark)
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        chain = ep.run_gate_ladder(
            repo, {"MEMORY_GUIDANCE": "Short."}, regression_threshold=0.02
        )
        assert chain.passed

    def test_a_failing_pytest_gate_stops_the_ladder(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.FAILED, "3 failed")
        )
        monkeypatch.setattr(
            ep,
            "run_benchmark_gate",
            lambda *a, **k: pytest.fail("benchmarks must not run after pytest fails"),
        )
        chain = ep.run_gate_ladder(
            repo, {"MEMORY_GUIDANCE": "Short."}, benchmarks=()
        )
        assert not chain.passed

    def test_the_candidate_is_staged_while_gates_run(self, repo, builder, monkeypatch):
        seen = {}

        def spy_pytest(repo_arg, **kwargs):
            seen["source"] = builder.read_text(encoding="utf-8")
            return gate("pytest", GateStatus.PASSED, "ok")

        monkeypatch.setattr(ep, "run_pytest_gate", spy_pytest)
        monkeypatch.setattr(
            ep, "run_benchmark_gate", lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "-")
        )
        original = builder.read_text(encoding="utf-8")
        ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "STAGED CANDIDATE TEXT."})
        assert "STAGED CANDIDATE TEXT." in seen["source"]
        assert builder.read_text(encoding="utf-8") == original

    def test_a_backup_is_left_behind(self, repo, builder, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: gate("pytest", GateStatus.PASSED, "ok")
        )
        monkeypatch.setattr(
            ep, "run_benchmark_gate", lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "-")
        )
        backup = tmp_path / "out" / "prompt_builder.py.bak"
        original = builder.read_text(encoding="utf-8")
        ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "STAGED."}, backup_path=backup)
        assert backup.read_text(encoding="utf-8") == original

    def test_pytest_gate_can_be_skipped(self, repo, monkeypatch):
        monkeypatch.setattr(
            ep, "run_pytest_gate", lambda *a, **k: pytest.fail("pytest should be skipped")
        )
        monkeypatch.setattr(
            ep, "run_benchmark_gate", lambda *a, **k: gate(a[1], GateStatus.UNAVAILABLE, "-")
        )
        chain = ep.run_gate_ladder(repo, {"MEMORY_GUIDANCE": "S."}, run_pytest=False)
        assert chain.passed and len(chain.results) == 2

    def test_default_pytest_subset_is_narrow(self):
        assert "-k" in ep.DEFAULT_PYTEST_SUBSET
        assert "prompt" in ep.DEFAULT_PYTEST_SUBSET


# ──────────────────────────────────────────────────────────────────────────
# SectionOutcome bookkeeping
# ──────────────────────────────────────────────────────────────────────────


class TestSectionOutcome:
    def test_growth_and_improvement_maths(self):
        outcome = ep.SectionOutcome(
            name="MEMORY_GUIDANCE",
            baseline_text="y" * 100,
            evolved_text="y" * 110,
            baseline_score=0.4,
            evolved_score=0.6,
        )
        assert outcome.growth == pytest.approx(0.10)
        assert outcome.improvement == pytest.approx(0.2)
        assert outcome.holdout_improvement is None

    def test_holdout_improvement_when_measured(self):
        outcome = ep.SectionOutcome(
            name="M", baseline_text="a", evolved_text="b",
            holdout_baseline=0.5, holdout_evolved=0.75,
        )
        assert outcome.holdout_improvement == pytest.approx(0.25)

    def test_to_dict_is_json_serialisable(self):
        outcome = ep.SectionOutcome(name="M", baseline_text="a", evolved_text="bb")
        json.dumps(outcome.to_dict())


# ──────────────────────────────────────────────────────────────────────────
# CLI argument handling
# ──────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_dry_run_touches_nothing(self, repo, builder):
        before = builder.read_text(encoding="utf-8")
        result = CliRunner().invoke(
            ep.main,
            ["--section", "MEMORY_GUIDANCE", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in flat(result.output)
        assert builder.read_text(encoding="utf-8") == before

    def test_dry_run_reports_the_zero_tolerance_setting(self, repo):
        result = CliRunner().invoke(
            ep.main,
            ["--all-sections", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert "0% regression tolerance" in flat(result.output)

    def test_dry_run_states_the_next_session_rule(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        assert "NEXT session" in flat(result.output)

    def test_platform_hints_is_refused_with_an_explanation(self, repo):
        result = CliRunner().invoke(
            ep.main,
            ["--section", "PLATFORM_HINTS", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert result.exit_code == 1
        assert "not an evolvable prompt section" in flat(result.output)
        assert "not a plain string" in flat(result.output)

    def test_unknown_section_is_refused(self, repo):
        result = CliRunner().invoke(
            ep.main,
            ["--section", "KANBAN_GUIDANCE", "--hermes-repo", str(repo), "--dry-run"],
        )
        assert result.exit_code == 1
        assert "KANBAN_GUIDANCE" in flat(result.output)

    def test_no_target_is_refused(self, repo):
        result = CliRunner().invoke(ep.main, ["--hermes-repo", str(repo), "--dry-run"])
        assert result.exit_code == 1
        assert "--all-sections" in flat(result.output)

    def test_missing_prompt_builder_is_refused(self, tmp_path):
        empty = tmp_path / "empty-repo"
        empty.mkdir()
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(empty), "--dry-run"]
        )
        assert result.exit_code == 1
        assert "No evolvable prompt sections" in flat(result.output)

    def test_all_sections_lists_every_discovered_section(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        for name in (
            "DEFAULT_AGENT_IDENTITY",
            "MEMORY_GUIDANCE",
            "SESSION_SEARCH_GUIDANCE",
            "SKILLS_GUIDANCE",
        ):
            assert name in flat(result.output)

    def test_platform_hints_is_reported_as_out_of_scope(self, repo):
        result = CliRunner().invoke(
            ep.main, ["--all-sections", "--hermes-repo", str(repo), "--dry-run"]
        )
        assert "PLATFORM_HINTS" in flat(result.output)
        assert "out of scope for this phase" in flat(result.output)

    def test_write_defaults_to_off(self):
        params = {p.name: p for p in ep.main.params}
        assert params["write"].default is False


# ──────────────────────────────────────────────────────────────────────────
# Full run with stubbed optimizer, harness, and gates
# ──────────────────────────────────────────────────────────────────────────


EVOLVED_MARKER = "Prefer facts that survive a week."


@pytest.fixture
def stubbed(monkeypatch):
    """Replace everything that would need a model or a benchmark."""

    def fake_optimize(section_name, baseline_text, trainset, valset, iterations, optimizer_model):
        return f"{baseline_text} {EVOLVED_MARKER}", "stub"

    def fake_evaluate(self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None):
        targets = list(scenarios if scenarios is not None else self.scenarios)
        score = 0.9 if EVOLVED_MARKER in system_prompt else 0.4
        return BehavioralReport(
            outcomes=[
                BehavioralOutcome(
                    scenario_id=s.scenario_id,
                    category=s.category,
                    section=s.section_under_test,
                    score=score,
                    passed=score >= 0.6,
                )
                for s in targets
            ],
            harness="stub",
        )

    def fake_gates(**kwargs):
        return GateChain(strict=kwargs.get("strict", False)).run(
            gate("pytest", GateStatus.PASSED, "ok")
        )

    monkeypatch.setattr(ep, "_optimize_section", fake_optimize)
    monkeypatch.setattr(BehavioralSuite, "evaluate", fake_evaluate)
    monkeypatch.setattr(ep, "run_gate_ladder", fake_gates)
    monkeypatch.setattr(
        ep, "detect_active_session", lambda *a, **k: ActiveSessionReport(active=False)
    )
    return monkeypatch


class TestFullRun:
    def test_no_write_by_default(self, repo, builder, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_write_applies_the_evolved_section(self, repo, builder, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert EVOLVED_MARKER in load_sections(repo).get("MEMORY_GUIDANCE").baseline_text

    def test_write_leaves_neighbouring_constants_untouched(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        before = constant_strings(builder.read_text(encoding="utf-8"))
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True)
        after = constant_strings(builder.read_text(encoding="utf-8"))
        changed = [k for k in before if after[k] != before[k]]
        assert changed == ["MEMORY_GUIDANCE"]
        assert after["KANBAN_GUIDANCE"] == before["KANBAN_GUIDANCE"]
        assert after["DEFAULT_AGENT_IDENTITY"] == before["DEFAULT_AGENT_IDENTITY"]

    def test_an_active_session_refuses_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ep,
            "detect_active_session",
            lambda *a, **k: ActiveSessionReport(
                active=True, evidence=("HERMES_SESSION_ID=abc",)
            ),
        )
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 2
        assert builder.read_text(encoding="utf-8") == before

    def test_a_blocked_gate_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ep,
            "run_gate_ladder",
            lambda **kwargs: GateChain().run(
                gate("tblite", GateStatus.FAILED, "regressed -3%")
            ),
        )
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_a_constraint_violation_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                baseline_text * 4,
                "stub",
            ),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_identity_trait_loss_prevents_the_write(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                "You are a bot. Do tasks.",
                "stub",
            ),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["DEFAULT_AGENT_IDENTITY"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_metrics_and_artifacts_are_saved(self, repo, stubbed, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo))
        runs = sorted((tmp_path / "output" / "prompts").iterdir())
        assert runs
        metrics = json.loads((runs[-1] / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["regression_threshold"] == 0.0
        assert metrics["sections"][0]["name"] == "MEMORY_GUIDANCE"
        assert (runs[-1] / "evolved_MEMORY_GUIDANCE.txt").exists()
        assert (runs[-1] / "baseline_MEMORY_GUIDANCE.txt").exists()
        assert (runs[-1] / "scenarios.jsonl").exists()

    def _block_crossing_inventory(self, repo):
        """An inventory whose baseline sits one edit away from a new cache block."""
        baseline = "m" * (CACHE_BLOCK_TOKENS * 4 - 100)
        return SectionInventory(
            prompt_builder=repo / "agent" / "prompt_builder.py",
            sections=[
                EvolvableSection(
                    name="MEMORY_GUIDANCE",
                    path=repo / "agent" / "prompt_builder.py",
                    baseline_text=baseline,
                    span=None,
                )
            ],
        )

    def test_strict_mode_rejects_a_cache_block_crossing(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        inventory = self._block_crossing_inventory(repo)
        monkeypatch.setattr(ep, "load_sections", lambda *a, **k: inventory)
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                "m" * int(len(baseline_text) * 1.2),
                "stub",
            ),
        )
        monkeypatch.chdir(tmp_path)
        before = builder.read_text(encoding="utf-8")
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(repo),
            write=True,
            strict_gates=True,
        )
        assert code == 0
        assert builder.read_text(encoding="utf-8") == before

    def test_permissive_mode_accepts_the_same_crossing(
        self, repo, builder, stubbed, tmp_path, monkeypatch
    ):
        inventory = self._block_crossing_inventory(repo)
        grown = "m" * int(len(inventory.get("MEMORY_GUIDANCE").baseline_text) * 1.2)
        monkeypatch.setattr(ep, "load_sections", lambda *a, **k: inventory)
        monkeypatch.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, trainset, valset, iterations, optimizer_model: (
                grown,
                "stub",
            ),
        )

        def score_by_length(self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None):
            targets = list(scenarios if scenarios is not None else self.scenarios)
            score = 0.9 if grown in system_prompt else 0.4
            return BehavioralReport(
                outcomes=[
                    BehavioralOutcome(
                        s.scenario_id, s.category, s.section_under_test, score, score >= 0.6
                    )
                    for s in targets
                ]
            )

        monkeypatch.setattr(BehavioralSuite, "evaluate", score_by_length)
        monkeypatch.chdir(tmp_path)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), write=True
        )
        assert code == 0
        assert load_sections(repo).get("MEMORY_GUIDANCE").baseline_text == grown

    def test_strict_mode_is_carried_into_the_ladder(self, repo, stubbed, tmp_path, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return GateChain(strict=kwargs["strict"]).run(
                gate("pytest", GateStatus.PASSED, "ok")
            )

        monkeypatch.setattr(ep, "run_gate_ladder", spy)
        monkeypatch.chdir(tmp_path)
        ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(repo), strict_gates=True
        )
        assert seen["strict"] is True
        assert set(seen["updates"]) == {"MEMORY_GUIDANCE"}
