"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset-path datasets/skills/arxiv/
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintResult, ConstraintValidator
from evolution.core.dataset_builder import EvalDataset, GoldenDatasetLoader, SyntheticDatasetBuilder
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import LLMJudge, make_skill_fitness_metric
from evolution.core.lm import make_lm
from evolution.skills.skill_module import (
    SkillModule,
    extract_evolved_skill_text,
    find_skill,
    load_skill,
    reassemble_skill,
)

console = Console()


@contextmanager
def _temporary_file_contents(path: Path, contents: str):
    """Temporarily write a file and always restore its original contents."""

    original = path.read_text()
    try:
        path.write_text(contents)
        yield
    finally:
        path.write_text(original)


def _all_pass(results: list[ConstraintResult]) -> bool:
    return all(result.passed for result in results)


def _print_constraint_results(results: list[ConstraintResult]) -> None:
    for c in results:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if c.details and not c.passed:
            console.print(f"    [dim]{c.details}[/dim]")


def _fail(message: str, exit_code: int = 1) -> None:
    console.print(f"[red]✗ {message}[/red]")
    raise SystemExit(exit_code)


def _build_dataset(
    config: EvolutionConfig,
    skill_name: str,
    skill_raw: str,
    eval_source: str,
    dataset_path: Optional[str],
    judge_model: str,
) -> EvalDataset:
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
        return dataset

    if eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill_raw,
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=judge_model,
        )
        if not dataset.all_examples:
            _fail("No relevant examples found from session history")
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
        return dataset

    if eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(artifact_text=skill_raw, artifact_type="skill")
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
        return dataset

    if dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
        return dataset

    _fail("Specify --dataset-path or use --eval-source synthetic")


def _require_usable_dataset(dataset: EvalDataset) -> None:
    """Phase 1 must have train/val/holdout data before claiming improvement."""

    missing = []
    if not dataset.train:
        missing.append("train")
    if not dataset.val:
        missing.append("val")
    if not dataset.holdout:
        missing.append("holdout")
    if missing:
        _fail(
            "Evaluation dataset is missing required split(s): "
            + ", ".join(missing)
            + ". Phase 1 must prove improvement on a held-out set."
        )


def _adjust_size_limit_for_baseline(config: EvolutionConfig, baseline_text: str) -> None:
    """Avoid rejecting existing large skills while still enforcing growth limits."""

    required_limit = int(len(baseline_text) * (1 + config.max_prompt_growth))
    if required_limit > config.max_skill_size:
        console.print(
            "  [yellow]Increasing max_skill_size for this run from "
            f"{config.max_skill_size:,} to {required_limit:,} chars so the baseline "
            "skill is not rejected solely for already being large.[/yellow]"
        )
        config.max_skill_size = required_limit


def _compile_with_gepa_or_fallback(
    baseline_module: SkillModule,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    gepa_metric,
    scalar_metric,
    config: EvolutionConfig,
) -> tuple[dspy.Module, str]:
    """Run GEPA with the current DSPy 3.x API, falling back visibly if needed."""

    reflection_lm = make_lm(config.optimizer_model, config)
    try:
        optimizer = dspy.GEPA(
            metric=gepa_metric,
            max_full_evals=max(1, config.iterations),
            reflection_lm=reflection_lm,
        )
        return optimizer.compile(baseline_module, trainset=trainset, valset=valset), "GEPA"
    except Exception as e:
        console.print(
            "[yellow]WARNING: GEPA unavailable or failed "
            f"({type(e).__name__}: {e}); falling back to MIPROv2[/yellow]"
        )

    optimizer = dspy.MIPROv2(metric=scalar_metric, auto="light")
    # MIPROv2.compile prompts on stdin by default (requires_permission_to_run=True),
    # which raises EOFError / hangs in a non-interactive evolution run. The fallback
    # must stay fully automated.
    return (
        optimizer.compile(
            baseline_module,
            trainset=trainset,
            requires_permission_to_run=False,
        ),
        "MIPROv2",
    )


def _score_holdout(
    module: dspy.Module,
    examples: list[dspy.Example],
    judge: LLMJudge,
    fallback_skill_text: str,
    config: EvolutionConfig,
) -> list[float]:
    """Score a module on holdout examples with LLM-as-judge."""

    scores: list[float] = []
    for ex in examples:
        prediction = module(task_input=ex.task_input)
        skill_text = getattr(prediction, "skill_text", "") or fallback_skill_text
        score = judge.score(
            task_input=ex.task_input,
            expected_behavior=getattr(ex, "expected_behavior", ""),
            agent_output=getattr(prediction, "output", "") or "",
            skill_text=skill_text,
            artifact_size=len(skill_text),
            max_size=config.max_skill_size,
        )
        scores.append(score.composite)
    return scores


def _save_candidate(
    skill_name: str,
    timestamp: str,
    baseline_full: str,
    evolved_full: str,
    metrics: dict,
) -> Path:
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evolved_skill.md").write_text(evolved_full)
    (output_dir / "baseline_skill.md").write_text(baseline_full)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return output_dir


def _run_pytest_gate(
    validator: ConstraintValidator,
    hermes_repo: Path,
    skill_path: Path,
    evolved_full: str,
) -> ConstraintResult:
    console.print("\n[bold]Running hermes-agent test gate against temporary evolved skill[/bold]")
    with _temporary_file_contents(skill_path, evolved_full):
        return validator.run_test_suite(hermes_repo)


def _run_tblite_gate(
    config: EvolutionConfig,
    hermes_repo: Path,
    skill_path: Path,
    evolved_full: str,
) -> BenchmarkResult:
    console.print("\n[bold]Running TBLite benchmark gate[/bold]")
    gate = BenchmarkGate(config)
    try:
        if config.tblite_baseline_score is None:
            console.print("  Measuring baseline TBLite score before writing evolved skill")
            baseline_score = gate.run_tblite_score(hermes_repo)
        else:
            baseline_score = config.tblite_baseline_score
            console.print(f"  Using configured baseline TBLite score: {baseline_score:.3f}")

        with _temporary_file_contents(skill_path, evolved_full):
            evolved_score = gate.run_tblite_score(hermes_repo)
        return gate.compare_tblite_scores(baseline_score, evolved_score)
    except Exception as exc:
        return BenchmarkResult(
            passed=False,
            name="tblite",
            message=f"TBLite gate failed: {exc}",
        )


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: str = "openai/gpt-4.1",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    run_tblite: bool = False,
    tblite_command: Optional[str] = None,
    tblite_baseline_score: Optional[float] = None,
    min_skill_improvement: float = 0.10,
    allow_heuristic_fallback: bool = False,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        run_pytest=run_tests,
        run_tblite=run_tblite,
        min_skill_improvement=min_skill_improvement,
        allow_heuristic_fallback=allow_heuristic_fallback,
    )
    if tblite_command is not None:
        config.tblite_command = tblite_command
    if tblite_baseline_score is not None:
        config.tblite_baseline_score = tblite_baseline_score

    console.print(
        f"\n[bold cyan]Hermes Agent Self-Evolution[/bold cyan] — "
        f"Evolving skill: [bold]{skill_name}[/bold]\n"
    )

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        _fail(f"Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}")

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    _adjust_size_limit_for_baseline(config, skill["raw"])

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} full evals)")
        console.print("  Would validate constraints, holdout score, and configured gates")
        return

    dataset = _build_dataset(config, skill_name, skill["raw"], eval_source, dataset_path, judge_model)
    console.print(
        f"  Split: {len(dataset.train)} train / "
        f"{len(dataset.val)} val / {len(dataset.holdout)} holdout"
    )
    _require_usable_dataset(dataset)

    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    _print_constraint_results(baseline_constraints)
    if not _all_pass(baseline_constraints):
        _fail("Baseline skill failed constraints; refusing to optimize from an invalid baseline")

    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} full evals; MIPROv2 fallback)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval/student model: {eval_model}")
    console.print(f"  Judge/dataset model: {judge_model}")
    console.print(f"  Promotion threshold: ≥{min_skill_improvement:.1%} relative holdout lift")

    student_lm = make_lm(config.eval_model, config)
    dspy.configure(lm=student_lm)

    baseline_module = SkillModule(skill["body"])
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")
    gepa_metric = make_skill_fitness_metric(
        config, fallback_skill_text=skill["body"], return_feedback=True
    )
    scalar_metric = make_skill_fitness_metric(
        config, fallback_skill_text=skill["body"], return_feedback=False
    )

    console.print(f"\n[bold cyan]Running optimization ({iterations} full evals)...[/bold cyan]\n")
    start_time = time.time()
    optimized_module, optimizer_name = _compile_with_gepa_or_fallback(
        baseline_module=baseline_module,
        trainset=trainset,
        valset=valset,
        gepa_metric=gepa_metric,
        scalar_metric=scalar_metric,
        config=config,
    )
    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed with {optimizer_name} in {elapsed:.1f}s")

    evolved_body = extract_evolved_skill_text(optimized_module, fallback=skill["body"])
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    _print_constraint_results(evolved_constraints)
    if not _all_pass(evolved_constraints):
        console.print("[red]✗ Evolved skill FAILED constraints — not promoting[/red]")
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    holdout_examples = dataset.to_dspy_examples("holdout")
    judge = LLMJudge(config)
    baseline_scores = _score_holdout(baseline_module, holdout_examples, judge, skill["body"], config)
    evolved_scores = _score_holdout(optimized_module, holdout_examples, judge, evolved_body, config)

    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_evolved = sum(evolved_scores) / len(evolved_scores)
    improvement = avg_evolved - avg_baseline
    relative_improvement = improvement / max(0.001, avg_baseline)
    promotion_passed = avg_evolved > avg_baseline and relative_improvement >= config.min_skill_improvement

    test_result: Optional[ConstraintResult] = None
    benchmark_result: Optional[BenchmarkResult] = None

    if promotion_passed and config.run_pytest:
        test_result = _run_pytest_gate(validator, config.hermes_agent_path, skill_path, evolved_full)
        _print_constraint_results([test_result])
        if not test_result.passed:
            promotion_passed = False
            console.print("[red]✗ Evolved skill FAILED test gate — not promoting[/red]")

    if promotion_passed and config.run_tblite:
        benchmark_result = _run_tblite_gate(config, config.hermes_agent_path, skill_path, evolved_full)
        color = "green" if benchmark_result.passed else "red"
        icon = "✓" if benchmark_result.passed else "✗"
        console.print(f"  [{color}]{icon} {benchmark_result.name}[/{color}]: {benchmark_result.message}")
        if not benchmark_result.passed:
            promotion_passed = False
            console.print("[red]✗ Evolved skill FAILED benchmark gate — not promoting[/red]")

    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f} ({relative_improvement:+.1%})[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Optimizer", "", optimizer_name, "")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")
    table.add_row("Promotion Gate", "", "PASS" if promotion_passed else "FAIL", "")
    console.print()
    console.print(table)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer": optimizer_name,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "min_skill_improvement": config.min_skill_improvement,
        "promotion_passed": promotion_passed,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "max_skill_size_used": config.max_skill_size,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": _all_pass(evolved_constraints),
        "pytest_gate_ran": test_result is not None,
        "pytest_gate_passed": None if test_result is None else test_result.passed,
        "tblite_gate_ran": benchmark_result is not None,
        "tblite_gate_passed": None if benchmark_result is None else benchmark_result.passed,
        "tblite_baseline_score": None if benchmark_result is None else benchmark_result.baseline_score,
        "tblite_evolved_score": None if benchmark_result is None else benchmark_result.evolved_score,
    }
    output_dir = _save_candidate(skill_name, timestamp, skill["raw"], evolved_full, metrics)

    console.print(f"\n  Output saved to {output_dir}/")
    if promotion_passed:
        console.print(
            "\n[bold green]✓ Phase-1 promotion gate passed: "
            f"{relative_improvement:+.1%} relative holdout lift[/bold green]"
        )
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(
            "\n[yellow]⚠ Phase-1 promotion gate did not pass "
            f"(requires ≥{config.min_skill_improvement:.0%} relative holdout lift and all enabled gates)[/yellow]"
        )
        console.print("  Candidate saved for inspection but should not be promoted.")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA full evals")
@click.option(
    "--eval-source",
    default="synthetic",
    type=click.Choice(["synthetic", "golden", "sessiondb"]),
    help="Source for evaluation dataset",
)
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL or split dir)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Student model used to execute candidate skills")
@click.option("--judge-model", default="openai/gpt-4.1", help="Model for LLM-as-judge scoring and dataset generation")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as promotion gate")
@click.option("--run-tblite", is_flag=True, help="Run configured TBLite benchmark as promotion gate")
@click.option("--tblite-command", default=None, help="Command that prints TBLite JSON score/pass_rate")
@click.option("--tblite-baseline-score", default=None, type=float, help="Optional precomputed baseline TBLite score")
@click.option("--min-skill-improvement", default=0.10, type=float, help="Required relative holdout lift")
@click.option("--allow-heuristic-fallback", is_flag=True, help="Allow lexical metric fallback if LLM judge fails")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(
    skill,
    iterations,
    eval_source,
    dataset_path,
    optimizer_model,
    eval_model,
    judge_model,
    hermes_repo,
    run_tests,
    run_tblite,
    tblite_command,
    tblite_baseline_score,
    min_skill_improvement,
    allow_heuristic_fallback,
    dry_run,
):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""

    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        run_tblite=run_tblite,
        tblite_command=tblite_command,
        tblite_baseline_score=tblite_baseline_score,
        min_skill_improvement=min_skill_improvement,
        allow_heuristic_fallback=allow_heuristic_fallback,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
