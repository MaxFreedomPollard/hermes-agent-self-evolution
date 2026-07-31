"""Evolve a hermes-agent system prompt section with DSPy + GEPA.

Usage:
    python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE
    python -m evolution.prompts.evolve_prompt_section --all-sections --iterations 5 --strict-gates
    python -m evolution.prompts.evolve_prompt_section --section SKILLS_GUIDANCE --write

This is the riskiest tier in PLAN.md, and the code is shaped by that rather
than by the optimizer. A skill affects the sessions that load it. A tool
description affects the sessions that see the tool. A system prompt section is
in front of every turn of every session, so the interesting questions here are
all about what the run refuses to do:

* Only the four allowlisted string constants can be targeted. ``PLATFORM_HINTS``
  is reported and skipped, because in the real ``agent/prompt_builder.py`` it is
  a dict of per-platform strings with its own accuracy rules.
* A candidate that grows a section past +20%, drops a core identity trait, or
  blows the prompt caching budget is rejected before it is ever scored.
* Benchmark regression tolerance is 0.0, not the 2% Phases 2 and 3 use
  elsewhere. PLAN.md is explicit: "zero tolerance for regression here."
  ``--strict-gates`` goes one step further and treats a benchmark that could
  not be run, or a cache block boundary crossing, as blocking - which is the
  honest setting given that none of PLAN.md's benchmarks exist in hermes-agent
  today.
* Writing is off by default, and even with ``--write`` the run refuses while a
  Hermes session looks live. A running session has already assembled and cached
  its system prompt; an evolved section deploys on the NEXT session and is never
  hot-swapped into a running one.

Nothing here mutates the hermes-agent working tree except the explicit
write-back step and the gate ladder's staged write, which restores the original
in a ``finally`` and leaves a backup in the run's output directory first.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.artifact_io import EVOLVABLE_PROMPT_SECTIONS
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.gates import GateChain, run_benchmark_gate, run_pytest_gate
from evolution.prompts.behavioral_eval import (
    BehavioralJudge,
    BehavioralReport,
    BehavioralSuite,
    SectionBehaviorModule,
    make_behavioral_metric,
    scenarios_to_dspy_examples,
    select_harness,
)
from evolution.prompts.sections import (
    NEXT_SESSION_NOTICE,
    SectionValidation,
    UnknownSection,
    detect_active_session,
    load_sections,
    staged_prompt_write,
    validate_section_names,
    write_sections,
)

console = Console()

# PLAN.md: "Benchmarks hold or improve (zero tolerance for regression here)."
ZERO_REGRESSION_TOLERANCE = 0.0

# Narrow slice of hermes-agent's suite. The full suite is 2550+ tests and none
# of the rest can be affected by prompt text, so the affordable subset is the
# one that reads prompt_builder.
DEFAULT_PYTEST_SUBSET = ("tests/", "-k", "prompt")

DEFAULT_BENCHMARKS = ("tblite", "yc_bench")


@dataclass
class SectionOutcome:
    """What happened to one section during a run."""

    name: str
    baseline_text: str
    evolved_text: str
    baseline_score: float = 0.0
    evolved_score: float = 0.0
    holdout_baseline: Optional[float] = None
    holdout_evolved: Optional[float] = None
    validation: Optional[SectionValidation] = None
    accepted: bool = False
    reason: str = ""
    elapsed_s: float = 0.0
    optimizer: str = ""

    @property
    def improvement(self) -> float:
        return self.evolved_score - self.baseline_score

    @property
    def holdout_improvement(self) -> Optional[float]:
        if self.holdout_baseline is None or self.holdout_evolved is None:
            return None
        return self.holdout_evolved - self.holdout_baseline

    @property
    def growth(self) -> float:
        base = max(1, len(self.baseline_text))
        return (len(self.evolved_text) - len(self.baseline_text)) / base

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline_chars": len(self.baseline_text),
            "evolved_chars": len(self.evolved_text),
            "growth": round(self.growth, 4),
            "baseline_score": round(self.baseline_score, 4),
            "evolved_score": round(self.evolved_score, 4),
            "improvement": round(self.improvement, 4),
            "holdout_baseline": self.holdout_baseline,
            "holdout_evolved": self.holdout_evolved,
            "accepted": self.accepted,
            "reason": self.reason,
            "elapsed_s": round(self.elapsed_s, 2),
            "optimizer": self.optimizer,
            "validation": self.validation.to_dict() if self.validation else None,
        }


def _banner(text: str) -> None:
    console.print(f"\n[bold]── {text} " + "─" * max(0, 58 - len(text)) + "[/bold]")


def run_gate_ladder(
    hermes_repo: Path,
    updates: dict[str, str],
    strict: bool = False,
    python: Optional[str] = None,
    benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
    regression_threshold: float = ZERO_REGRESSION_TOLERANCE,
    pytest_subset: Sequence[str] = DEFAULT_PYTEST_SUBSET,
    backup_path: Optional[Path] = None,
    fast: bool = True,
    run_pytest: bool = True,
) -> GateChain:
    """Run the validation ladder against the candidate sections.

    The candidate has to be on disk for pytest or a benchmark to see it - there
    is no ephemeral-prompt channel for either - so the sections are staged, the
    gates run, and the original file is restored no matter how the block exits.
    A copy of the untouched file is written to *backup_path* first.

    Benchmark baselines are measured before staging, then re-measured with the
    candidate in place and compared at *regression_threshold*, which this phase
    sets to 0.0. In today's hermes-agent every benchmark reports UNAVAILABLE;
    that is a pass in permissive mode and a blocker under ``strict``.
    """
    baselines: dict[str, Optional[float]] = {}
    for name in benchmarks:
        result = run_benchmark_gate(hermes_repo, name, fast=fast)
        baselines[name] = result.score

    chain = GateChain(strict=strict)
    with staged_prompt_write(
        hermes_repo, updates, enabled=bool(updates), backup_path=backup_path
    ):
        gates = []
        if run_pytest:
            gates.append(
                partial(
                    run_pytest_gate,
                    hermes_repo,
                    subset=list(pytest_subset),
                    python=python,
                )
            )
        for name in benchmarks:
            gates.append(
                partial(
                    run_benchmark_gate,
                    hermes_repo,
                    name,
                    baseline=baselines[name],
                    regression_threshold=regression_threshold,
                    fast=fast,
                )
            )
        chain.run(*gates)

    return chain


def _optimize_section(
    section_name: str,
    baseline_text: str,
    trainset,
    valset,
    iterations: int,
    optimizer_model: str,
) -> tuple[str, str]:
    """Run GEPA over one section, falling back to MIPROv2.

    Returns the evolved text and the name of the optimizer that produced it.
    The section lives in the signature's instructions, so instruction mutation
    is section mutation and the evolved text reads straight back out.
    """
    module = SectionBehaviorModule(baseline_text, section_name)
    metric = make_behavioral_metric()

    try:
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=max(1, iterations) * max(1, len(trainset)),
            reflection_lm=dspy.LM(optimizer_model),
            track_stats=False,
        )
        optimized = optimizer.compile(module, trainset=trainset, valset=valset)
        return optimized.section_text, "GEPA"
    except Exception as exc:  # noqa: BLE001 - GEPA availability varies by dspy build
        console.print(f"  [yellow]GEPA unavailable ({exc}), falling back to MIPROv2[/yellow]")

    optimizer = dspy.MIPROv2(metric=metric, auto="light")
    optimized = optimizer.compile(module, trainset=trainset)
    return optimized.section_text, "MIPROv2"


def evolve(
    section_names: Sequence[str] = (),
    all_sections: bool = False,
    iterations: int = 5,
    hermes_repo: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    strict_gates: bool = False,
    dry_run: bool = False,
    write: bool = False,
    max_turns: int = 6,
    num_workers: int = 4,
    seed: int = 0,
) -> int:
    """Optimize one or more system prompt sections. Returns a process exit code."""

    console.print(
        "\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] "
        "- Phase 3: system prompt sections\n"
    )

    # ── 1. Resolve the repo and discover sections ───────────────────────
    _banner("Discovering sections")
    try:
        repo = resolve_hermes_agent_path(hermes_repo)
    except FileNotFoundError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    config = EvolutionConfig(
        hermes_agent_path=repo,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )

    try:
        inventory = load_sections(repo, max_growth=config.max_prompt_growth)
    except UnknownSection as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    if not inventory.sections:
        console.print(
            f"[red]✗ No evolvable prompt sections found under {repo}[/red]\n"
            f"  Expected string constants "
            f"({', '.join(EVOLVABLE_PROMPT_SECTIONS)}) in agent/prompt_builder.py"
        )
        return 1

    console.print(f"  Repo: {repo}")
    console.print(f"  File: {inventory.prompt_builder}")
    for section in inventory:
        console.print(
            f"  [green]✓[/green] {section.name}: {section.baseline_size:,} chars "
            f"(~{section.baseline_tokens:,} tokens, ceiling {section.max_chars:,})"
        )
    for missing in inventory.missing:
        console.print(f"  [yellow]○ {missing}: not found in this checkout[/yellow]")
    for structured in inventory.structured:
        console.print(f"  [yellow]○ {structured.name}: {structured.reason}[/yellow]")

    console.print(
        f"  Assembled prefix: ~{inventory.estimated_tokens():,} tokens "
        f"in {inventory.cache_blocks()} cache block(s) of "
        f"{inventory.cache_budget_tokens:,} budgeted"
    )

    # ── 2. Select targets ────────────────────────────────────────────────
    requested = list(section_names)
    if all_sections:
        requested = list(inventory.names)
    if not requested:
        console.print(
            "[red]✗ Nothing to do: pass --section NAME (repeatable) or --all-sections[/red]"
        )
        return 1

    unknown = validate_section_names(requested)
    if unknown:
        for name in unknown:
            hint = ""
            if name in {s.name for s in inventory.structured}:
                hint = " (present in prompt_builder.py, but not a plain string)"
            console.print(f"[red]✗ {name} is not an evolvable prompt section{hint}[/red]")
        console.print(f"  Allowed: {', '.join(EVOLVABLE_PROMPT_SECTIONS)}")
        return 1

    targets = []
    for name in requested:
        try:
            targets.append(inventory.get(name))
        except UnknownSection as exc:
            console.print(f"[red]✗ {exc}[/red]")
            return 1

    console.print(f"\n  Targeting: {', '.join(t.name for t in targets)}")

    # ── 3. Behavioral scenarios ──────────────────────────────────────────
    _banner("Building behavioral suite")
    target_names = [t.name for t in targets]
    suite = BehavioralSuite.from_seeds(
        sections=target_names + ["PLATFORM_HINTS"],
        include_platform=True,
    )
    train, val, holdout = suite.split(
        train_ratio=config.train_ratio, val_ratio=config.val_ratio, seed=seed
    )
    console.print(f"  Scenarios: {len(suite)} across {len(suite.categories())} categories")
    for category, count in suite.categories().items():
        console.print(f"    {category}: {count}")
    console.print(f"  Split: {len(train)} train / {len(val)} val / {len(holdout)} holdout")

    harness, harness_reason = select_harness(
        repo, model=eval_model, max_turns=max_turns, num_workers=num_workers
    )
    console.print(f"  Harness: {harness_reason}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / "prompts" / timestamp

    if dry_run:
        _banner("Dry run")
        console.print("  Setup validated. Nothing was optimized and nothing was written.")
        console.print(f"  Would optimize: {', '.join(target_names)}")
        console.print(f"  Would run {iterations} iteration(s) with {optimizer_model}")
        console.print(
            f"  Would gate on pytest + {', '.join(DEFAULT_BENCHMARKS)} at "
            f"{ZERO_REGRESSION_TOLERANCE:.0%} regression tolerance"
            f"{' (strict)' if strict_gates else ''}"
        )
        console.print(f"  Would write output to {output_dir}/")
        console.print(f"  [dim]{NEXT_SESSION_NOTICE}[/dim]")
        return 0

    # ── 4. Baseline behaviour ────────────────────────────────────────────
    _banner("Baseline behaviour")
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)
    fast_judge = BehavioralJudge(use_llm=False)

    baseline_prompt = inventory.assembled_prompt()
    baseline_reports: dict[str, BehavioralReport] = {}
    for section in targets:
        scenarios = [s for s in val if s.section_under_test == section.name] or [
            s for s in suite.for_section(section.name)
        ]
        report = suite.evaluate(
            baseline_prompt,
            harness,
            judge=fast_judge,
            section_name=section.name,
            run_name=f"hase-baseline-{section.name.lower()}-{timestamp}",
            scenarios=scenarios,
        )
        baseline_reports[section.name] = report
        console.print(
            f"  {section.name}: {report.mean_score:.3f} mean over {len(report)} scenario(s) "
            f"({report.pass_rate:.0%} pass)"
        )

    # ── 5. Optimize ──────────────────────────────────────────────────────
    _banner(f"Optimizing ({iterations} iteration(s))")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    outcomes: list[SectionOutcome] = []
    for section in targets:
        section_train = [s for s in train if s.section_under_test == section.name]
        section_val = [s for s in val if s.section_under_test == section.name]
        if not section_train:
            console.print(
                f"  [yellow]○ {section.name}: no training scenarios, skipping[/yellow]"
            )
            continue

        console.print(
            f"\n  [cyan]{section.name}[/cyan] "
            f"({len(section_train)} train / {len(section_val)} val)"
        )
        started = time.time()
        evolved_text, optimizer_name = _optimize_section(
            section_name=section.name,
            baseline_text=section.baseline_text,
            trainset=scenarios_to_dspy_examples(section_train),
            valset=scenarios_to_dspy_examples(section_val or section_train),
            iterations=iterations,
            optimizer_model=optimizer_model,
        )
        elapsed = time.time() - started

        outcome = SectionOutcome(
            name=section.name,
            baseline_text=section.baseline_text,
            evolved_text=evolved_text,
            baseline_score=baseline_reports[section.name].mean_score,
            elapsed_s=elapsed,
            optimizer=optimizer_name,
        )
        console.print(
            f"  {optimizer_name} finished in {elapsed:.1f}s "
            f"({len(section.baseline_text):,} -> {len(evolved_text):,} chars)"
        )
        outcomes.append(outcome)

    if not outcomes:
        console.print("[red]✗ No section had scenarios to optimize against[/red]")
        return 1

    # ── 6. Constraints ───────────────────────────────────────────────────
    _banner("Constraints")
    for outcome in outcomes:
        validation = inventory.validate(outcome.name, outcome.evolved_text)
        outcome.validation = validation
        console.print(f"  [cyan]{outcome.name}[/cyan]")
        for check in validation.checks:
            colour = "green" if check.passed else ("yellow" if check.is_warning else "red")
            console.print(f"    [{colour}]{'✓' if check.passed else '✗'} {check.name}[/{colour}]: {check.message}")

        if not validation.passed:
            outcome.accepted = False
            outcome.reason = "failed constraints: " + ", ".join(
                c.name for c in validation.errors
            )
        elif strict_gates and not validation.passed_strict:
            outcome.accepted = False
            outcome.reason = "strict mode: " + ", ".join(
                c.name for c in validation.warnings
            )
        else:
            outcome.accepted = True
            outcome.reason = "constraints passed"

    survivors = [o for o in outcomes if o.accepted]
    for rejected in (o for o in outcomes if not o.accepted):
        console.print(f"  [red]✗ {rejected.name} rejected: {rejected.reason}[/red]")

    # ── 7. Gate ladder ───────────────────────────────────────────────────
    _banner("Gate ladder")
    chain: Optional[GateChain] = None
    if not survivors:
        console.print("  [yellow]○ Skipped: no candidate survived the constraints[/yellow]")
    else:
        updates = {o.name: o.evolved_text for o in survivors}
        console.print(
            f"  Staging {len(updates)} section(s), regression tolerance "
            f"{ZERO_REGRESSION_TOLERANCE:.0%}{' , strict' if strict_gates else ''}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        backup_path = output_dir / "prompt_builder.py.bak"
        chain = run_gate_ladder(
            hermes_repo=repo,
            updates=updates,
            strict=strict_gates,
            python=sys.executable,
            backup_path=backup_path,
        )
        console.print(f"  Backup of the original: {backup_path}")
        for result in chain.results:
            colour = {
                "passed": "green",
                "failed": "red",
                "unavailable": "yellow",
                "skipped": "dim",
            }[result.status.value]
            console.print(f"  [{colour}]{result.name}[/{colour}]: {result.message}")

        if not chain.passed:
            blockers = ", ".join(r.name for r in chain.blockers)
            console.print(f"  [red]✗ Gate ladder blocked on: {blockers}[/red]")
            for outcome in survivors:
                outcome.accepted = False
                outcome.reason = f"gate blocked: {blockers}"
            survivors = []

    # ── 8. Holdout ───────────────────────────────────────────────────────
    _banner("Holdout")
    if not survivors:
        console.print("  [yellow]○ Skipped: nothing survived to evaluate[/yellow]")
    else:
        holdout_judge = BehavioralJudge(model=eval_model, use_llm=True)
        for outcome in survivors:
            scenarios = [s for s in holdout if s.section_under_test == outcome.name]
            if not scenarios:
                console.print(
                    f"  [yellow]○ {outcome.name}: no holdout scenarios for this section[/yellow]"
                )
                continue
            evolved_prompt = inventory.assembled_prompt({outcome.name: outcome.evolved_text})
            base_report = suite.evaluate(
                baseline_prompt,
                harness,
                judge=holdout_judge,
                section_name=outcome.name,
                run_name=f"hase-holdout-base-{outcome.name.lower()}-{timestamp}",
                scenarios=scenarios,
            )
            evolved_report = suite.evaluate(
                evolved_prompt,
                harness,
                judge=holdout_judge,
                section_name=outcome.name,
                run_name=f"hase-holdout-evolved-{outcome.name.lower()}-{timestamp}",
                scenarios=scenarios,
            )
            outcome.holdout_baseline = base_report.mean_score
            outcome.holdout_evolved = evolved_report.mean_score
            outcome.evolved_score = evolved_report.mean_score
            console.print(
                f"  {outcome.name}: {base_report.mean_score:.3f} -> "
                f"{evolved_report.mean_score:.3f} over {len(scenarios)} scenario(s)"
            )

    # ── 9. Results ───────────────────────────────────────────────────────
    table = Table(title="Phase 3 - System Prompt Evolution")
    table.add_column("Section", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Growth", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Holdout", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Verdict")

    for outcome in outcomes:
        change = outcome.holdout_improvement
        change_text = "-"
        if change is not None:
            colour = "green" if change > 0 else ("dim" if change == 0 else "red")
            change_text = f"[{colour}]{change:+.3f}[/{colour}]"
        verdict = (
            "[green]accepted[/green]" if outcome.accepted else f"[red]{outcome.reason}[/red]"
        )
        table.add_row(
            outcome.name,
            f"{len(outcome.baseline_text):,} → {len(outcome.evolved_text):,}",
            f"{outcome.growth:+.1%}",
            f"{outcome.baseline_score:.3f}",
            "-" if outcome.holdout_evolved is None else f"{outcome.holdout_evolved:.3f}",
            change_text,
            verdict,
        )

    console.print()
    console.print(table)

    # ── 10. Save ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    for outcome in outcomes:
        (output_dir / f"baseline_{outcome.name}.txt").write_text(
            outcome.baseline_text, encoding="utf-8"
        )
        (output_dir / f"evolved_{outcome.name}.txt").write_text(
            outcome.evolved_text, encoding="utf-8"
        )
    suite.save(output_dir / "scenarios.jsonl")
    metrics = {
        "timestamp": timestamp,
        "hermes_repo": str(repo),
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "strict_gates": strict_gates,
        "regression_threshold": ZERO_REGRESSION_TOLERANCE,
        "harness": harness_reason,
        "scenarios": {
            "total": len(suite),
            "train": len(train),
            "val": len(val),
            "holdout": len(holdout),
            "by_category": suite.categories(),
        },
        "inventory": inventory.to_dict(),
        "gates": chain.to_dict() if chain else None,
        "sections": [o.to_dict() for o in outcomes],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    console.print(f"\n  Output saved to {output_dir}/")

    # ── 11. Write-back ───────────────────────────────────────────────────
    _banner("Write-back")
    console.print(f"  [dim]{NEXT_SESSION_NOTICE}[/dim]")

    # Holdout evidence is the price of admission for a write. This tier is too
    # wide to deploy on a training-set improvement, and "no holdout scenarios"
    # is not evidence of anything.
    improved: list[SectionOutcome] = []
    for outcome in survivors:
        delta = outcome.holdout_improvement
        if delta is None:
            console.print(
                f"  [yellow]○ {outcome.name}: no holdout evidence, not deployable[/yellow]"
            )
        elif delta <= 0:
            console.print(
                f"  [yellow]○ {outcome.name}: holdout did not improve "
                f"({delta:+.3f}), not deployable[/yellow]"
            )
        else:
            improved.append(outcome)

    if not write:
        console.print("  No write requested (--no-write is the default).")
        if improved:
            console.print(
                f"  Review first: diff {output_dir}/baseline_<SECTION>.txt "
                f"{output_dir}/evolved_<SECTION>.txt"
            )
            console.print("  Then re-run with --write to apply.")
        return 0

    if not improved:
        console.print("  [yellow]Nothing to write: no section improved on holdout.[/yellow]")
        return 0

    session = detect_active_session()
    if session.active:
        console.print(f"  [red]✗ Refusing to write: {session.summary}[/red]")
        console.print(
            "  A live session has already assembled and cached its system prompt. "
            "Evolved sections deploy on the next session, so finish or stop the "
            "current one and re-run with --write."
        )
        return 2

    updates = {o.name: o.evolved_text for o in improved}
    result = write_sections(repo, updates)
    console.print(
        f"  [green]✓ Wrote {', '.join(result.updated)} to {result.path}[/green] "
        f"({result.before_chars:,} -> {result.after_chars:,} chars)"
    )
    console.print("  Neighbouring constants verified unchanged.")
    console.print(f"  [bold]{NEXT_SESSION_NOTICE}[/bold]")
    return 0


@click.command()
@click.option(
    "--section",
    "sections",
    multiple=True,
    help="Prompt section to evolve; repeatable. Must be on the evolvable allowlist.",
)
@click.option("--all-sections", is_flag=True, help="Evolve every discovered section")
@click.option("--iterations", default=5, help="Optimizer iterations per section")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluation and judging")
@click.option(
    "--strict-gates",
    is_flag=True,
    help="Treat an unavailable benchmark or a cache block crossing as blocking",
)
@click.option("--dry-run", is_flag=True, help="Validate setup without optimizing or writing")
@click.option(
    "--write/--no-write",
    default=False,
    help="Write accepted sections back to prompt_builder.py (default: no-write)",
)
def main(
    sections,
    all_sections,
    iterations,
    hermes_repo,
    optimizer_model,
    eval_model,
    strict_gates,
    dry_run,
    write,
):
    """Evolve hermes-agent system prompt sections with DSPy + GEPA."""
    code = evolve(
        section_names=list(sections),
        all_sections=all_sections,
        iterations=iterations,
        hermes_repo=hermes_repo,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        strict_gates=strict_gates,
        dry_run=dry_run,
        write=write,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
