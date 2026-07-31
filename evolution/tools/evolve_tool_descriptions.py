"""Evolve hermes-agent tool descriptions with DSPy + GEPA.

PLAN.md Phase 2. The pipeline, in order:

    catalogue -> selection dataset -> baseline -> GEPA -> constraints ->
    cross-tool regression check -> gate ladder -> holdout -> write-back

Three rules shape the whole thing:

1. **All tools are always evaluated together.** Even when ``--tool`` narrows
   what may be rewritten, the selector sees the entire catalogue and the
   cross-tool guard scores every tool. Optimizing one description in isolation
   is how one tool's gain quietly becomes another's loss.
2. **Writing to a real checkout is opt-in.** ``--no-write`` is the default and
   still runs the complete rewrite through ``artifact_io`` in dry-run mode, so a
   run that reports a clean write really would have written cleanly.
3. **A gate that could not run says so.** hermes-agent ships no benchmarks
   today, so the TBLite gate reports UNAVAILABLE. ``--strict-gates`` turns that
   into a blocker for anyone who needs the gate to have actually run.

Usage:
    python -m evolution.tools.evolve_tool_descriptions --dry-run
    python -m evolution.tools.evolve_tool_descriptions --toolset file --iterations 8
    python -m evolution.tools.evolve_tool_descriptions --tool read_file --tool search_files --write
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.gates import GateChain, run_benchmark_gate, run_pytest_gate
from evolution.tools.accuracy import (
    DescriptionEntailment,
    FactualAccuracyChecker,
    facts_from_catalog,
)
from evolution.tools.cross_tool import (
    DEFAULT_TOLERANCE,
    CrossToolGuard,
    CrossToolReport,
    CrossToolVerdict,
)
from evolution.tools.selection_eval import (
    NO_TOOL,
    ToolSelectionDataset,
    ToolSelectionDatasetBuilder,
    ToolSelector,
    catalog_signatures,
    evaluate_selection,
    extract_bundle,
    gepa_selection_metric,
    selector_predict_fn,
    tool_selection_metric,
)
from evolution.tools.tool_catalog import (
    ToolCatalog,
    ToolDescriptions,
    UnknownTool,
    bundle_to_dict,
    diff_bundles,
    load_catalog,
    write_bundle,
)

console = Console()

__all__ = [
    "ConstraintOutcome",
    "build_accuracy_checker",
    "enforce_constraints",
    "freeze_unselected",
    "evolve_tool_descriptions",
    "main",
]


def _banner(text: str) -> None:
    console.print(f"\n[bold]── {text} ─[/bold]")


# ──────────────────────────────────────────────────────────────────────────
# Candidate hygiene
# ──────────────────────────────────────────────────────────────────────────


def freeze_unselected(
    candidate: dict[str, ToolDescriptions],
    baseline: dict[str, ToolDescriptions],
    allowed: Sequence[str],
) -> dict[str, ToolDescriptions]:
    """Keep only the tools the run was asked to touch; restore the rest.

    An optimizer handed the whole catalogue will happily rewrite a tool nobody
    asked about. That is out of scope for the run, so it is reverted before the
    candidate is scored: the cross-tool comparison then measures the effect of
    the requested change and nothing else.
    """
    permitted = set(allowed)
    merged: dict[str, ToolDescriptions] = {}
    for tool_name, base in baseline.items():
        proposed = candidate.get(tool_name)
        if proposed is None or tool_name not in permitted:
            merged[tool_name] = base.copy()
        else:
            merged[tool_name] = proposed.copy()
    return merged


@dataclass
class ConstraintOutcome:
    """What the constraint validator said about one description."""

    target: str
    kind: str  # "tool_description" or "param_description"
    passed: bool
    reverted: bool
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "kind": self.kind,
            "passed": self.passed,
            "reverted": self.reverted,
            "messages": list(self.messages),
        }


def build_accuracy_checker(
    catalog: ToolCatalog,
    lm: object = None,
    entailment: object = None,
) -> FactualAccuracyChecker:
    """A factual-accuracy checker for this catalogue.

    The entailment predictor is built by default but only ever called when an
    LM is configured, so this is safe to construct in an offline run. Pass
    ``entailment`` to inject a stub, or ``entailment=False`` to run the
    deterministic checks alone.
    """
    if entailment is False:
        predictor = None
    elif entailment is not None:
        predictor = entailment
    else:
        predictor = dspy.ChainOfThought(DescriptionEntailment)
    return FactualAccuracyChecker(
        facts=facts_from_catalog(catalog), entailment=predictor, lm=lm
    )


def enforce_constraints(
    candidate: dict[str, ToolDescriptions],
    baseline: dict[str, ToolDescriptions],
    validator: ConstraintValidator,
    allowed: Optional[Sequence[str]] = None,
    accuracy: Optional[FactualAccuracyChecker] = None,
) -> tuple[dict[str, ToolDescriptions], list[ConstraintOutcome]]:
    """Revert any evolved description that busts its budget or its schema.

    Checked against the 500 / 200 char budgets and the growth limit from
    EvolutionConfig, and - when *accuracy* is supplied - against PLAN.md's
    remaining Phase 2 constraint, that a description "must remain factually
    accurate (can't claim a tool does something it doesn't)". A factual finding
    reverts the description exactly like a budget failure does: an inaccurate
    description is not a smaller problem than a long one, it is a larger one.

    A failure reverts that single description to baseline rather than throwing
    away the whole candidate, so one greedy rewrite does not cost the run every
    other improvement it found.

    Unchanged text is not re-validated. hermes-agent's ``read_file``
    description is already 539 chars and its ``write_file.cross_profile``
    parameter is already 302; failing the run on a violation that was there
    before evolution started would make the tool unusable on the real repo.
    """
    permitted = set(allowed) if allowed is not None else set(baseline)
    result: dict[str, ToolDescriptions] = {}
    outcomes: list[ConstraintOutcome] = []

    for tool_name, base in baseline.items():
        proposed = candidate.get(tool_name)
        if proposed is None or tool_name not in permitted:
            result[tool_name] = base.copy()
            continue

        kept = proposed.copy()

        if kept.description != base.description:
            checks = validator.validate_all(
                kept.description, "tool_description", baseline_text=base.description
            )
            failures = [c for c in checks if not c.passed]
            messages = [f"{c.constraint_name}: {c.message}" for c in checks]
            findings = (
                accuracy.check_tool(tool_name, kept.description, base.description)
                if accuracy
                else []
            )
            messages.extend(f"factual_accuracy: {f.describe()}" for f in findings)
            rejected = bool(failures or findings)
            if rejected:
                kept.description = base.description
            outcomes.append(
                ConstraintOutcome(
                    target=tool_name,
                    kind="tool_description",
                    passed=not rejected,
                    reverted=rejected,
                    messages=messages,
                )
            )

        for param, base_text in base.params.items():
            new_text = kept.params.get(param)
            if new_text is None or new_text == base_text:
                kept.params[param] = base_text
                continue
            checks = validator.validate_all(
                new_text, "param_description", baseline_text=base_text
            )
            failures = [c for c in checks if not c.passed]
            messages = [f"{c.constraint_name}: {c.message}" for c in checks]
            findings = (
                accuracy.check_param(tool_name, param, new_text, base_text)
                if accuracy
                else []
            )
            messages.extend(f"factual_accuracy: {f.describe()}" for f in findings)
            rejected = bool(failures or findings)
            if rejected:
                kept.params[param] = base_text
            outcomes.append(
                ConstraintOutcome(
                    target=f"{tool_name}.{param}",
                    kind="param_description",
                    passed=not rejected,
                    reverted=rejected,
                    messages=messages,
                )
            )

        # Parameters the optimizer invented are dropped: the schema is frozen.
        kept.params = {name: kept.params[name] for name in base.params}
        result[tool_name] = kept

    return result, outcomes


# ──────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────


def _catalogue_table(catalog: ToolCatalog, selected: ToolCatalog) -> Table:
    chosen = set(selected.names)
    table = Table(title="Tool catalogue")
    table.add_column("Tool", style="bold")
    table.add_column("Toolset")
    table.add_column("Module")
    table.add_column("Desc", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Budget")
    table.add_column("In run", justify="center")

    for entry in catalog:
        findings = entry.budget_findings()
        if findings:
            budget = "[red]" + "; ".join(f.describe() for f in findings) + "[/red]"
        else:
            budget = "[green]ok[/green]"
        table.add_row(
            entry.tool_name,
            f"{entry.toolset}" + ("" if entry.toolset_source == "registry" else " (inferred)"),
            entry.module,
            f"{entry.description_size}",
            f"{len(entry.param_names)}",
            budget,
            "✓" if entry.tool_name in chosen else "",
        )
    return table


def _rates_table(
    baseline: CrossToolReport,
    candidate: CrossToolReport,
    verdict: Optional[CrossToolVerdict] = None,
) -> Table:
    """Per-tool rates with the uncertainty that makes them readable.

    A rate change with no interval, no p-value and no power marker invites the
    reader to treat a one-example flip and a forty-example collapse as the same
    finding. The last three columns are what stop that.
    """
    table = Table(title="Per-tool selection rate")
    table.add_column("Tool", style="bold")
    table.add_column("Examples", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("95% CI on change", justify="right")
    table.add_column("p(worse)", justify="right")
    table.add_column("Power", justify="left")

    for tool in sorted(set(baseline.rates) | set(candidate.rates)):
        opportunities = baseline.opportunities(tool) or candidate.opportunities(tool)
        if opportunities == 0:
            continue
        before = baseline.rate(tool)
        after = candidate.rate(tool)
        delta = after - before
        colour = "green" if delta > 0 else ("red" if delta < 0 else "white")

        comparison = verdict.comparison(tool) if verdict else None
        interval = comparison.delta_interval() if comparison else None
        ci_text = (
            f"[{interval.low:+.1%}, {interval.high:+.1%}]" if interval else "[dim]-[/dim]"
        )
        p_worse = comparison.p_worse if comparison else None
        p_text = f"{p_worse:.3f}" if p_worse is not None else "[dim]-[/dim]"
        if comparison is None or comparison.paired is None:
            power = "[yellow]no pairing[/yellow]"
        elif comparison.underpowered:
            power = (
                f"[yellow]⚠ needs {comparison.min_detectable_shift:.0%}[/yellow]"
            )
        elif comparison.significant_regression:
            power = "[red]✗ significant[/red]"
        elif comparison.significant_improvement:
            power = "[green]✓ significant[/green]"
        else:
            power = "[green]✓[/green]"

        table.add_row(
            tool,
            str(opportunities),
            f"{before:.1%}",
            f"{after:.1%}",
            f"[{colour}]{delta:+.1%}[/{colour}]",
            ci_text,
            p_text,
            power,
        )
    return table


def _print_confusions(report: CrossToolReport) -> None:
    confusions = report.confusion.top_confusions(limit=5)
    if not confusions:
        console.print("  No misselections recorded.")
        return
    console.print("  Most common misselections (expected -> picked):")
    for expected, predicted, count in confusions:
        console.print(f"    {expected} -> {predicted}: {count}")


# ──────────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────────


def evolve_tool_descriptions(
    tools: Sequence[str] = (),
    toolset: Optional[str] = None,
    iterations: int = 10,
    dataset_path: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    run_tests: bool = False,
    strict_gates: bool = False,
    dry_run: bool = False,
    write: bool = False,
    regression_tolerance: float = DEFAULT_TOLERANCE,
    output_root: Optional[Path] = None,
) -> Optional[dict]:
    """Run the full Phase 2 optimization. Returns the metrics it saved."""

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=optimizer_model,  # Dataset generation deserves the strong model
        run_pytest=run_tests,
    )
    repo = Path(config.hermes_agent_path)

    # ── 1. Load the catalogue ───────────────────────────────────────────
    console.print(
        "\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] "
        "- Phase 2: tool descriptions\n"
    )
    _banner("1. Tool catalogue")
    console.print(f"  Repo: {repo}")

    catalog = load_catalog(repo, config)
    if not len(catalog):
        console.print(f"[red]✗ No literal tool schemas found under {repo / 'tools'}[/red]")
        sys.exit(1)

    try:
        selected = catalog.select(tools=list(tools), toolset=toolset)
    except UnknownTool as exc:
        console.print(f"[red]✗ {exc}[/red]")
        sys.exit(1)

    if not len(selected):
        console.print(f"[red]✗ No tools matched (toolset={toolset!r}, tools={list(tools)})[/red]")
        sys.exit(1)

    console.print(_catalogue_table(catalog, selected))
    console.print(
        f"  {len(catalog)} tool(s) in {len(catalog.by_toolset())} toolset(s), "
        f"{catalog.total_description_chars:,} chars of description total"
    )
    console.print(f"  Evolving {len(selected)}: {', '.join(selected.names)}")

    over_budget = catalog.budget_findings()
    if over_budget:
        console.print(
            f"[yellow]⚠ {len(over_budget)} description(s) already over budget "
            f"before evolution:[/yellow]"
        )
        for finding in over_budget:
            console.print(f"    {finding.describe()}")

    if dry_run:
        console.print("\n[bold green]DRY RUN - setup validated successfully.[/bold green]")
        console.print(f"  Would build a tool-selection dataset over all {len(catalog)} tools")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print("  Would check constraints, cross-tool regressions, and the gate ladder")
        console.print(
            f"  Would reject any tool regressing past a {regression_tolerance:.1%} "
            f"tolerance, or any regression significant at alpha=0.05"
        )
        console.print(
            "  Would " + ("write results back to the repo" if write else "leave the repo untouched")
        )
        return None

    baseline_bundle = catalog.bundle()
    signatures = catalog_signatures(catalog)

    # ── 2. Selection dataset ────────────────────────────────────────────
    _banner("2. Tool-selection dataset")
    dataset_dir = Path(dataset_path) if dataset_path else Path("datasets") / "tools" / (toolset or "all")

    if (dataset_dir / "train.jsonl").exists():
        dataset = ToolSelectionDataset.load(dataset_dir)
        console.print(f"  Loaded {len(dataset)} examples from {dataset_dir}/")
    else:
        console.print(f"  Generating with {config.judge_model} (all {len(catalog)} tools)")
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog,
            config=config,
            lm=dspy.LM(config.judge_model, temperature=0.0),
        )
        dataset = builder.generate()
        if not dataset.all_examples:
            console.print("[red]✗ Dataset generation produced no usable examples[/red]")
            sys.exit(1)
        dataset.save(dataset_dir)
        console.print(f"  Generated {len(dataset)} examples, saved to {dataset_dir}/")
        if builder.rejected:
            console.print(f"  [yellow]Rejected {len(builder.rejected)} generated case(s)[/yellow]")
            for target, reason in builder.rejected[:5]:
                console.print(f"    {target}: {reason}")

    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / "
        f"{len(dataset.holdout)} holdout"
    )
    console.print(f"  Categories: {dataset.category_counts()}")

    if not dataset.val:
        console.print("[red]✗ The val split is empty; cannot measure a regression[/red]")
        sys.exit(1)

    # ── 3. Baseline measurement ─────────────────────────────────────────
    _banner("3. Baseline measurement")
    lm = dspy.LM(config.eval_model, temperature=0.0)
    dspy.configure(lm=lm)

    baseline_module = ToolSelector(baseline_bundle, signatures)
    all_tools = list(catalog.names) + [NO_TOOL]

    with dspy.context(lm=lm):
        baseline_val = evaluate_selection(dataset.val, selector_predict_fn(baseline_module))
    baseline_report = CrossToolReport.from_report(baseline_val, tools=all_tools)

    console.print(f"  Selection accuracy: {baseline_report.describe_accuracy()}")
    console.print(f"  Parameter correctness: {baseline_val.param_accuracy:.1%}")
    _print_confusions(baseline_report)

    # ── 4. Optimize ─────────────────────────────────────────────────────
    _banner("4. GEPA optimization")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")
    console.print(f"  Iterations: {iterations}")

    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")
    start_time = time.time()
    optimizer_used = "GEPA"

    try:
        optimizer = dspy.GEPA(
            metric=gepa_selection_metric,
            max_full_evals=iterations,
            reflection_lm=dspy.LM(optimizer_model),
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as exc:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({exc}), falling back to MIPROv2[/yellow]")
        optimizer_used = "MIPROv2"
        optimizer = dspy.MIPROv2(
            metric=tool_selection_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s using {optimizer_used}")

    raw_bundle = extract_bundle(optimized_module, baseline_bundle)
    candidate_bundle = freeze_unselected(raw_bundle, baseline_bundle, selected.names)

    # ── 5. Constraints ──────────────────────────────────────────────────
    _banner("5. Constraint validation")
    validator = ConstraintValidator(config)
    accuracy = build_accuracy_checker(catalog, lm=lm)
    candidate_bundle, constraint_outcomes = enforce_constraints(
        candidate_bundle,
        baseline_bundle,
        validator,
        allowed=selected.names,
        accuracy=accuracy,
    )
    factual_reverts = sum(
        1
        for outcome in constraint_outcomes
        if any(m.startswith("factual_accuracy:") for m in outcome.messages)
    )
    console.print(
        "  Factual accuracy: schema-structural checks"
        + (
            " plus LLM entailment"
            if accuracy.entailment_ran
            else f" only ({accuracy.skipped_reason or 'entailment not run'})"
        )
    )

    if not constraint_outcomes:
        console.print("  No description changed, so nothing to validate.")
    for outcome in constraint_outcomes:
        icon = "✓" if outcome.passed else "✗"
        colour = "green" if outcome.passed else "red"
        note = "" if outcome.passed else " [reverted to baseline]"
        console.print(f"  [{colour}]{icon} {outcome.target}[/{colour}]{note}")
        for message in outcome.messages:
            console.print(f"      {message}")

    changes = diff_bundles(baseline_bundle, candidate_bundle)
    console.print(f"  {len(changes)} description(s) changed after validation")

    # ── 6. Cross-tool regression check ──────────────────────────────────
    _banner("6. Cross-tool regression check")
    candidate_module = ToolSelector(candidate_bundle, signatures)
    with dspy.context(lm=lm):
        candidate_val = evaluate_selection(dataset.val, selector_predict_fn(candidate_module))
    candidate_report = CrossToolReport.from_report(candidate_val, tools=all_tools)

    guard = CrossToolGuard(tolerance=regression_tolerance)
    verdict = guard.compare(baseline_report, candidate_report)

    console.print(_rates_table(baseline_report, candidate_report, verdict))
    console.print(f"  Overall: {candidate_report.describe_accuracy()}")
    icon = "✓" if verdict.accepted else "✗"
    colour = "green" if verdict.accepted else "red"
    console.print(f"  [{colour}]{icon} {verdict.summary()}[/{colour}]")
    for regression in verdict.regressions:
        console.print(f"    [red]{regression.describe()}[/red]")
    if verdict.underpowered:
        console.print(f"  [yellow]⚠ {verdict.power_note()}[/yellow]")
    _print_confusions(candidate_report)

    # ── 7. Gate ladder ──────────────────────────────────────────────────
    _banner("7. Gate ladder")
    chain = GateChain(strict=strict_gates)
    gates = [lambda: verdict.to_gate_result()]
    if run_tests:
        gates.append(lambda: run_pytest_gate(repo))
    gates.append(
        lambda: run_benchmark_gate(
            repo,
            "tblite",
            baseline=None,
            regression_threshold=config.tblite_regression_threshold,
            fast=True,
        )
    )
    chain.run(*gates)
    console.print(chain.summary())

    # ── 8. Holdout ──────────────────────────────────────────────────────
    _banner(f"8. Holdout evaluation ({len(dataset.holdout)} examples)")
    if dataset.holdout:
        with dspy.context(lm=lm):
            baseline_holdout = evaluate_selection(
                dataset.holdout, selector_predict_fn(baseline_module)
            )
            candidate_holdout = evaluate_selection(
                dataset.holdout, selector_predict_fn(candidate_module)
            )
        holdout_baseline_report = CrossToolReport.from_report(baseline_holdout, tools=all_tools)
        holdout_candidate_report = CrossToolReport.from_report(candidate_holdout, tools=all_tools)
        holdout_verdict = guard.compare(holdout_baseline_report, holdout_candidate_report)
        console.print(f"  {holdout_verdict.summary()}")
    else:
        baseline_holdout = candidate_holdout = None
        holdout_verdict = None
        console.print("  [yellow]No holdout examples; skipping[/yellow]")

    # ── 9. Results ──────────────────────────────────────────────────────
    _banner("9. Results")
    baseline_chars = sum(d.total_chars for d in baseline_bundle.values())
    candidate_chars = sum(d.total_chars for d in candidate_bundle.values())

    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    def _row(label: str, before: float, after: float) -> None:
        delta = after - before
        colour = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(
            label,
            f"{before:.1%}",
            f"{after:.1%}",
            f"[{colour}]{delta:+.1%}[/{colour}]",
        )

    _row("Selection accuracy (val)", baseline_report.overall_accuracy, candidate_report.overall_accuracy)
    baseline_ci = baseline_report.accuracy_interval()
    candidate_ci = candidate_report.accuracy_interval()
    table.add_row(
        "  95% CI on accuracy",
        f"[{baseline_ci.low:.1%}, {baseline_ci.high:.1%}]",
        f"[{candidate_ci.low:.1%}, {candidate_ci.high:.1%}]",
        "",
    )
    # Raw accuracy is not interpretable on its own: 40% is poor against two
    # tools and excellent against thirty.
    table.add_row(
        f"  Chance ({candidate_report.num_options} options)",
        f"{baseline_report.chance_accuracy:.1%}",
        f"{candidate_report.chance_accuracy:.1%}",
        "",
    )
    _row("Parameter correctness (val)", baseline_val.param_accuracy, candidate_val.param_accuracy)
    if baseline_holdout is not None and candidate_holdout is not None:
        _row("Selection accuracy (holdout)", baseline_holdout.tool_accuracy, candidate_holdout.tool_accuracy)
    table.add_row(
        "Description chars",
        f"{baseline_chars:,}",
        f"{candidate_chars:,}",
        f"{candidate_chars - baseline_chars:+,}",
    )
    table.add_row("Descriptions changed", "", str(len(changes)), "")
    table.add_row("Factual reverts", "", str(factual_reverts), "")
    table.add_row(
        "Underpowered tools",
        "",
        str(len(verdict.underpowered)),
        ", ".join(verdict.underpowered),
    )
    table.add_row("Optimizer", "", optimizer_used, "")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")

    console.print()
    console.print(table)

    # ── 10. Write-back ──────────────────────────────────────────────────
    _banner("10. Write-back")
    may_write = write and verdict.accepted and chain.passed and bool(changes)

    if not changes:
        console.print("  Nothing to write: no description survived validation unchanged.")
        write_report = None
    else:
        # Always exercise the rewrite. A dry run that verifies is evidence.
        write_report = write_bundle(
            repo, candidate_bundle, dry_run=not may_write, baseline=baseline_bundle
        )
        console.print(f"  {write_report.summary()}")
        for change in write_report.changes:
            console.print(f"    {change.target} ({change.delta_chars:+d} chars)")
        for target, reason in write_report.skipped:
            console.print(f"    [yellow]skipped {target}: {reason}[/yellow]")
        if not may_write:
            if not write:
                console.print("  [yellow]--no-write is the default; re-run with --write to apply[/yellow]")
            elif not verdict.accepted:
                console.print("  [red]Not written: the cross-tool guard rejected this candidate[/red]")
            elif not chain.passed:
                console.print("  [red]Not written: a gate blocked this candidate[/red]")

    # ── 11. Save artifacts ──────────────────────────────────────────────
    _banner("11. Artifacts")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root or config.output_dir) / "tools" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "baseline_descriptions.json").write_text(
        json.dumps(bundle_to_dict(baseline_bundle), indent=2)
    )
    (output_dir / "evolved_descriptions.json").write_text(
        json.dumps(bundle_to_dict(candidate_bundle), indent=2)
    )
    (output_dir / "cross_tool_report.json").write_text(
        json.dumps(
            {
                "baseline": baseline_report.to_dict(),
                "candidate": candidate_report.to_dict(),
                "verdict": verdict.to_dict(),
                "holdout_verdict": holdout_verdict.to_dict() if holdout_verdict else None,
            },
            indent=2,
        )
    )
    (output_dir / "gates.json").write_text(json.dumps(chain.to_dict(), indent=2))
    (output_dir / "changes.json").write_text(
        json.dumps([c.to_dict() for c in changes], indent=2)
    )

    metrics = {
        "phase": "tool_descriptions",
        "timestamp": timestamp,
        "hermes_repo": str(repo),
        "tools_evolved": selected.names,
        "toolset": toolset,
        "iterations": iterations,
        "optimizer": optimizer_used,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_accuracy": baseline_report.overall_accuracy,
        "candidate_accuracy": candidate_report.overall_accuracy,
        "baseline_param_accuracy": baseline_val.param_accuracy,
        "candidate_param_accuracy": candidate_val.param_accuracy,
        "holdout_baseline_accuracy": baseline_holdout.tool_accuracy if baseline_holdout else None,
        "holdout_candidate_accuracy": candidate_holdout.tool_accuracy if candidate_holdout else None,
        "baseline_accuracy_ci": baseline_report.accuracy_interval().to_dict(),
        "candidate_accuracy_ci": candidate_report.accuracy_interval().to_dict(),
        "chance_accuracy": candidate_report.chance_accuracy,
        "num_options": candidate_report.num_options,
        "cross_tool_accepted": verdict.accepted,
        "regression_tolerance": regression_tolerance,
        # Per tool: baseline rate, candidate rate, delta with its interval, the
        # one-sided p-value, and whether this many examples could ever have
        # detected the tolerance being enforced.
        "per_tool": [comparison.to_dict() for comparison in verdict.comparisons],
        "underpowered_tools": list(verdict.underpowered),
        "unpaired_tools": list(verdict.unpaired),
        "significant_regressions": [r.tool for r in verdict.significant_regressions],
        "gates_passed": chain.passed,
        "descriptions_changed": len(changes),
        "baseline_chars": baseline_chars,
        "candidate_chars": candidate_chars,
        "constraint_reverts": sum(1 for o in constraint_outcomes if o.reverted),
        "factual_reverts": factual_reverts,
        "entailment_ran": accuracy.entailment_ran,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "written": bool(may_write and write_report and write_report.files_written),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n  Output saved to {output_dir}/")

    improvement = candidate_report.overall_accuracy - baseline_report.overall_accuracy
    if verdict.accepted and improvement > 0:
        console.print(
            f"\n[bold green]✓ Selection accuracy improved {improvement:+.1%} "
            f"with no per-tool regression[/bold green]"
        )
    elif verdict.accepted:
        console.print(
            f"\n[yellow]⚠ No accuracy improvement ({improvement:+.1%}), "
            f"but nothing regressed[/yellow]"
        )
    else:
        console.print(
            "\n[red]✗ Candidate rejected by the cross-tool guard - "
            "one tool's gain came out of another's[/red]"
        )

    return metrics


@click.command()
@click.option("--tool", "tools", multiple=True, help="Tool to evolve (repeatable, default all)")
@click.option("--toolset", default=None, help="Limit to one toolset, e.g. 'file'")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--dataset-path", default=None, help="Directory holding train/val/holdout JSONL")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--run-tests", is_flag=True, help="Run the hermes-agent pytest suite as a gate")
@click.option("--strict-gates", is_flag=True, help="Treat an unavailable gate as a failure")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--write/--no-write", default=False, help="Write evolved descriptions into the repo")
@click.option(
    "--regression-tolerance",
    default=DEFAULT_TOLERANCE,
    type=float,
    help="How far one tool's selection rate may fall before rejection (0 = not at all)",
)
def main(
    tools,
    toolset,
    iterations,
    dataset_path,
    hermes_repo,
    optimizer_model,
    eval_model,
    run_tests,
    strict_gates,
    dry_run,
    write,
    regression_tolerance,
):
    """Evolve hermes-agent tool descriptions using DSPy + GEPA optimization."""
    evolve_tool_descriptions(
        tools=tools,
        toolset=toolset,
        iterations=iterations,
        dataset_path=dataset_path,
        hermes_repo=hermes_repo,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        run_tests=run_tests,
        strict_gates=strict_gates,
        dry_run=dry_run,
        write=write,
        regression_tolerance=regression_tolerance,
    )


if __name__ == "__main__":
    main()
