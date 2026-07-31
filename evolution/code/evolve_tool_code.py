"""Evolve hermes-agent tool implementation code against a real bug.

Usage:
    python -m evolution.code.evolve_tool_code --tool file_tools --bug-issue 742 \\
        --repro-script repros/issue_742.py --iterations 10

This is Phase 4, the tier PLAN.md calls the highest risk: everything else
evolves text that an LLM reads, this evolves code that the interpreter runs.
The shape of the run reflects that.

    resolve target      one file, inside the hermes-agent checkout
    load reproduction   the specific bug this run is aimed at
    snapshot baseline   tests green, bug reproducing - or there is nothing to do
    ask the evolver     Darwinian Evolver, as an external CLI subprocess
    guardrails first    safety.py, before anything expensive runs
    then fitness        pytest as a hard gate, then benchmark, bug, quality
    emit a branch       and a diff, and stop

Two hard constraints shape the code below.

**Licensing.** Darwinian Evolver is AGPL v3. It is invoked as an external
process and nothing from it is ever imported, so no AGPL code is linked into
this MIT-licensed package. When it is not installed the run stops with a
non-zero exit and says so. It does not quietly substitute a weaker mutation
source and present the result as evolution.

**No auto-merge.** PLAN.md requires human review of every line of evolved
code, so the deliverable is a git branch plus a diff. This command never
merges, never pushes, and always puts the operator back on the branch they
started on.

Exit codes:
    0   the run completed (a winner, or nothing that survived the guardrails)
    1   setup problem: no repo, no target, dirty tree, red baseline
    2   Darwinian Evolver is not installed
    3   the evolver ran but produced no candidate to score
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import resolve_hermes_agent_path
from evolution.core.gates import GateStatus
from evolution.code.fitness_code import (
    BugReproduction,
    CodeFitness,
    CodeFitnessEvaluator,
    ReproStatus,
)
from evolution.code.organism import (
    CodeOrganism,
    Mutation,
    OrganismError,
    git_available,
    is_git_repo,
)

console = Console()

__all__ = [
    "EvolverError",
    "EvolverNotInstalled",
    "Candidate",
    "EvolverJob",
    "ExternalEvolver",
    "BugFixBrief",
    "MUTATION_CONSTRAINTS",
    "find_evolver",
    "resolve_tool_file",
    "build_objective",
    "evolve_tool_code",
    "main",
]


# Commands Darwinian Evolver is plausibly installed as. An explicit
# --evolver-cmd or DARWINIAN_EVOLVER_CMD beats all of them.
EVOLVER_CANDIDATE_COMMANDS = ("darwinian-evolver", "darwinian_evolver", "devolve")
EVOLVER_ENV_VAR = "DARWINIAN_EVOLVER_CMD"

# Handed to the evolver in the job spec and enforced afterwards by safety.py.
# Telling the mutation engine the rules is cheaper than rejecting everything
# it sends back, but the enforcement is what actually holds.
MUTATION_CONSTRAINTS = (
    "Do not change any function signature: names, parameters, defaults and "
    "star-args are frozen.",
    "Do not change, add or remove any registry.register(...) call.",
    "Do not reduce error handling: try, except, raise and finally coverage may "
    "not decrease, module-wide or in any individual function.",
    "Do not remove assertions, validation guards or early error returns.",
    "The full pytest suite must pass; a single failing test rejects the change.",
    "Change one file only, and change as little of it as possible.",
)


class EvolverError(RuntimeError):
    """Raised when the external evolver fails or returns nothing usable."""


class EvolverNotInstalled(EvolverError):
    """Raised when no Darwinian Evolver CLI can be found."""


class TargetNotFound(RuntimeError):
    """Raised when the requested tool file does not exist in the repo."""


# ──────────────────────────────────────────────────────────────────────────
# Target resolution
# ──────────────────────────────────────────────────────────────────────────


def resolve_tool_file(repo: Path, tool: str) -> Path:
    """Resolve ``--tool`` to a file inside *repo*.

    Accepts a bare module name (``file_tools``), a filename
    (``file_tools.py``), a repo-relative path (``tools/file_tools.py``) or an
    absolute path. Refuses anything outside the repo, since the whole point of
    the organism is that exactly one tracked file moves.
    """
    repo = Path(repo).expanduser().resolve()
    raw = tool.strip()

    if Path(raw).is_absolute():
        candidates = [Path(raw)]
    else:
        stem = raw[:-3] if raw.endswith(".py") else raw
        candidates = [
            repo / raw,
            repo / f"{stem}.py",
            repo / "tools" / f"{stem}.py",
            repo / "agent" / f"{stem}.py",
        ]

    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(repo)
            except ValueError as exc:
                raise TargetNotFound(
                    f"{resolved} is outside the hermes-agent repo {repo}"
                ) from exc
            return resolved

    raise TargetNotFound(
        f"could not find a source file for --tool {tool!r} under {repo} "
        "(tried the repo root, tools/ and agent/)"
    )


# ──────────────────────────────────────────────────────────────────────────
# External evolver
# ──────────────────────────────────────────────────────────────────────────


def find_evolver(explicit: Optional[str] = None, env: Optional[dict] = None) -> list[str]:
    """Locate the Darwinian Evolver CLI, or raise :class:`EvolverNotInstalled`.

    Returns the command as an argv list so an operator can point at a wrapper
    script with its own flags (``--evolver-cmd "uvx darwinian-evolver --quiet"``).
    """
    import os

    environ = env if env is not None else os.environ
    sources = [explicit, environ.get(EVOLVER_ENV_VAR)]

    for source in sources:
        if not source:
            continue
        argv = shlex.split(source)
        if not argv:
            continue
        if shutil.which(argv[0]) or Path(argv[0]).expanduser().is_file():
            return argv
        raise EvolverNotInstalled(
            f"evolver command not executable: {argv[0]} "
            f"(from {'--evolver-cmd' if source == explicit else EVOLVER_ENV_VAR})"
        )

    for name in EVOLVER_CANDIDATE_COMMANDS:
        found = shutil.which(name)
        if found:
            return [found]

    raise EvolverNotInstalled(
        "Darwinian Evolver is not installed. Tried: "
        + ", ".join(EVOLVER_CANDIDATE_COMMANDS)
        + f". Install it separately (it is AGPL v3 and is only ever run as an "
        f"external process), then point at it with --evolver-cmd or "
        f"{EVOLVER_ENV_VAR}."
    )


@dataclass
class Candidate:
    """One proposed rewrite of the target file."""

    index: int
    source: str
    notes: str = ""
    origin: str = ""

    @property
    def label(self) -> str:
        return f"c{self.index:02d}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "notes": self.notes,
            "origin": self.origin,
            "chars": len(self.source),
        }


@dataclass
class EvolverJob:
    """The job spec handed to the external evolver.

    This is our adapter contract, written to a JSON file and passed by path.
    Keeping it a file rather than a pipe means a failed run leaves the exact
    request behind for inspection.
    """

    target_path: str
    source: str
    objective: str
    iterations: int
    constraints: tuple[str, ...] = MUTATION_CONSTRAINTS
    bug_issue: Optional[str] = None
    reproduction: Optional[str] = None
    reproduction_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "target_path": self.target_path,
            "source": self.source,
            "objective": self.objective,
            "iterations": self.iterations,
            "constraints": list(self.constraints),
            "bug_issue": self.bug_issue,
            "reproduction": self.reproduction,
            "reproduction_path": self.reproduction_path,
        }


class ProposesCandidates(Protocol):
    """What :func:`evolve_tool_code` needs from a mutation source."""

    def propose(self, job: EvolverJob) -> list[Candidate]:  # pragma: no cover
        ...


class ExternalEvolver:
    """Drive Darwinian Evolver as a subprocess and collect its candidates.

    Never imported, only executed: the package is AGPL v3 and this one is MIT.

    The adapter reads candidates from, in order of preference:

    1. ``<output>/candidates/*.py`` - one file per candidate
    2. ``<output>/candidates.jsonl`` - one JSON object per line, with a
       ``source`` string or a ``path`` to read
    3. stdout, as JSON lines of the same shape

    A non-zero exit with no candidates is an error. A non-zero exit that still
    produced candidates is reported and the candidates are scored anyway; the
    guardrails decide, not the evolver's opinion of its own run.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        repo: Path,
        workdir: Path,
        timeout: int = 3600,
    ) -> None:
        self.cmd = list(cmd)
        self.repo = Path(repo)
        # Absolute: the evolver runs with cwd set to the hermes-agent repo, so
        # a relative job or output path would resolve against the wrong root.
        self.workdir = Path(workdir).expanduser().resolve()
        self.timeout = timeout
        self.last_stdout = ""
        self.last_stderr = ""
        self.last_returncode: Optional[int] = None

    def propose(self, job: EvolverJob) -> list[Candidate]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        job_path = self.workdir / "job.json"
        out_dir = self.workdir / "evolver_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        job_path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

        cmd = [*self.cmd, "--job", str(job_path), "--output", str(out_dir)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.repo),
            )
        except subprocess.TimeoutExpired as exc:
            raise EvolverError(
                f"evolver timed out after {self.timeout}s"
            ) from exc
        except OSError as exc:
            raise EvolverNotInstalled(f"could not run {cmd[0]}: {exc}") from exc

        self.last_stdout = proc.stdout or ""
        self.last_stderr = proc.stderr or ""
        self.last_returncode = proc.returncode

        candidates = self._collect(out_dir, self.last_stdout)
        if not candidates:
            tail = "\n".join(
                (self.last_stderr or self.last_stdout).strip().splitlines()[-15:]
            )
            raise EvolverError(
                f"evolver exited {proc.returncode} and produced no candidates"
                + (f":\n{tail}" if tail else "")
            )
        return candidates

    # ── candidate collection ────────────────────────────────────────────

    def _collect(self, out_dir: Path, stdout: str) -> list[Candidate]:
        candidates = self._from_directory(out_dir)
        if candidates:
            return candidates
        candidates = self._from_jsonl(out_dir / "candidates.jsonl")
        if candidates:
            return candidates
        return self._from_stdout(stdout)

    def _from_directory(self, out_dir: Path) -> list[Candidate]:
        folder = out_dir / "candidates"
        if not folder.is_dir():
            return []
        out: list[Candidate] = []
        for index, path in enumerate(sorted(folder.glob("*.py")), start=1):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            out.append(Candidate(index=index, source=source, origin=str(path)))
        return out

    def _from_jsonl(self, path: Path) -> list[Candidate]:
        if not path.is_file():
            return []
        out: list[Candidate] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            candidate = self._candidate_from_line(line, len(out) + 1, str(path))
            if candidate:
                out.append(candidate)
        return out

    def _from_stdout(self, stdout: str) -> list[Candidate]:
        out: list[Candidate] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            candidate = self._candidate_from_line(line, len(out) + 1, "stdout")
            if candidate:
                out.append(candidate)
        return out

    def _candidate_from_line(
        self, line: str, index: int, origin: str
    ) -> Optional[Candidate]:
        line = line.strip()
        if not line:
            return None
        try:
            blob = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(blob, dict):
            return None

        source = blob.get("source")
        if source is None and blob.get("path"):
            path = Path(blob["path"])
            if not path.is_absolute():
                path = self.repo / path
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None
        if not isinstance(source, str) or not source.strip():
            return None

        return Candidate(
            index=index,
            source=source,
            notes=str(blob.get("notes") or blob.get("rationale") or ""),
            origin=origin,
        )


# ──────────────────────────────────────────────────────────────────────────
# Mutation brief
# ──────────────────────────────────────────────────────────────────────────


class BugFixBrief(dspy.Signature):
    """Turn a bug report and its reproduction into a precise mutation brief.

    The brief is the objective handed to the external evolver. It should name
    the observable defect, the input that triggers it and the behaviour that
    would be correct. It must not propose a design change, and must not ask
    for anything the constraints forbid.
    """

    tool_module: str = dspy.InputField(desc="The tool module being evolved")
    bug_report: str = dspy.InputField(desc="Issue number, title and description")
    reproduction: str = dspy.InputField(desc="Source of the reproduction script")
    constraints: str = dspy.InputField(desc="Rules the mutation must respect")
    objective: str = dspy.OutputField(
        desc="A short, concrete statement of what the mutation must achieve"
    )


def _lm_configured() -> bool:
    """True when DSPy has a language model to call.

    Checked rather than assumed: this command is useful offline, and a fitness
    run must never die because nobody exported an API key.
    """
    try:
        return getattr(dspy.settings, "lm", None) is not None
    except Exception:
        return False


def _template_objective(
    tool_module: str, bug_issue: Optional[str], reproduction: Optional[str]
) -> str:
    lines = [
        f"Fix the defect in {tool_module} without changing its public shape.",
    ]
    if bug_issue:
        lines.append(f"Target bug: {bug_issue}.")
    if reproduction:
        lines.append(
            "The reproduction script exits non-zero while the bug is present "
            "and zero once it is fixed."
        )
    lines.append("Make the smallest change that resolves it.")
    return " ".join(lines)


def build_objective(
    tool_module: str,
    bug_issue: Optional[str] = None,
    reproduction: Optional[str] = None,
    constraints: Iterable[str] = MUTATION_CONSTRAINTS,
    predictor=None,
) -> str:
    """Compose the objective handed to the evolver.

    Uses the LLM when one is configured, and a deterministic template when it
    is not. Either way the constraints are enforced afterwards by safety.py,
    so a weak brief costs candidates, never correctness.
    """
    if predictor is None:
        if not _lm_configured():
            return _template_objective(tool_module, bug_issue, reproduction)
        predictor = dspy.Predict(BugFixBrief)

    try:
        result = predictor(
            tool_module=tool_module,
            bug_report=bug_issue or "no issue supplied",
            reproduction=reproduction or "no reproduction supplied",
            constraints="\n".join(constraints),
        )
    except Exception as exc:  # a brief is a nicety, not a dependency
        console.print(f"[yellow]![/yellow] Could not draft an LLM brief ({exc})")
        return _template_objective(tool_module, bug_issue, reproduction)

    objective = str(getattr(result, "objective", "") or "").strip()
    return objective or _template_objective(tool_module, bug_issue, reproduction)


# ──────────────────────────────────────────────────────────────────────────
# Run record
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class CandidateOutcome:
    """A candidate, its commit and its verdict, kept together for the report."""

    candidate: Candidate
    fitness: CodeFitness
    mutation: Optional[Mutation] = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "fitness": self.fitness.to_dict(),
            "mutation": self.mutation.to_dict() if self.mutation else None,
        }


# ──────────────────────────────────────────────────────────────────────────
# Console helpers
# ──────────────────────────────────────────────────────────────────────────


def _step(title: str) -> None:
    console.print(f"\n[bold cyan]── {title} ─────────────────────────────[/bold cyan]")


def _gate_icon(status: GateStatus) -> str:
    return {
        GateStatus.PASSED: "[green]✓[/green]",
        GateStatus.FAILED: "[red]✗[/red]",
        GateStatus.UNAVAILABLE: "[yellow]○[/yellow]",
        GateStatus.SKIPPED: "[dim]-[/dim]",
    }[status]


# ──────────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────────


def evolve_tool_code(
    tool: str,
    bug_issue: Optional[str] = None,
    repro_script: Optional[str] = None,
    iterations: int = 10,
    hermes_repo: Optional[str] = None,
    evolver_cmd: Optional[str] = None,
    strict_gates: bool = False,
    dry_run: bool = False,
    benchmarks: Sequence[str] = (),
    python: Optional[str] = None,
    pytest_subset: Optional[Sequence[str]] = None,
    allow_dirty: bool = False,
    output_root: Optional[Path] = None,
    evolver: Optional[ProposesCandidates] = None,
) -> int:
    """Run one code-evolution pass. Returns a process exit code.

    *evolver* is an injection point: pass an object with ``propose(job)`` to
    drive a different mutation source, or a fake one in tests. Left as None,
    the Darwinian Evolver CLI is discovered and used, and its absence stops
    the run.
    """
    console.print(
        "\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] - "
        f"Phase 4: evolving code in [bold]{tool}[/bold]\n"
    )

    # ── 1. Resolve the target ───────────────────────────────────────────
    try:
        repo = resolve_hermes_agent_path(hermes_repo)
    except FileNotFoundError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    if not Path(repo).is_dir():
        console.print(f"[red]✗ hermes-agent repo not found: {repo}[/red]")
        return 1

    try:
        target = resolve_tool_file(repo, tool)
    except TargetNotFound as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    relpath = target.relative_to(Path(repo).resolve()).as_posix()
    console.print(f"  Repo:   {repo}")
    console.print(f"  Target: {relpath} ({len(target.read_text(encoding='utf-8')):,} chars)")

    if not git_available():
        console.print("[red]✗ git is not installed - code evolution needs it[/red]")
        return 1
    if not is_git_repo(Path(repo)):
        console.print(f"[red]✗ {repo} is not a git repository[/red]")
        return 1

    # ── 2. Bug reproduction ─────────────────────────────────────────────
    repro: Optional[BugReproduction] = None
    repro_source: Optional[str] = None
    if repro_script:
        repro_path = Path(repro_script).expanduser()
        if not repro_path.is_file():
            console.print(f"[red]✗ reproduction script not found: {repro_path}[/red]")
            return 1
        repro = BugReproduction(script=repro_path.resolve(), issue=bug_issue)
        try:
            repro_source = repro.script.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            repro_source = None
        console.print(f"  Repro:  {repro.script}")
    else:
        console.print(
            "  Repro:  [yellow]none supplied - fitness cannot prove any bug was "
            "fixed[/yellow]"
        )

    # ── 3. Locate the evolver ───────────────────────────────────────────
    evolver_argv: Optional[list[str]] = None
    if evolver is None:
        try:
            evolver_argv = find_evolver(evolver_cmd)
        except EvolverNotInstalled as exc:
            console.print(f"\n[red]✗ {exc}[/red]")
            console.print(
                "[dim]  Nothing was mutated. This command does not substitute a "
                "different mutation engine when the requested one is absent.[/dim]"
            )
            return 2
        console.print(f"  Evolver: {' '.join(evolver_argv)}")
    else:
        console.print(f"  Evolver: injected ({type(evolver).__name__})")

    if dry_run:
        console.print("\n[bold green]DRY RUN - setup validated successfully.[/bold green]")
        console.print(f"  Would branch from HEAD and mutate {relpath}")
        console.print(f"  Would request {iterations} iteration(s) from the evolver")
        console.print(
            "  Would gate each candidate on: safety guardrails, pytest, "
            + (", ".join(benchmarks) if benchmarks else "no benchmarks")
            + (", bug reproduction" if repro else "")
        )
        console.print("  Would emit a branch and a diff. Never a merge.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root or Path("output")) / "code" / Path(relpath).stem / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    outcomes: list[CandidateOutcome] = []
    winner: Optional[CandidateOutcome] = None
    exit_code = 0

    try:
        organism = CodeOrganism(repo, target, allow_dirty=allow_dirty)
    except OrganismError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    try:
        organism.start()
    except OrganismError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    try:
        baseline_source = organism.baseline_source
        console.print(f"  Branch: [bold]{organism.branch}[/bold] (from {organism.original_ref})")

        evaluator = CodeFitnessEvaluator(
            repo=Path(repo),
            target=target,
            repro=repro,
            benchmarks=benchmarks,
            python=python,
            pytest_subset=pytest_subset,
            strict=strict_gates,
        )

        # ── 4. Baseline snapshot ────────────────────────────────────────
        _step("Baseline")
        baseline = evaluator.snapshot_baseline(baseline_source)
        console.print(
            f"  {_gate_icon(baseline.pytest_result.status)} pytest: "
            f"{baseline.pytest_result.message}"
        )
        for bench in baseline.benchmark_results:
            console.print(f"  {_gate_icon(bench.status)} {bench.name}: {bench.message}")
        if baseline.repro:
            icon = "[green]✓[/green]" if baseline.bug_reproduces else "[yellow]![/yellow]"
            console.print(f"  {icon} repro: {baseline.repro.message}")

        if baseline.pytest_result.status is GateStatus.FAILED:
            console.print(
                "\n[red]✗ The baseline test suite is already failing. "
                "A red baseline cannot gate anything - fix it first.[/red]"
            )
            return 1
        if baseline.pytest_result.status is GateStatus.UNAVAILABLE:
            message = (
                "pytest could not run against this repo, so the hard gate is "
                "not actually gating."
            )
            if strict_gates:
                console.print(f"\n[red]✗ {message} Refusing under --strict-gates.[/red]")
                return 1
            console.print(f"[yellow]⚠ {message}[/yellow]")

        if repro is not None and baseline.repro is not None:
            if baseline.repro.status is ReproStatus.FIXED:
                message = (
                    "the reproduction script already passes at baseline, so it "
                    "does not reproduce the bug"
                )
                if strict_gates:
                    console.print(f"\n[red]✗ {message}. Refusing under --strict-gates.[/red]")
                    return 1
                console.print(f"[yellow]⚠ {message} - bug fitness will be meaningless[/yellow]")
            elif baseline.repro.status is ReproStatus.UNAVAILABLE:
                console.print(f"[yellow]⚠ {baseline.repro.message}[/yellow]")

        # ── 5. Ask the evolver ──────────────────────────────────────────
        _step("Mutation")
        objective = build_objective(
            tool_module=relpath,
            bug_issue=f"issue {bug_issue}" if bug_issue else None,
            reproduction=repro_source,
        )
        console.print(f"  Objective: {objective}")

        job = EvolverJob(
            target_path=relpath,
            source=baseline_source,
            objective=objective,
            iterations=iterations,
            bug_issue=str(bug_issue) if bug_issue else None,
            reproduction=repro_source,
            reproduction_path=str(repro.script) if repro else None,
        )

        engine: ProposesCandidates = evolver or ExternalEvolver(
            cmd=evolver_argv or [],
            repo=Path(repo),
            workdir=out_dir,
        )

        try:
            candidates = engine.propose(job)
        except EvolverNotInstalled as exc:
            console.print(f"[red]✗ {exc}[/red]")
            return 2
        except EvolverError as exc:
            console.print(f"[red]✗ {exc}[/red]")
            return 3

        console.print(f"  Received {len(candidates)} candidate(s)")

        # ── 6. Guardrails, then fitness, one candidate at a time ────────
        _step("Evaluation")
        for candidate in candidates:
            console.print(f"\n  [bold]{candidate.label}[/bold] {candidate.notes}".rstrip())
            mutation = organism.mutate(
                candidate.source,
                label=candidate.label,
                message=(
                    f"evolve({Path(relpath).stem}): candidate {candidate.label}"
                    + (f" for issue {bug_issue}" if bug_issue else "")
                ),
            )
            if mutation.is_empty:
                console.print("    [dim]no textual change[/dim]")

            fitness = evaluator.evaluate(
                baseline_source, candidate.source, label=candidate.label
            )
            for line in fitness.safety.summary().splitlines():
                console.print(f"    {line}")
            if fitness.pytest_result.status is not GateStatus.SKIPPED:
                console.print(
                    f"    {_gate_icon(fitness.pytest_result.status)} pytest: "
                    f"{fitness.pytest_result.message}"
                )
            if fitness.repro:
                console.print(f"    repro: {fitness.repro.message}")
            if fitness.accepted:
                console.print(f"    [green]accepted[/green] score {fitness.total:.3f}")
            else:
                console.print(f"    [red]rejected[/red] {fitness.rejection_reason}")

            outcomes.append(CandidateOutcome(candidate, fitness, mutation))
            # Candidates are alternatives generated from the same baseline, not
            # a sequence. Rewind so the next one is scored against baseline too.
            organism.revert_last()

        # ── 7. Pick a winner and re-apply it ────────────────────────────
        accepted = [o for o in outcomes if o.fitness.accepted]
        if accepted:
            winner = max(accepted, key=lambda o: o.fitness.total)
            final = organism.reapply(
                winner.mutation,
                label=winner.candidate.label,
            ) if winner.mutation else None
            winner_diff = organism.diff_from_baseline()
        else:
            final = None
            winner_diff = ""

        elapsed = time.time() - started

        # ── 8. Report ───────────────────────────────────────────────────
        _step("Results")
        table = Table(title=f"Code evolution: {relpath}")
        table.add_column("Candidate", style="bold")
        table.add_column("Safety")
        table.add_column("pytest")
        table.add_column("Bug")
        table.add_column("Quality", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Verdict")

        for outcome in outcomes:
            fitness = outcome.fitness
            safety_cell = (
                "[green]✓[/green]"
                if fitness.safety.passed
                else f"[red]✗ {len(fitness.safety.violations)}[/red]"
            )
            bug_cell = "-"
            if fitness.repro:
                bug_cell = (
                    "[green]fixed[/green]"
                    if fitness.repro.fixed
                    else f"[red]{fitness.repro.status.value}[/red]"
                )
            table.add_row(
                outcome.candidate.label,
                safety_cell,
                _gate_icon(fitness.pytest_result.status),
                bug_cell,
                f"{fitness.quality.score:.2f}",
                f"{fitness.total:.3f}",
                "[green]accepted[/green]" if fitness.accepted else "[red]rejected[/red]",
            )

        console.print()
        console.print(table)

        # ── 9. Emit the branch and the diff ─────────────────────────────
        (out_dir / "baseline.py").write_text(baseline_source, encoding="utf-8")
        for outcome in outcomes:
            (out_dir / f"{outcome.candidate.label}.py").write_text(
                outcome.candidate.source, encoding="utf-8"
            )

        metrics = {
            "tool": tool,
            "target": relpath,
            "repo": str(repo),
            "branch": organism.branch,
            "baseline_sha": organism.baseline_sha,
            "bug_issue": bug_issue,
            "repro_script": str(repro.script) if repro else None,
            "iterations": iterations,
            "strict_gates": strict_gates,
            "benchmarks": list(benchmarks),
            "objective": objective,
            "elapsed_seconds": round(elapsed, 2),
            "baseline": baseline.to_dict(),
            "candidates": [o.to_dict() for o in outcomes],
            "winner": winner.candidate.label if winner else None,
            "winner_sha": final.sha if final else None,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if winner and winner_diff.strip():
            (out_dir / "winner.py").write_text(winner.candidate.source, encoding="utf-8")
            (out_dir / "winner.diff").write_text(winner_diff, encoding="utf-8")
            console.print(
                Panel(
                    f"Branch:  [bold]{organism.branch}[/bold]\n"
                    f"Commit:  {final.short_sha if final else '-'}\n"
                    f"Diff:    {out_dir / 'winner.diff'}\n"
                    f"Score:   {winner.fitness.total:.3f}\n\n"
                    "Nothing was merged. PLAN.md requires human review of every "
                    "line of evolved code:\n"
                    f"  git diff {organism.baseline_sha[:8] if organism.baseline_sha else 'HEAD'} "
                    f"{organism.branch} -- {relpath}",
                    title="✓ Candidate ready for review",
                    border_style="green",
                )
            )
        elif winner:
            console.print(
                "\n[yellow]⚠ The winning candidate is identical to the baseline - "
                "nothing to review.[/yellow]"
            )
        else:
            console.print(
                "\n[yellow]⚠ No candidate survived the guardrails. "
                "Nothing to review, nothing changed.[/yellow]"
            )
            for outcome in outcomes:
                console.print(
                    f"    {outcome.candidate.label}: {outcome.fitness.rejection_reason}"
                )

        console.print(f"\n  Run artifacts: {out_dir}/")
        console.print(f"  Elapsed: {elapsed:.1f}s")
    finally:
        # Restoring the operator's branch matters more than anything above it,
        # so a failure here is reported rather than raised over the real result.
        original = organism.original_ref
        try:
            organism.close()
            console.print(f"  Restored branch: {original}")
        except OrganismError as exc:
            console.print(
                f"[red]✗ Could not restore branch {original}: {exc}[/red]\n"
                f"[red]  You are still on {organism.branch}.[/red]"
            )

    return exit_code


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--tool", required=True, help="Tool module to evolve (e.g. file_tools)")
@click.option("--bug-issue", default=None, help="GitHub issue number this run targets")
@click.option("--repro-script", default=None, help="Script that reproduces the bug")
@click.option("--iterations", default=10, help="Iterations to request from the evolver")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--evolver-cmd", default=None, help="Path to the Darwinian Evolver CLI")
@click.option("--benchmark", "benchmarks", multiple=True,
              help="Benchmark to gate on (repeatable, e.g. --benchmark tblite)")
@click.option("--python", "python_bin", default=None,
              help="Interpreter used for the hermes-agent test suite and repro")
@click.option("--pytest-subset", multiple=True,
              help="Narrow the pytest gate (repeatable path or -k expression)")
@click.option("--allow-dirty", is_flag=True,
              help="Evolve on top of uncommitted changes in the hermes-agent repo")
@click.option("--strict-gates", is_flag=True,
              help="Treat an unavailable gate as a failure instead of a warning")
@click.option("--dry-run", is_flag=True, help="Validate setup without mutating anything")
def main(
    tool,
    bug_issue,
    repro_script,
    iterations,
    hermes_repo,
    evolver_cmd,
    benchmarks,
    python_bin,
    pytest_subset,
    allow_dirty,
    strict_gates,
    dry_run,
):
    """Evolve hermes-agent tool code with Darwinian Evolver, under guardrails."""
    code = evolve_tool_code(
        tool=tool,
        bug_issue=bug_issue,
        repro_script=repro_script,
        iterations=iterations,
        hermes_repo=hermes_repo,
        evolver_cmd=evolver_cmd,
        strict_gates=strict_gates,
        dry_run=dry_run,
        benchmarks=tuple(benchmarks),
        python=python_bin,
        pytest_subset=tuple(pytest_subset) or None,
        allow_dirty=allow_dirty,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
