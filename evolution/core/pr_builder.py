"""Turn a finished run into a reviewable pull request.

PLAN.md constraint 5 is "Deployment via PR (Never Direct Commit)": every
evolved change reaches hermes-agent as a branch and a PR whose body carries the
before/after scores on train, validation and holdout, the full diff, the cost of
the run, and any constraint violations caught and rejected along the way. Until
now the pipeline stopped one step short of that - it wrote evolved text and a
metrics file into an output directory and left the reviewer to assemble the rest
by hand. ``EvolutionConfig.create_pr`` has defaulted to ``True`` the whole time
with nothing reading it.

**Nothing here reaches the network on its own.** Building a branch and writing a
PR body are local operations and happen by default; pushing that branch and
opening the PR are separate, explicitly requested steps. An optimization run
that phoned out to GitHub because a config field defaulted to True would be a
bad surprise, and "never direct commit" is a rule about review, not a licence to
publish automatically. :meth:`PullRequestPlan.push` and
:meth:`PullRequestPlan.open` exist for a caller that has been told to, and both
refuse rather than guess when the remote or the CLI is missing.

The branch name follows PLAN.md's ``evolve/<target>-<timestamp>``. The timestamp
is passed in rather than read from the clock so a run is reproducible and a test
can assert on the name.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from evolution.core.cost import CostReport

__all__ = [
    "ScoreLine",
    "RejectedCandidate",
    "PullRequestPlan",
    "build_pull_request",
    "GitError",
]


class GitError(RuntimeError):
    """A git or gh invocation failed, with its stderr attached."""


def _run(args: Sequence[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(
            list(args), cwd=str(cwd), capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError as exc:
        raise GitError(f"{args[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"{' '.join(args)} timed out") from exc
    if proc.returncode != 0:
        raise GitError(f"{' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


@dataclass(frozen=True)
class ScoreLine:
    """One split's before and after, for the PR body's headline table."""

    split: str
    baseline: float
    evolved: float
    detail: str = ""

    @property
    def delta(self) -> float:
        return self.evolved - self.baseline

    def row(self) -> str:
        detail = f" | {self.detail}" if self.detail else " |"
        return (
            f"| {self.split} | {self.baseline:.3f} | {self.evolved:.3f} "
            f"| {self.delta:+.3f}{detail}"
        )


@dataclass(frozen=True)
class RejectedCandidate:
    """A variant the run threw away, and why.

    PLAN.md asks for these in the body on purpose. A PR that shows only the
    winner hides how hard the gates were working, and a reviewer who cannot see
    what was rejected cannot tell a careful run from a lucky one.
    """

    label: str
    reason: str


@dataclass
class PullRequestPlan:
    """A branch, a commit message, and a PR body, all on disk and nothing sent."""

    repo: Path
    branch: str
    title: str
    body: str
    commit_message: str
    files: tuple[str, ...] = ()
    created_branch: bool = False
    original_ref: str = ""
    body_path: Optional[Path] = None

    def write_body(self, output_dir: Path) -> Path:
        """Save the PR body next to the run's other artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "PULL_REQUEST.md"
        path.write_text(self.body, encoding="utf-8")
        self.body_path = path
        return path

    def push(self, remote: str = "origin") -> str:
        """Push the branch. Only call this when the operator asked for it."""
        return _run(["git", "push", "-u", remote, self.branch], self.repo)

    def open(self, base: str = "main") -> str:
        """Open the PR with ``gh``. Only call this when the operator asked for it."""
        if shutil.which("gh") is None:
            raise GitError(
                "gh is not installed, so the PR cannot be opened from here. "
                "The branch and PULL_REQUEST.md are ready to use by hand."
            )
        return _run(
            [
                "gh", "pr", "create",
                "--base", base,
                "--head", self.branch,
                "--title", self.title,
                "--body", self.body,
            ],
            self.repo,
        )

    def restore(self) -> None:
        """Return the checkout to the ref it was on before the branch was made."""
        if self.created_branch and self.original_ref:
            _run(["git", "checkout", self.original_ref], self.repo)

    def to_dict(self) -> dict:
        return {
            "branch": self.branch,
            "title": self.title,
            "files": list(self.files),
            "created_branch": self.created_branch,
            "original_ref": self.original_ref,
            "body_path": str(self.body_path) if self.body_path else None,
        }


def _current_ref(repo: Path) -> str:
    """The branch name, or the commit sha when the checkout is detached."""
    ref = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).strip()
    if ref and ref != "HEAD":
        return ref
    return _run(["git", "rev-parse", "HEAD"], repo).strip()


def render_body(
    *,
    target: str,
    phase: str,
    scores: Sequence[ScoreLine],
    diff: str,
    cost: Optional[CostReport] = None,
    rejected: Sequence[RejectedCandidate] = (),
    gates: Sequence[str] = (),
    dataset: str = "",
    optimizer: str = "",
    iterations: Optional[int] = None,
    statistics: str = "",
    notes: Sequence[str] = (),
    max_diff_lines: int = 400,
) -> str:
    """Render the PR body PLAN.md specifies, in that order."""
    lines: list[str] = []

    lines.append(f"Evolved `{target}` with {phase}.")
    lines.append("")

    if scores:
        lines.append("## Scores")
        lines.append("")
        lines.append("| Split | Before | After | Change | Notes |")
        lines.append("|---|---:|---:|---:|---|")
        lines.extend(s.row() for s in scores)
        lines.append("")

    if statistics:
        lines.append("## Evidence")
        lines.append("")
        lines.append(statistics)
        lines.append("")

    if gates:
        lines.append("## Gates")
        lines.append("")
        lines.extend(f"- {g}" for g in gates)
        lines.append("")

    if rejected:
        lines.append("## Rejected along the way")
        lines.append("")
        lines.append(
            f"{len(rejected)} candidate(s) were produced and refused before this one:"
        )
        lines.append("")
        lines.extend(f"- `{r.label}`: {r.reason}" for r in rejected)
        lines.append("")

    lines.append("## Run")
    lines.append("")
    if optimizer:
        detail = f"{optimizer}"
        if iterations is not None:
            detail += f", {iterations} iteration(s)"
        lines.append(f"- Optimizer: {detail}")
    if dataset:
        lines.append(f"- Eval dataset: {dataset}")
    lines.append(f"- Cost: {cost.describe() if cost else 'not measured'}")
    lines.extend(f"- {n}" for n in notes)
    lines.append("")

    if diff:
        diff_lines = diff.splitlines()
        clipped = len(diff_lines) > max_diff_lines
        shown = diff_lines[:max_diff_lines]
        lines.append("## Diff")
        lines.append("")
        lines.append("```diff")
        lines.extend(shown)
        lines.append("```")
        if clipped:
            lines.append("")
            lines.append(
                f"_Diff clipped at {max_diff_lines} lines of "
                f"{len(diff_lines)}; the branch has all of it._"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_pull_request(
    *,
    repo: Path,
    target: str,
    phase: str,
    timestamp: str,
    files: Sequence[str],
    scores: Sequence[ScoreLine] = (),
    cost: Optional[CostReport] = None,
    rejected: Sequence[RejectedCandidate] = (),
    gates: Sequence[str] = (),
    dataset: str = "",
    optimizer: str = "",
    iterations: Optional[int] = None,
    statistics: str = "",
    notes: Sequence[str] = (),
    commit: bool = True,
) -> PullRequestPlan:
    """Create ``evolve/<target>-<timestamp>``, commit *files*, and render the body.

    The working tree is expected to already hold the evolved content: this
    stages what it is given and does not decide what changed. Nothing is pushed
    and no PR is opened.

    Raises :class:`GitError` when *repo* is not a git repository, so a caller
    never believes a branch exists that does not.
    """
    repo = Path(repo)
    if not (repo / ".git").exists():
        raise GitError(f"{repo} is not a git repository")

    original = _current_ref(repo)
    branch = f"evolve/{target}-{timestamp}"
    _run(["git", "checkout", "-b", branch], repo)

    diff = ""
    if commit and files:
        _run(["git", "add", "--", *files], repo)
        diff = _run(["git", "diff", "--cached"], repo)

    headline = ""
    if scores:
        best = scores[-1]
        headline = f" - {best.split} {best.baseline:.3f} to {best.evolved:.3f}"

    title = f"evolve: {target}"
    body = render_body(
        target=target,
        phase=phase,
        scores=scores,
        diff=diff,
        cost=cost,
        rejected=rejected,
        gates=gates,
        dataset=dataset,
        optimizer=optimizer,
        iterations=iterations,
        statistics=statistics,
        notes=notes,
    )

    message_lines = [f"evolve: {target}{headline}", ""]
    if optimizer:
        detail = optimizer
        if iterations is not None:
            detail += f" ({iterations} iterations)"
        message_lines.append(f"Optimizer: {detail}")
    if dataset:
        message_lines.append(f"Eval dataset: {dataset}")
    for score in scores:
        message_lines.append(
            f"{score.split}: {score.baseline:.3f} -> {score.evolved:.3f} "
            f"({score.delta:+.3f})"
        )
    if cost is not None:
        message_lines.append(f"Cost: {cost.describe()}")
    commit_message = "\n".join(message_lines).rstrip() + "\n"

    if commit and files:
        _run(["git", "commit", "-m", commit_message], repo)

    return PullRequestPlan(
        repo=repo,
        branch=branch,
        title=title,
        body=body,
        commit_message=commit_message,
        files=tuple(files),
        created_branch=True,
        original_ref=original,
    )
