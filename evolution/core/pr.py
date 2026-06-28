"""Turn a validated, improved skill into a reviewable pull request.

This is the last step of the pipeline that the README advertises
("Best variant -> PR against hermes-agent") and that PLAN.md specifies
("All commands output a PR branch + summary against hermes-agent. Human
merges."), but which was never actually implemented: the only trace of it
in the codebase was the ``create_pr`` config flag and a dry-run print.

Until now ``evolve_skill`` stopped after writing ``output/<skill>/.../evolved_skill.md``,
so a successful run left the operator to hand-copy the file into the
hermes-agent repo and stitch together the git/gh commands themselves.

Design notes
------------
* Safe by default. Preparing a pull request means writing the evolved skill
  into the hermes-agent working tree, creating a fresh branch, and committing
  *only that one file*. Pushing and opening the PR over the network is opt-in
  (``open_remote=True``); the default produces a local branch plus the exact
  commands to finish by hand. This mirrors the PLAN.md contract of emitting "a
  PR branch + summary" that a human reviews.
* Never destructive. We only ever ``git add`` the single skill file, never
  ``git add -A``, so an unrelated dirty working tree is left untouched. We
  create a new branch rather than committing onto the current one, and we
  never force-push.
* Never fatal. A missing git repo, a missing ``gh`` binary, or a failed push
  degrades to a clear message plus copy-pasteable follow-up commands instead
  of raising. A best-effort convenience must not sink a run that already
  produced a good skill.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from evolution.core.constraints import ConstraintResult


# A runner is anything that behaves like subprocess.run with
# capture_output=True, text=True. Injectable so tests need no network.
Runner = Callable[[Sequence[str], Path], "subprocess.CompletedProcess"]


def _subprocess_run(cmd: Sequence[str], cwd: Path) -> "subprocess.CompletedProcess":
    return subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True)


@dataclass
class PREmissionResult:
    """Outcome of preparing (and optionally opening) a pull request."""

    prepared: bool                       # local branch + commit created
    branch: str
    base: str
    skill_rel_path: str
    commit_sha: Optional[str] = None
    pushed: bool = False
    pr_url: Optional[str] = None
    follow_up_commands: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def slugify(name: str) -> str:
    """Reduce a skill name to a branch-safe slug."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name.lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "skill"


def build_branch_name(skill_name: str, timestamp: str) -> str:
    return f"evolve/{slugify(skill_name)}-{timestamp}"


def build_pr_title(skill_name: str, improvement: float) -> str:
    return f"evolve: {skill_name} skill (holdout {improvement:+.3f})"


def _pct(delta: float, baseline: float) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{delta / baseline * 100:+.1f}%"


def build_pr_body(
    skill_name: str,
    metrics: dict,
    skill_rel_path: str,
    evolved_full: str,
    constraint_results: Optional[Sequence[ConstraintResult]] = None,
) -> str:
    """Render a deterministic, human-readable PR body from a run's metrics.

    Deterministic so the same run always produces the same description, which
    keeps the output reviewable and easy to diff.
    """
    baseline = float(metrics.get("baseline_score", 0.0))
    evolved = float(metrics.get("evolved_score", 0.0))
    improvement = float(metrics.get("improvement", evolved - baseline))
    base_size = int(metrics.get("baseline_size", 0))
    new_size = int(metrics.get("evolved_size", len(evolved_full)))
    sha = hashlib.sha256(evolved_full.encode("utf-8")).hexdigest()

    lines = [
        f"This updates the `{skill_name}` skill with a version produced by the "
        "self-evolution pipeline (DSPy + GEPA). It is offered for human review, "
        "not auto-merge.",
        "",
        "## Results (holdout set)",
        "",
        "| Metric | Baseline | Evolved | Change |",
        "| --- | ---: | ---: | ---: |",
        f"| Holdout score | {baseline:.3f} | {evolved:.3f} | "
        f"{improvement:+.3f} ({_pct(improvement, baseline)}) |",
        f"| Skill size | {base_size:,} chars | {new_size:,} chars | "
        f"{new_size - base_size:+,} chars |",
        "",
        "## How it was produced",
        "",
        f"- Optimizer: GEPA, {metrics.get('iterations', 'n/a')} iterations",
        f"- Optimizer model: {metrics.get('optimizer_model', 'n/a')}",
        f"- Eval model: {metrics.get('eval_model', 'n/a')}",
        f"- Eval dataset: {metrics.get('train_examples', 0)} train / "
        f"{metrics.get('val_examples', 0)} val / "
        f"{metrics.get('holdout_examples', 0)} holdout "
        f"(source: {metrics.get('eval_source', 'n/a')})",
        f"- Evolved file SHA-256: `{sha}`",
    ]

    if constraint_results:
        lines += ["", "## Constraints", ""]
        for c in constraint_results:
            box = "x" if c.passed else " "
            lines.append(f"- [{box}] {c.constraint_name}: {c.message}")

    lines += [
        "",
        "## Files",
        "",
        f"- `{skill_rel_path}`",
        "",
        f"Reproduce with `python -m evolution.skills.evolve_skill --skill {skill_name}`. "
        "Please read the diff before merging.",
    ]
    return "\n".join(lines)


def _is_git_repo(repo: Path, run: Runner) -> bool:
    try:
        res = run(["git", "rev-parse", "--is-inside-work-tree"], repo)
    except FileNotFoundError:
        return False
    return res.returncode == 0 and (res.stdout or "").strip() == "true"


def _follow_up_commands(repo: Path, branch: str, base: str, title: str) -> list[str]:
    """The commands a human can run to push and open the PR by hand."""
    quoted_title = shlex.quote(title)
    return [
        f"git -C {shlex.quote(str(repo))} push -u origin {shlex.quote(branch)}",
        f"gh pr create --base {shlex.quote(base)} --head {shlex.quote(branch)} "
        f"--title {quoted_title} --body-file <(...)  # from the hermes-agent repo",
    ]


def emit_pr(
    hermes_repo: Path,
    skill_rel_path: str,
    evolved_full: str,
    metrics: dict,
    constraint_results: Optional[Sequence[ConstraintResult]] = None,
    *,
    base: str = "main",
    open_remote: bool = False,
    timestamp: str = "",
    run: Runner = _subprocess_run,
) -> PREmissionResult:
    """Prepare a pull request for an evolved skill against the hermes-agent repo.

    By default this writes the evolved skill into ``hermes_repo`` at
    ``skill_rel_path``, creates a fresh ``evolve/<skill>-<timestamp>`` branch,
    and commits that single file. With ``open_remote=True`` it also pushes the
    branch and opens the PR with ``gh``. Anything that cannot be done (no git
    repo, no ``gh``, push rejected) becomes a message plus follow-up commands;
    this function does not raise on those conditions.
    """
    hermes_repo = Path(hermes_repo)
    skill_name = str(metrics.get("skill_name", Path(skill_rel_path).parent.name))
    improvement = float(metrics.get("improvement", 0.0))
    branch = build_branch_name(skill_name, timestamp or "manual")
    title = build_pr_title(skill_name, improvement)
    body = build_pr_body(skill_name, metrics, skill_rel_path, evolved_full, constraint_results)

    result = PREmissionResult(
        prepared=False,
        branch=branch,
        base=base,
        skill_rel_path=skill_rel_path,
    )

    if not _is_git_repo(hermes_repo, run):
        result.messages.append(
            f"{hermes_repo} is not a git repository, so no branch was created. "
            "The evolved skill is saved in the run's output directory; commit it "
            "into a hermes-agent checkout to open a PR."
        )
        result.follow_up_commands = [
            f"cp <output>/evolved_skill.md {shlex.quote(str(hermes_repo / skill_rel_path))}",
            *_follow_up_commands(hermes_repo, branch, base, title),
        ]
        return result

    # Write the evolved skill into the working tree.
    skill_file = hermes_repo / skill_rel_path
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(evolved_full)

    co = run(["git", "checkout", "-b", branch], hermes_repo)
    if co.returncode != 0:
        result.messages.append(
            f"Could not create branch {branch}: {(co.stderr or co.stdout or '').strip()}. "
            "The evolved skill was written to the working tree; commit it manually."
        )
        result.follow_up_commands = _follow_up_commands(hermes_repo, branch, base, title)
        return result

    # Stage only the skill file so an unrelated dirty tree is left alone.
    run(["git", "add", skill_rel_path], hermes_repo)
    commit = run(["git", "commit", "-m", title, "-m", body], hermes_repo)
    if commit.returncode != 0:
        result.messages.append(
            f"Branch {branch} was created but the commit failed: "
            f"{(commit.stderr or commit.stdout or '').strip()}."
        )
        result.follow_up_commands = _follow_up_commands(hermes_repo, branch, base, title)
        return result

    rev = run(["git", "rev-parse", "HEAD"], hermes_repo)
    result.prepared = True
    result.commit_sha = (rev.stdout or "").strip() if rev.returncode == 0 else None
    result.messages.append(f"Prepared branch {branch} with the evolved {skill_name} skill.")
    result.follow_up_commands = _follow_up_commands(hermes_repo, branch, base, title)

    if not open_remote:
        return result

    # Opt-in network mutation from here on.
    try:
        push = run(["git", "push", "-u", "origin", branch], hermes_repo)
    except FileNotFoundError:
        push = None
    if push is None or push.returncode != 0:
        detail = "" if push is None else (push.stderr or push.stdout or "").strip()
        result.messages.append(
            f"Could not push {branch} to origin{(': ' + detail) if detail else ''}. "
            "Run the follow-up commands once the remote is reachable."
        )
        return result
    result.pushed = True

    try:
        pr = run(
            [
                "gh", "pr", "create",
                "--base", base,
                "--head", branch,
                "--title", title,
                "--body", body,
            ],
            hermes_repo,
        )
    except FileNotFoundError:
        pr = None
    if pr is None or pr.returncode != 0:
        detail = "gh not found" if pr is None else (pr.stderr or pr.stdout or "").strip()
        result.messages.append(
            f"Pushed {branch}, but could not open the PR automatically ({detail}). "
            "Open it from the hermes-agent repo with gh or the web UI."
        )
        return result

    result.pr_url = (pr.stdout or "").strip().splitlines()[-1] if pr.stdout else None
    result.messages.append(f"Opened pull request: {result.pr_url}")
    return result
