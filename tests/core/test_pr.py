"""Tests for pull request emission (evolution/core/pr.py).

Background
----------
The README advertises "Best variant -> PR against hermes-agent" and PLAN.md
specifies that every command should "output a PR branch + summary against
hermes-agent" for a human to merge. That step was never implemented: the only
trace was the ``create_pr`` config flag and a dry-run print, so a successful
``evolve_skill`` run stopped at writing ``output/<skill>/.../evolved_skill.md``
and left the operator to copy the file and stitch the git/gh commands by hand.

These tests lock in the new behavior at both layers: the deterministic title and
body builders, and ``emit_pr`` itself. The local-prep paths run against a real
temporary git repo (faithful, no network). The remote paths use an injected
runner so push and ``gh`` are exercised without touching the network, including
the degradation paths (no git repo, missing ``gh``).
"""

import subprocess
from pathlib import Path

import pytest

from evolution.core.constraints import ConstraintResult
from evolution.core.pr import (
    build_branch_name,
    build_pr_body,
    build_pr_title,
    emit_pr,
    slugify,
)


SKILL_REL = "skills/testing/demo/SKILL.md"
EVOLVED = "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo (evolved)\n\n1. Do the better thing.\n"

METRICS = {
    "skill_name": "demo",
    "iterations": 10,
    "optimizer_model": "openai/gpt-4.1",
    "eval_model": "openai/gpt-4.1-mini",
    "eval_source": "synthetic",
    "baseline_score": 0.785,
    "evolved_score": 0.812,
    "improvement": 0.027,
    "baseline_size": 14836,
    "evolved_size": 11407,
    "train_examples": 10,
    "val_examples": 5,
    "holdout_examples": 5,
}

CONSTRAINTS = [
    ConstraintResult(passed=True, constraint_name="size", message="within limit"),
    ConstraintResult(passed=True, constraint_name="non_empty", message="ok"),
]


# ── helpers ─────────────────────────────────────────────────────────────────

def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _make_git_repo(tmp_path) -> Path:
    """A minimal hermes-agent-style git repo with one committed skill."""
    repo = tmp_path / "hermes-agent"
    skill_dir = repo / "skills" / "testing" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\n\n1. Do the thing.\n"
    )
    _git(["init", "-b", "main"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)
    _git(["add", "."], repo)
    _git(["commit", "-m", "initial"], repo)
    return repo


def _committed_files(repo, sha="HEAD"):
    out = _git(["show", "--name-only", "--pretty=format:", sha], repo).stdout
    return [line for line in out.splitlines() if line.strip()]


class FakeRunner:
    """Records commands and returns canned results; no real process spawned."""

    def __init__(self, missing=()):
        self.calls = []
        self.missing = set(missing)

    def __call__(self, cmd, cwd):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[0] in self.missing:
            raise FileNotFoundError(cmd[0])
        joined = " ".join(cmd)
        if "rev-parse --is-inside-work-tree" in joined:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        if "rev-parse HEAD" in joined:
            return subprocess.CompletedProcess(cmd, 0, "deadbeef\n", "")
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(
                cmd, 0, "https://github.com/NousResearch/hermes-agent/pull/999\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")


# ── builders ──────────────────────────────────────────────────────────────

def test_slugify_makes_branch_safe_names():
    assert slugify("github-code-review") == "github-code-review"
    assert slugify("My Skill!!") == "my-skill"
    assert slugify("---") == "skill"


def test_build_branch_name_and_title():
    assert build_branch_name("demo", "20260628-101500") == "evolve/demo-20260628-101500"
    assert build_pr_title("demo", 0.027) == "evolve: demo skill (holdout +0.027)"


def test_build_pr_body_is_deterministic_and_clean():
    body_a = build_pr_body("demo", METRICS, SKILL_REL, EVOLVED, CONSTRAINTS)
    body_b = build_pr_body("demo", METRICS, SKILL_REL, EVOLVED, CONSTRAINTS)
    assert body_a == body_b  # deterministic

    assert "demo" in body_a
    assert "0.785" in body_a and "0.812" in body_a   # before/after
    assert "+3.4%" in body_a                          # percentage change
    assert SKILL_REL in body_a
    assert "size" in body_a and "non_empty" in body_a  # constraint names
    assert "SHA-256" in body_a

    # House rules for this contribution: no em dashes, no tool branding.
    assert "—" not in body_a
    assert "claude" not in body_a.lower()


# ── emit_pr: local preparation (real git) ────────────────────────────────

def test_emit_pr_prepares_branch_and_commit(tmp_path):
    repo = _make_git_repo(tmp_path)
    result = emit_pr(repo, SKILL_REL, EVOLVED, METRICS, CONSTRAINTS, timestamp="ts1")

    assert result.prepared is True
    assert result.branch == "evolve/demo-ts1"
    assert result.pushed is False and result.pr_url is None
    assert result.commit_sha
    assert result.follow_up_commands  # how to finish by hand

    # Branch is checked out and the evolved content is committed.
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], repo).stdout.strip() == "evolve/demo-ts1"
    assert (repo / SKILL_REL).read_text() == EVOLVED
    assert SKILL_REL in _committed_files(repo)


def test_emit_pr_commits_only_the_skill_file(tmp_path):
    repo = _make_git_repo(tmp_path)
    (repo / "unrelated.txt").write_text("dirty working tree")  # untracked noise

    emit_pr(repo, SKILL_REL, EVOLVED, METRICS, CONSTRAINTS, timestamp="ts2")

    committed = _committed_files(repo)
    assert committed == [SKILL_REL]
    assert "unrelated.txt" not in committed


# ── emit_pr: degradation paths (never fatal) ─────────────────────────────

def test_emit_pr_non_git_repo_degrades(tmp_path):
    plain = tmp_path / "not-a-repo"
    (plain / "skills" / "testing" / "demo").mkdir(parents=True)

    result = emit_pr(plain, SKILL_REL, EVOLVED, METRICS, CONSTRAINTS, timestamp="ts3")

    assert result.prepared is False
    assert any("not a git repository" in m for m in result.messages)
    assert result.follow_up_commands  # cp + push + gh, so the user can finish
    # Nothing was written into a non-repo, so we cannot clobber anything.
    assert not (plain / SKILL_REL).exists()


def test_emit_pr_open_remote_pushes_and_opens_pr(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "skills" / "testing" / "demo").mkdir(parents=True)
    runner = FakeRunner()

    result = emit_pr(
        repo, SKILL_REL, EVOLVED, METRICS, CONSTRAINTS,
        base="main", open_remote=True, timestamp="ts4", run=runner,
    )

    assert result.prepared is True
    assert result.pushed is True
    assert result.pr_url == "https://github.com/NousResearch/hermes-agent/pull/999"

    gh_calls = [c for c in runner.calls if c[0] == "gh"]
    assert gh_calls, "gh pr create should have been invoked"
    gh = gh_calls[0]
    assert "--head" in gh and "evolve/demo-ts4" in gh
    assert "--base" in gh and "main" in gh


def test_emit_pr_missing_gh_degrades_after_push(tmp_path):
    repo = tmp_path / "hermes-agent"
    (repo / "skills" / "testing" / "demo").mkdir(parents=True)
    runner = FakeRunner(missing={"gh"})

    result = emit_pr(
        repo, SKILL_REL, EVOLVED, METRICS, CONSTRAINTS,
        open_remote=True, timestamp="ts5", run=runner,
    )

    assert result.pushed is True       # push still succeeded
    assert result.pr_url is None       # but the PR could not be opened
    assert any("could not open the PR" in m for m in result.messages)
