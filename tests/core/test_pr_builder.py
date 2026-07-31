"""Tests for the deployment step.

Real git repositories in tmp_path, because the branch handling is the part that
can damage someone's checkout and mocking it would test nothing. Nothing here
touches a network: no test pushes, and ``gh`` is never invoked.
"""

import subprocess

import pytest

from evolution.core.cost import CostReport, LMCall
from evolution.core.pr_builder import (
    GitError,
    RejectedCandidate,
    ScoreLine,
    build_pull_request,
    render_body,
)


def git(repo, *args):
    subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "Tester")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "tool.py").write_text("DESCRIPTION = 'before'\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


@pytest.fixture
def evolved(repo):
    (repo / "tool.py").write_text("DESCRIPTION = 'after'\n")
    return repo


def build(repo, **kw):
    kw.setdefault("target", "read_file")
    kw.setdefault("phase", "Phase 2")
    kw.setdefault("timestamp", "20260731_010203")
    kw.setdefault("files", ["tool.py"])
    return build_pull_request(repo=repo, **kw)


class TestBranch:
    def test_branch_follows_the_plan_naming(self, evolved):
        plan = build(evolved)
        assert plan.branch == "evolve/read_file-20260731_010203"

    def test_the_branch_really_exists_and_is_checked_out(self, evolved):
        plan = build(evolved)
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert head == plan.branch

    def test_the_change_is_committed(self, evolved):
        build(evolved)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert status == ""

    def test_restore_returns_to_the_original_ref(self, evolved):
        before = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        plan = build(evolved)
        plan.restore()
        after = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert after == before

    def test_a_non_git_directory_is_refused(self, tmp_path):
        with pytest.raises(GitError, match="not a git repository"):
            build(tmp_path)

    def test_the_timestamp_is_supplied_not_read_from_the_clock(self, evolved):
        """Two runs with the same timestamp produce the same branch name."""
        plan = build(evolved)
        plan.restore()
        assert "20260731_010203" in plan.branch


class TestBody:
    def test_carries_every_split_plan_asks_for(self, evolved):
        plan = build(
            evolved,
            scores=[
                ScoreLine("train", 0.6, 0.7),
                ScoreLine("val", 0.55, 0.68),
                ScoreLine("holdout", 0.5, 0.64),
            ],
        )
        for split in ("train", "val", "holdout"):
            assert split in plan.body

    def test_includes_the_diff(self, evolved):
        body = build(evolved).body
        assert "```diff" in body
        assert "-DESCRIPTION = 'before'" in body
        assert "+DESCRIPTION = 'after'" in body

    def test_includes_the_cost(self, evolved):
        cost = CostReport(calls=[LMCall("m", 100, 50, 0.02)])
        assert "$0.0200" in build(evolved, cost=cost).body

    def test_an_unpriced_run_is_not_reported_as_cheap(self, evolved):
        cost = CostReport(calls=[LMCall("m", 100, 50, None)])
        body = build(evolved, cost=cost).body
        assert "at least" in body
        assert "no price available" in body

    def test_unmeasured_cost_says_so(self, evolved):
        assert "not measured" in build(evolved).body

    def test_lists_rejected_candidates(self, evolved):
        plan = build(
            evolved,
            rejected=[
                RejectedCandidate("cand-1", "size_limit: 528/500 chars"),
                RejectedCandidate("cand-2", "factual_accuracy: unknown parameter"),
            ],
        )
        assert "Rejected along the way" in plan.body
        assert "size_limit: 528/500 chars" in plan.body
        assert "factual_accuracy" in plan.body

    def test_lists_gates(self, evolved):
        plan = build(evolved, gates=["pytest: 2550 passed", "tblite: unavailable"])
        assert "tblite: unavailable" in plan.body

    def test_a_long_diff_is_clipped_with_a_pointer_to_the_branch(self):
        body = render_body(
            target="t", phase="p", scores=[], diff="\n".join(f"+line {i}" for i in range(900))
        )
        assert "clipped at 400 lines of 900" in body

    def test_body_is_written_next_to_the_run_artifacts(self, evolved, tmp_path):
        plan = build(evolved)
        path = plan.write_body(tmp_path / "out")
        assert path.name == "PULL_REQUEST.md"
        assert path.read_text() == plan.body


class TestCommitMessage:
    def test_follows_the_plan_shape(self, evolved):
        plan = build(
            evolved,
            scores=[ScoreLine("holdout", 0.5, 0.64)],
            optimizer="GEPA",
            iterations=10,
            dataset="synthetic, 120 examples",
        )
        assert plan.commit_message.startswith("evolve: read_file")
        assert "Optimizer: GEPA (10 iterations)" in plan.commit_message
        assert "Eval dataset: synthetic, 120 examples" in plan.commit_message
        assert "holdout: 0.500 -> 0.640 (+0.140)" in plan.commit_message

    def test_the_real_commit_carries_it(self, evolved):
        plan = build(evolved, scores=[ScoreLine("holdout", 0.5, 0.64)])
        message = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout
        assert "evolve: read_file" in message


class TestNothingIsPublished:
    def test_building_does_not_add_a_remote_or_push(self, evolved):
        build(evolved)
        remotes = subprocess.run(
            ["git", "remote"], cwd=str(evolved), capture_output=True, text=True
        ).stdout.strip()
        assert remotes == ""

    def test_push_is_a_separate_explicit_call(self, evolved):
        plan = build(evolved)
        # No remote configured, so a push must fail loudly rather than silently
        # succeed or be attempted as part of building.
        with pytest.raises(GitError):
            plan.push()

    def test_open_refuses_without_gh(self, evolved, monkeypatch):
        monkeypatch.setattr("evolution.core.pr_builder.shutil.which", lambda _: None)
        plan = build(evolved)
        with pytest.raises(GitError, match="gh is not installed"):
            plan.open()
