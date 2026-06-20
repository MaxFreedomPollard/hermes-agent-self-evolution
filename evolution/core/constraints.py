"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be considered
valid. Failed constraints = immediate rejection.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.core.config import EvolutionConfig


_SKILL_FRONTMATTER_RE = re.compile(
    r"\A[ \t]*---[ \t]*\r?\n(?P<frontmatter>.*?)(?:\r?\n)---[ \t]*(?:\r?\n|\Z)(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass
class ConstraintResult:
    """Result of constraint validation."""

    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
    ) -> list[ConstraintResult]:
        """Run all applicable constraints."""

        results = []
        results.append(self._check_size(artifact_text, artifact_type))
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))
        results.append(self._check_non_empty(artifact_text))
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))
        return results

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(hermes_repo),
            )
            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )

            last_lines = result.stdout.strip().split("\n")[-5:] if result.stdout else []
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite failed",
                details="\n".join(last_lines),
            )
        except subprocess.TimeoutExpired:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite timed out (300s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        size = len(text)
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        else:
            limit = self.config.max_skill_size

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="size_limit",
            message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
        )

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth
        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_limit",
            message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
        )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="non_empty",
            message="Artifact is empty",
        )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a full skill file has a closed YAML frontmatter block and body."""

        stripped = text.strip()
        match = _SKILL_FRONTMATTER_RE.match(stripped)
        frontmatter = match.group("frontmatter").strip() if match else ""
        body = match.group("body").strip() if match else ""

        has_frontmatter = match is not None
        has_name = bool(re.search(r"(?m)^name\s*:\s*\S+", frontmatter))
        has_description = bool(re.search(r"(?m)^description\s*:\s*\S+", frontmatter))
        has_body = bool(body)

        if has_frontmatter and has_name and has_description and has_body:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has closed frontmatter (name + description) and body",
            )

        missing = []
        if not has_frontmatter:
            missing.append("closed YAML frontmatter block (--- ... ---)")
        if not has_name:
            missing.append("name field")
        if not has_description:
            missing.append("description field")
        if not has_body:
            missing.append("markdown body")
        return ConstraintResult(
            passed=False,
            constraint_name="skill_structure",
            message=f"Skill missing: {', '.join(missing)}",
        )
