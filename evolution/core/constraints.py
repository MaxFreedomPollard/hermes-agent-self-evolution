"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig

# Common words that carry no topical signal. Deliberately small: the
# 4+ char term filter already drops most function words, and an
# aggressive stopword list would risk erasing domain vocabulary.
_STOPWORDS = frozenset({
    "about", "after", "also", "always", "been", "before", "being", "between",
    "both", "cannot", "could", "does", "each", "either", "every", "from",
    "have", "here", "instead", "into", "just", "like", "make", "more", "most",
    "must", "never", "only", "other", "over", "should", "some", "such",
    "than", "that", "them", "then", "there", "these", "they", "this",
    "those", "under", "very", "want", "well", "were", "what", "when",
    "where", "which", "while", "will", "with", "without", "would", "your",
})


def _content_terms(text: str) -> Counter:
    """Frequency of topical terms: lowercase words of 4+ chars minus stopwords."""
    words = re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower())
    return Counter(w for w in words if w not in _STOPWORDS)


def semantic_similarity(baseline: str, evolved: str) -> float:
    """Cosine similarity between the content-term frequencies of two texts.

    A deterministic, dependency-free proxy for topical drift: a rewrite of
    the same skill keeps most of its domain vocabulary (commands, API names,
    concepts), while text that drifted to another purpose does not. Returns
    a value in [0, 1].
    """
    a = _content_terms(baseline)
    b = _content_terms(evolved)
    if not a or not b:
        return 1.0 if not a and not b else 0.0

    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


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
        """Run all applicable constraints. Returns list of results."""
        results = []

        # 1. Size limits
        results.append(self._check_size(artifact_text, artifact_type))

        # 2. Baseline-relative checks (growth + semantic preservation)
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))
            results.append(self._check_semantic_preservation(artifact_text, baseline_text))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
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
            else:
                # Extract failure summary
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
            limit = self.config.max_skill_size  # Default

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
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
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="growth_limit",
                message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
            )

    def _check_semantic_preservation(self, text: str, baseline: str) -> ConstraintResult:
        """Reject evolved text that drifted away from the baseline's purpose.

        PLAN.md constraint 4: a skill for GitHub code review must still
        perform code reviews, not drift into something else. Measured as
        content-term cosine similarity against the baseline. Calibrated on
        real hermes-agent skills: rewrites of the same skill score 0.84+,
        while pairs of unrelated skills score below 0.16, so the default
        threshold of 0.4 separates them with wide margin on both sides.
        A threshold of 0 disables the check.
        """
        threshold = self.config.min_semantic_similarity
        similarity = semantic_similarity(baseline, text)

        if similarity >= threshold:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message=f"Topic preserved: similarity {similarity:.2f} (min {threshold:.2f})",
            )

        # Show which topical vocabulary disappeared, for human review.
        baseline_terms = _content_terms(baseline)
        evolved_terms = _content_terms(text)
        lost = [t for t, _ in baseline_terms.most_common(30) if t not in evolved_terms][:8]
        return ConstraintResult(
            passed=False,
            constraint_name="semantic_preservation",
            message=f"Topic drift: similarity {similarity:.2f} below minimum {threshold:.2f}",
            details=f"Baseline terms missing from evolved text: {', '.join(lost)}" if lost else None,
        )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        has_frontmatter = text.strip().startswith("---")
        has_name = "name:" in text[:500] if has_frontmatter else False
        has_description = "description:" in text[:500] if has_frontmatter else False

        if has_frontmatter and has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        else:
            missing = []
            if not has_frontmatter:
                missing.append("YAML frontmatter (---)")
            if not has_name:
                missing.append("name field")
            if not has_description:
                missing.append("description field")
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill missing: {', '.join(missing)}",
            )
