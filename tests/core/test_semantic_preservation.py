"""Tests for the semantic preservation constraint (PLAN.md constraint 4)."""

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator, semantic_similarity

# Two texts about the same skill, written very differently.
REVIEW_SKILL = """
# GitHub Code Review

Fetch the pull request diff with `gh pr diff`. Read every changed file.
Look for bugs, security issues, and missing tests. Post review comments
with `gh pr review` explaining each finding with file and line references.
"""

REVIEW_SKILL_REWRITTEN = """
# Reviewing Pull Requests on GitHub

Start by pulling the diff (`gh pr diff`) and reading the changed files
carefully. Hunt for security problems, bugs, and gaps in test coverage.
Then submit findings as review comments (`gh pr review`), each anchored
to a file and line so the author can act on them.
"""

# A different topic entirely.
BAKING_TEXT = """
# Sourdough Baking

Feed the starter twice daily. Mix flour, water, and salt, then rest the
dough for autolyse. Fold every thirty minutes, shape the boule, proof
overnight in the fridge, and bake in a dutch oven at high heat.
"""


def _config():
    return EvolutionConfig(hermes_agent_path=None)


class TestSemanticSimilarity:
    def test_identical_text_is_one(self):
        assert semantic_similarity(REVIEW_SKILL, REVIEW_SKILL) == pytest.approx(1.0)

    def test_rewrite_of_same_skill_scores_high(self):
        assert semantic_similarity(REVIEW_SKILL, REVIEW_SKILL_REWRITTEN) > 0.5

    def test_unrelated_topic_scores_low(self):
        assert semantic_similarity(REVIEW_SKILL, BAKING_TEXT) < 0.2

    def test_empty_evolved_text_is_zero(self):
        assert semantic_similarity(REVIEW_SKILL, "") == pytest.approx(0.0)

    def test_two_empty_texts_are_identical(self):
        assert semantic_similarity("", "") == pytest.approx(1.0)

    def test_symmetry(self):
        forward = semantic_similarity(REVIEW_SKILL, BAKING_TEXT)
        backward = semantic_similarity(BAKING_TEXT, REVIEW_SKILL)
        assert forward == pytest.approx(backward)


class TestSemanticPreservationConstraint:
    def test_runs_only_with_baseline(self):
        validator = ConstraintValidator(_config())
        names = [r.constraint_name for r in validator.validate_all(REVIEW_SKILL, "skill")]
        assert "semantic_preservation" not in names

        names = [
            r.constraint_name
            for r in validator.validate_all(REVIEW_SKILL, "skill", baseline_text=REVIEW_SKILL)
        ]
        assert "semantic_preservation" in names

    def test_faithful_rewrite_passes(self):
        validator = ConstraintValidator(_config())
        results = validator.validate_all(
            REVIEW_SKILL_REWRITTEN, "skill", baseline_text=REVIEW_SKILL
        )
        result = next(r for r in results if r.constraint_name == "semantic_preservation")
        assert result.passed
        assert "similarity" in result.message

    def test_topic_drift_fails_with_lost_terms(self):
        validator = ConstraintValidator(_config())
        results = validator.validate_all(BAKING_TEXT, "skill", baseline_text=REVIEW_SKILL)
        result = next(r for r in results if r.constraint_name == "semantic_preservation")
        assert not result.passed
        assert "drift" in result.message.lower()
        # The details should name baseline vocabulary that disappeared.
        assert result.details
        assert "review" in result.details or "diff" in result.details

    def test_zero_threshold_disables_the_check(self):
        config = _config()
        config.min_semantic_similarity = 0.0
        validator = ConstraintValidator(config)
        results = validator.validate_all(BAKING_TEXT, "skill", baseline_text=REVIEW_SKILL)
        result = next(r for r in results if r.constraint_name == "semantic_preservation")
        assert result.passed
