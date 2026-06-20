"""Tests for strict skill constraint validation."""

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator


def test_skill_structure_rejects_body_only_text():
    result = ConstraintValidator(EvolutionConfig())._check_skill_structure("# Body only")

    assert not result.passed
    assert "frontmatter" in result.message


def test_skill_structure_rejects_unclosed_frontmatter():
    text = "---\nname: x\ndescription: y\n# Body accidentally inside yaml"

    result = ConstraintValidator(EvolutionConfig())._check_skill_structure(text)

    assert not result.passed
    assert "closed YAML frontmatter" in result.message


def test_skill_structure_accepts_full_skill_file():
    text = "---\nname: x\ndescription: y\n---\n\n# Body"

    result = ConstraintValidator(EvolutionConfig())._check_skill_structure(text)

    assert result.passed
