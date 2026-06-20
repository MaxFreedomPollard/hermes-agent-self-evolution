"""Tests for skill module loading, parsing, and evolved text extraction."""

from pathlib import Path

from evolution.skills.skill_module import (
    SkillModule,
    extract_evolved_skill_text,
    load_skill,
    reassemble_skill,
    split_frontmatter,
    strip_leading_frontmatter,
)

SAMPLE_SKILL = """---
name: test-skill
description: A skill for testing things
version: 1.0.0
metadata:
  hermes: true
  tags: [testing]
---

# Test Skill — Testing Things

## When to Use
Use this when you need to test things.

## Procedure
1. First, do the thing
2. Then, verify it worked
3. Report results

## Pitfalls
- Don't forget to check edge cases
"""


class TestLoadSkill:
    def test_parses_frontmatter(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)

        skill = load_skill(skill_file)

        assert skill["name"] == "test-skill"
        assert skill["description"] == "A skill for testing things"
        assert "version: 1.0.0" in skill["frontmatter"]

    def test_parses_body(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)

        skill = load_skill(skill_file)

        assert "# Test Skill" in skill["body"]
        assert "## Procedure" in skill["body"]
        assert "Don't forget" in skill["body"]

    def test_raw_contains_everything(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)

        skill = load_skill(skill_file)

        assert skill["raw"] == SAMPLE_SKILL

    def test_path_is_stored(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)

        skill = load_skill(skill_file)

        assert skill["path"] == skill_file


class TestFrontmatterHandling:
    def test_split_frontmatter_preserves_internal_horizontal_rules(self):
        raw = """---
name: x
description: y
---

# Body

---

Not frontmatter.
"""

        frontmatter, body = split_frontmatter(raw)

        assert "name: x" in frontmatter
        assert "---\n\nNot frontmatter." in body

    def test_strip_leading_frontmatter_is_noop_for_plain_body(self):
        body = "# Body\n\nNo frontmatter here."

        assert strip_leading_frontmatter(body) == body

    def test_strip_leading_frontmatter_handles_crlf(self):
        text = "---\r\nname: emitted\r\ndescription: emitted\r\n---\r\n\r\n# Body\r\nStep."

        assert strip_leading_frontmatter(text) == "# Body\r\nStep."


class TestReassembleSkill:
    def test_roundtrip(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        reassembled = reassemble_skill(skill["frontmatter"], skill["body"])

        assert reassembled.startswith("---\n")
        assert "name: test-skill" in reassembled
        assert "# Test Skill" in reassembled

    def test_preserves_frontmatter(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        body = "# My Skill\nDo the thing."

        result = reassemble_skill(frontmatter, body)

        assert result.startswith("---\n")
        assert "name: my-skill" in result
        assert "# My Skill" in result

    def test_reassemble_strips_optimizer_emitted_frontmatter(self):
        frontmatter = "name: baseline-skill\ndescription: Baseline metadata"
        evolved_body = """---
name: optimizer-skill
description: Optimizer hallucinated metadata
---

# Evolved Body
Use the better procedure.
"""

        result = reassemble_skill(frontmatter, evolved_body)

        assert result == (
            "---\n"
            "name: baseline-skill\n"
            "description: Baseline metadata\n"
            "---\n\n"
            "# Evolved Body\n"
            "Use the better procedure.\n"
        )
        assert "optimizer-skill" not in result


class TestSkillModule:
    def test_forward_attaches_current_skill_text_to_prediction(self):
        module = SkillModule("# Baseline\nFollow the procedure.")

        prediction = module(task_input="Do it")

        assert prediction.skill_text == "# Baseline\nFollow the procedure."


class TestExtractEvolvedSkillText:
    def test_extracts_nested_dspy_signature_instructions(self):
        class Signature:
            instructions = "# Evolved\n\nDo the better process."

        class Predict:
            signature = Signature()

        class Predictor:
            predict = Predict()

        class Module:
            predictor = Predictor()
            skill_text = "# Baseline"

        assert extract_evolved_skill_text(Module(), fallback="# Fallback") == "# Evolved\n\nDo the better process."

    def test_extract_strips_optimizer_frontmatter(self):
        class Signature:
            instructions = "---\nname: x\ndescription: y\n---\n\n# Body"

        class Predictor:
            signature = Signature()

        class Module:
            predictor = Predictor()

        assert extract_evolved_skill_text(Module()) == "# Body"
