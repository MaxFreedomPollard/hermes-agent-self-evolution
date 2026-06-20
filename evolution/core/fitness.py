"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs. Supports length
penalties and multi-dimensional scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.lm import make_lm


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""

    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""

        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, min(1.0, raw - self.length_penalty))


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation."""

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """

        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(
            desc="Rubric describing what a good response looks like"
        )
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The current skill/instructions being evaluated")

        correctness: float = dspy.OutputField(
            desc="Score 0.0-1.0: Did the response correctly address the task?"
        )
        procedure_following: float = dspy.OutputField(
            desc="Score 0.0-1.0: Did it follow the expected procedure?"
        )
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(
            desc="Specific, actionable feedback on what could be improved"
        )

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = make_lm(self.config.judge_model, self.config)
        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        length_penalty = 0.0
        if artifact_size is not None and max_size is not None and max_size > 0:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def make_skill_fitness_metric(
    config: EvolutionConfig,
    fallback_skill_text: str = "",
    *,
    return_feedback: bool = False,
):
    """Build the Phase-1 metric using LLM-as-judge scoring.

    This is intentionally stricter than the legacy lexical proxy. The whole
    point of Phase 1 is to optimize skill quality, not keyword overlap. If the
    judge cannot run, the pipeline fails closed unless
    ``allow_heuristic_fallback`` is explicitly enabled.

    ``SkillModule.forward`` returns the candidate's current skill instructions
    on the prediction object. That allows the judge to evaluate each GEPA
    candidate against its own evolved instructions rather than against the
    original baseline skill.

    GEPA can use textual feedback for reflection, so callers running GEPA pass
    ``return_feedback=True`` to receive ``dspy.Prediction(score=...,
    feedback=...)``. Scalar-only optimizers such as the MIPROv2 fallback use
    the same judge with ``return_feedback=False``.
    """

    judge = LLMJudge(config)

    def _format_score(score: FitnessScore):
        if return_feedback:
            return dspy.Prediction(score=score.composite, feedback=score.feedback)
        return score.composite

    def _format_zero(feedback: str):
        if return_feedback:
            return dspy.Prediction(score=0.0, feedback=feedback)
        return 0.0

    def metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ):
        agent_output = getattr(prediction, "output", "") or ""
        if not agent_output.strip():
            return _format_zero("The candidate produced an empty output.")

        task_input = getattr(example, "task_input", "") or ""
        expected_behavior = getattr(example, "expected_behavior", "") or ""
        candidate_skill_text = (
            getattr(prediction, "skill_text", "")
            or getattr(example, "skill_text", "")
            or fallback_skill_text
        )
        try:
            score = judge.score(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=candidate_skill_text,
                artifact_size=len(candidate_skill_text),
                max_size=config.max_skill_size,
            )
            return _format_score(score)
        except Exception as exc:
            if config.allow_heuristic_fallback:
                scalar = skill_fitness_metric(
                    example,
                    prediction,
                    trace=trace,
                    pred_name=pred_name,
                    pred_trace=pred_trace,
                )
                if return_feedback:
                    return dspy.Prediction(
                        score=scalar,
                        feedback=f"LLM judge failed ({type(exc).__name__}); used lexical fallback.",
                    )
                return scalar
            raise

    return metric


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """Legacy lexical proxy metric retained as an explicit fallback.

    This function is not the default Phase-1 optimization objective anymore.
    It exists for cheap smoke tests and for explicitly configured fallback mode.
    """

    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    if not agent_output.strip():
        return 0.0

    expected_words = _content_words(expected)
    output_words = _content_words(agent_output)
    if not expected_words:
        return 0.5

    overlap = len(expected_words & output_words) / len(expected_words)
    return min(1.0, max(0.0, 0.3 + (0.7 * overlap)))


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9_]+", text.lower()) if len(word) > 2}


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""

    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))

    text = str(value).strip()
    try:
        return min(1.0, max(0.0, float(text)))
    except (ValueError, TypeError):
        pass

    # Handle common judge outputs like "0.8/1" or "score: 80%".
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(/\s*1|%)?", text)
    if not match:
        return 0.5

    number = float(match.group(1))
    suffix = match.group(2) or ""
    if "%" in suffix or number > 1.0:
        number /= 100.0
    return min(1.0, max(0.0, number))
