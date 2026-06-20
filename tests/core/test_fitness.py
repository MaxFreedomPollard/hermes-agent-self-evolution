"""Tests for fitness parsing, fallback scoring, and candidate-skill metric behavior."""

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core import fitness
from evolution.core.fitness import FitnessScore, _parse_score, make_skill_fitness_metric, skill_fitness_metric


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.25, 0.25),
        ("0.75", 0.75),
        ("80%", 0.8),
        ("score: 0.6/1", 0.6),
        ("not parseable", 0.5),
        ("150%", 1.0),
    ],
)
def test_parse_score(value, expected):
    assert _parse_score(value) == pytest.approx(expected)


def test_composite_clamps_after_penalty():
    score = FitnessScore(correctness=1, procedure_following=1, conciseness=1, length_penalty=2)

    assert score.composite == 0.0


def test_legacy_metric_is_keyword_based_fallback_only():
    class Example:
        expected_behavior = "must identify sql injection and recommend parameterized queries"

    class Prediction:
        output = "This identifies SQL injection and recommends parameterized queries."

    assert skill_fitness_metric(Example(), Prediction()) > 0.7


def test_metric_uses_candidate_skill_text_from_prediction(monkeypatch):
    seen = {}

    class FakeJudge:
        def __init__(self, config):
            pass

        def score(self, **kwargs):
            seen.update(kwargs)
            return FitnessScore(correctness=1, procedure_following=1, conciseness=1)

    monkeypatch.setattr(fitness, "LLMJudge", FakeJudge)
    metric = make_skill_fitness_metric(EvolutionConfig(), fallback_skill_text="# Baseline")

    class Example:
        task_input = "task"
        expected_behavior = "expected"

    class Prediction:
        output = "answer"
        skill_text = "# Candidate"

    assert metric(Example(), Prediction()) == 1.0
    assert seen["skill_text"] == "# Candidate"


def test_gepa_metric_returns_score_and_feedback_prediction(monkeypatch):
    class FakeJudge:
        def __init__(self, config):
            pass

        def score(self, **kwargs):
            return FitnessScore(
                correctness=0.8,
                procedure_following=0.7,
                conciseness=0.9,
                feedback="Tighten the procedure."
            )

    monkeypatch.setattr(fitness, "LLMJudge", FakeJudge)
    metric = make_skill_fitness_metric(
        EvolutionConfig(), fallback_skill_text="# Baseline", return_feedback=True
    )

    class Example:
        task_input = "task"
        expected_behavior = "expected"

    class Prediction:
        output = "answer"
        skill_text = "# Candidate"

    result = metric(Example(), Prediction(), trace=[], pred_name="predictor", pred_trace=[])

    assert result.score == pytest.approx(0.79)
    assert result.feedback == "Tighten the procedure."


def test_scalar_metric_accepts_gepa_extra_arguments(monkeypatch):
    class FakeJudge:
        def __init__(self, config):
            pass

        def score(self, **kwargs):
            return FitnessScore(correctness=1, procedure_following=1, conciseness=1)

    monkeypatch.setattr(fitness, "LLMJudge", FakeJudge)
    metric = make_skill_fitness_metric(EvolutionConfig(), fallback_skill_text="# Baseline")

    class Example:
        task_input = "task"
        expected_behavior = "expected"

    class Prediction:
        output = "answer"

    assert metric(Example(), Prediction(), trace=[], pred_name="p", pred_trace=[]) == 1.0
