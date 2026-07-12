"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig


def _dedup_key(text: str) -> str:
    """Normalize a task for duplicate detection: casefold, strip punctuation,
    collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between the token sets of two normalized tasks."""
    tokens_a = set(_dedup_key(a).split())
    tokens_b = set(_dedup_key(b).split())
    if not tokens_a or not tokens_b:
        return 1.0 if tokens_a == tokens_b else 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def dedupe_examples(examples: list["EvalExample"], jaccard_threshold: float = 0.9) -> list["EvalExample"]:
    """Drop duplicate and near-duplicate tasks, keeping the first occurrence.

    LLM generation and mined session history both produce repeats. If two
    copies of the same task land on opposite sides of the train/holdout
    split, the holdout is contaminated and "improvement" can be pure
    memorization. Exact duplicates are matched on the normalized task text;
    near-duplicates on token-set Jaccard similarity.
    """
    kept: list["EvalExample"] = []
    seen_keys: set[str] = set()
    for example in examples:
        key = _dedup_key(example.task_input)
        if key in seen_keys:
            continue
        if any(_token_jaccard(example.task_input, k.task_input) >= jaccard_threshold for k in kept):
            continue
        seen_keys.add(key)
        kept.append(example)
    return kept


def split_examples(
    examples: list["EvalExample"],
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    seed: int = 13,
) -> "EvalDataset":
    """Deterministic shuffled train/val/holdout split.

    Seeded so the same examples always split the same way: re-running a
    build cannot silently move an example from holdout into train.
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return EvalDataset()
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    return EvalDataset(
        train=shuffled[:n_train],
        val=shuffled[n_train:n_train + n_val],
        holdout=shuffled[n_train + n_val:],
    )


@dataclass
class EvalExample:
    """A single evaluation example."""
    task_input: str  # What the user asks
    expected_behavior: str  # Rubric — what a good response looks like
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # Category for stratified eval
    source: str = "synthetic"  # synthetic, sessiondb, golden

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalDataset:
    """Train/val/holdout split of evaluation examples."""
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)

    @property
    def all_examples(self) -> list[EvalExample]:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        """Save dataset splits to JSONL files."""
        path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split_name}.jsonl", "w") as f:
                for ex in split_data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset splits from JSONL files."""
        dataset = cls()
        for split_name in ["train", "val", "holdout"]:
            split_file = path / f"{split_name}.jsonl"
            if split_file.exists():
                examples = []
                with open(split_file) as f:
                    for line in f:
                        if line.strip():
                            examples.append(EvalExample.from_dict(json.loads(line)))
                setattr(dataset, split_name, examples)
        return dataset

    def to_dspy_examples(self, split: str = "train") -> list[dspy.Example]:
        """Convert a split to DSPy Example objects."""
        data = getattr(self, split)
        return [
            dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("task_input")
            for ex in data
        ]


class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        Given the full text of a skill/tool description, generate diverse test cases
        that would exercise different aspects of the skill. Each test case should include:
        - A realistic task_input (what a user would actually ask)
        - An expected_behavior rubric (what a good response should contain/do, NOT exact text)
        - A difficulty level (easy, medium, hard)
        - A category (what aspect of the skill this tests)
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases, each with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits."""

        n = num_cases or self.config.eval_dataset_size

        # Configure DSPy to use the judge model for generation
        lm = dspy.LM(self.config.judge_model)

        with dspy.context(lm=lm):
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )

        # Parse the generated test cases
        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\[.*\]', result.test_cases, re.DOTALL)
            if match:
                cases_raw = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse test cases from LLM output: {result.test_cases[:200]}")

        examples = [
            EvalExample(
                task_input=c.get("task_input", ""),
                expected_behavior=c.get("expected_behavior", ""),
                difficulty=c.get("difficulty", "medium"),
                category=c.get("category", "general"),
                source="synthetic",
            )
            for c in cases_raw
            if c.get("task_input") and c.get("expected_behavior")
        ]

        # Dedupe (LLMs generate near-identical cases) and split with a
        # fixed seed so holdout membership is stable across runs.
        examples = dedupe_examples(examples, self.config.dedup_jaccard)
        return split_examples(
            examples,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            seed=self.config.dataset_split_seed,
        )


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        # Single file — auto-split
        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        return split_examples(dedupe_examples(examples))
