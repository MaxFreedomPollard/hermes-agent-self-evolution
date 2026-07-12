"""Run manifests: the evidence record for every evolution run.

PLAN.md's deployment process requires each evolved artifact to ship with
before/after scores on every split, the eval dataset used, the cost of
the optimization run, and any constraint violations that were caught.
None of that can be reported honestly unless it is recorded at run time.

A RunManifest captures, in one machine-readable JSON file next to the
evolved output:

  - what ran: skill, config snapshot (models, iterations, thresholds)
  - on what data: per-split counts and a content fingerprint of the
    eval dataset, so two runs can be compared apples-to-apples
  - what came out: content digests and sizes of baseline and evolved
    text, constraint results, holdout scores per example
  - what it cost: LLM calls, tokens, and dollars, summed from dspy's
    call history
  - whether it was deployable: constraints all passed or not

Manifests are written on failed runs too; a rejected candidate with its
violations on record is exactly the audit trail the PLAN asks for.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from dspy.clients.base_lm import GLOBAL_HISTORY, MAX_HISTORY_SIZE
except ImportError:  # dspy internals moved; degrade to no usage tracking
    GLOBAL_HISTORY: list = []
    MAX_HISTORY_SIZE = 10_000

SCHEMA_VERSION = 1


def text_digest(text: str) -> str:
    """Short, stable content digest for baseline/evolved text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def config_snapshot(config: Any) -> dict:
    """JSON-safe snapshot of an EvolutionConfig (Paths become strings)."""
    raw = asdict(config) if is_dataclass(config) else dict(vars(config))
    return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}


@dataclass
class DatasetFingerprint:
    """Identity of an eval dataset: split sizes plus a content hash.

    The hash covers (split, task) pairs, order-independent within a
    split but sensitive to split membership: moving one example from
    holdout to train changes the fingerprint. Two runs with the same
    fingerprint were measured on the same data.
    """

    train: int
    val: int
    holdout: int
    sha256: str

    @classmethod
    def from_dataset(cls, dataset: Any) -> "DatasetFingerprint":
        items = sorted(
            (split, " ".join(example.task_input.split()))
            for split in ("train", "val", "holdout")
            for example in getattr(dataset, split)
        )
        digest = hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()[:16]
        return cls(
            train=len(dataset.train),
            val=len(dataset.val),
            holdout=len(dataset.holdout),
            sha256=digest,
        )


@dataclass
class LMUsageSummary:
    """LLM spend across a tracked window: calls, tokens, and dollars."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Sum over calls whose cost the provider reported. None when no call
    # had a known cost (unknown models, cache hits).
    cost_usd: Optional[float] = None
    calls_with_unknown_cost: int = 0
    # True when dspy's bounded history may have evicted entries from
    # this window, so the numbers are a lower bound.
    possibly_incomplete: bool = False


class UsageTracker:
    """Sums LLM usage recorded by dspy between start() and finish().

    dspy appends every LM call to a global, size-bounded history where
    each entry carries token usage, provider-reported cost, and an ISO
    timestamp. The tracker snapshots the clock at start() and attributes
    every entry stamped at or after it to this run.

    The history and its size cap are injectable for testing.
    """

    def __init__(self, history: Optional[list] = None, max_size: Optional[int] = None):
        self._history = GLOBAL_HISTORY if history is None else history
        self._max_size = MAX_HISTORY_SIZE if max_size is None else max_size
        self._start_iso: Optional[str] = None

    def start(self) -> "UsageTracker":
        self._start_iso = datetime.now().isoformat()
        return self

    def finish(self) -> LMUsageSummary:
        if self._start_iso is None:
            return LMUsageSummary()

        summary = LMUsageSummary()
        cost_total = 0.0
        cost_seen = False

        for entry in self._history:
            if entry.get("timestamp", "") < self._start_iso:
                continue
            summary.calls += 1

            usage = entry.get("usage") or {}
            for attr, key in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    setattr(summary, attr, getattr(summary, attr) + int(value))

            cost = entry.get("cost")
            if isinstance(cost, (int, float)):
                cost_total += float(cost)
                cost_seen = True
            else:
                summary.calls_with_unknown_cost += 1

        if cost_seen:
            summary.cost_usd = round(cost_total, 6)

        # If the bounded history filled up and its oldest surviving entry
        # is newer than our start, entries from this window were evicted.
        if (
            len(self._history) >= self._max_size
            and self._history
            and self._history[0].get("timestamp", "") > self._start_iso
        ):
            summary.possibly_incomplete = True

        return summary


@dataclass
class RunManifest:
    """Everything needed to audit or reproduce one evolution run."""

    run_id: str
    created_at: str
    skill_name: str
    skill_path: str
    config: dict
    dataset: DatasetFingerprint
    baseline: dict  # {"sha256", "chars"}
    evolved: dict  # {"sha256", "chars"}
    constraints: list = field(default_factory=list)
    # None when the run was rejected before holdout evaluation.
    scores: Optional[dict] = None
    deployed: bool = False
    elapsed_seconds: Optional[float] = None
    usage: LMUsageSummary = field(default_factory=LMUsageSummary)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        skill_name: str,
        skill_path: Path,
        config: Any,
        dataset: Any,
        baseline_text: str,
        evolved_text: str,
        constraint_results: list,
        scores: Optional[dict],
        deployed: bool,
        elapsed_seconds: Optional[float],
        usage: LMUsageSummary,
    ) -> "RunManifest":
        return cls(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            skill_name=skill_name,
            skill_path=str(skill_path),
            config=config_snapshot(config),
            dataset=DatasetFingerprint.from_dataset(dataset),
            baseline={"sha256": text_digest(baseline_text), "chars": len(baseline_text)},
            evolved={"sha256": text_digest(evolved_text), "chars": len(evolved_text)},
            constraints=[
                {
                    "name": getattr(r, "constraint_name", ""),
                    "passed": bool(getattr(r, "passed", False)),
                    "message": getattr(r, "message", ""),
                    "details": getattr(r, "details", None),
                }
                for r in constraint_results
            ],
            scores=scores,
            deployed=deployed,
            elapsed_seconds=elapsed_seconds,
            usage=usage,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> dict:
        """Read a saved manifest as a plain dict (for reports and tooling)."""
        return json.loads(path.read_text())
