"""Durable performance history for the continuous self-improvement loop.

Phases 1-4 are one-shot tools: a human picks a target, runs an optimizer, reads
a diff. Phase 5 only works if the pipeline can answer "which target is worth a
run this week?" on its own, and that question is unanswerable without a memory
of how things have been going. This module is that memory.

PLAN.md names four signals to track:

    skill_success_rate         per-skill success, mined from real sessions
    tool_selection_accuracy    did the agent reach for the right tool
    benchmark_score            periodic benchmark runs, scored over time
    user_correction            "no, use X instead" is a labelled failure

They share one shape - a value attached to a named target at a moment in time -
so they share one store: a JSONL file that is only ever appended to. Append-only
matters more than it sounds. A monitor that rewrites its own history can quietly
erase the evidence that a regression happened, and the loop's whole claim to
usefulness rests on that evidence being trustworthy. Rotation exists
(:meth:`MetricStore.archive_before`) but it moves old points to a sibling file
rather than dropping them.

Two design rules the tests depend on:

1. **The clock is injected.** Every function that needs "now" takes it as an
   argument, defaulting to the store's clock. Nothing calls ``datetime.now()``
   from inside logic worth testing, so trend detection is reproducible.
2. **Nothing is inferred that was not measured.** An absent benchmark records no
   point at all rather than a zero. A zero would read as "the agent failed every
   task", which is a very different claim from "we did not run".

Pure local I/O throughout: no network, no model, no hermes-agent checkout.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

__all__ = [
    "SECONDS_PER_DAY",
    "SKILL_SUCCESS_RATE",
    "TOOL_SELECTION_ACCURACY",
    "BENCHMARK_SCORE",
    "USER_CORRECTION",
    "OPTIMIZATION_RUN",
    "TRACKED_METRICS",
    "HIGHER_IS_BETTER",
    "utc_now",
    "MetricPoint",
    "MetricStore",
    "Aggregate",
    "Trend",
    "TrendDirection",
    "compute_trend",
]

SECONDS_PER_DAY = 86_400.0

# The four signals PLAN.md Phase 5 lists under "Performance monitor".
SKILL_SUCCESS_RATE = "skill_success_rate"
TOOL_SELECTION_ACCURACY = "tool_selection_accuracy"
BENCHMARK_SCORE = "benchmark_score"
USER_CORRECTION = "user_correction"

# Bookkeeping the loop writes about itself, so a later cycle can see that a
# target was already optimized and does not need optimizing again this week.
OPTIMIZATION_RUN = "optimization_run"

TRACKED_METRICS: tuple[str, ...] = (
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    BENCHMARK_SCORE,
    USER_CORRECTION,
)

# Direction of goodness per metric. Corrections are the odd one out: more of
# them is worse, so a rising correction count is a deterioration, not progress.
HIGHER_IS_BETTER: dict[str, bool] = {
    SKILL_SUCCESS_RATE: True,
    TOOL_SELECTION_ACCURACY: True,
    BENCHMARK_SCORE: True,
    USER_CORRECTION: False,
    OPTIMIZATION_RUN: True,
}

Selector = Optional[Union[str, Iterable[str]]]


def utc_now() -> float:
    """Current UTC time as unix seconds.

    The only clock reader in this module. Everything else takes a timestamp or
    a clock callable, which is what keeps the tests deterministic.
    """
    return datetime.now(timezone.utc).timestamp()


def _as_set(value: Selector) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


# ──────────────────────────────────────────────────────────────────────────
# Points
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MetricPoint:
    """One observation: a value for a named target at a moment in time.

    *samples* is how many underlying observations the value summarizes. A 50%
    success rate over two sessions and a 50% success rate over five hundred are
    the same number carrying wildly different weight, and triage ranks by usage
    frequency, so the count has to travel with the value.
    """

    metric: str
    target: str
    value: float
    timestamp: float
    source: str = "unknown"
    samples: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric = str(self.metric).strip()
        self.target = str(self.target).strip()
        if not self.metric:
            raise ValueError("MetricPoint.metric must be a non-empty name")
        if not self.target:
            raise ValueError("MetricPoint.target must be a non-empty name")
        self.value = float(self.value)
        self.timestamp = float(self.timestamp)
        self.samples = int(self.samples)
        if self.samples < 0:
            raise ValueError("MetricPoint.samples cannot be negative")
        if self.metadata is None:
            self.metadata = {}

    @property
    def when(self) -> datetime:
        """The timestamp as an aware UTC datetime."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def higher_is_better(self) -> bool:
        return HIGHER_IS_BETTER.get(self.metric, True)

    def to_dict(self) -> dict:
        record = {
            "metric": self.metric,
            "target": self.target,
            "value": self.value,
            "timestamp": self.timestamp,
            # Redundant with timestamp, kept because a human tailing this file
            # should be able to read it without doing epoch arithmetic.
            "at": self.when.isoformat(),
            "source": self.source,
            "samples": self.samples,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, blob: dict) -> "MetricPoint":
        timestamp = blob.get("timestamp")
        if timestamp is None:
            at = blob.get("at")
            if not at:
                raise ValueError("record has neither 'timestamp' nor 'at'")
            timestamp = datetime.fromisoformat(at).timestamp()
        return cls(
            metric=blob["metric"],
            target=blob["target"],
            value=blob.get("value", 0.0),
            timestamp=float(timestamp),
            source=blob.get("source", "unknown"),
            samples=int(blob.get("samples", 1)),
            metadata=dict(blob.get("metadata") or {}),
        )

    @classmethod
    def from_json_line(cls, line: str) -> "MetricPoint":
        return cls.from_dict(json.loads(line))


# ──────────────────────────────────────────────────────────────────────────
# Trends
# ──────────────────────────────────────────────────────────────────────────


class TrendDirection(str, Enum):
    RISING = "rising"
    FLAT = "flat"
    DECLINING = "declining"
    UNKNOWN = "unknown"


@dataclass
class Trend:
    """Where a series is heading, and whether anyone should care.

    ``direction`` describes the *value*: rising means the number went up, which
    is good news for a success rate and bad news for a correction count.
    ``significant`` folds that in - it is True only when the series has moved in
    the bad direction by at least the significance threshold, which is the
    condition triage treats as an actionable decline.
    """

    metric: str
    target: str
    direction: TrendDirection
    slope_per_day: float
    change: float
    first_value: Optional[float]
    last_value: Optional[float]
    n: int
    span_days: float
    higher_is_better: bool = True
    significant: bool = False
    note: str = ""

    @property
    def is_deterioration(self) -> bool:
        if self.direction is TrendDirection.DECLINING:
            return self.higher_is_better
        if self.direction is TrendDirection.RISING:
            return not self.higher_is_better
        return False

    def describe(self) -> str:
        if self.direction is TrendDirection.UNKNOWN:
            return self.note or f"not enough history ({self.n} points)"
        return (
            f"{self.direction.value} {self.change:+.3f} over {self.span_days:.1f}d "
            f"({self.slope_per_day:+.4f}/day, n={self.n})"
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "target": self.target,
            "direction": self.direction.value,
            "slope_per_day": self.slope_per_day,
            "change": self.change,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "n": self.n,
            "span_days": round(self.span_days, 4),
            "higher_is_better": self.higher_is_better,
            "significant": self.significant,
            "note": self.note,
        }


def compute_trend(
    points: Sequence[MetricPoint],
    *,
    metric: str = "",
    target: str = "",
    higher_is_better: Optional[bool] = None,
    min_points: int = 3,
    flat_tolerance: float = 0.02,
    significant_change: float = 0.05,
) -> Trend:
    """Least-squares trend over *points*, expressed as change per day.

    The slope is fitted against time in days rather than sample index, so an
    irregular reporting cadence does not distort it. ``change`` is the modelled
    movement across the observed span, which is the number a human actually
    cares about: "this skill lost nine points of success rate in three weeks".

    Fewer than *min_points* observations yields ``UNKNOWN`` rather than a guess.
    Two noisy readings are not a trend, and acting on them would burn an
    optimization cycle on nothing.
    """
    ordered = sorted(points, key=lambda p: p.timestamp)
    metric = metric or (ordered[0].metric if ordered else "")
    target = target or (ordered[0].target if ordered else "")
    if higher_is_better is None:
        higher_is_better = HIGHER_IS_BETTER.get(metric, True)

    if len(ordered) < max(2, min_points):
        return Trend(
            metric=metric,
            target=target,
            direction=TrendDirection.UNKNOWN,
            slope_per_day=0.0,
            change=0.0,
            first_value=ordered[0].value if ordered else None,
            last_value=ordered[-1].value if ordered else None,
            n=len(ordered),
            span_days=0.0,
            higher_is_better=higher_is_better,
            note=f"need {max(2, min_points)} points, have {len(ordered)}",
        )

    t0 = ordered[0].timestamp
    xs = [(p.timestamp - t0) / SECONDS_PER_DAY for p in ordered]
    ys = [p.value for p in ordered]
    span_days = xs[-1] - xs[0]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator > 0:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        change = slope * span_days
    else:
        # Every point shares a timestamp. There is no rate to fit, but the
        # values still moved, so report the raw difference and no slope.
        slope = 0.0
        change = ys[-1] - ys[0]

    if change > flat_tolerance:
        direction = TrendDirection.RISING
    elif change < -flat_tolerance:
        direction = TrendDirection.DECLINING
    else:
        direction = TrendDirection.FLAT

    trend = Trend(
        metric=metric,
        target=target,
        direction=direction,
        slope_per_day=slope,
        change=change,
        first_value=ys[0],
        last_value=ys[-1],
        n=len(ordered),
        span_days=span_days,
        higher_is_better=higher_is_better,
    )
    trend.significant = trend.is_deterioration and abs(change) >= significant_change
    return trend


# ──────────────────────────────────────────────────────────────────────────
# Aggregates
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Aggregate:
    """Summary of one (metric, target) series over a window."""

    metric: str
    target: str
    count: int = 0
    samples: int = 0
    mean: float = 0.0
    weighted_mean: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    total: float = 0.0
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    last_value: Optional[float] = None
    sources: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return self.count == 0

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "target": self.target,
            "count": self.count,
            "samples": self.samples,
            "mean": self.mean,
            "weighted_mean": self.weighted_mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "total": self.total,
            "last_value": self.last_value,
            "sources": list(self.sources),
        }


def summarize(metric: str, target: str, points: Sequence[MetricPoint]) -> Aggregate:
    """Fold *points* into an :class:`Aggregate`.

    ``weighted_mean`` weights each point by its sample count, so a rate computed
    over four hundred sessions is not outvoted by a rate computed over three.
    """
    if not points:
        return Aggregate(metric=metric, target=target)

    ordered = sorted(points, key=lambda p: p.timestamp)
    values = [p.value for p in ordered]
    samples = sum(p.samples for p in ordered)
    weight_total = sum(max(p.samples, 0) for p in ordered)
    if weight_total > 0:
        weighted = sum(p.value * max(p.samples, 0) for p in ordered) / weight_total
    else:
        weighted = sum(values) / len(values)

    return Aggregate(
        metric=metric,
        target=target,
        count=len(ordered),
        samples=samples,
        mean=sum(values) / len(values),
        weighted_mean=weighted,
        minimum=min(values),
        maximum=max(values),
        total=sum(values),
        first_timestamp=ordered[0].timestamp,
        last_timestamp=ordered[-1].timestamp,
        last_value=ordered[-1].value,
        sources=tuple(sorted({p.source for p in ordered})),
    )


# ──────────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────────


class MetricStore:
    """Append-only JSONL history of performance observations.

    One JSON object per line, appended and flushed immediately, so a cycle that
    dies halfway through still leaves every point it had already written. A line
    that fails to parse is skipped rather than fatal - a truncated final line
    from a killed process should not blind the monitor to the year of history
    above it - and the count lands in :attr:`skipped_lines` so the damage is
    visible instead of silent.
    """

    def __init__(
        self,
        path: Union[str, Path],
        clock: Callable[[], float] = utc_now,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self.skipped_lines = 0

    # -- clock ------------------------------------------------------------

    def now(self) -> float:
        return float(self._clock())

    # -- writing ----------------------------------------------------------

    def append(self, point: MetricPoint) -> MetricPoint:
        """Append one point and return it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(point.to_json_line() + "\n")
            handle.flush()
        return point

    def extend(self, points: Iterable[MetricPoint]) -> list[MetricPoint]:
        written: list[MetricPoint] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for point in points:
                handle.write(point.to_json_line() + "\n")
                written.append(point)
            handle.flush()
        return written

    def record(
        self,
        metric: str,
        target: str,
        value: float,
        *,
        source: str = "unknown",
        samples: int = 1,
        timestamp: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> MetricPoint:
        """Build a point from the store's clock and append it.

        An explicit *timestamp* always wins, which is how backfilled history and
        deterministic tests get written.
        """
        point = MetricPoint(
            metric=metric,
            target=target,
            value=value,
            timestamp=self.now() if timestamp is None else timestamp,
            source=source,
            samples=samples,
            metadata=dict(metadata or {}),
        )
        return self.append(point)

    # -- reading ----------------------------------------------------------

    def load(self) -> list[MetricPoint]:
        """Every point in file order. Missing file reads as empty history."""
        self.skipped_lines = 0
        if not self.path.exists():
            return []

        points: list[MetricPoint] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(MetricPoint.from_json_line(line))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    self.skipped_lines += 1
        return points

    def query(
        self,
        metric: Selector = None,
        target: Selector = None,
        source: Selector = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> list[MetricPoint]:
        """Points matching every supplied filter, oldest first.

        *since* and *until* are both inclusive.
        """
        metrics = _as_set(metric)
        targets = _as_set(target)
        sources = _as_set(source)

        selected = []
        for point in self.load():
            if metrics is not None and point.metric not in metrics:
                continue
            if targets is not None and point.target not in targets:
                continue
            if sources is not None and point.source not in sources:
                continue
            if since is not None and point.timestamp < since:
                continue
            if until is not None and point.timestamp > until:
                continue
            selected.append(point)
        selected.sort(key=lambda p: p.timestamp)
        return selected

    def window(
        self,
        days: float,
        *,
        now: Optional[float] = None,
        metric: Selector = None,
        target: Selector = None,
        source: Selector = None,
    ) -> list[MetricPoint]:
        """Points inside the trailing *days*-long window ending at *now*."""
        end = self.now() if now is None else now
        return self.query(
            metric=metric,
            target=target,
            source=source,
            since=end - days * SECONDS_PER_DAY,
            until=end,
        )

    def pairs(self, points: Optional[Sequence[MetricPoint]] = None) -> list[tuple[str, str]]:
        """Sorted (metric, target) pairs present in the history."""
        source = self.load() if points is None else points
        return sorted({(p.metric, p.target) for p in source})

    def targets(
        self,
        metric: Selector = None,
        points: Optional[Sequence[MetricPoint]] = None,
    ) -> list[str]:
        metrics = _as_set(metric)
        source = self.load() if points is None else points
        return sorted(
            {p.target for p in source if metrics is None or p.metric in metrics}
        )

    def latest(self, metric: str, target: str) -> Optional[MetricPoint]:
        found = self.query(metric=metric, target=target)
        return found[-1] if found else None

    def summarize(
        self,
        metric: str,
        target: str,
        *,
        window_days: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Aggregate:
        if window_days is None:
            points = self.query(metric=metric, target=target)
        else:
            points = self.window(window_days, now=now, metric=metric, target=target)
        return summarize(metric, target, points)

    def trend(
        self,
        metric: str,
        target: str,
        *,
        window_days: Optional[float] = None,
        now: Optional[float] = None,
        min_points: int = 3,
        flat_tolerance: float = 0.02,
        significant_change: float = 0.05,
    ) -> Trend:
        if window_days is None:
            points = self.query(metric=metric, target=target)
        else:
            points = self.window(window_days, now=now, metric=metric, target=target)
        return compute_trend(
            points,
            metric=metric,
            target=target,
            min_points=min_points,
            flat_tolerance=flat_tolerance,
            significant_change=significant_change,
        )

    # -- rotation ---------------------------------------------------------

    @property
    def archive_path(self) -> Path:
        return self.path.with_suffix(".archive" + self.path.suffix)

    def archive_before(self, cutoff: float) -> int:
        """Move points older than *cutoff* into the sibling archive file.

        Nothing is deleted. An unattended weekly job writing to one JSONL file
        forever needs somewhere for the old lines to go, but a monitor that
        discards its own evidence is worse than one that grows, so the old
        points are appended to ``<name>.archive.jsonl`` and the live file is
        rewritten atomically with what remains. Returns the number moved.
        """
        points = self.load()
        if not points:
            return 0

        old = [p for p in points if p.timestamp < cutoff]
        if not old:
            return 0
        kept = [p for p in points if p.timestamp >= cutoff]

        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a", encoding="utf-8") as handle:
            for point in old:
                handle.write(point.to_json_line() + "\n")

        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for point in kept:
                    handle.write(point.to_json_line() + "\n")
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return len(old)
