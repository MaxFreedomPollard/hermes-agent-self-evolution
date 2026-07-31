"""Statistics for deciding whether an evolved variant is actually better.

Every comparison this pipeline makes is **paired**: the baseline and the
candidate are evaluated on the identical example set, so each example yields a
matched pair of outcomes. That structure is the single most useful fact
available, and discarding it is expensive in both directions.

* Comparing two independent rates when the data are paired throws away the
  pairing and loses power. The variance of a paired difference depends only on
  the examples where the two versions *disagreed*; examples both got right, or
  both got wrong, carry no information about which is better.
* Comparing point estimates against a fixed tolerance models no noise at all.
  With 10 examples, a selection rate of 0.8 has a standard error of 0.126, so a
  swing of 12 points is one standard error of nothing happening.

So the tests here are paired throughout: McNemar's exact test for binary
outcomes, Wilcoxon signed-rank and a seeded paired bootstrap for continuous
scores, and ordinary least squares with a real t-test on the slope for trends.

**Power is reported, not assumed.** :func:`min_detectable_paired_shift` answers
the question a small eval set makes unavoidable: given this many examples, what
is the smallest regression that could possibly have reached significance? When
that number is larger than the tolerance being enforced, the gate cannot do its
job, and saying so is more useful than returning a confident pass.

**Why an intersection-union test, and why no multiplicity correction.**
PLAN.md's requirement for Phase 2 is that *no individual tool's selection rate
regresses*. Accepting a candidate therefore means accepting a conjunction of
per-tool claims. Under the intersection-union principle, a conjunction of
claims each tested at level alpha is itself valid at level alpha, so no
Bonferroni or Benjamini-Hochberg adjustment is applied or needed. Correcting
here would make the gate more permissive as the tool count grows, which is
exactly backwards for a safety gate.

Standard library only: no numpy, no scipy. The incomplete beta function is
implemented directly because a Student-t tail is needed and stdlib has none.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional, Sequence

__all__ = [
    "Interval",
    "wilson_interval",
    "mcnemar_exact",
    "PairedBinary",
    "compare_paired_binary",
    "min_detectable_paired_shift",
    "paired_bootstrap_ci",
    "wilcoxon_signed_rank",
    "PairedContinuous",
    "compare_paired_continuous",
    "OLSTrend",
    "ols_trend",
    "cohens_h",
    "cohens_d",
    "chance_accuracy",
    "student_t_sf",
    "binomial_sf",
    "binomial_cdf",
]

_NORM = NormalDist()


# ──────────────────────────────────────────────────────────────────────────
# Distribution primitives
# ──────────────────────────────────────────────────────────────────────────


def _z_for(confidence: float) -> float:
    """Two-sided critical value for a normal at *confidence* (e.g. 0.95)."""
    confidence = min(max(confidence, 1e-6), 1 - 1e-9)
    return _NORM.inv_cdf(1 - (1 - confidence) / 2)


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, p). Exact, not approximated.

    Used for McNemar's conditional test, where p is always 0.5 and the counts
    are small enough that exactness matters more than speed.
    """
    if n <= 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def binomial_sf(k: int, n: int, p: float = 0.5) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p)."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _betacf(a: float, b: float, x: float, iterations: int = 300) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) for Student's t with *df* degrees of freedom."""
    if df <= 0:
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    two_sided = _betainc(df / 2.0, 0.5, df / (df + t * t))
    return two_sided / 2.0 if t > 0 else 1.0 - two_sided / 2.0


# ──────────────────────────────────────────────────────────────────────────
# Intervals
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    """A confidence interval on a point estimate."""

    point: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def describe(self, as_percent: bool = True) -> str:
        if as_percent:
            return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def to_dict(self) -> dict:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "confidence": self.confidence,
        }


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval because eval sets here are small and rates
    sit near the boundaries, exactly where Wald misbehaves: at 10/10 successes
    Wald reports the absurd [1.0, 1.0], while Wilson reports [0.72, 1.0].
    """
    if n <= 0:
        return Interval(0.0, 0.0, 1.0, confidence)
    z = _z_for(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), confidence)


# ──────────────────────────────────────────────────────────────────────────
# Paired binary outcomes (McNemar)
# ──────────────────────────────────────────────────────────────────────────


def mcnemar_exact(b: int, c: int, alternative: str = "two-sided") -> float:
    """Exact McNemar test p-value on discordant counts.

    *b* is the number of examples the baseline got right and the candidate got
    wrong; *c* is the reverse. Concordant examples are uninformative and are
    correctly excluded: under the null the b + c discordant pairs split like
    fair coin flips, so the test is an exact binomial on that conditional.

    ``alternative="worse"`` is the one-sided test for the candidate having
    regressed, which is the direction a safety gate cares about.
    """
    n = b + c
    if n == 0:
        return 1.0
    if alternative == "worse":
        # P(at least b losses) under a fair split.
        return binomial_sf(b, n, 0.5)
    if alternative == "better":
        return binomial_sf(c, n, 0.5)
    # Two-sided exact: double the smaller tail, clamped at 1.
    return min(1.0, 2.0 * binomial_cdf(min(b, c), n, 0.5))


def min_detectable_paired_shift(n: int, alpha: float = 0.05) -> float:
    """Smallest regression an exact paired test could ever call significant.

    The best case for detection is every disagreement pointing the same way. If
    k of n examples flip against the candidate and none flip for it, the
    one-sided exact p-value is 0.5**k, so significance needs
    ``k >= -log2(alpha)``. The answer is that k over n.

    This is the number that makes a small eval set honest. At n = 10 and
    alpha = 0.05 it is 0.5: nothing short of a fifty point collapse could reach
    significance, so a 5% tolerance is not being enforced by evidence.
    """
    if n <= 0:
        return 1.0
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil(-math.log2(alpha))
    return min(1.0, k / n)


@dataclass
class PairedBinary:
    """Baseline vs candidate over paired binary outcomes on one population."""

    n: int
    both_correct: int
    baseline_only: int  # b: baseline right, candidate wrong
    candidate_only: int  # c: baseline wrong, candidate right
    both_wrong: int
    alpha: float = 0.05
    confidence: float = 0.95

    @property
    def baseline_correct(self) -> int:
        return self.both_correct + self.baseline_only

    @property
    def candidate_correct(self) -> int:
        return self.both_correct + self.candidate_only

    @property
    def baseline_rate(self) -> float:
        return self.baseline_correct / self.n if self.n else 0.0

    @property
    def candidate_rate(self) -> float:
        return self.candidate_correct / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        return self.candidate_rate - self.baseline_rate

    @property
    def discordant(self) -> int:
        return self.baseline_only + self.candidate_only

    @property
    def p_worse(self) -> float:
        """One-sided p for 'the candidate regressed'."""
        return mcnemar_exact(self.baseline_only, self.candidate_only, "worse")

    @property
    def p_two_sided(self) -> float:
        return mcnemar_exact(self.baseline_only, self.candidate_only, "two-sided")

    @property
    def significant_regression(self) -> bool:
        return self.p_worse < self.alpha

    @property
    def significant_improvement(self) -> bool:
        return mcnemar_exact(self.baseline_only, self.candidate_only, "better") < self.alpha

    def delta_interval(self) -> Interval:
        """Confidence interval on the paired difference in rates.

        Uses the standard McNemar variance for a difference of correlated
        proportions. Only discordant pairs contribute, which is why a candidate
        that changes nothing has an interval of zero width regardless of n.
        """
        if self.n <= 0:
            return Interval(0.0, 0.0, 0.0, self.confidence)
        b, c, n = self.baseline_only, self.candidate_only, self.n
        variance = (b + c - (c - b) ** 2 / n) / (n * n)
        se = math.sqrt(max(0.0, variance))
        z = _z_for(self.confidence)
        d = self.delta
        return Interval(d, max(-1.0, d - z * se), min(1.0, d + z * se), self.confidence)

    def min_detectable_shift(self) -> float:
        return min_detectable_paired_shift(self.n, self.alpha)

    def underpowered_for(self, tolerance: float) -> bool:
        """True when no result on this many examples could detect *tolerance*."""
        return self.min_detectable_shift() > abs(tolerance)

    def describe(self) -> str:
        ci = self.delta_interval()
        return (
            f"{self.baseline_rate:.1%} -> {self.candidate_rate:.1%} "
            f"({self.delta:+.1%}, 95% CI [{ci.low:+.1%}, {ci.high:+.1%}], "
            f"n={self.n}, discordant={self.discordant}, p={self.p_two_sided:.3f})"
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "baseline_rate": round(self.baseline_rate, 6),
            "candidate_rate": round(self.candidate_rate, 6),
            "delta": round(self.delta, 6),
            "delta_ci": self.delta_interval().to_dict(),
            "baseline_only": self.baseline_only,
            "candidate_only": self.candidate_only,
            "discordant": self.discordant,
            "p_worse": round(self.p_worse, 6),
            "p_two_sided": round(self.p_two_sided, 6),
            "significant_regression": self.significant_regression,
            "min_detectable_shift": round(self.min_detectable_shift(), 6),
        }


def compare_paired_binary(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> PairedBinary:
    """Build a :class:`PairedBinary` from two aligned outcome sequences.

    The sequences must be the same length and in the same example order: index
    i in both must be the same example, or the pairing is meaningless.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and {len(candidate)}"
        )
    both = only_b = only_c = neither = 0
    for base, cand in zip(baseline, candidate):
        if base and cand:
            both += 1
        elif base and not cand:
            only_b += 1
        elif cand and not base:
            only_c += 1
        else:
            neither += 1
    return PairedBinary(
        n=len(baseline),
        both_correct=both,
        baseline_only=only_b,
        candidate_only=only_c,
        both_wrong=neither,
        alpha=alpha,
        confidence=confidence,
    )


# ──────────────────────────────────────────────────────────────────────────
# Paired continuous outcomes
# ──────────────────────────────────────────────────────────────────────────


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 20260731,
) -> Interval:
    """Percentile bootstrap CI on the mean paired difference.

    Resamples *pairs*, not individual observations, which is what preserves the
    pairing. Distribution-free, which matters because LLM-judge scores are
    bounded, lumpy, and nowhere near normal.

    The seed is fixed and explicit so a gate decision is reproducible: the same
    inputs must always produce the same verdict, or a rerun could flip a
    borderline result and nobody could audit it.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap needs equal lengths")
    n = len(baseline)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, confidence)

    diffs = [c - b for b, c in zip(baseline, candidate)]
    point = sum(diffs) / n
    if n == 1 or all(d == diffs[0] for d in diffs):
        return Interval(point, point, point, confidence)

    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo_index = int((1 - confidence) / 2 * iterations)
    hi_index = min(iterations - 1, int((1 + confidence) / 2 * iterations))
    return Interval(point, means[lo_index], means[hi_index], confidence)


def wilcoxon_signed_rank(
    baseline: Sequence[float], candidate: Sequence[float]
) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired differences.

    Returns ``(statistic, two_sided_p)``. Zero differences are dropped and ties
    receive average ranks, both standard. The p-value uses the normal
    approximation with a tie correction, which is reasonable from about eight
    non-zero differences; below that it is reported but should be read as
    indicative, and the exact McNemar path is the better tool for binary data.
    """
    if len(baseline) != len(candidate):
        raise ValueError("wilcoxon needs equal lengths")
    diffs = [c - b for b, c in zip(baseline, candidate) if c != b]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    ordered = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[ordered[j + 1]]) == abs(diffs[ordered[i]]):
            j += 1
        average = (i + j + 2) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[ordered[k]] = average
        i = j + 1

    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    statistic = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    tie_groups: dict[float, int] = {}
    for d in diffs:
        tie_groups[abs(d)] = tie_groups.get(abs(d), 0) + 1
    tie_term = sum(t**3 - t for t in tie_groups.values())
    variance = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if variance <= 0:
        return statistic, 1.0
    z = (statistic - mean) / math.sqrt(variance)
    return statistic, min(1.0, 2.0 * _NORM.cdf(-abs(z)))


@dataclass
class PairedContinuous:
    """Baseline vs candidate over paired continuous scores."""

    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    delta_ci: Interval
    wilcoxon_p: float
    effect_size: float
    alpha: float = 0.05

    @property
    def significant_improvement(self) -> bool:
        return self.wilcoxon_p < self.alpha and self.delta > 0

    @property
    def significant_regression(self) -> bool:
        return self.wilcoxon_p < self.alpha and self.delta < 0

    @property
    def inconclusive(self) -> bool:
        """True when the interval straddles zero: no evidence either way."""
        return self.delta_ci.contains(0.0)

    def describe(self) -> str:
        return (
            f"{self.baseline_mean:.3f} -> {self.candidate_mean:.3f} "
            f"({self.delta:+.3f}, 95% CI [{self.delta_ci.low:+.3f}, "
            f"{self.delta_ci.high:+.3f}], n={self.n}, p={self.wilcoxon_p:.3f}, "
            f"d={self.effect_size:+.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            "delta": round(self.delta, 6),
            "delta_ci": self.delta_ci.to_dict(),
            "wilcoxon_p": round(self.wilcoxon_p, 6),
            "effect_size": round(self.effect_size, 6),
            "significant_improvement": self.significant_improvement,
            "significant_regression": self.significant_regression,
            "inconclusive": self.inconclusive,
        }


def compare_paired_continuous(
    baseline: Sequence[float],
    candidate: Sequence[float],
    alpha: float = 0.05,
    confidence: float = 0.95,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260731,
) -> PairedContinuous:
    """Full paired comparison for continuous scores: CI, test, and effect size."""
    if len(baseline) != len(candidate):
        raise ValueError("paired comparison needs equal lengths")
    n = len(baseline)
    if n == 0:
        return PairedContinuous(
            0, 0.0, 0.0, 0.0, Interval(0.0, 0.0, 0.0, confidence), 1.0, 0.0, alpha
        )
    base_mean = sum(baseline) / n
    cand_mean = sum(candidate) / n
    _, p = wilcoxon_signed_rank(baseline, candidate)
    return PairedContinuous(
        n=n,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        delta=cand_mean - base_mean,
        delta_ci=paired_bootstrap_ci(
            baseline, candidate, confidence, bootstrap_iterations, seed
        ),
        wilcoxon_p=p,
        effect_size=cohens_d(baseline, candidate),
        alpha=alpha,
    )


# ──────────────────────────────────────────────────────────────────────────
# Trends
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class OLSTrend:
    """Least-squares fit of value against time, with the uncertainty attached."""

    n: int
    slope: float
    intercept: float
    stderr: float
    t_statistic: float
    p_value: float
    r_squared: float
    span: float
    change: float
    slope_ci: Interval
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        """True when the slope differs from zero beyond chance.

        This is a real test, not a magnitude threshold. A steep slope through
        three scattered points is not significant, and a shallow one through
        forty tight points is.
        """
        return self.n >= 3 and self.p_value < self.alpha

    def describe(self) -> str:
        return (
            f"slope {self.slope:+.4f}/day (95% CI [{self.slope_ci.low:+.4f}, "
            f"{self.slope_ci.high:+.4f}]), change {self.change:+.3f} over "
            f"{self.span:.1f}d, n={self.n}, R²={self.r_squared:.2f}, "
            f"p={self.p_value:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "slope": round(self.slope, 8),
            "intercept": round(self.intercept, 8),
            "stderr": round(self.stderr, 8),
            "t_statistic": round(self.t_statistic, 6),
            "p_value": round(self.p_value, 6),
            "r_squared": round(self.r_squared, 6),
            "span": round(self.span, 6),
            "change": round(self.change, 6),
            "slope_ci": self.slope_ci.to_dict(),
            "significant": self.significant,
        }


def ols_trend(
    xs: Sequence[float],
    ys: Sequence[float],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> OLSTrend:
    """Fit ``y = a + b*x`` and test whether the slope is distinguishable from zero.

    Needs at least three points: with two, the line passes through both exactly,
    the residual variance is zero, and there is no error term left to test
    against. Returning a confident answer from two points is how noise gets
    promoted to a trend.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("ols_trend needs equal lengths")

    empty = Interval(0.0, 0.0, 0.0, confidence)
    if n < 3:
        return OLSTrend(n, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, empty, alpha)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        # All observations share a timestamp: no slope is identifiable.
        return OLSTrend(n, 0.0, mean_y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, empty, alpha)

    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - sse / sst if sst > 0 else 1.0

    df = n - 2
    if df <= 0 or sse <= 0:
        # A perfect fit has no residual scatter. Treat it as significant only
        # when there is genuine spread in y; a flat line through flat data is not.
        perfect = sst > 0
        return OLSTrend(
            n, slope, intercept, 0.0,
            math.inf if perfect else 0.0,
            0.0 if perfect else 1.0,
            r_squared,
            max(xs) - min(xs),
            slope * (max(xs) - min(xs)),
            Interval(slope, slope, slope, confidence),
            alpha,
        )

    stderr = math.sqrt(sse / df / sxx)
    t_stat = slope / stderr if stderr > 0 else 0.0
    p_value = 2.0 * student_t_sf(abs(t_stat), df)

    # Critical t via the inverse of the survival function, bisected. Avoids
    # pulling in a full inverse-CDF implementation for one use.
    target = (1 - confidence) / 2
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if student_t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    t_crit = (lo + hi) / 2

    span = max(xs) - min(xs)
    return OLSTrend(
        n=n,
        slope=slope,
        intercept=intercept,
        stderr=stderr,
        t_statistic=t_stat,
        p_value=min(1.0, p_value),
        r_squared=r_squared,
        span=span,
        change=slope * span,
        slope_ci=Interval(
            slope, slope - t_crit * stderr, slope + t_crit * stderr, confidence
        ),
        alpha=alpha,
    )


# ──────────────────────────────────────────────────────────────────────────
# Effect sizes and baselines
# ──────────────────────────────────────────────────────────────────────────


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions.

    The arcsine transform keeps the scale meaningful near 0 and 1, where a raw
    difference badly understates the change: 0.95 to 0.99 is a far larger
    practical move than 0.50 to 0.54, despite both being four points.
    """
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Paired Cohen's d: mean difference over the standard deviation of differences."""
    if len(baseline) != len(candidate) or not baseline:
        return 0.0
    diffs = [c - b for b, c in zip(baseline, candidate)]
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return 0.0
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return mean / sd


def chance_accuracy(num_options: int) -> float:
    """Accuracy a coin-flipping selector would reach across *num_options*.

    Reported alongside tool-selection accuracy because raw accuracy is not
    interpretable without it. 40% looks poor until you know there were twelve
    tools to choose between, where chance is 8%.
    """
    return 1.0 / num_options if num_options > 0 else 0.0
