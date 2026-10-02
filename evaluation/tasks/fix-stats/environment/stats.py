"""Tiny statistics helpers."""

import math


def mean(xs: list[float]) -> float:
    """Arithmetic mean. Raises ValueError on an empty list."""
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    """Median. For an even number of values, the mean of the two middle values.
    Does not modify the input. Raises ValueError on an empty list."""
    if not xs:
        raise ValueError("median of empty list")
    xs.sort()
    return xs[len(xs) // 2]


def stdev(xs: list[float]) -> float:
    """*Sample* standard deviation (divides by n - 1). Needs at least 2 values."""
    if len(xs) < 2:
        raise ValueError("stdev needs at least two values")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def percentile(xs: list[float], p: float) -> float:
    """p-th percentile (0 <= p <= 100) using linear interpolation between
    closest ranks, i.e. the same as numpy's default ("linear") method."""
    if not xs:
        raise ValueError("percentile of empty list")
    s = sorted(xs)
    k = (len(s)) * p / 100
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
