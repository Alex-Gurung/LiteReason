"""Generic reporting metrics: per-repetition stats + Wilson interval."""

import math
from typing import Dict, List

# Two-sided 95% Student-t critical values (t_0.975) by degrees of freedom (k-1),
# tabulated for df 1..30. For df >= 30 the normal approximation (z = 1.96) is
# close, so the .get(df, 1.96) fallback handles larger df.
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
         15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
         22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
         29: 2.045, 30: 2.042}


def compute_rep_stats(values: List[float]) -> Dict[str, float]:
    """Mean/std and a 95% t-CI over per-repetition values."""
    k = len(values)
    if k == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "k": 0}
    mean = sum(values) / k
    if k == 1:
        return {"mean": mean, "std": 0.0, "ci_lo": mean, "ci_hi": mean, "k": 1}
    std = (sum((v - mean) ** 2 for v in values) / (k - 1)) ** 0.5
    sem = std / math.sqrt(k)
    t = _T_95.get(k - 1, 1.96)
    return {"mean": mean, "std": std, "ci_lo": mean - t * sem, "ci_hi": mean + t * sem, "k": k}


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion ``k``/``n``."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half
