"""Group-fairness diagnostics for a matching.

Deliberately lightweight and transparent:

* **Top-20% mentor match rate** — the share of proposers matched to one of their most
  preferred receivers (top `tier_fraction` of the receiver list). A simple, auditable
  proxy for match quality.
* **Disparate-impact (DI) ratio** — the ratio of Top-20% rates between demographic
  groups, compared against the EEOC "four-fifths" (0.80) rule of thumb.
* **Paired equivalence check (TOST)** — Tango's (1998) score confidence interval for
  the difference in proportions under a *paired* design, used to ask whether two
  similarity measures produce practically equivalent Top-20% rates within a
  prespecified margin.

The design is paired: both measures score the *same* participants, so the two Top-20%
rates are correlated. An independent-samples interval would be the wrong model — it
ignores that correlation and is generally too wide. Tango's score interval is built
from the discordant pairs instead (the participants who are Top-20% under one measure
but not the other), which is what carries the information about the difference.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence


def top20_flags(matching: Dict[int, int],
                   proposer_prefs: Sequence[Sequence[int]],
                   tier_fraction: float = 0.20) -> List[bool]:
    """For each proposer, whether their matched receiver is in their top `tier_fraction`.

    Proposers with no match are counted as not Top-20%.
    """
    n_receivers = len(proposer_prefs[0]) if proposer_prefs else 0
    cutoff = max(1, math.ceil(tier_fraction * n_receivers))
    flags: List[bool] = []
    for p in range(len(proposer_prefs)):
        r = matching.get(p)
        if r is None:
            flags.append(False)
            continue
        rank = proposer_prefs[p].index(r)      # 0 = most preferred
        flags.append(rank < cutoff)
    return flags


def group_rates(flags: Sequence[bool], groups: Sequence[str]) -> Dict[str, float]:
    """Top-20% selection rate within each group label."""
    hit: Dict[str, int] = defaultdict(int)
    tot: Dict[str, int] = defaultdict(int)
    for f, g in zip(flags, groups):
        tot[g] += 1
        hit[g] += 1 if f else 0
    return {g: (hit[g] / tot[g] if tot[g] else 0.0) for g in tot}


def disparate_impact(flags: Sequence[bool], groups: Sequence[str]) -> Dict[str, float]:
    """Disparate-impact summary: per-group rates, the min/max DI ratio, and pass flag."""
    rates = group_rates(flags, groups)
    if len(rates) < 2:
        return {"rates": rates, "di_ratio": 1.0, "passes_four_fifths": True}
    lo, hi = min(rates.values()), max(rates.values())
    di = (lo / hi) if hi > 0 else 1.0
    return {"rates": rates, "di_ratio": di, "passes_four_fifths": di >= 0.80}


def _rmle_q12(b: int, c: int, n: int, lam: float) -> float:
    """Restricted MLE of the discordant-cell probability q12, given lambda.

    Larger root of Tango's (1998) Eq. 25-26 quadratic. Writing the discordant cells as
    q21 (positive under method 1 only) and q12 (positive under method 2 only), with
    lambda = q21 - q12, the constrained likelihood gives

        2n q12^2 - [b + c - lambda(b + 3c + 2d)] q12 - c*lambda(1 - lambda) = 0

    where d = n - b - c is the concordant count.
    """
    d = n - b - c
    B = b + c - lam * (b + 3 * c + 2 * d)
    disc = B * B + 8 * n * c * lam * (1 - lam)
    disc = max(disc, 0.0)
    return (B + math.sqrt(disc)) / (4 * n)


def _tango_z(b: int, c: int, n: int, lam: float) -> float:
    """Tango's score statistic evaluated at lambda (his Eq. 24 at Delta = -lambda)."""
    q12 = _rmle_q12(b, c, n, lam)
    var = n * (2 * q12 + lam * (1 - lam))
    if var <= 0:
        return math.inf if (b - c - n * lam) > 0 else -math.inf
    return (b - c - n * lam) / math.sqrt(var)


def tango_score_ci(b: int, c: int, n: int,
                   conf: float = 0.90,
                   tol: float = 1e-10) -> Dict[str, float]:
    """Tango (1998) score confidence interval for the paired difference of proportions.

    `b` counts pairs positive under method 1 only, `c` pairs positive under method 2
    only, `n` the number of pairs. The estimand is lambda = (b - c) / n, i.e.
    Rate(method 1) - Rate(method 2). The interval is {lambda : |Z(lambda)| <= z}, found
    by bisection because Z is monotone decreasing in lambda.

    Equivalent to R's `PropCIs::scoreci.mp(c, b, n, conf)` (note the argument order:
    that function's estimand is (arg2 - arg1) / n).
    """
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}.get(round(conf, 2), 1.6448536269514722)
    lam_hat = (b - c) / n if n else 0.0

    def solve(target: float, lo: float, hi: float) -> float:
        """Find lambda in [lo, hi] with Z(lambda) = target."""
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            # Z decreases in lambda, so shrink toward the side that brackets `target`.
            if _tango_z(b, c, n, mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    ci_low = solve(z, -1.0, lam_hat)      # Z = +z at the lower limit
    ci_high = solve(-z, lam_hat, 1.0)     # Z = -z at the upper limit
    return {"lambda_hat": lam_hat, "ci_low": ci_low, "ci_high": ci_high,
            "b": b, "c": c, "n": n}


def equivalence_check(flags_1: Sequence[bool],
                      flags_2: Sequence[bool],
                      margin: float = 0.05,
                      conf: float = 0.90) -> Dict[str, Optional[float]]:
    """Paired TOST equivalence of two methods' Top-20% proportions.

    `flags_1` and `flags_2` are the per-participant Top-20% outcomes under the two
    methods, in the same participant order. Returns the two rates, the paired
    difference lambda-hat = Rate(1) - Rate(2), Tango's score interval on it, the
    discordant counts, and whether the interval sits entirely inside +/- `margin`
    (the TOST "practically equivalent" verdict).
    """
    n = len(flags_1)
    if n != len(flags_2):
        raise ValueError("paired design requires equal-length flag sequences")

    def prop(flags: Sequence[bool]) -> float:
        return sum(1 for f in flags if f) / len(flags) if flags else 0.0

    # discordant pairs: b = positive under method 1 only, c = under method 2 only
    b = sum(1 for f1, f2 in zip(flags_1, flags_2) if f1 and not f2)
    c = sum(1 for f1, f2 in zip(flags_1, flags_2) if f2 and not f1)

    p_1, p_2 = prop(flags_1), prop(flags_2)
    ci = tango_score_ci(b, c, n, conf=conf) if n else {
        "lambda_hat": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    lo, hi = ci["ci_low"], ci["ci_high"]
    equivalent = (lo > -margin) and (hi < margin)
    return {"prop_1": p_1, "prop_2": p_2, "diff": ci["lambda_hat"],
            "b": b, "c": c, "n": n,
            "ci_low": lo, "ci_high": hi, "margin": margin,
            "practically_equivalent": equivalent}
