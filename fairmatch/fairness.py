"""Group-fairness diagnostics for a matching.

Deliberately lightweight and transparent:

* **Top-20% mentor match rate** — the share of proposers matched to one of their most
  preferred receivers (top `tier_fraction` of the receiver list). A simple, auditable
  proxy for match quality.
* **Disparate-impact (DI) ratio** — the ratio of Top-20% rates between demographic
  groups, compared against the EEOC "four-fifths" (0.80) rule of thumb.
* **Illustrative equivalence check** — a plain two-proportion difference with a
  normal-approximation interval, to ask whether two matching methods produce
  practically equivalent Top-20% rates within a chosen margin.

NOTE: the equivalence check here is an *illustration*, not the paired score-interval
TOST used in the referenced paper. It is intended to convey the idea, not to reproduce
the full statistical apparatus.
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


def equivalence_check(flags_a: Sequence[bool],
                      flags_b: Sequence[bool],
                      margin: float = 0.05,
                      conf: float = 0.90) -> Dict[str, Optional[float]]:
    """Illustrative equivalence of two methods' overall Top-20% proportions.

    Returns the two proportions, their difference, a normal-approx confidence interval
    on the difference, and whether that interval sits entirely within +/- `margin`
    (a TOST-style "practically equivalent" verdict). Illustrative only.
    """
    def prop(flags: Sequence[bool]) -> float:
        return sum(1 for f in flags if f) / len(flags) if flags else 0.0

    n_a, n_b = len(flags_a), len(flags_b)
    p_a, p_b = prop(flags_a), prop(flags_b)
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b) if n_a and n_b else 0.0
    # two-sided z for the given confidence (0.90 -> ~1.645)
    z = {0.90: 1.645, 0.95: 1.960}.get(round(conf, 2), 1.645)
    lo, hi = diff - z * se, diff + z * se
    equivalent = (lo > -margin) and (hi < margin)
    return {"prop_a": p_a, "prop_b": p_b, "diff": diff,
            "ci_low": lo, "ci_high": hi, "margin": margin,
            "practically_equivalent": equivalent}
