"""End-to-end demo pipeline: text -> scores -> preferences -> matching -> fairness."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import fairness, matching, similarity
from .synthetic import Person


@dataclass
class MatchResult:
    measure: str
    matching: Dict[int, int]
    top20_flags: List[bool]
    top20_rate: float
    disparate_impact: dict
    stable: bool


def run_matching(mentees: Sequence[Person],
                 mentors: Sequence[Person],
                 measure: str = "cosine",
                 tier_fraction: float = 0.20,
                 seed: int = 20260724) -> MatchResult:
    """Run one full matching under a similarity measure and score its fairness."""
    rng = random.Random(seed)

    mentee_tokens = [similarity.tokenize(p.bio) for p in mentees]
    mentor_tokens = [similarity.tokenize(p.bio) for p in mentors]

    scores = similarity.score_matrix(mentee_tokens, mentor_tokens, measure=measure)

    mentee_prefs = matching.preferences_from_scores(scores, rng)
    mentor_prefs = matching.preferences_from_scores(matching.transpose(scores), rng)

    m = matching.gale_shapley(mentee_prefs, mentor_prefs)

    flags = fairness.top20_flags(m, mentee_prefs, tier_fraction=tier_fraction)
    rate = sum(1 for f in flags if f) / len(flags) if flags else 0.0
    groups = [p.group for p in mentees]
    di = fairness.disparate_impact(flags, groups)
    stable = matching.is_stable(m, mentee_prefs, mentor_prefs)

    return MatchResult(measure=measure, matching=m, top20_flags=flags,
                       top20_rate=rate, disparate_impact=di, stable=stable)


def compare_measures(mentees: Sequence[Person],
                     mentors: Sequence[Person],
                     margin: float = 0.05,
                     seed: int = 20260724):
    """Run both measures and return (results_by_measure, equivalence_summary)."""
    results = {name: run_matching(mentees, mentors, measure=name, seed=seed)
               for name in ("cosine", "matching_words")}
    # Orientation follows the paper: lambda-hat = Rate(MW) - Rate(CS).
    equiv = fairness.equivalence_check(
        results["matching_words"].top20_flags,
        results["cosine"].top20_flags,
        margin=margin,
    )
    return results, equiv
