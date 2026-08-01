"""Gale-Shapley deferred-acceptance stable matching (Gale & Shapley, 1962).

Proposer-optimal: with complete strict preference lists the proposing side receives
its best achievable stable partner. Implemented from the textbook algorithm; supports
unequal sides (extra agents on the larger side stay unmatched) and produces a matching
with no blocking pair.

Ties in the underlying scores are broken with small seeded random jitter when building
preferences, which yields the strict orderings the algorithm requires and avoids the
systematic bias that fixed tie-breaking can introduce.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence


def preferences_from_scores(scores: Sequence[Sequence[float]],
                            rng: random.Random,
                            jitter: float = 1e-9) -> List[List[int]]:
    """Rank receivers for each proposer by descending score, jittering to break ties.

    `scores` is a proposers x receivers matrix. Returns, for each proposer, a list of
    receiver indices from most to least preferred.
    """
    n_receivers = len(scores[0]) if scores else 0
    prefs: List[List[int]] = []
    for row in scores:
        order = sorted(range(n_receivers),
                       key=lambda j: row[j] + rng.uniform(0.0, jitter),
                       reverse=True)
        prefs.append(order)
    return prefs


def transpose(scores: Sequence[Sequence[float]]) -> List[List[float]]:
    """Transpose a score matrix (proposers x receivers -> receivers x proposers)."""
    if not scores:
        return []
    n_rows, n_cols = len(scores), len(scores[0])
    return [[scores[i][j] for i in range(n_rows)] for j in range(n_cols)]


def gale_shapley(proposer_prefs: Sequence[Sequence[int]],
                 receiver_prefs: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Return a stable matching as {proposer_index: receiver_index}.

    `proposer_prefs[p]` is p's ranked list of receiver indices; `receiver_prefs[r]`
    is r's ranked list of proposer indices. Unmatched proposers are absent from the
    returned dict.
    """
    n_p = len(proposer_prefs)
    # receiver_rank[r][p] = position of proposer p in receiver r's list (lower = better)
    receiver_rank: List[Dict[int, int]] = [
        {p: rank for rank, p in enumerate(receiver_prefs[r])}
        for r in range(len(receiver_prefs))
    ]
    free = list(range(n_p))
    next_choice = [0] * n_p          # index of next receiver p will propose to
    match_of_receiver: Dict[int, int] = {}
    match_of_proposer: Dict[int, int] = {}

    while free:
        p = free.pop(0)
        if next_choice[p] >= len(proposer_prefs[p]):
            continue                  # p exhausted its list; stays unmatched
        r = proposer_prefs[p][next_choice[p]]
        next_choice[p] += 1
        if r not in match_of_receiver:
            match_of_receiver[r] = p
            match_of_proposer[p] = r
        else:
            incumbent = match_of_receiver[r]
            big = len(receiver_rank[r]) + 1
            if receiver_rank[r].get(p, big) < receiver_rank[r].get(incumbent, big):
                match_of_receiver[r] = p          # r trades up to p
                match_of_proposer[p] = r
                del match_of_proposer[incumbent]
                free.append(incumbent)
            else:
                free.append(p)                    # r keeps incumbent; p tries again
    return match_of_proposer


def is_stable(matching: Dict[int, int],
              proposer_prefs: Sequence[Sequence[int]],
              receiver_prefs: Sequence[Sequence[int]]) -> bool:
    """True if `matching` has no blocking pair (verifier used by the tests)."""
    prop_rank = [
        {r: k for k, r in enumerate(proposer_prefs[p])}
        for p in range(len(proposer_prefs))
    ]
    recv_rank = [
        {p: k for k, p in enumerate(receiver_prefs[r])}
        for r in range(len(receiver_prefs))
    ]
    match_of_receiver = {r: p for p, r in matching.items()}
    big = 10 ** 9
    for p in range(len(proposer_prefs)):
        cur_r = matching.get(p)
        cur_r_rank = prop_rank[p].get(cur_r, big) if cur_r is not None else big
        for r in proposer_prefs[p]:
            if prop_rank[p][r] >= cur_r_rank:
                break                              # p likes cur_r at least as much
            incumbent = match_of_receiver.get(r)
            incumbent_rank = recv_rank[r].get(incumbent, big) if incumbent is not None else big
            if recv_rank[r].get(p, big) < incumbent_rank:
                return False                       # (p, r) block the matching
    return True
