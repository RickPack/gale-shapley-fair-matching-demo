"""Tests for the fairmatch demo. Run with `python tests/test_fairmatch.py` or `pytest`."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmatch import fairness, matching, similarity, synthetic
from fairmatch.pipeline import run_matching


def test_cosine_bounds_and_symmetry():
    a = similarity.tokenize("data analytics and statistics modeling")
    b = similarity.tokenize("statistics modeling and research methods")
    s = similarity.cosine_similarity(a, b)
    assert 0.0 <= s <= 1.0
    assert abs(s - similarity.cosine_similarity(b, a)) < 1e-12   # symmetric
    assert similarity.cosine_similarity(a, a) > 0.999            # self ~ 1
    assert similarity.cosine_similarity(a, []) == 0.0            # empty -> 0


def test_matching_words_overlap():
    a = similarity.tokenize("cloud azure scaling reliability")
    b = similarity.tokenize("azure reliability finance markets")
    assert similarity.matching_words(a, b) == 2                  # azure, reliability


def test_gale_shapley_known_instance():
    # 3x3 instance with a unique-ish stable outcome; check it is stable.
    proposer_prefs = [[0, 1, 2], [1, 0, 2], [0, 1, 2]]
    receiver_prefs = [[1, 0, 2], [0, 1, 2], [0, 1, 2]]
    m = matching.gale_shapley(proposer_prefs, receiver_prefs)
    assert len(m) == 3
    assert set(m.values()) == {0, 1, 2}                         # perfect matching
    assert matching.is_stable(m, proposer_prefs, receiver_prefs)


def test_gale_shapley_random_instances_are_stable():
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(2, 8)
        pp = [rng.sample(range(n), n) for _ in range(n)]
        rp = [rng.sample(range(n), n) for _ in range(n)]
        m = matching.gale_shapley(pp, rp)
        assert matching.is_stable(m, pp, rp)


def test_disparate_impact_math():
    flags = [True, True, False, False, True, False]
    groups = ["a", "a", "a", "b", "b", "b"]
    di = fairness.disparate_impact(flags, groups)
    assert abs(di["rates"]["a"] - 2 / 3) < 1e-9
    assert abs(di["rates"]["b"] - 1 / 3) < 1e-9
    assert abs(di["di_ratio"] - 0.5) < 1e-9
    assert di["passes_four_fifths"] is False


def test_tango_score_ci_matches_r_propcis():
    """Reproduce R's PropCIs::scoreci.mp(25, 28, 618) -> [-0.0148, +0.0247].

    Those are the pooled discordant counts from the referenced paper (b = 28 positive
    under matching words only, c = 25 under cosine only, n = 618 mentees). scoreci.mp's
    estimand is (arg2 - arg1)/n, hence the swapped argument order.
    """
    ci = fairness.tango_score_ci(b=28, c=25, n=618, conf=0.90)
    assert abs(ci["lambda_hat"] - 3 / 618) < 1e-12
    assert abs(ci["ci_low"] - (-0.0148)) < 5e-5
    assert abs(ci["ci_high"] - 0.0247) < 5e-5


def test_tango_ci_brackets_the_estimate_and_is_symmetric_under_swap():
    ci = fairness.tango_score_ci(b=28, c=25, n=618)
    assert ci["ci_low"] < ci["lambda_hat"] < ci["ci_high"]
    flipped = fairness.tango_score_ci(b=25, c=28, n=618)
    assert abs(flipped["ci_low"] + ci["ci_high"]) < 1e-9   # mirror image
    assert abs(flipped["ci_high"] + ci["ci_low"]) < 1e-9


def test_equivalence_check_counts_discordant_pairs():
    #                 1  2  3  4  5
    flags_1 = [True,  True,  False, False, True]
    flags_2 = [True,  False, True,  False, False]
    e = fairness.equivalence_check(flags_1, flags_2, margin=0.05)
    assert e["b"] == 2                      # positive under method 1 only (#2, #5)
    assert e["c"] == 1                      # positive under method 2 only (#3)
    assert abs(e["diff"] - 1 / 5) < 1e-12   # lambda-hat = (b - c)/n
    assert e["practically_equivalent"] is False


def test_pipeline_runs_and_is_stable():
    mentees, mentors = synthetic.generate(n_mentees=40, n_mentors=40, seed=7)
    r = run_matching(mentees, mentors, measure="cosine", seed=7)
    assert r.stable is True
    assert 0.0 <= r.top20_rate <= 1.0
    assert 0.0 <= r.disparate_impact["di_ratio"] <= 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
