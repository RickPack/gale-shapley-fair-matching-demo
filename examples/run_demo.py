"""Run the full clean-room demo on synthetic data and print a readable report.

    python examples/run_demo.py
"""
import os
import sys

# allow running directly from the repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmatch import synthetic
from fairmatch.pipeline import compare_measures


def main() -> None:
    mentees, mentors = synthetic.generate(n_mentees=250, n_mentors=250, seed=20260724)
    print(f"Synthetic cohort: {len(mentees)} mentees, {len(mentors)} mentors "
          f"(all data randomly generated).\n")

    results, equiv = compare_measures(mentees, mentors, margin=0.05)

    print("Match quality and fairness by similarity measure")
    print("-" * 62)
    for name, r in results.items():
        di = r.disparate_impact
        rates = ", ".join(f"{g}={v:.2f}" for g, v in sorted(di["rates"].items()))
        print(f"  {name:>15}:  top-tier rate = {r.top_tier_rate:.3f}   "
              f"stable = {r.stable}")
        print(f"  {'':>15}   group rates ({rates})")
        print(f"  {'':>15}   DI ratio = {di['di_ratio']:.3f}   "
              f"four-fifths pass = {di['passes_four_fifths']}\n")

    print("Illustrative equivalence of the two measures (top-tier proportion)")
    print("-" * 62)
    print(f"  cosine = {equiv['prop_a']:.3f}   matching_words = {equiv['prop_b']:.3f}   "
          f"diff = {equiv['diff']:+.3f}")
    print(f"  90% CI on diff = [{equiv['ci_low']:+.3f}, {equiv['ci_high']:+.3f}]   "
          f"margin = +/-{equiv['margin']:.2f}")
    print(f"  practically equivalent within margin: "
          f"{equiv['practically_equivalent']}")
    print("\n(Equivalence here is a simple illustration, not the paired-score TOST "
          "in the paper.)")


if __name__ == "__main__":
    main()
