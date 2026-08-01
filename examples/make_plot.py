"""Plot the paired equivalence interval as the cohort shrinks.

    pip install -e ".[viz]"
    python examples/make_plot.py

Writes two PNGs to assets/:
  equivalence_vs_cohort.png         1200x675 (1.91:1, social/LinkedIn)
  equivalence_vs_cohort_readme.png  1000x620 (README)

Everything is seeded, so the committed images match what you get from a fresh run.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running directly from the repo root without installing
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fairmatch import synthetic
from fairmatch.pipeline import compare_measures

COHORTS = (50, 100, 250, 500)
MARGIN = 0.05
SEED = 20260724

# Okabe-Ito, colour-blind safe (matches the JSM poster palette)
BLUE = "#0072B2"        # equivalent
VERMILLION = "#D55E00"  # not equivalent
GREY = "#666666"
BAND = "#009E73"        # green: the equivalence margin


def collect():
    """Return one equivalence summary per cohort size."""
    rows = []
    for n in COHORTS:
        mentees, mentors = synthetic.generate(n_mentees=n, n_mentors=n, seed=SEED)
        _, equiv = compare_measures(mentees, mentors, margin=MARGIN, seed=SEED)
        rows.append((n, equiv))
    return rows


def draw(rows, path, size_in, dpi, base_font):
    plt.rcParams.update({
        "font.size": base_font,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": GREY,
    })
    fig, ax = plt.subplots(figsize=size_in, dpi=dpi)

    x = list(range(len(rows)))

    # equivalence margin band
    ax.axhspan(-MARGIN, MARGIN, color=BAND, alpha=0.15, zorder=0)
    ax.axhline(0.0, color=GREY, lw=1, ls=":", zorder=1)
    for edge in (-MARGIN, MARGIN):
        ax.axhline(edge, color=BAND, lw=1.4, zorder=1)

    for xi, (n, e) in zip(x, rows):
        eq = e["practically_equivalent"]
        c = BLUE if eq else VERMILLION
        ax.plot([xi, xi], [e["ci_low"], e["ci_high"]], color=c, lw=6, zorder=3,
                solid_capstyle="butt")
        ax.plot([xi], [e["diff"]], "o", color=c, ms=9, mec="white", mew=1.6, zorder=4)
        label = "equivalent" if eq else "verdict flips"
        ax.annotate(label, (xi, e["ci_high"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=base_font - 1,
                    color=c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}" for n, _ in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Cohort size (mentees = mentors)", labelpad=8)
    ax.set_ylabel("$\\hat{\\lambda}$ = Top-20% rate,\nmatching words $-$ cosine")
    ax.set_title("Small cohorts flip the equivalence verdict",
                 fontsize=base_font + 5, fontweight="bold", loc="left", pad=34)
    ax.text(0.0, 1.012,
            f"90% Tango score CI vs. a ±{MARGIN:.2f} equivalence margin"
            "  ·  SYNTHETIC DATA",
            transform=ax.transAxes, fontsize=base_font - 1, color=GREY, va="bottom")

    lo = min(e["ci_low"] for _, e in rows)
    hi = max(e["ci_high"] for _, e in rows)
    pad = 0.22 * max(MARGIN, hi - lo)
    ax.set_ylim(min(lo, -MARGIN) - pad, max(hi, MARGIN) + pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.8)
    ax.set_axisbelow(True)

    ax.text(1.0, -0.17,
            "Synthetic data · github.com/RickPack/gale-shapley-fair-matching-demo",
            transform=ax.transAxes, ha="right", fontsize=base_font - 2, color=GREY)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    rows = collect()
    for n, e in rows:
        print(f"n = {n:>3}:  diff = {e['diff']:+.3f}   "
              f"90% CI = [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]   "
              f"equivalent = {e['practically_equivalent']}")

    out = os.path.join(REPO, "assets")
    os.makedirs(out, exist_ok=True)
    # 1200x675 at dpi 100 -> 1.91:1 social card
    draw(rows, os.path.join(out, "equivalence_vs_cohort.png"),
         size_in=(12.0, 6.75), dpi=100, base_font=15)
    # README copy
    draw(rows, os.path.join(out, "equivalence_vs_cohort_readme.png"),
         size_in=(10.0, 6.2), dpi=100, base_font=14)


if __name__ == "__main__":
    main()
