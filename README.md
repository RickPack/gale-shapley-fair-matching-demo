# Fair Two-Sided Matching: a clean-room demo

Stable matching (Gale–Shapley) driven by **NLP text similarity**, with **group-fairness
diagnostics**. Implemented from scratch in pure Python, on fully synthetic data.

> **Clean-room / independent work.** This repository is an original, educational
> reimplementation built **only from the public description** of the method (see the
> talk and papers linked below). It contains **no employer code, data, weights, survey
> instrument, or configuration**, is **not derived from or affiliated with any
> employer's proprietary implementation**, and every participant here is **randomly
> generated**. It exists to demonstrate the *technique and reasoning*, not to reproduce
> any production system.

## Why synthetic data?

Real matching pipelines run on survey responses that contain free-text descriptions,
demographic indicators, and organizational metadata — none of which can be published.
Synthetic data lets you validate the statistical properties of the matching algorithm
(stability, fairness, equivalence) and share the full pipeline publicly, without
exposing any personally identifiable information. Every participant in this repository
is randomly generated; the bios, group labels, and cohort sizes are chosen to exercise
the same code paths and fairness boundaries that matter on real data.

## What it demonstrates

A small, auditable pipeline that goes from free text to a fair set of pairings:

1. **Text similarity** (`fairmatch/similarity.py`): two interpretable compatibility
   signals computed from participants' short "bios":
   - **Cosine similarity** over term-frequency vectors (the classic Vector Space Model),
     length-normalized to `[0, 1]`.
   - **Matching words**: a raw count of shared tokens (simple, but unnormalized, so it
     carries a length bias).
2. **Stable matching** (`fairmatch/matching.py`): **Gale–Shapley deferred acceptance**
   (proposer-optimal), with seeded jitter to break score ties into the strict orderings
   the algorithm needs. Includes an independent `is_stable()` verifier (no blocking pair).
3. **Fairness diagnostics** (`fairmatch/fairness.py`): Top-20% mentor match rate, a
   **disparate-impact ratio** checked against the EEOC four-fifths (0.80) rule, and a
   **paired equivalence test (TOST)** asking whether two matching methods produce
   practically equivalent outcomes within a prespecified margin. Built on **Tango's
   (1998) score interval** for the paired difference in proportions, ported from scratch
   in ~40 lines of standard library.

## Quick start

No third-party dependencies. Standard library only (Python 3.9+).

```bash
python examples/run_demo.py      # end-to-end demo on synthetic data
python tests/test_fairmatch.py   # run the tests (or: pytest)
```

### Example output

```
Synthetic cohort: 250 mentees, 250 mentors (all data randomly generated).

Match quality and fairness by similarity measure
--------------------------------------------------------------
           cosine:  Top-20% rate = 0.976   stable = True
                    group rates (junior=0.97, senior=0.98)
                    DI ratio = 0.993   four-fifths pass = True

   matching_words:  Top-20% rate = 0.964   stable = True
                    group rates (junior=0.97, senior=0.96)
                    DI ratio = 0.994   four-fifths pass = True

Paired equivalence of the two measures (TOST, Top-20% proportion)
--------------------------------------------------------------
  matching_words = 0.964   cosine = 0.976   lambda-hat = -0.012
  discordant pairs: b (MW only) = 5, c (CS only) = 8, n = 250
  90% Tango score CI = [-0.039, +0.013]   margin = +/-0.05
  practically equivalent within margin: True
```

Both similarity measures place ~96–98% of participants in a Top-20% mentor match at
n = 250, both matchings are provably stable, both clear the four-fifths
disparate-impact rule across groups, and the two measures are practically equivalent
within a 5-point margin. That last verdict depends on how many participants you test,
which the next section takes apart.

### The verdict flips as the cohort shrinks

![90% Tango score confidence intervals on lambda-hat, the difference in Top-20% match
rate between matching-words and cosine similarity, at synthetic cohort sizes of 50, 100,
250 and 500. A shaded band marks the plus-or-minus 0.05 equivalence margin. At n = 250
the interval [-0.039, +0.013] and at n = 500 the interval [-0.012, +0.012] sit entirely
inside the band, so the two measures are declared practically equivalent. At n = 100 the
interval [-0.044, +0.066] and at n = 50 the interval [-0.108, +0.064] spill outside the
band and the equivalence verdict flips. All data is synthetic.](assets/equivalence_vs_cohort_readme.png)

| Cohort | λ̂ = Rate(MW) − Rate(CS) | 90% Tango score CI | Within ±0.05? |
|---|---|---|---|
| n = 50 | −0.020 | [−0.108, +0.064] | no |
| n = 100 | +0.010 | [−0.044, +0.066] | no |
| n = 250 | −0.012 | [−0.039, +0.013] | **yes** |
| n = 500 | +0.000 | [−0.012, +0.012] | **yes** |

Same seed (20260724), same margin (±0.05). Only the cohort size changes. The point estimate
barely moves; the *interval* does, and that is what decides the verdict.

Lo, Datta & Salami (2025, §4, *AI and Ethics*) argue directly that fairness tests near
a threshold require enough power to distinguish near-compliance from breach, and that
small cohorts commonly fail tests that larger ones pass, not because the algorithm
behaved differently but because the interval widened. The table above is that argument
run to ground: one algorithm, one seed, four cohort sizes, and the verdict turns over
between the second row and the third. The
[full R analysis](https://github.com/RickPack/gale-shapley-fair-matching-synthetic) pools
588 matched pairs across three synthetic survey years and fails equivalence for a
different reason: signal collapse, not sample size. Both failure modes are explained
with measured data, not asserted.

Reproduce the figure with:

```bash
pip install -e ".[viz]"     # matplotlib is an optional extra; core install stays clean
python examples/make_plot.py
```

## Project structure

```
fairmatch/
  similarity.py   cosine similarity + matching-words overlap (from scratch)
  matching.py     Gale-Shapley deferred acceptance + stability verifier
  fairness.py     Top-20% rate, disparate-impact ratio, Tango paired TOST
  synthetic.py    reproducible synthetic mentor/mentee generator
  pipeline.py     text -> scores -> preferences -> matching -> fairness
examples/run_demo.py    readable end-to-end report
examples/make_plot.py   equivalence-vs-cohort-size figure (needs the [viz] extra)
assets/                 committed PNGs produced by make_plot.py
tests/test_fairmatch.py stability, similarity, and fairness-math tests
```

## Scope and honesty notes

- The **equivalence check is the paired score-interval TOST**: both measures score the
  same participants, so the design is paired and an independent-samples interval would be
  the wrong model. `tango_score_ci()` is a from-scratch port of Tango (1998), pinned by a
  test that reproduces R's `PropCIs::scoreci.mp(25, 28, 618)` to four decimals.
  The wider apparatus of the referenced research (year strata, the survey-weight grid,
  sensitivity margins) is in the full R analysis linked above, not here.
- Group labels ("junior"/"senior") and bios are synthetic stand-ins, not modeled on any
  real population.
- The goal is clarity and correctness, not completeness. The code is short enough to read
  in one sitting.

## Background and references

- D. Gale and L. S. Shapley (1962). *College Admissions and the Stability of Marriage.*
  American Mathematical Monthly.
- G. Salton, A. Wong, C. S. Yang (1975). *A Vector Space Model for Automatic Indexing.*
- EEOC Uniform Guidelines on Employee Selection Procedures (the "four-fifths" rule).
- Equivalence testing (TOST): Schuirmann (1987); Lakens (2017).
- T. Tango (1998). *Equivalence test and confidence interval for the difference in
  proportions for the paired-sample design.* Statistics in Medicine, 17(8), 891-908.
- J.-P. Liu, H.-M. Hsueh, E. Hsieh, J. J. Chen (2002). *Tests for equivalence or
  non-inferiority for paired binary data.* Statistics in Medicine, 21(2), 231-245.
- V. S. Y. Lo, S. Datta & Y. Salami (2025). *Bringing practical statistical science to
  AI and predictive model fairness testing.* AI and Ethics, 5, 2149-2164.

Related public work by the author:

- JSM 2026 Speed session: "Fair Mentor Matching with Gale–Shapley Pairing, Transparent
  Design, and Statistical Validation."
  ⟶ https://ww3.aievolution.com/JSMAnnual2026/Events/viewEv?ev=9095
- "Surrogate Membership for Inferred Metrics in Fairness Evaluation," LION 2023 (coauthor).
  ⟶ https://link.springer.com/chapter/10.1007/978-3-031-44505-7_29

## License

MIT. See [LICENSE](LICENSE).

*Author: Rick Pack · [linkedin.com/in/rick-pack-mappstat-5320387](https://www.linkedin.com/in/rick-pack-mappstat-5320387/) · github.com/RickPack*
