# Fair Two-Sided Matching — a clean-room demo

Stable matching (Gale–Shapley) driven by **NLP text similarity**, with **group-fairness
diagnostics** — implemented from scratch in pure-Python, on fully synthetic data.

> **Clean-room / independent work.** This repository is an original, educational
> reimplementation built **only from the public description** of the method (see the
> talk and papers linked below). It contains **no employer code, data, weights, survey
> instrument, or configuration**, is **not derived from or affiliated with any
> employer's proprietary implementation**, and every participant here is **randomly
> generated**. It exists to demonstrate the *technique and reasoning*, not to reproduce
> any production system.

## What it demonstrates

A small, auditable pipeline that goes from free text to a fair set of pairings:

1. **Text similarity** (`fairmatch/similarity.py`) — two interpretable compatibility
   signals computed from participants' short "bios":
   - **Cosine similarity** over term-frequency vectors (the classic Vector Space Model),
     length-normalized to `[0, 1]`.
   - **Matching words** — a raw count of shared tokens (simple, but unnormalized, so it
     carries a length bias).
2. **Stable matching** (`fairmatch/matching.py`) — **Gale–Shapley deferred acceptance**
   (proposer-optimal), with seeded jitter to break score ties into the strict orderings
   the algorithm needs. Includes an independent `is_stable()` verifier (no blocking pair).
3. **Fairness diagnostics** (`fairmatch/fairness.py`) — Top-20% mentor match rate, a
   **disparate-impact ratio** checked against the EEOC four-fifths (0.80) rule, and an
   **illustrative equivalence check** asking whether two matching methods produce
   practically equivalent outcomes within a chosen margin.

## Quick start

No third-party dependencies — standard library only (Python 3.9+).

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

Illustrative equivalence of the two measures (Top-20% proportion)
--------------------------------------------------------------
  cosine = 0.976   matching_words = 0.964   diff = +0.012
  90% CI on diff = [-0.013, +0.037]   margin = +/-0.05
  practically equivalent within margin: True
```

Both similarity measures place ~96–98% of participants in a Top-20% mentor match at this cohort size, both matchings
are provably stable, both clear the four-fifths disparate-impact rule across groups, and —
at this cohort size — the two measures are practically equivalent within a 5-point margin.
The equivalence verdict is intentionally sensitive to the margin and cohort size; shrink
the cohort or the margin and it will flip, which is the point of reporting an interval
rather than a bare yes/no.

## Project structure

```
fairmatch/
  similarity.py   cosine similarity + matching-words overlap (from scratch)
  matching.py     Gale-Shapley deferred acceptance + stability verifier
  fairness.py     Top-20% rate, disparate-impact ratio, illustrative equivalence
  synthetic.py    reproducible synthetic mentor/mentee generator
  pipeline.py     text -> scores -> preferences -> matching -> fairness
examples/run_demo.py    readable end-to-end report
tests/test_fairmatch.py stability, similarity, and fairness-math tests
```

## Scope and honesty notes

- The **equivalence check is an illustration**, not the paired score-interval TOST used in
  the referenced research; it conveys the idea with a plain two-proportion interval.
- Group labels ("junior"/"senior") and bios are synthetic stand-ins chosen so the metrics
  have something to measure — they are not modeled on any real population.
- The goal is clarity and correctness over completeness: the code is short enough to read
  in one sitting.

## Background and references

- D. Gale and L. S. Shapley (1962). *College Admissions and the Stability of Marriage.*
  American Mathematical Monthly.
- G. Salton, A. Wong, C. S. Yang (1975). *A Vector Space Model for Automatic Indexing.*
- EEOC Uniform Guidelines on Employee Selection Procedures — the "four-fifths" rule.
- Equivalence testing (TOST): Schuirmann (1987); Lakens (2017).

Related public work by the author (fill in links):

- JSM 2026 Speed session — "Fair Mentor Matching with Gale–Shapley Pairing, Transparent
  Design, and Statistical Validation."  ⟶ `<link>`
- "Surrogate Membership for Inferred Metrics in Fairness Evaluation," LION 2023 (coauthor).
  ⟶ `<link>`

## License

MIT — see [LICENSE](LICENSE).

*Author: Rick Pack · [linkedin.com/in/rick-pack-mappstat-5320387](https://www.linkedin.com/in/rick-pack-mappstat-5320387/) · github.com/RickPack*
