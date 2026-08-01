"""fairmatch — a clean-room demo of fair two-sided matching.

Independent, educational reimplementation built only from the PUBLIC description of
the method (stable matching + NLP text similarity + group-fairness diagnostics). It
contains no employer code, data, weights, or configuration and is not derived from any
proprietary implementation. All data used with this package is synthetically generated.

Modules
-------
similarity : text tokenization, cosine similarity, and matching-words overlap
matching   : Gale-Shapley deferred-acceptance stable matching (proposer-optimal)
fairness   : Top-20% mentor match quality, disparate-impact ratio, illustrative equivalence
synthetic  : reproducible synthetic mentor/mentee generator
pipeline   : end-to-end text -> scores -> preferences -> matching -> fairness
"""
from . import similarity, matching, fairness, synthetic, pipeline  # noqa: F401

__all__ = ["similarity", "matching", "fairness", "synthetic", "pipeline"]
__version__ = "0.1.0"
