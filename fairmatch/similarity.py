"""Text similarity measures, implemented from first principles (standard library only).

Two measures, mirroring the two NLP compatibility signals described publicly:

* **Cosine similarity** over term-frequency vectors (the classic Vector Space Model;
  Salton, Wong & Yang, 1975) — captures relatedness even when exact words differ, and
  is length-normalized to [0, 1].
* **Matching words** — a raw count of shared unique tokens. Simple and interpretable,
  but unnormalized, so it carries a length bias (longer texts share more by chance).

Neither uses any proprietary vocabulary, stop-word list, or weighting; the small
stop-word set below is a generic English list included only so the demo is self-contained.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

# A short, generic English stop-word list (not tied to any specific application).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "i", "you", "we",
    "they", "he", "she", "them", "my", "our", "your", "their", "me", "us",
    "do", "does", "did", "have", "has", "had", "not", "no", "so", "than",
    "then", "there", "here", "about", "into", "over", "up", "out", "am", "will",
}

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> List[str]:
    """Lower-case, keep alphabetic tokens, drop stop-words and single characters."""
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) > 1 and t not in STOPWORDS]


def cosine_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Cosine similarity of two token sequences via term-frequency vectors, in [0, 1]."""
    va, vb = Counter(tokens_a), Counter(tokens_b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in va.values()))
    norm_b = math.sqrt(sum(v * v for v in vb.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def matching_words(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    """Number of distinct tokens shared by the two sequences (unnormalized overlap)."""
    return len(set(tokens_a) & set(tokens_b))


# Registry so the pipeline can select a measure by name.
MEASURES = {
    "cosine": lambda a, b: cosine_similarity(a, b),
    "matching_words": lambda a, b: float(matching_words(a, b)),
}


def score_matrix(proposer_tokens: Iterable[Sequence[str]],
                 receiver_tokens: Iterable[Sequence[str]],
                 measure: str = "cosine") -> List[List[float]]:
    """Build a proposers x receivers score matrix under the named measure."""
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}; choose from {sorted(MEASURES)}")
    fn = MEASURES[measure]
    prow = list(proposer_tokens)
    rcol = list(receiver_tokens)
    return [[fn(pt, rt) for rt in rcol] for pt in prow]
