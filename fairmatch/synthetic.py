"""Reproducible synthetic mentor/mentee generator.

Produces fake participants, each with an id, a group label (a synthetic "grade band"),
and a short free-text "bio" assembled from random interest topics. Because bios are
built from shared topic vocabularies, the text-similarity measures have real signal to
work with. Everything here is randomly generated and seeded — there is no real person,
survey, or organizational data involved.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

# Generic interest "topics", each a small bag of words a bio might draw from.
TOPICS = {
    "data": ["data", "analytics", "statistics", "modeling", "pipelines", "sql", "python"],
    "leadership": ["leadership", "mentoring", "coaching", "strategy", "teams", "growth"],
    "product": ["product", "roadmap", "customers", "design", "discovery", "delivery"],
    "cloud": ["cloud", "azure", "infrastructure", "scaling", "reliability", "systems"],
    "research": ["research", "experiments", "fairness", "methods", "writing", "review"],
    "finance": ["finance", "investing", "risk", "markets", "planning", "portfolio"],
    "wellbeing": ["wellbeing", "balance", "community", "volunteering", "networking"],
}
_CONNECTORS = ["interested in", "excited about", "focused on", "hoping to grow in",
               "looking to learn", "passionate about"]


@dataclass
class Person:
    pid: str
    group: str
    bio: str


def _make_bio(rng: random.Random, n_topics: int = 3, words_per_topic: int = 4) -> str:
    chosen = rng.sample(list(TOPICS), k=n_topics)
    parts = []
    for topic in chosen:
        words = rng.sample(TOPICS[topic], k=min(words_per_topic, len(TOPICS[topic])))
        parts.append(f"{rng.choice(_CONNECTORS)} {' '.join(words)}")
    return "; ".join(parts) + "."


def generate(n_mentees: int = 120,
             n_mentors: int = 120,
             groups=("junior", "senior"),
             group_weights=(0.6, 0.4),
             seed: int = 20260724):
    """Return (mentees, mentors) as lists of Person, reproducible for a given seed."""
    rng = random.Random(seed)

    def make(prefix: str, n: int) -> List[Person]:
        people = []
        for i in range(n):
            g = rng.choices(list(groups), weights=list(group_weights), k=1)[0]
            people.append(Person(pid=f"{prefix}{i:03d}", group=g, bio=_make_bio(rng)))
        return people

    mentees = make("mentee_", n_mentees)
    mentors = make("mentor_", n_mentors)
    return mentees, mentors
