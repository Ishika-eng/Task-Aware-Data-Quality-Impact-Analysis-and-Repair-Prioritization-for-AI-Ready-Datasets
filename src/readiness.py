"""
AI-readiness assessment.

After we've prioritized and repaired the highest-impact issues, the natural
question is: "is this dataset now good enough for the task?" We answer that
with a simple, interpretable ratio rather than inventing a new opaque
metric -- readiness is just how much of the original clean-vs-dirty gap we
recovered.
"""

def readiness_score(clean_f1: float, dirty_f1: float, repaired_f1: float) -> float:
    """Fraction of the clean/dirty performance gap recovered by repair.

    1.0  -> repaired data performs exactly as well as the clean baseline.
    0.0  -> repair recovered none of the lost performance.
    Can exceed 1.0 if repaired performance beats the original clean run
    (possible due to run-to-run noise, or because repair also mitigated
    an error type that was always lurking in the "clean" split).
    """
    gap = clean_f1 - dirty_f1
    if gap <= 0:
        return 1.0  # error caused no measurable damage to begin with
    recovered = repaired_f1 - dirty_f1
    return recovered / gap


def readiness_verdict(score: float, threshold: float = 0.9) -> str:
    if score >= threshold:
        return "AI-ready"
    elif score >= 0.5:
        return "Partially ready -- further repair recommended"
    else:
        return "Not ready -- repair did not sufficiently recover performance"
