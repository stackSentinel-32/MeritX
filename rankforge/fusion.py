"""
rankforge/fusion.py
-------------------
Reciprocal Rank Fusion (RRF) of the three scoring signals.

rrf_fusion(kw, tfidf, bm25, k=60) -> list[float]
  Fuses Signal A (keyword), Signal B (TF-IDF), and Signal C (BM25)
  into a single normalised score in [0.0, 1.0].
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_ranks(scores: list[float]) -> list[int]:
    """
    Convert a list of scores to 1-based ranks (rank 1 = highest score).

    Parameters
    ----------
    scores : list[float]
        Raw signal scores.

    Returns
    -------
    list[int]
        Rank for each position; ties broken by index order (stable).
    """
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank + 1
    return ranks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rrf_fusion(
    kw: list[float],
    tfidf: list[float],
    bm25: list[float],
    k: int = 60,
) -> list[float]:
    """
    Reciprocal Rank Fusion of three scoring signals.

    RRF formula for candidate i:
        score_i = 1/(k + rank_kw_i) + 1/(k + rank_tfidf_i) + 1/(k + rank_bm25_i)

    The fused scores are then min-max normalised to [0.0, 1.0].

    Parameters
    ----------
    kw : list[float]
        Signal A scores (keyword depth), one per candidate.
    tfidf : list[float]
        Signal B scores (TF-IDF cosine similarity), one per candidate.
    bm25 : list[float]
        Signal C scores (BM25 relevance), one per candidate.
    k : int, optional
        RRF smoothing constant. Default 60 (standard literature value).

    Returns
    -------
    list[float]
        Fused scores normalised to [0.0, 1.0], same order as inputs.

    Raises
    ------
    ValueError
        If the three input lists have different lengths.
    """
    n = len(kw)
    if len(tfidf) != n or len(bm25) != n:
        raise ValueError(
            f"Input lists must have equal length. "
            f"Got kw={n}, tfidf={len(tfidf)}, bm25={len(bm25)}."
        )

    if n == 0:
        return []

    kw_ranks    = _get_ranks(kw)
    tfidf_ranks = _get_ranks(tfidf)
    bm25_ranks  = _get_ranks(bm25)

    rrf_scores = [
        1.0 / (k + kw_ranks[i])
        + 1.0 / (k + tfidf_ranks[i])
        + 1.0 / (k + bm25_ranks[i])
        for i in range(n)
    ]

    # Normalise to [0, 1]
    min_r = min(rrf_scores)
    max_r = max(rrf_scores)

    if max_r > min_r:
        return [(s - min_r) / (max_r - min_r) for s in rrf_scores]

    # Degenerate: all candidates tied (single candidate or identical scores)
    return [0.5] * n
