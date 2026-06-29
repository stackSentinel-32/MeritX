"""
rankforge/scorer.py
-------------------
Scoring signals for the candidate ranking pipeline.

Signal A: score_keywords(features) -> float [0.0, 1.0]
  Multi-component keyword / skill depth score.

Signal B: score_tfidf_batch(features_list) -> list[float]
  Batch TF-IDF cosine similarity against a curated JD corpus.
  Call ONCE after pre-filtering — do NOT call per-record.
"""

from __future__ import annotations

import math
from datetime import date

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rankforge.constants import (
    ALL_JD_SKILLS,
    JD_TITLES,
    NEGATIVE_CV_SPEECH,
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    TIER2_RECSYS,
    TIER3_LLM,
    TIER3_MLOPS,
)

# ---------------------------------------------------------------------------
# JD reference text (Signal B)
# ---------------------------------------------------------------------------

JD_TEXT = """
faiss pinecone qdrant milvus weaviate faiss pinecone qdrant milvus
sentence-transformers vector search dense retrieval semantic search
sentence-transformers vector search dense retrieval semantic search
embedding retrieval ann approximate nearest neighbor vector database
nlp bm25 elasticsearch information retrieval text ranking
ranking model learning to rank ndcg mrr passage retrieval
recommendation system two-tower bi-encoder cross-encoder
applied machine learning product company python production ml
a b testing experimentation reranking hybrid retrieval rrf
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_POSSIBLE: float = 60.0

_PROFICIENCY_MULT: dict[str, float] = {
    "advanced":     1.0,
    "intermediate": 0.7,
    "beginner":     0.3,
}

_COMPANY_ML_MULT: dict[str, float] = {
    "witch":           0.25,
    "consulting":      0.45,
    "product_startup": 1.00,
    "startup":         0.90,
    "product":         0.85,
}
_COMPANY_ML_MULT_DEFAULT: float = 0.60


def _get_tier_weight(skill_name: str) -> float:
    """Return the tier weight for a skill, or 0.0 if not in any tier."""
    if skill_name in TIER1_RETRIEVAL:
        return 4.0
    if skill_name in TIER2_NLP_IR:
        return 3.0
    if skill_name in TIER2_RECSYS:
        return 3.0
    if skill_name in TIER3_LLM:
        return 2.5
    if skill_name in TIER3_MLOPS:
        return 2.0
    return 0.0


def _duration_multiplier(months: int | float) -> float:
    months = float(months)
    if months >= 36:
        return 1.00
    if months >= 18:
        return 0.85
    if months >= 6:
        return 0.70
    return 0.50


def _assessment_multiplier(assess: float) -> float:
    """Map an assessment score (0-1) to a weight multiplier."""
    if assess >= 0.90:
        return 1.50
    if assess >= 0.75:
        return 1.25
    if assess >= 0.60:
        return 1.00
    if assess >= 0.40:
        return 0.70
    return 0.40


def _endorsement_multiplier(endorse: int | float) -> float:
    endorse = int(endorse)
    if endorse >= 30:
        return 1.15
    if endorse >= 15:
        return 1.08
    if endorse >= 5:
        return 1.03
    return 1.00


def _days_since(end_date: date | None) -> int:
    """Days between end_date and today. Returns large int if None."""
    if end_date is None:
        return 99999
    return max((date.today() - end_date).days, 0)


# ---------------------------------------------------------------------------
# Signal A — score_keywords
# ---------------------------------------------------------------------------

def score_keywords(features: dict) -> float:
    """
    Compute the keyword / skill-depth score for one candidate.

    Parameters
    ----------
    features : dict
        Output of ``rankforge.parser.extract_features``.

    Returns
    -------
    float
        Normalised score in [0.0, 1.0].
    """
    skill_map: dict[str, dict]  = features.get("skill_map", {})
    skill_set: set[str]         = features.get("skill_set", set())
    assessment_scores: dict     = features.get("assessment_scores", {})
    description_text: str       = features.get("description_text", "")
    career_roles: list[dict]    = features.get("career_roles", [])
    years_exp: float            = float(features.get("years_exp", 0))
    ml_yoe: float               = float(features.get("ml_yoe", 0))
    current_title: str          = (features.get("current_title") or "").lower()
    trajectory: float           = float(features.get("career_trajectory", 0.0))

    # ------------------------------------------------------------------
    # Step 1 — Skill depth per skill
    # ------------------------------------------------------------------
    total_skill_points: float = 0.0

    for skill_name, skill_data in skill_map.items():
        tier_w = _get_tier_weight(skill_name)
        if tier_w == 0.0:
            continue  # not a relevant skill

        prof     = _proficiency_mult = _PROFICIENCY_MULT.get(
                       skill_data.get("proficiency", ""), 0.3)
        months   = skill_data.get("duration_months", 0) or 0
        dur_m    = _duration_multiplier(months)

        assess   = skill_data.get("assessment_score")
        assess_m = _assessment_multiplier(assess) if assess not in (None, -1) else 1.0

        endorse  = skill_data.get("endorsements", 0) or 0
        endorse_m = _endorsement_multiplier(endorse)

        skill_points = tier_w * prof * dur_m * assess_m * endorse_m
        total_skill_points += skill_points

    # ------------------------------------------------------------------
    # Step 2 — Global assessment bonus
    # ------------------------------------------------------------------
    if assessment_scores:
        best_assess = max(assessment_scores.values())
        assessment_global_bonus = max(0.0, (best_assess - 0.5) * 8.0)
    else:
        assessment_global_bonus = 0.0

    # ------------------------------------------------------------------
    # Step 3 — Description text bonus (lower trust)
    # ------------------------------------------------------------------
    description_bonus: float = 0.0
    for keyword in ALL_JD_SKILLS:
        if keyword in description_text and keyword not in skill_set:
            description_bonus += _get_tier_weight(keyword) * 0.35
    description_bonus = min(description_bonus, 4.0)

    # ------------------------------------------------------------------
    # Step 4 — ML YOE bonus (quality-adjusted)
    # ------------------------------------------------------------------
    ml_credit: float = 0.0
    for role in career_roles:
        if not role.get("has_ml_signal"):
            continue
        company_type  = role.get("company_type", "other")
        actual_months = float(role.get("actual_months", 0) or 0)
        is_current    = bool(role.get("is_current"))
        end_date      = role.get("end_date")          # may be date or None

        company_ml_mult = _COMPANY_ML_MULT.get(company_type, _COMPANY_ML_MULT_DEFAULT)

        if is_current:
            recency_factor = 1.3
        else:
            days_ago = _days_since(end_date if isinstance(end_date, date) else None)
            recency_factor = 1.1 if days_ago < 365 else 1.0

        ml_credit += (actual_months / 12.0) * company_ml_mult * recency_factor

    ml_yoe_bonus = min(ml_credit / 4.0, 1.0) * 10.0

    # ------------------------------------------------------------------
    # Step 5 — YOE fit score (flat top for 5-9yr range)
    # ------------------------------------------------------------------
    yoe = years_exp
    if 5.0 <= yoe <= 9.0:
        yoe_score = 5.0
    elif yoe < 5.0:
        yoe_score = 5.0 * math.exp(-((yoe - 5.0) ** 2) / 4.0)
    else:  # yoe > 9.0
        yoe_score = 5.0 * math.exp(-((yoe - 9.0) ** 2) / 20.0)

    # ------------------------------------------------------------------
    # Step 6 — Title match bonus
    # ------------------------------------------------------------------
    if any(t in current_title for t in JD_TITLES):
        title_bonus = 4.0
    elif "data scientist" in current_title or "software engineer" in current_title:
        title_bonus = 1.5
    else:
        title_bonus = 0.0

    # ------------------------------------------------------------------
    # Step 7 — Career trajectory bonus
    # ------------------------------------------------------------------
    trajectory_bonus = trajectory * 3.0  # max +3 for strong upward AI trajectory

    # ------------------------------------------------------------------
    # Step 8 — CV / speech penalty
    # ------------------------------------------------------------------
    neg_hits   = len(skill_set & NEGATIVE_CV_SPEECH)
    tier1_hits = len(skill_set & TIER1_RETRIEVAL)
    cv_penalty = neg_hits * 2.0 if (neg_hits > 2 and tier1_hits == 0) else 0.0

    # ------------------------------------------------------------------
    # Step 9 — Total + normalize
    # ------------------------------------------------------------------
    raw = (
        total_skill_points
        + assessment_global_bonus
        + description_bonus
        + ml_yoe_bonus
        + yoe_score
        + title_bonus
        + trajectory_bonus
        - cv_penalty
    )

    return max(0.0, min(1.0, raw / _MAX_POSSIBLE))


# ---------------------------------------------------------------------------
# Signal B — score_tfidf_batch
# ---------------------------------------------------------------------------

def score_tfidf_batch(features_list: list[dict]) -> list[float]:
    """
    Batch TF-IDF cosine similarity of candidate texts against JD_TEXT.

    IMPORTANT: This is a **batch** function. Call it ONCE on the full
    post-filter candidate list. Do NOT call inside a per-record loop.

    Parameters
    ----------
    features_list : list[dict]
        List of feature dicts from ``rankforge.parser.extract_features``.

    Returns
    -------
    list[float]
        Cosine similarity scores in [0.0, 1.0], one per candidate,
        in the same order as ``features_list``.
    """
    if not features_list:
        return []

    # ------------------------------------------------------------------
    # Build enriched candidate texts
    # ------------------------------------------------------------------
    candidate_texts: list[str] = []

    for features in features_list:
        skill_set: set[str] = features.get("skill_set", set())

        tier1 = [s for s in skill_set if s in TIER1_RETRIEVAL]
        tier2 = [s for s in skill_set if s in TIER2_NLP_IR | TIER2_RECSYS]
        tier3 = [s for s in skill_set if s in TIER3_LLM | TIER3_MLOPS]
        title: str = features.get("current_title") or ""
        desc: str  = (features.get("description_text") or "")[:600]

        candidate_text = (
            " ".join(tier1 * 4) + " "   # TIER1 repeated 4× for TF boost
            + " ".join(tier2 * 2) + " " # TIER2 repeated 2×
            + " ".join(tier3) + " "
            + title + " " + title + " " # title 2×
            + desc
        )
        candidate_texts.append(candidate_text)

    # ------------------------------------------------------------------
    # Build corpus = candidates + JD, vectorize, compute similarities
    # ------------------------------------------------------------------
    corpus = candidate_texts + [JD_TEXT]

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    candidate_matrix = tfidf_matrix[:-1]   # all rows except last
    jd_vector        = tfidf_matrix[-1]    # last row = JD

    scores = cosine_similarity(jd_vector, candidate_matrix).flatten()

    return [float(s) for s in scores]
