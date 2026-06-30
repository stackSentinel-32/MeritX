"""
rankforge/output.py
-------------------
Human-readable reasoning generation and CSV output writer.

generate_reasoning(...) -> str
  Produces a single-line reasoning string for one ranked candidate.

write_csv(ranked_candidates, output_path) -> None
  Writes the final top-100 CSV with assertions before any I/O.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from rankforge.constants import (
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    TIER2_RECSYS,
    TIER_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GEO_LABEL: dict[str, str] = {
    "preferred_city": "India (preferred city)",
    "india_other":    "India",
    "international":  "international",
}

_TYPE_DISPLAY: dict[str, str] = {
    "product_startup": "startup",
    "startup":         "startup",
    "product":         "product",
    "big_tech":        "big tech",
    "witch":           "outsourcing",
    "consulting":      "consulting",
    "other":           "other",
}


def _tier1_sort_key(skill: str) -> float:
    """Sort tier1 skills by descending TIER_WEIGHTS value (most relevant first)."""
    return -TIER_WEIGHTS.get("tier1", 4.0)   # all tier1 have the same weight = 4.0


def _title_case(s: str) -> str:
    """Title-case a string, preserving short prepositions lower-cased."""
    SMALL = {"a", "an", "and", "at", "but", "by", "for", "in", "of",
             "on", "or", "so", "the", "to", "up", "yet"}
    words = s.split()
    result = []
    for i, w in enumerate(words):
        result.append(w if (i != 0 and w.lower() in SMALL) else w.capitalize())
    return " ".join(result)


# ---------------------------------------------------------------------------
# Public: generate_reasoning
# ---------------------------------------------------------------------------

def generate_reasoning(
    features: dict,
    kw: float,
    tfidf: float,
    bm25: float,
    fused: float,
    availability: float,
    geo: float,
    final: float,
    is_honeypot: bool,
) -> str:
    """
    Build a single-line reasoning string for one ranked candidate.

    Parameters mirror the pipeline outputs for that candidate.

    Returns
    -------
    str
        Human-readable reasoning; ends with " [HONEYPOT]" when flagged.
    """
    # ------------------------------------------------------------------
    # Current title and company
    # ------------------------------------------------------------------
    title_raw = features.get("current_title") or "unknown title"
    title     = _title_case(title_raw.strip())

    company_raw = features.get("current_company") or "unknown company"
    company     = _title_case(company_raw.strip())

    yoe: float = float(features.get("years_exp", 0))

    # ------------------------------------------------------------------
    # Company type from most recent role (first in list is most recent
    # per pipeline convention — fall back to "other")
    # ------------------------------------------------------------------
    career_roles: list[dict] = features.get("career_roles") or []
    if career_roles:
        ctype_raw = career_roles[0].get("company_type", "other")
    else:
        ctype_raw = "other"
    type_display = _TYPE_DISPLAY.get(ctype_raw, "other")

    # ------------------------------------------------------------------
    # TIER1 skills — listed, sorted by relevance, capped at 4
    # ------------------------------------------------------------------
    skill_set: set[str] = features.get("skill_set", set())
    tier1_hits = sorted(
        [s for s in skill_set if s in TIER1_RETRIEVAL],
        key=lambda s: -TIER_WEIGHTS.get("tier1", 4.0),  # all equal weight; alphabetical fallback
    )
    # Secondary sort: alphabetical for determinism within same weight
    tier1_hits = sorted(
        [s for s in skill_set if s in TIER1_RETRIEVAL],
    )[:4]
    tier1_count = len(tier1_hits)
    # NEVER show empty parentheses
    tier1_str = f" ({', '.join(tier1_hits)})" if tier1_hits else ""

    # ------------------------------------------------------------------
    # TIER2 — only shown when tier1 < 3 for context
    # ------------------------------------------------------------------
    tier2_hits = sorted(
        [s for s in skill_set if s in TIER2_NLP_IR | TIER2_RECSYS]
    )[:3]
    tier2_str = f"NLP/IR: {', '.join(tier2_hits)}; " if tier2_hits else ""

    # ------------------------------------------------------------------
    # Assessment note
    # ------------------------------------------------------------------
    assessment_scores: dict = features.get("assessment_scores") or {}
    if assessment_scores:
        best = max(assessment_scores.values())
        assess_note = f"assessed {best:.0%}; "
    else:
        assess_note = ""

    # ------------------------------------------------------------------
    # ML YOE and notice
    # ------------------------------------------------------------------
    ml_yoe: float  = float(features.get("ml_yoe", 0))
    notice_days: int = int(features.get("notice_days", 0) or 0)

    # ------------------------------------------------------------------
    # Geo label
    # ------------------------------------------------------------------
    geo_bucket: str = features.get("geo_bucket", "international")
    geo_str = _GEO_LABEL.get(geo_bucket, "unknown")

    # ------------------------------------------------------------------
    # GitHub signal
    # ------------------------------------------------------------------
    gs = features.get("github_score", -1)
    if gs is None or gs == -1:
        github_str = "GitHub not linked"
    elif gs > 0:
        github_str = "GitHub active"
    else:
        github_str = "GitHub inactive"

    # ------------------------------------------------------------------
    # Honeypot suffix
    # ------------------------------------------------------------------
    hp_note = " [HONEYPOT]" if is_honeypot else ""

    # ------------------------------------------------------------------
    # Assemble reasoning string
    # ------------------------------------------------------------------
    reasoning = (
        f"{title} with {yoe:.1f} yrs at {company} ({type_display}); "
        f"{tier1_count} retrieval skill{'s' if tier1_count != 1 else ''}{tier1_str}; "
        f"{tier2_str}{assess_note}"
        f"ML YOE {ml_yoe:.1f} yrs; notice {notice_days}d; "
        f"{geo_str}; {github_str}; score {final:.3f}.{hp_note}"
    )

    return reasoning


# ---------------------------------------------------------------------------
# Public: write_csv
# ---------------------------------------------------------------------------

def write_csv(ranked_candidates: list[dict], output_path: str | Path) -> None:
    """
    Assert integrity of the top-100 ranked list, then write it as CSV.

    Parameters
    ----------
    ranked_candidates : list[dict]
        Each dict must have keys:
          candidate_id, rank, score, reasoning
    output_path : str | Path
        Destination file path. Parent directories will be created if needed.

    Raises
    ------
    AssertionError
        If any integrity check fails. All checks run before any I/O.
    """
    rows = ranked_candidates

    # ------------------------------------------------------------------
    # Assertions — ALL checks before any file I/O
    # ------------------------------------------------------------------

    # 1. Exactly 100 rows
    assert len(rows) == 100, (
        f"Expected 100 ranked candidates, got {len(rows)}"
    )

    # 2. No duplicate candidate_ids
    ids = [r["candidate_id"] for r in rows]
    assert len(ids) == len(set(ids)), (
        f"Duplicate candidate_ids found: "
        f"{[x for x in ids if ids.count(x) > 1]}"
    )

    # 3. Scores strictly decreasing
    scores = [float(r["score"]) for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], (
            f"Scores not strictly decreasing at positions {i} and {i+1}: "
            f"{scores[i]:.6f} vs {scores[i+1]:.6f}"
        )

    # 4. Ranks == 1 to 100 in order
    ranks = [int(r["rank"]) for r in rows]
    assert ranks == list(range(1, 101)), (
        f"Ranks must be exactly 1–100 in order. Got: {ranks[:5]}..."
    )

    # 5. No empty reasoning strings
    assert all(r["reasoning"] for r in rows), (
        "One or more rows have empty reasoning strings."
    )

    # 6. At most 8 honeypot flags in top-100
    hp_count = sum(1 for r in rows if "[HONEYPOT]" in r["reasoning"])
    assert hp_count <= 8, (
        f"Too many honeypot-flagged candidates in top-100: {hp_count} (max 8)"
    )

    # 7. No empty parentheses in any reasoning
    empty_paren_pattern = re.compile(r"\(\s*\)|,\s*\)")
    for i, r in enumerate(rows):
        assert not empty_paren_pattern.search(r["reasoning"]), (
            f"Empty or malformed parentheses in reasoning at rank {i+1}: "
            f"{r['reasoning'][:80]!r}"
        )

    # 8. Score in CSV must match score embedded in reasoning ±0.001
    score_in_text = re.compile(r"score ([0-9]+\.[0-9]+)")
    for i, r in enumerate(rows):
        m = score_in_text.search(r["reasoning"])
        assert m, (
            f"No 'score X.XXX' found in reasoning at rank {i+1}: "
            f"{r['reasoning'][:80]!r}"
        )
        text_score = float(m.group(1))
        assert abs(text_score - float(r["score"])) <= 0.001, (
            f"Score mismatch at rank {i+1}: "
            f"CSV={float(r['score']):.6f}, reasoning={text_score:.6f}"
        )

    # ------------------------------------------------------------------
    # Write CSV — only reached if all assertions pass
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                round(float(r["score"]), 6),
                r["reasoning"],
            ])
