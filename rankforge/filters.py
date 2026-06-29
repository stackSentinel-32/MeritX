"""
rankforge/filters.py
--------------------
Hard-discard filters for the candidate ranking pipeline.
Filters run cheapest-first; execution stops at the first discard.

Honeypot detection runs independently — it flags but does NOT discard alone.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from rankforge.constants import (
    ALL_JD_SKILLS,
    NEGATIVE_CV_SPEECH,
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    TIER2_RECSYS,
)

# ---------------------------------------------------------------------------
# Module-level filter hit counters
# ---------------------------------------------------------------------------

FILTER_COUNTS: dict[str, int] = {
    "witch_only_no_ml": 0,
    "zero_ai_signal": 0,
    "wrong_domain": 0,
    "cv_speech_only_no_nlp": 0,
    "insufficient_ml_yoe": 0,
    "zero_ml_yoe_wrong_title": 0,
    "honeypot": 0,
}

# ---------------------------------------------------------------------------
# Constants used inside filters
# ---------------------------------------------------------------------------

_WRONG_DOMAIN_TITLES: frozenset[str] = frozenset({
    "civil engineer",
    "accountant",
    "hr manager",
    "mechanical engineer",
    "marketing manager",
    "content writer",
    "operations manager",
    "brand designer",
    "graphic designer",
    "project manager",
})

_ZERO_ML_WRONG_TITLES: frozenset[str] = frozenset({
    "civil engineer",
    "hr manager",
    "accountant",
    "content writer",
    "mechanical engineer",
    "marketing manager",
    "operations manager",
})

# Honeypot flags that are individually sufficient to trigger is_honeypot=True
_HARD_HONEYPOT_FLAGS: frozenset[str] = frozenset({
    "salary_inverted",
    "tenure_mismatch",
    "future_start_date",
    "education_timeline",
})

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    should_discard: bool
    discard_reason: str
    is_honeypot: bool
    honeypot_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _title_contains_any(title: str, bad_titles: frozenset[str]) -> bool:
    """True if the lowercased title contains any of the bad-title strings."""
    title_lc = title.lower().strip()
    return any(bad in title_lc for bad in bad_titles)


def _detect_honeypot(flags: list[str]) -> tuple[bool, list[str]]:
    """
    Evaluate honeypot flags from the feature dict.

    Rules
    -----
    - Any hard flag (salary_inverted / tenure_mismatch /
      future_start_date / education_timeline) → honeypot.
    - Two or more flags of any kind → honeypot.
    - Single "instant_expert" alone → NOT honeypot.
    - Single "yoe_sum_mismatch" alone → NOT honeypot.
    """
    reasons: list[str] = []

    for f in flags:
        if f in _HARD_HONEYPOT_FLAGS:
            reasons.append(f)

    # Multiple soft signals also trigger
    if len(flags) >= 2 and not reasons:
        reasons = list(flags)

    is_hp = bool(reasons)
    return is_hp, reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_filters(features: dict) -> FilterResult:  # noqa: C901
    """
    Apply hard-discard filters to an already-extracted feature dict.

    Filters are evaluated in order; the first discard wins.
    Honeypot detection always runs regardless of discard outcome.

    Parameters
    ----------
    features : dict
        Output of ``rankforge.parser.extract_features``.

    Returns
    -------
    FilterResult
    """
    # ---- Unpack commonly-used fields ------------------------------------
    years_exp: float        = features.get("years_exp", 0.0)
    ml_yoe: float           = features.get("ml_yoe", 0.0)
    skill_set: set[str]     = features.get("skill_set", set())
    description_text: str   = features.get("description_text", "")
    current_title: str      = (features.get("current_title") or "").lower().strip()
    career_roles: list[dict] = features.get("career_roles", [])
    honeypot_flags: list[str] = features.get("honeypot_flags", [])

    # ---- Honeypot (runs always, independently of discard) ---------------
    is_honeypot, honeypot_reasons = _detect_honeypot(honeypot_flags)
    if is_honeypot:
        FILTER_COUNTS["honeypot"] += 1

    # ---- Filter 1 — WITCH-only + no redemption --------------------------
    if career_roles:
        all_witch = all(r.get("company_type") == "witch" for r in career_roles)
        if all_witch and years_exp > 4 and ml_yoe < 1.0:
            FILTER_COUNTS["witch_only_no_ml"] += 1
            return FilterResult(
                should_discard=True,
                discard_reason="witch_only_no_ml",
                is_honeypot=is_honeypot,
                honeypot_reasons=honeypot_reasons,
            )

    # ---- Filter 2 — Zero AI signal --------------------------------------
    ai_skill_hits = len(skill_set & ALL_JD_SKILLS)
    desc_has_ai = any(kw in description_text for kw in ALL_JD_SKILLS)
    if ai_skill_hits == 0 and not desc_has_ai:
        FILTER_COUNTS["zero_ai_signal"] += 1
        return FilterResult(
            should_discard=True,
            discard_reason="zero_ai_signal",
            is_honeypot=is_honeypot,
            honeypot_reasons=honeypot_reasons,
        )

    # ---- Filter 3 — Wrong domain with no ML career evidence -------------
    if (
        _title_contains_any(current_title, _WRONG_DOMAIN_TITLES)
        and ai_skill_hits < 2
        and ml_yoe < 0.5
    ):
        FILTER_COUNTS["wrong_domain"] += 1
        return FilterResult(
            should_discard=True,
            discard_reason="wrong_domain",
            is_honeypot=is_honeypot,
            honeypot_reasons=honeypot_reasons,
        )

    # ---- Filter 4 — CV/speech dominant ----------------------------------
    neg_hits  = len(skill_set & NEGATIVE_CV_SPEECH)
    tier1_hits = len(skill_set & TIER1_RETRIEVAL)
    tier2_hits = len(skill_set & (TIER2_NLP_IR | TIER2_RECSYS))
    if neg_hits > 4 and tier1_hits == 0 and tier2_hits == 0:
        FILTER_COUNTS["cv_speech_only_no_nlp"] += 1
        return FilterResult(
            should_discard=True,
            discard_reason="cv_speech_only_no_nlp",
            is_honeypot=is_honeypot,
            honeypot_reasons=honeypot_reasons,
        )

    # ---- Filter 5 — ML YOE floor ----------------------------------------
    if years_exp < 3.5 and ml_yoe < 0.5:
        FILTER_COUNTS["insufficient_ml_yoe"] += 1
        return FilterResult(
            should_discard=True,
            discard_reason="insufficient_ml_yoe",
            is_honeypot=is_honeypot,
            honeypot_reasons=honeypot_reasons,
        )

    if ml_yoe == 0.0 and years_exp > 3.0 and _title_contains_any(current_title, _ZERO_ML_WRONG_TITLES):
        FILTER_COUNTS["zero_ml_yoe_wrong_title"] += 1
        return FilterResult(
            should_discard=True,
            discard_reason="zero_ml_yoe_wrong_title",
            is_honeypot=is_honeypot,
            honeypot_reasons=honeypot_reasons,
        )

    # ---- Passed all filters ---------------------------------------------
    return FilterResult(
        should_discard=False,
        discard_reason="",
        is_honeypot=is_honeypot,
        honeypot_reasons=honeypot_reasons,
    )


# ---------------------------------------------------------------------------
# Debug summary (triggered when module is run with --debug)
# ---------------------------------------------------------------------------

def print_filter_summary() -> None:
    """Print a formatted summary of FILTER_COUNTS to stdout."""
    print("\n=== Filter Hit Summary ===")
    total = sum(FILTER_COUNTS.values())
    for reason, count in FILTER_COUNTS.items():
        pct = (count / total * 100) if total else 0.0
        print(f"  {reason:<30s} {count:>6d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<30s} {total:>6d}")
    print("=" * 42)


if "--debug" in sys.argv:
    import atexit
    atexit.register(print_filter_summary)
