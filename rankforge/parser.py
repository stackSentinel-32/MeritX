"""
rankforge/parser.py
-------------------
Takes one raw JSON dict (as loaded from the platform dump) and returns a
normalised feature dict ready for downstream scoring modules.

No ML libraries. No scoring. Pure extraction and normalisation only.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from dateutil import parser as dateutil_parser

from rankforge.constants import (
    PREFERRED_CITIES,
    PRODUCT_STARTUPS_INDIA,
    TIER1_RETRIEVAL,
    TIER2_NLP_IR,
    WITCH_COMPANIES,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ML_SIGNAL_KEYWORDS = TIER1_RETRIEVAL | TIER2_NLP_IR


def _parse_date(value: Any) -> date | None:
    """Try ISO-8601 first, then dateutil, then return None."""
    if not value:
        return None
    s = str(value).strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        pass
    try:
        return dateutil_parser.parse(s).date()
    except Exception:
        return None


def _months_between(start: date, end: date) -> float:
    """Fractional months between two dates (always >= 0)."""
    delta_days = (end - start).days
    return max(delta_days / 30.4375, 0.0)


def _classify_company(company_lower: str, industry_lower: str, size: str) -> str:
    """Return company_type string based on classification rules (ordered)."""
    if company_lower in WITCH_COMPANIES:
        return "witch"
    if company_lower in PRODUCT_STARTUPS_INDIA:
        return "product_startup"
    if industry_lower == "software" and size in {"11-50", "51-200"}:
        return "startup"
    if industry_lower == "software" and size in {"201-500", "501-1000"}:
        return "product"
    if industry_lower in {"it services", "consulting"}:
        return "consulting"
    return "other"


def _has_ml_signal(text: str) -> bool:
    """True if any TIER1 or TIER2_NLP_IR keyword appears in the text."""
    if not text:
        return False
    for kw in _ML_SIGNAL_KEYWORDS:
        if kw in text:
            return True
    return False


def _role_relevance_score(description: str) -> float:
    """
    Simple 0-1 AI relevance score for a single role description.
    Counts hits from TIER1 | TIER2_NLP_IR, normalised by 5 as per spec.
    Clamped to [0, 1].
    """
    if not description:
        return 0.0
    hits = sum(1 for kw in _ML_SIGNAL_KEYWORDS if kw in description)
    return min(hits / 5.0, 1.0)


def _compute_trajectory(career_roles: list[dict]) -> float:
    """
    trajectory = avg(last 2 roles) - avg(first 2 roles).
    Returns value in [-1, 1].
    Roles are expected to be in chronological order (oldest first).
    """
    scores = [r.get("_relevance_score", 0.0) for r in career_roles]
    if not scores:
        return 0.0

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    first_two = scores[:2]
    last_two = scores[-2:] if len(scores) >= 2 else scores
    traj = _avg(last_two) - _avg(first_two)
    return max(-1.0, min(1.0, traj))


def _normalize_assessment(value: Any, max_val: float = 100.0) -> float:
    """Normalize a numeric assessment value to [0, 1]."""
    try:
        v = float(value)
        return max(0.0, min(v / max_val, 1.0))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _geo_bucket(location: str, country: str) -> str:
    """Classify location into geo bucket."""
    loc_lower = location.lower() if location else ""
    # "gurgaon" and "gurugram" are both treated as preferred (both in constant)
    for city in PREFERRED_CITIES:
        if city in loc_lower:
            return "preferred_city"
    country_lower = (country or "").lower()
    if country_lower in {"india", "in"}:
        return "india_other"
    return "international"


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def extract_features(raw: dict) -> dict:  # noqa: C901  (complexity expected here)
    """
    Parse one raw candidate JSON dict and return a normalised feature dict.

    Parameters
    ----------
    raw : dict
        A single candidate record as loaded from the platform dump.

    Returns
    -------
    dict
        Flat feature dict ready for downstream scoring modules.
    """
    today = date.today()
    honeypot_flags: list[str] = []

    # ------------------------------------------------------------------
    # Support both flat schema (synthetic) and nested schema (real data)
    # Real data: profile fields are under raw["profile"]
    # ------------------------------------------------------------------
    profile: dict = raw.get("profile") or {}

    candidate_id = raw.get("candidate_id") or raw.get("id")
    name = (profile.get("anonymized_name") or raw.get("name") or "")
    years_exp: float = float(
        profile.get("years_of_experience")
        or raw.get("years_of_experience")
        or raw.get("years_exp")
        or 0
    )
    current_title: str = (
        profile.get("current_title") or raw.get("current_title") or ""
    ).lower().strip()
    current_company: str = (
        profile.get("current_company") or raw.get("current_company") or ""
    ).lower().strip()
    current_industry: str = (
        profile.get("current_industry") or raw.get("current_industry") or ""
    ).lower().strip()
    company_size: str = (
        profile.get("current_company_size") or raw.get("company_size") or ""
    )

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------
    skills_raw: list[dict] = raw.get("skills") or []
    skill_set: set[str] = set()
    skill_map: dict[str, dict] = {}

    # Collect redrob_signals assessment scores for skill augmentation
    # Real data key: "skill_assessment_scores"; synthetic key: "skill_assessments"
    redrob_signals: dict = raw.get("redrob_signals") or {}
    signal_assessments: dict = (
        redrob_signals.get("skill_assessment_scores")
        or redrob_signals.get("skill_assessments")
        or {}
    )

    for skill_item in skills_raw:
        sname_raw: str = skill_item.get("name") or skill_item.get("skill") or ""
        sname = sname_raw.lower().strip()
        if not sname:
            continue
        skill_set.add(sname)

        duration_months: int = int(skill_item.get("duration_months") or 0)
        proficiency: str = (skill_item.get("proficiency") or "").lower()
        endorsements: int = int(skill_item.get("endorsements") or 0)

        # Pull assessment_score from redrob_signals if available
        assessment_raw = signal_assessments.get(sname_raw) or signal_assessments.get(sname)
        assessment_score: float | None = None
        if assessment_raw is not None:
            assessment_score = _normalize_assessment(assessment_raw)

        skill_map[sname] = {
            "proficiency": proficiency,
            "duration_months": duration_months,
            "endorsements": endorsements,
            "assessment_score": assessment_score,
        }

        # Honeypot: instant expert — advanced proficiency but <= 1 month experience
        if proficiency in {"advanced", "expert"} and duration_months <= 1:
            honeypot_flags.append("instant_expert")

    # ------------------------------------------------------------------
    # Assessment scores from redrob_signals (normalized 0-1)
    # ------------------------------------------------------------------
    assessment_scores: dict[str, float] = {}
    for skill_key, score_val in signal_assessments.items():
        assessment_scores[skill_key.lower()] = _normalize_assessment(score_val)

    # ------------------------------------------------------------------
    # Education timeline check
    # ------------------------------------------------------------------
    education_list: list[dict] = raw.get("education") or []
    for edu in education_list:
        start_yr = edu.get("start_year")
        end_yr = edu.get("end_year")
        try:
            if start_yr is not None and end_yr is not None:
                if int(end_yr) < int(start_yr):
                    if "education_timeline" not in honeypot_flags:
                        honeypot_flags.append("education_timeline")
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Career roles
    # ------------------------------------------------------------------
    roles_raw: list[dict] = raw.get("career_history") or raw.get("experience") or []
    career_roles: list[dict] = []
    description_parts: list[str] = []
    total_actual_months: float = 0.0
    ml_months: float = 0.0

    for role_raw in roles_raw:
        company_raw: str = role_raw.get("company") or ""
        company_lc: str = company_raw.lower().strip()
        title_raw: str = role_raw.get("title") or role_raw.get("job_title") or ""
        industry_raw: str = (role_raw.get("industry") or "").lower().strip()
        size_raw: str = role_raw.get("company_size") or ""
        desc_raw: str = role_raw.get("description") or ""
        desc_lc: str = desc_raw.lower().strip()
        is_current: bool = bool(role_raw.get("is_current") or role_raw.get("current"))

        start_date = _parse_date(role_raw.get("start_date"))
        end_date_parsed = _parse_date(role_raw.get("end_date"))

        # For current roles use today as end_date
        effective_end: date | None = today if is_current else end_date_parsed

        # Honeypot: future start date for non-current role
        if start_date and not is_current and start_date > today:
            if "future_start_date" not in honeypot_flags:
                honeypot_flags.append("future_start_date")

        # Compute actual months
        actual_months: float = 0.0
        if start_date and effective_end:
            actual_months = _months_between(start_date, effective_end)

        # Claimed months (if platform provides it)
        claimed_months: float = float(role_raw.get("duration_months") or 0)

        # Tenure mismatch check
        if claimed_months > 0 and abs(actual_months - claimed_months) > 6:
            if "tenure_mismatch" not in honeypot_flags:
                honeypot_flags.append("tenure_mismatch")

        company_type = _classify_company(company_lc, industry_raw, size_raw)
        ml_signal = _has_ml_signal(desc_lc)
        relevance_score = _role_relevance_score(desc_lc)

        if desc_lc:
            description_parts.append(desc_lc)

        total_actual_months += actual_months
        if ml_signal:
            ml_months += actual_months

        career_roles.append({
            "company": company_lc,
            "title": title_raw,
            "industry": industry_raw,
            "company_size": size_raw,
            "start_date": start_date,
            "end_date": effective_end,
            "actual_months": actual_months,
            "claimed_months": claimed_months,
            "description": desc_lc,
            "company_type": company_type,
            "is_current": is_current,
            "has_ml_signal": ml_signal,
            # Private field used for trajectory; stripped from public API below
            "_relevance_score": relevance_score,
        })

    # Compute trajectory before stripping private key
    career_trajectory: float = _compute_trajectory(career_roles)

    # Strip private keys
    for role in career_roles:
        role.pop("_relevance_score", None)

    # ------------------------------------------------------------------
    # Derived YOE fields
    # ------------------------------------------------------------------
    computed_yoe: float = total_actual_months / 12.0
    ml_yoe: float = ml_months / 12.0
    yoe_delta: float = abs(computed_yoe - years_exp)

    if yoe_delta > 3:
        if "yoe_sum_mismatch" not in honeypot_flags:
            honeypot_flags.append("yoe_sum_mismatch")

    # ------------------------------------------------------------------
    # Description text (all career descriptions joined)
    # ------------------------------------------------------------------
    description_text: str = " ".join(description_parts)

    # ------------------------------------------------------------------
    # Platform behavioural signals
    # Supports both flat schema (synthetic) and nested redrob_signals (real)
    # ------------------------------------------------------------------
    def _sig(key_real: str, key_flat: str, sentinel: float = -1.0) -> float:
        """Read from redrob_signals first, then flat raw, else sentinel."""
        v = redrob_signals.get(key_real) if redrob_signals else None
        if v is None:
            v = raw.get(key_flat)
        if v is None:
            return sentinel
        try:
            return float(v)
        except (TypeError, ValueError):
            return sentinel

    last_active_raw = _parse_date(
        redrob_signals.get("last_active_date") or raw.get("last_active_date")
    )
    last_active_days: int = (today - last_active_raw).days if last_active_raw else -1

    notice_days: int = int(
        redrob_signals.get("notice_period_days")
        or raw.get("notice_period_days")
        or 0
    )
    response_rate: float  = _sig("recruiter_response_rate",  "recruiter_response_rate")
    avg_response_hrs: float = _sig("avg_response_time_hours", "avg_response_time_hours")
    github_score: float   = _sig("github_activity_score",    "github_activity_score")
    interview_rate: float = _sig("interview_completion_rate", "interview_completion_rate")
    offer_acceptance: float = _sig("offer_acceptance_rate",  "offer_acceptance_rate")

    # open_to_work: real key is "open_to_work_flag"; synthetic key is "open_to_work"
    open_to_work: bool = bool(
        redrob_signals.get("open_to_work_flag")
        or raw.get("open_to_work")
        or False
    )
    willing_relocate: bool = bool(
        redrob_signals.get("willing_to_relocate")
        or raw.get("willing_to_relocate")
        or False
    )
    work_mode: str = (
        redrob_signals.get("preferred_work_mode")
        or raw.get("work_mode")
        or ""
    ).lower().strip()

    # profile_completeness_score: real is in redrob_signals; synthetic is flat
    profile_raw = (
        redrob_signals.get("profile_completeness_score")
        or raw.get("profile_completeness_score")
    )
    profile_complete: float = 0.0
    if profile_raw is not None:
        try:
            profile_complete = float(profile_raw) / 100.0
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Location / Geo
    # Real data: location and country are inside profile dict
    # ------------------------------------------------------------------
    country: str = (
        profile.get("country") or raw.get("country") or ""
    ).lower().strip()
    location: str = (
        profile.get("location") or raw.get("location") or raw.get("city") or ""
    ).lower().strip()
    geo_bucket: str = _geo_bucket(location, country)

    # ------------------------------------------------------------------
    # Salary — real data uses expected_salary_range_inr_lpa dict
    # ------------------------------------------------------------------
    salary_range: dict = redrob_signals.get("expected_salary_range_inr_lpa") or {}
    salary_min: float = float(
        salary_range.get("min") or raw.get("expected_salary_min") or 0
    )
    salary_max: float = float(
        salary_range.get("max") or raw.get("expected_salary_max") or 0
    )
    salary_inverted: bool = salary_min > salary_max and salary_min > 0 and salary_max > 0

    if salary_inverted:
        if "salary_inverted" not in honeypot_flags:
            honeypot_flags.append("salary_inverted")

    # ------------------------------------------------------------------
    # Verification flags
    # ------------------------------------------------------------------
    verified_email: bool = bool(
        redrob_signals.get("verified_email") or raw.get("verified_email") or False
    )
    verified_phone: bool = bool(
        redrob_signals.get("verified_phone") or raw.get("verified_phone") or False
    )

    # ------------------------------------------------------------------
    # Assemble and return
    # ------------------------------------------------------------------
    return {
        # Identity
        "candidate_id": candidate_id,
        "name": name,
        # Experience
        "years_exp": years_exp,
        "computed_yoe": computed_yoe,
        "ml_yoe": ml_yoe,
        "yoe_delta": yoe_delta,
        # Current role
        "current_title": current_title,
        "current_company": current_company,
        "current_industry": current_industry,
        "company_size": company_size,
        # Skills
        "skill_set": skill_set,
        "skill_map": skill_map,
        "assessment_scores": assessment_scores,
        # Text
        "description_text": description_text,
        # Career
        "career_roles": career_roles,
        "career_trajectory": career_trajectory,
        # Availability / engagement
        "last_active_days": last_active_days,
        "notice_days": notice_days,
        "response_rate": response_rate,
        "avg_response_hrs": avg_response_hrs,
        "github_score": github_score,
        "interview_rate": interview_rate,
        "offer_acceptance": offer_acceptance,
        "open_to_work": open_to_work,
        "willing_relocate": willing_relocate,
        "work_mode": work_mode,
        "profile_complete": profile_complete,
        # Location
        "country": country,
        "location": location,
        "geo_bucket": geo_bucket,
        # Salary
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_inverted": salary_inverted,
        # Verification
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        # Quality signals
        "honeypot_flags": honeypot_flags,
    }
