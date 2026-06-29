"""
rankforge/signals.py
--------------------
Batch computation of availability and geo signals.

compute_availability_batch(features_list) -> tuple[np.ndarray, np.ndarray]
  Returns (availability_array, geo_bonus_array), both shape (N,).

All computation is NumPy-vectorised — no per-record Python loops
inside the signal logic itself.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_availability_batch(
    features_list: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute availability and geo-bonus signals for a batch of candidates.

    Parameters
    ----------
    features_list : list[dict]
        List of feature dicts from ``rankforge.parser.extract_features``.

    Returns
    -------
    availability : np.ndarray, shape (N,)
        Composite availability score in [0.05, 1.0].
    geo_bonus : np.ndarray, shape (N,)
        Additive geo + verification bonus in [0.0, 0.09].
    """
    if not features_list:
        empty = np.array([], dtype=float)
        return empty, empty

    # ------------------------------------------------------------------
    # Extract raw fields into numpy arrays
    # ------------------------------------------------------------------
    last_active  = np.array([f["last_active_days"]   for f in features_list], dtype=float)
    notice       = np.array([f["notice_days"]         for f in features_list], dtype=float)
    response     = np.array([f["response_rate"]       for f in features_list], dtype=float)
    avg_hrs      = np.array([f["avg_response_hrs"]    for f in features_list], dtype=float)
    completion   = np.array([f["interview_rate"]      for f in features_list], dtype=float)
    github       = np.array([f["github_score"]        for f in features_list], dtype=float)
    openwork     = np.array([f["open_to_work"]        for f in features_list], dtype=float)
    offer_acc    = np.array([f["offer_acceptance"]    for f in features_list], dtype=float)
    completeness = np.array([f["profile_complete"]    for f in features_list], dtype=float)

    # ------------------------------------------------------------------
    # Recency  (exponential decay — unknown last_active treated as 0.80)
    # ------------------------------------------------------------------
    recency = np.where(
        last_active < 0,
        0.80,
        np.clip(np.exp(-last_active / 180.0), 0.10, 1.0),
    )

    # ------------------------------------------------------------------
    # Notice period multiplier
    # notice=0 → BONUS (immediately available)
    # ------------------------------------------------------------------
    notice_m = np.where(notice <= 0,   1.05,
               np.where(notice <= 15,  1.02,
               np.where(notice <= 30,  1.00,
               np.where(notice <= 60,  0.85,
               np.where(notice <= 90,  0.70,
               np.where(notice <= 120, 0.55,
                                       0.40))))))

    # ------------------------------------------------------------------
    # Recruiter response rate  (-1 = unknown → neutral 0.80)
    # ------------------------------------------------------------------
    resp_m = np.where(
        response == -1,
        0.80,
        np.clip(response, 0.50, 1.0),
    )
    # Slow responder penalty (>120 hrs and not missing)
    slow_penalty = np.where((avg_hrs > 120) & (avg_hrs != -1), 0.90, 1.0)
    resp_m = resp_m * slow_penalty

    # ------------------------------------------------------------------
    # Interview completion rate  (-1 = unknown → neutral 0.80)
    # ------------------------------------------------------------------
    comp_m = np.where(completion == -1,  0.80,
             np.where(completion >= 0.70, 1.00,
             np.where(completion >= 0.50, 0.85,
             np.where(completion >= 0.30, 0.65,
                                          0.50))))

    # ------------------------------------------------------------------
    # GitHub activity score  (-1 = not linked → neutral 0.72)
    # ------------------------------------------------------------------
    gh_m = np.where(
        github == -1,
        0.72,
        np.clip(0.72 + github / 100.0, 0.72, 1.0),
    )

    # ------------------------------------------------------------------
    # Open-to-work flag
    # ------------------------------------------------------------------
    ow_m = np.where(openwork == 1, 1.0, 0.88)

    # ------------------------------------------------------------------
    # Offer acceptance rate  (-1 = unknown → neutral 0.95)
    # ------------------------------------------------------------------
    oar_m = np.where(offer_acc == -1,   0.95,
            np.where(offer_acc >= 0.70, 1.05,
            np.where(offer_acc >= 0.40, 1.00,
            np.where(offer_acc >= 0.20, 0.90,
                                        0.80))))

    # ------------------------------------------------------------------
    # Profile completeness penalty (incomplete profiles are risky)
    # ------------------------------------------------------------------
    comp_pen = np.where(completeness < 0.40, 0.5 + completeness * 0.5, 1.0)

    # ------------------------------------------------------------------
    # Composite availability
    # ------------------------------------------------------------------
    availability = (
        recency * notice_m * resp_m * comp_m
        * gh_m * ow_m * oar_m * comp_pen
    )
    availability = np.clip(availability, 0.05, 1.0)

    # ------------------------------------------------------------------
    # Geo bonus  (additive, not a multiplier)
    # ------------------------------------------------------------------
    geo = np.array([
        0.08 if f["geo_bucket"] == "preferred_city"
        else 0.05 if f["geo_bucket"] == "india_other"
        else 0.02 if f.get("willing_relocate")
        else 0.00
        for f in features_list
    ], dtype=float)

    # Verification bonus
    verified = np.array([
        0.01 if f.get("verified_email") and f.get("verified_phone") else 0.0
        for f in features_list
    ], dtype=float)

    geo_bonus = geo + verified

    return availability, geo_bonus
