"""
rank.py — single entry point for the candidate ranking pipeline.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv [--verbose]
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv --workers 4

Phases:
  0  Startup + logging
  1  Parallel parse / filter / keyword score
  2  5K pre-filter before batch scoring
  3  Batch scoring (TF-IDF + BM25 + RRF fusion)
  4  Final scores  (fused × availability + geo)
  5  Pick top-100 with honeypot budget = 8
  6  Write CSV + time check
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import orjson

from rankforge.filters import apply_filters
from rankforge.fusion import rrf_fusion
from rankforge.output import generate_reasoning, write_csv
from rankforge.parser import extract_features
from rankforge.scorer import score_bm25_batch, score_keywords, score_tfidf_batch
from rankforge.signals import compute_availability_batch

# ---------------------------------------------------------------------------
# Module-level worker (must be at module scope for multiprocessing pickling)
# ---------------------------------------------------------------------------

def process_one(line: bytes) -> dict | None:
    """
    Parse one raw JSON line, filter it, and compute Signal A (keyword score).

    Returns None if the candidate is discarded by filters.
    Returns a dict with features, kw score, and honeypot info otherwise.
    """
    try:
        raw = orjson.loads(line)
    except Exception:
        return None

    features = extract_features(raw)
    result = apply_filters(features)

    if result.should_discard:
        return None

    kw_score = score_keywords(features)

    return {
        "features":         features,
        "kw":               kw_score,
        "is_honeypot":      result.is_honeypot,
        "honeypot_reasons": result.honeypot_reasons,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="MeritX candidate ranking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        metavar="PATH",
        help="Path to input .jsonl file (one candidate JSON per line)",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Path to output CSV file (top-100 ranked candidates)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging (default: WARNING only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        metavar="N",
        help="Number of parallel worker processes (default: cpu_count()-1)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # PHASE 0 — Startup
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("rank")

    log.info("MeritX ranking pipeline starting")
    log.info("Input : %s", args.candidates)
    log.info("Output: %s", args.out)
    log.info("Workers: %d", args.workers)

    # ------------------------------------------------------------------
    # PHASE 1 — Parallel parse + filter + keyword score
    # ------------------------------------------------------------------
    log.info("Phase 1: reading candidates …")

    with open(args.candidates, "rb") as fh:
        lines = fh.readlines()

    log.info("Read %d lines in %.2fs", len(lines), time.perf_counter() - t0)

    n_workers = min(args.workers, cpu_count())
    log.info("Spawning pool of %d workers …", n_workers)

    with Pool(n_workers) as pool:
        results = pool.map(process_one, lines, chunksize=500)

    survivors = [r for r in results if r is not None]
    honeypot_ids = {
        r["features"]["candidate_id"]
        for r in survivors
        if r["is_honeypot"]
    }

    log.info(
        "Phase 1 done: %d / %d passed filters | %d honeypots flagged | %.1fs elapsed",
        len(survivors), len(lines), len(honeypot_ids),
        time.perf_counter() - t0,
    )

    if len(survivors) < 100:
        logging.error(
            "Only %d candidates survived filtering — cannot produce top-100. Exiting.",
            len(survivors),
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # PHASE 2 — 5K pre-filter before batch scoring
    # ------------------------------------------------------------------
    log.info("Phase 2: pre-filtering to ≤5200 candidates for batch scoring …")

    non_hp = [r for r in survivors if not r["is_honeypot"]]
    hp     = [r for r in survivors if r["is_honeypot"]]

    non_hp.sort(key=lambda r: r["kw"], reverse=True)

    batch_candidates = non_hp[:4900] + hp[:300]

    log.info(
        "Pre-filtered to %d candidates (non-hp: %d, hp: %d)",
        len(batch_candidates), min(len(non_hp), 4900), min(len(hp), 300),
    )

    # ------------------------------------------------------------------
    # PHASE 3 — Batch scoring
    # ------------------------------------------------------------------
    log.info("Phase 3: batch scoring (TF-IDF + BM25 + RRF) …")

    features_batch = [r["features"] for r in batch_candidates]
    kw_scores      = [r["kw"]       for r in batch_candidates]

    tfidf_scores = score_tfidf_batch(features_batch)
    bm25_scores  = score_bm25_batch(features_batch)
    fused_scores = rrf_fusion(kw_scores, tfidf_scores, bm25_scores)

    log.info(
        "Batch scoring done. Signal ranges: "
        "kw=%.3f  tfidf=%.3f  bm25=%.3f | %.1fs elapsed",
        max(kw_scores)    - min(kw_scores),
        max(tfidf_scores) - min(tfidf_scores),
        max(bm25_scores)  - min(bm25_scores),
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # PHASE 4 — Final scores  (fused × availability + geo)
    # ------------------------------------------------------------------
    log.info("Phase 4: computing final scores …")

    availability, geo = compute_availability_batch(features_batch)

    fused_arr = np.array(fused_scores)
    final_arr = np.clip(fused_arr * availability + geo, 0.0, 1.0)

    final_records: list[dict] = []
    for i, r in enumerate(batch_candidates):
        final_records.append({
            "candidate_id": r["features"]["candidate_id"],
            "features":     r["features"],
            "kw":           kw_scores[i],
            "tfidf":        tfidf_scores[i],
            "bm25":         bm25_scores[i],
            "fused":        fused_scores[i],
            "availability": float(availability[i]),
            "geo":          float(geo[i]),
            "final":        float(final_arr[i]),
            "is_honeypot":  r["is_honeypot"],
        })

    log.info(
        "Final score range: %.4f – %.4f",
        min(r["final"] for r in final_records),
        max(r["final"] for r in final_records),
    )

    # ------------------------------------------------------------------
    # PHASE 5 — Pick top-100 with honeypot budget = 8
    # ------------------------------------------------------------------
    log.info("Phase 5: selecting top-100 (honeypot budget = 8) …")

    final_records.sort(key=lambda r: r["final"], reverse=True)

    top_100: list[dict] = []
    honeypot_count = 0

    for rec in final_records:
        if len(top_100) == 100:
            break
        if rec["is_honeypot"]:
            if honeypot_count < 8:
                top_100.append(rec)
                honeypot_count += 1
            # else: skip — budget exhausted
        else:
            top_100.append(rec)

    log.info(
        "Top-100 selected: %d honeypots included, "
        "score range %.4f – %.4f",
        honeypot_count,
        top_100[-1]["final"],
        top_100[0]["final"],
    )

    # Assign ranks and build reasoning strings
    csv_rows: list[dict] = []
    for rank, rec in enumerate(top_100, 1):
        reasoning = generate_reasoning(
            rec["features"],
            rec["kw"],
            rec["tfidf"],
            rec["bm25"],
            rec["fused"],
            rec["availability"],
            rec["geo"],
            rec["final"],
            rec["is_honeypot"],
        )
        csv_rows.append({
            "candidate_id": rec["candidate_id"],
            "rank":         rank,
            "score":        round(float(rec["final"]), 6),
            "reasoning":    reasoning,
        })

    # ------------------------------------------------------------------
    # PHASE 6 — Write CSV + time check
    # ------------------------------------------------------------------
    log.info("Phase 6: writing output CSV …")

    write_csv(csv_rows, args.out)

    elapsed = time.perf_counter() - t0
    log.info("Done in %.1fs → %s", elapsed, args.out)

    if elapsed > 250:
        logging.warning(
            "WARNING: pipeline took %.1fs — close to 300s limit!", elapsed
        )

    # Always print completion (even in non-verbose mode)
    print(f"[rank.py] Completed in {elapsed:.1f}s | output: {args.out}")


if __name__ == "__main__":
    main()
