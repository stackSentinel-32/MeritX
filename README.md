# RankForge — Senior AI Engineer Candidate Ranker

## Architecture
Three-signal hybrid ranker:
- Signal A: Skill Depth scoring (tier weight × proficiency × duration × assessment × endorsements)
- Signal B: TF-IDF with enriched candidate text (tier1 skills repeated for TF boost)
- Signal C: BM25 (term-frequency ranking — 5x more differentiation than neural embeddings for this domain)
- Fusion: Reciprocal Rank Fusion (RRF, k=60) — same algorithm used in production hybrid search
- Availability: 8-factor multiplier (recency, notice, response rate, completion, github, openwork, offer acceptance, profile completeness)

## Setup
pip install -r requirements.txt
(No model downloads needed — BM25 requires no weights)

## Reproduce submission
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Runtime
~21s on Intel i3 1005G1 (2C/4T, 1.2GHz)
~8s on Intel i5-13450HX (10C/16T, 2.4GHz)
