docker build -t rankforge .
docker run --network none --memory 14g --cpus 4 \
  -v $(pwd)/data:/data \
  rankforge \
  --candidates /data/candidates.jsonl \
  --out /data/submission.csv \
  --verbose
