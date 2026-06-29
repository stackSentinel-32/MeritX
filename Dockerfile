FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# No model download needed — BM25 has no model weights
ENTRYPOINT ["python", "rank.py"]
