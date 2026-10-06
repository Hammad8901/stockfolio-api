# Hugging Face Spaces (Docker SDK) — serves the Flask API on port 7860.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scientific wheels install fine on slim.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Deep-learning extras (HF Spaces have the RAM): CPU-only torch for LSTM/GRU,
# and transformers for FinBERT finance-news sentiment. Falls back gracefully if
# these fail — the app imports them in try/except.
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers==4.44.2"

COPY . .

ENV PORT=7860
EXPOSE 7860

# 1 worker (free tier RAM), long timeout for first prediction/model fits.
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:7860"]
