# Hugging Face Spaces (Docker SDK) — serves the Flask API on port 7860.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scientific wheels install fine on slim.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

# 1 worker (free tier RAM), long timeout for first prediction/model fits.
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:7860"]
