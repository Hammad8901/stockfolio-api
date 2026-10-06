---
title: Stockfolio API
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Stockfolio API

PSX (Pakistan Stock Exchange) stock analysis backend for the Stockfolio app.

**Features**
- Live prices & OHLC candles (yfinance)
- Technical indicators: RSI, EMA, Bollinger Bands
- **Ensemble predictions** (Ridge + RandomForest + GradientBoosting + Holt trend,
  plus optional LSTM & GRU) with **inverse-error weighting** to reduce bias
- **Explainable-AI (XAI)** feature drivers + per-model weights
- News + **VADER sentiment**
- Concrete **Buy / Hold / Sell** action

**Key endpoints**
- `GET /api/health`
- `GET /api/companies`
- `GET /api/stocks`
- `GET /api/stock/<symbol>?range=1M`
- `GET /api/portfolio/stock/<symbol>?range=1M`
- `GET /api/predict/<symbol>?range=3M`  — full ensemble + XAI + action
- `GET /api/sentiment/<symbol>`
- `GET /api/newsfeed`

To enable LSTM/GRU, uncomment `torch` in `requirements.txt` (Hugging Face free
Spaces have enough RAM). The ensemble runs fine without it.
