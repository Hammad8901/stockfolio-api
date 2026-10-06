"""
Finance-domain news sentiment.

Uses FinBERT (ProsusAI/finbert) when transformers + torch are installed —
far better than generic lexicons for market news — and falls back to VADER
otherwise, so the service deploys anywhere. The FinBERT model is loaded lazily
on first use and cached for the process.
"""
from __future__ import annotations

import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()

_finbert = None
_finbert_tried = False


def _load_finbert():
    global _finbert, _finbert_tried
    if _finbert_tried:
        return _finbert
    _finbert_tried = True
    try:
        from transformers import pipeline  # noqa
        _finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert', truncation=True, max_length=256)
        logger.info('FinBERT loaded for news sentiment.')
    except Exception as e:
        logger.warning(f'FinBERT unavailable, using VADER. ({e})')
        _finbert = None
    return _finbert


def backend_name() -> str:
    return 'finbert' if _load_finbert() is not None else 'vader'


_SIGN = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}


def score_texts(texts: list[str]) -> list[float]:
    """Signed sentiment per text in [-1, 1] (positive conf − negative conf)."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    fb = _load_finbert()
    if fb is not None:
        try:
            out = fb([t[:512] for t in texts])
            return [round(_SIGN.get(r['label'].lower(), 0.0) * float(r['score']), 4) for r in out]
        except Exception as e:
            logger.warning(f'FinBERT scoring failed, VADER fallback. ({e})')
    return [round(_vader.polarity_scores(t)['compound'], 4) for t in texts]


def score_text(text: str) -> float:
    r = score_texts([text])
    return r[0] if r else 0.0
