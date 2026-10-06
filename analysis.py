"""
Deep analysis layer for Stockfolio.

Fuses four independent pillars into one verdict with a transparent breakdown:

    1. Prediction  — the ensemble's expected next-day return (prediction.py)
    2. Technical   — composite of RSI, MACD, Bollinger position, EMA trend, volume
    3. Sentiment   — recency-weighted VADER over headlines + summaries
    4. Macro       — USD/PKR exposure tilt

Each pillar outputs a signal in [-1, 1]; a weighted blend gives a final
Buy / Hold / Sell with confidence and human-readable reasons.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

import fin_sentiment


# ───────────────────────────────────────────────────────── sentiment (deep)

def deep_sentiment(news: list[dict]) -> dict:
    """Recency-weighted FinBERT (or VADER) sentiment over headline + summary."""
    if not news:
        return {'score': 0.0, 'label': 'neutral', 'positive': 0, 'negative': 0,
                'neutral': 0, 'strength': 0.0, 'article_count': 0, 'model': fin_sentiment.backend_name()}

    now = datetime.now(timezone.utc)
    texts = [(n.get('headline', '') + '. ' + (n.get('summary', '') or '')).strip() for n in news]
    scores = fin_sentiment.score_texts(texts)
    if len(scores) < len(news):  # some empty texts dropped — pad
        scores = (scores + [0.0] * len(news))[:len(news)]

    weighted, wsum = 0.0, 0.0
    pos = neg = neu = 0
    for i, n in enumerate(news):
        s = scores[i]
        # recency weight: newer = heavier (list is newest-first); decay by rank
        w = math.exp(-i * 0.25)
        # best-effort recency from published_at
        try:
            pub = n.get('published_at', '')
            if pub:
                dt = datetime(*[int(x) for x in _parse_date(pub)][:6], tzinfo=timezone.utc)
                days = max(0.0, (now - dt).total_seconds() / 86400)
                w *= math.exp(-days / 7)  # half-life ~1 week
        except Exception:
            pass
        weighted += s * w
        wsum += w
        if s > 0.05:
            pos += 1
        elif s < -0.05:
            neg += 1
        else:
            neu += 1

    score = weighted / wsum if wsum else 0.0
    label = 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'
    # agreement strength: how one-sided the coverage is
    total = max(1, pos + neg + neu)
    strength = abs(pos - neg) / total
    return {
        'score': round(score, 3), 'label': label,
        'positive': pos, 'negative': neg, 'neutral': neu,
        'strength': round(strength, 3), 'article_count': len(news),
        'model': fin_sentiment.backend_name(),
    }


def _parse_date(s: str):
    """Very loose RFC-822 / ISO date → (Y, M, D, h, m, s). Returns now on fail."""
    import email.utils
    try:
        t = email.utils.parsedate_tz(s)
        if t:
            return t[:6]
    except Exception:
        pass
    n = datetime.now(timezone.utc)
    return (n.year, n.month, n.day, n.hour, n.minute, n.second)


# ───────────────────────────────────────────────────────── technical score

def _ema(x: np.ndarray, p: int) -> np.ndarray:
    k = 2 / (p + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def technical_score(ohlc: list[dict]) -> dict:
    """Composite technical signal in [-1, 1] with per-indicator notes."""
    if len(ohlc) < 30:
        return {'score': 0.0, 'signals': [], 'rsi': None}
    closes = np.array([e['close'] for e in ohlc], dtype=float)
    vols = np.array([e.get('volume', 0) or 0 for e in ohlc], dtype=float)
    signals, parts = [], []

    # RSI(14)
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.convolve(gain, np.ones(14) / 14, 'valid')
    al = np.convolve(loss, np.ones(14) / 14, 'valid')
    rs = ag[-1] / al[-1] if al[-1] != 0 else 100
    rsi = 100 - 100 / (1 + rs)
    if rsi < 30:
        parts.append(0.8); signals.append(f'RSI {rsi:.0f} — oversold (bullish)')
    elif rsi > 70:
        parts.append(-0.8); signals.append(f'RSI {rsi:.0f} — overbought (bearish)')
    else:
        parts.append((50 - rsi) / 50 * 0.3)
        signals.append(f'RSI {rsi:.0f} — neutral')

    # MACD
    ema12, ema26 = _ema(closes, 12), _ema(closes, 26)
    macd = ema12 - ema26
    sig = _ema(macd, 9)
    hist = macd[-1] - sig[-1]
    hist_prev = macd[-2] - sig[-2]
    if hist > 0 and hist > hist_prev:
        parts.append(0.7); signals.append('MACD — bullish momentum rising')
    elif hist < 0 and hist < hist_prev:
        parts.append(-0.7); signals.append('MACD — bearish momentum')
    else:
        parts.append(math.copysign(0.2, hist)); signals.append('MACD — flat')

    # Bollinger position (mean reversion)
    win = closes[-20:]
    mid, std = win.mean(), win.std()
    pos = (closes[-1] - (mid - 2 * std)) / (4 * std) if std > 1e-9 else 0.5
    if pos < 0.1:
        parts.append(0.6); signals.append('Price at lower Bollinger band (bounce likely)')
    elif pos > 0.9:
        parts.append(-0.6); signals.append('Price at upper Bollinger band (pullback likely)')
    else:
        parts.append((0.5 - pos) * 0.4)

    # Trend: EMA10 vs EMA20 + price vs EMA20
    e10, e20 = _ema(closes, 10)[-1], _ema(closes, 20)[-1]
    if e10 > e20 and closes[-1] > e20:
        parts.append(0.6); signals.append('Uptrend — EMA-10 above EMA-20')
    elif e10 < e20 and closes[-1] < e20:
        parts.append(-0.6); signals.append('Downtrend — EMA-10 below EMA-20')
    else:
        parts.append(0.0)

    # Volume confirmation
    if len(vols) > 20 and vols[-1] > vols[-20:].mean() * 1.3:
        conf = 0.3 if closes[-1] > closes[-2] else -0.3
        parts.append(conf)
        signals.append('High volume ' + ('confirms up-move' if conf > 0 else 'on down-move'))

    score = float(np.clip(np.mean(parts), -1, 1))
    return {'score': round(score, 3), 'signals': signals, 'rsi': round(float(rsi), 1)}


# ───────────────────────────────────────────────────────── fused verdict

def deep_analysis(ohlc: list[dict], news: list[dict], change_pct: float,
                  prediction: dict, usd_exposure: str = 'medium') -> dict:
    sent = deep_sentiment(news)
    tech = technical_score(ohlc)

    pred_ret = prediction.get('predicted_return_pct', 0.0) / 100.0
    pred_sig = float(np.clip(math.tanh(pred_ret * 30), -1, 1))
    tech_sig = tech['score']
    sent_sig = float(np.clip(sent['score'], -1, 1))
    macro_sig = {'high': -0.15, 'medium': 0.0, 'low': 0.1}.get(usd_exposure, 0.0)

    pillars = {
        'prediction': {'signal': round(pred_sig, 3), 'weight': 0.35},
        'technical': {'signal': round(tech_sig, 3), 'weight': 0.30},
        'sentiment': {'signal': round(sent_sig, 3), 'weight': 0.25},
        'macro': {'signal': round(macro_sig, 3), 'weight': 0.10},
    }
    final = sum(p['signal'] * p['weight'] for p in pillars.values())

    if final > 0.33:
        verdict = 'Strong Buy'
    elif final > 0.08:
        verdict = 'Buy'
    elif final < -0.33:
        verdict = 'Strong Sell'
    elif final < -0.08:
        verdict = 'Sell'
    else:
        verdict = 'Hold'

    # confidence: pillar agreement + ensemble confidence
    sigs = [pillars[k]['signal'] for k in pillars]
    agreement = 1 - float(np.std(sigs))
    conf = float(np.clip(0.5 * max(0, agreement) + 0.5 * prediction.get('confidence', 0.5), 0.3, 0.95))

    reasons = []
    reasons.append(f"Model expects {prediction.get('predicted_return_pct', 0):+.2f}% next session "
                   f"(to Rs {prediction.get('predicted_price', 0)}).")
    if tech['signals']:
        reasons.append('Technicals: ' + '; '.join(tech['signals'][:3]) + '.')
    reasons.append(f"News sentiment {sent['label']} "
                   f"({sent['positive']}+/{sent['negative']}- of {sent['article_count']} articles).")
    reasons.append(f"Today {change_pct:+.2f}%. USD/PKR exposure: {usd_exposure}.")

    return {
        'verdict': verdict,
        'confidence': round(conf, 3),
        'score': round(float(final), 3),
        'pillars': pillars,
        'sentiment': sent,
        'technical': tech,
        'prediction': prediction,
        'reasons': reasons,
        'summary': f"{verdict} — {reasons[0]} {reasons[2]}",
    }
