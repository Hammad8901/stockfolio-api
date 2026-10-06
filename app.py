"""
Stockfolio Flask API — PSX stock analysis backend
Deploy to Render: set start command to `gunicorn app:app --workers 1 --timeout 120`
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_compress import Compress
import yfinance as yf
import feedparser
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from psx_companies import PSX_COMPANIES, PSX_COMPANY_MAP, DEFAULT_SYMBOLS
from prediction import advanced_prediction
from analysis import deep_analysis

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
Compress(app)

# ─── Cache ────────────────────────────────────────────────────────────────────

_cache: dict = {}
_cache_lock = threading.Lock()

NEWS_TTL = 1800    # 30 min
STOCK_TTL = 60     # 1 min
PORTFOLIO_TTL = 300  # 5 min


def _cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry['ts'] < entry['ttl']:
            return entry['data']
        return None


def _cache_set(key, data, ttl=300):
    with _cache_lock:
        _cache[key] = {'data': data, 'ts': time.time(), 'ttl': ttl}


# ─── Sentiment (VADER, lightweight) ──────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()


def _vader_sentiment(texts: list[str]) -> tuple[float, str]:
    if not texts:
        return 0.0, 'neutral'
    scores = [_vader.polarity_scores(t)['compound'] for t in texts]
    avg = sum(scores) / len(scores)
    label = 'positive' if avg > 0.05 else 'negative' if avg < -0.05 else 'neutral'
    return avg, label


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _usd_rate() -> float:
    cached = _cache_get('usd_pkr')
    if cached:
        return cached
    try:
        tk = yf.Ticker('PKR=X')
        hist = tk.history(period='2d')
        rate = float(hist['Close'].iloc[-1]) if not hist.empty else 278.0
    except Exception:
        rate = 278.0
    _cache_set('usd_pkr', rate, ttl=3600)
    return rate


def _fetch_news(symbol: str, max_results: int = 5) -> list[dict]:
    ckey = f'news_{symbol}'
    cached = _cache_get(ckey)
    if cached:
        return cached

    query = symbol.split('.')[0]
    url = f'https://news.google.com/rss/search?q={query}+PSX+Pakistan+stock&hl=en&gl=PK&ceid=PK:en'
    articles = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_results]:
            summary = entry.get('summary', '') or ''
            import re as _re
            summary = _re.sub(r'<[^>]+>', '', summary)[:300]
            articles.append({
                'headline': entry.get('title', ''),
                'summary': summary,
                'source': entry.get('source', {}).get('title', 'Google News') if isinstance(entry.get('source'), dict) else 'Google News',
                'published_at': entry.get('published', ''),
                'url': entry.get('link', ''),
            })
    except Exception as e:
        logger.warning(f'News fetch failed for {symbol}: {e}')

    _cache_set(ckey, articles, ttl=NEWS_TTL)
    return articles


def _ohlc_data(symbol: str, period: str) -> list[dict]:
    period_map = {'1W': '7d', '1M': '1mo', '6M': '6mo', '1Y': '1y', '5Y': '5y'}
    yf_period = period_map.get(period, '1mo')
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=yf_period, interval='1d')
        if hist.empty:
            return []
        result = []
        for idx, row in hist.iterrows():
            result.append({
                'timestamp': int(idx.timestamp() * 1000),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume']),
            })
        return result
    except Exception as e:
        logger.error(f'OHLC fetch failed {symbol}: {e}')
        return []


def _current_price(symbol: str) -> tuple[float, float]:
    """Returns (price, change_percent)"""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period='2d')
        if len(hist) >= 2:
            prev = float(hist['Close'].iloc[-2])
            curr = float(hist['Close'].iloc[-1])
            return curr, round((curr - prev) / prev * 100, 2)
        elif len(hist) == 1:
            curr = float(hist['Close'].iloc[-1])
            return curr, 0.0
    except Exception:
        pass
    return 0.0, 0.0


# ─── Technical indicators ────────────────────────────────────────────────────

def _calc_rsi(closes: list[float], period: int = 14) -> list[dict]:
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))

    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    rsi_vals = []
    for i in range(period, len(closes)):
        if i > period:
            ag = (ag * (period - 1) + gains[i - 1]) / period
            al = (al * (period - 1) + losses[i - 1]) / period
        rs = ag / al if al != 0 else 100
        rsi_vals.append(100 - 100 / (1 + rs))
    return rsi_vals


def _calc_ema(prices: list[float], period: int) -> list[float]:
    if not prices:
        return []
    k = 2.0 / (period + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def _calc_bb(closes: list[float], period: int = 20, mult: float = 2.0):
    upper, middle, lower = [], [], []
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        sma = sum(window) / period
        std = (sum((p - sma) ** 2 for p in window) / period) ** 0.5
        middle.append(sma)
        upper.append(sma + mult * std)
        lower.append(sma - mult * std)
    return upper, middle, lower


def _build_indicators(ohlc: list[dict], timestamps: list[int]) -> dict:
    if len(ohlc) < 30:
        return {}
    closes = [e['close'] for e in ohlc]

    # RSI
    rsi_vals = _calc_rsi(closes)
    rsi_offset = len(ohlc) - len(rsi_vals)
    rsi_data = [{'date': timestamps[rsi_offset + i], 'value': round(v, 2)}
                for i, v in enumerate(rsi_vals)]

    # MACD
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal_src = macd_line[25:]
    signal_line = _calc_ema(signal_src, 9)
    macd_offset = 25
    macd_data = []
    for i, sig in enumerate(signal_line):
        di = macd_offset + i
        macd_data.append({
            'date': timestamps[di],
            'macd': round(macd_line[di], 4),
            'signal': round(sig, 4),
            'histogram': round(macd_line[di] - sig, 4),
        })

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = _calc_bb(closes)
    bb_offset = len(ohlc) - len(bb_upper)
    bb_upper_data = [{'date': timestamps[bb_offset + i], 'value': round(v, 2)} for i, v in enumerate(bb_upper)]
    bb_mid_data = [{'date': timestamps[bb_offset + i], 'value': round(v, 2)} for i, v in enumerate(bb_middle)]
    bb_lower_data = [{'date': timestamps[bb_offset + i], 'value': round(v, 2)} for i, v in enumerate(bb_lower)]

    return {
        'rsi': rsi_data,
        'macd': macd_data,
        'bb_upper': bb_upper_data,
        'bb_middle': bb_mid_data,
        'bb_lower': bb_lower_data,
    }


def _linear_prediction(ohlc: list[dict]) -> tuple[float, float]:
    """Simple linear regression on last 30 days → next-day prediction."""
    if len(ohlc) < 5:
        return 0.0, 0.5
    closes = [e['close'] for e in ohlc[-30:]]
    n = len(closes)
    x = list(range(n))
    xm = sum(x) / n
    ym = sum(closes) / n
    num = sum((x[i] - xm) * (closes[i] - ym) for i in range(n))
    den = sum((x[i] - xm) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0
    predicted = ym + slope * (n - xm)  # one step ahead
    # Confidence: based on R-squared
    ss_res = sum((closes[i] - (ym + slope * (x[i] - xm))) ** 2 for i in range(n))
    ss_tot = sum((closes[i] - ym) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    conf = max(0.4, min(0.92, abs(r2)))
    return round(predicted, 2), round(conf, 3)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})


@app.route('/api/companies')
def companies():
    return jsonify(PSX_COMPANIES)


@app.route('/api/stocks')
def stocks():
    cached = _cache_get('stocks_default')
    if cached:
        return jsonify(cached)

    results = []
    for sym in DEFAULT_SYMBOLS:
        try:
            price, change = _current_price(sym)
            info = PSX_COMPANY_MAP.get(sym, {'name': sym, 'sector': '', 'usd_exposure': 'medium'})
            news = _fetch_news(sym, 3)
            headlines = [n['headline'] for n in news]
            score, sentiment = _vader_sentiment(headlines)
            rec = 'Buy' if score > 0.1 else 'Sell' if score < -0.1 else 'Hold'
            if change > 2 and score > 0:
                rec = 'Strong Buy'
            elif change < -2 and score < 0:
                rec = 'Strong Sell'

            results.append({
                'symbol': sym,
                'name': info['name'],
                'sector': info['sector'],
                'usd_exposure': info['usd_exposure'],
                'current_price': round(price, 2),
                'price': round(price, 2),
                'change_percent': round(change, 2),
                'sentiment': sentiment,
                'sentiment_score': round(score, 3),
                'confidence': round(abs(score), 3),
                'recommendation': rec,
            })
        except Exception as e:
            logger.error(f'Stock data error for {sym}: {e}')

    _cache_set('stocks_default', results, ttl=STOCK_TTL)
    return jsonify(results)


@app.route('/api/portfolio')
def portfolio():
    return stocks()


@app.route('/api/stock/<symbol>')
def stock_ohlc(symbol):
    range_ = request.args.get('range', '1M')
    ckey = f'ohlc_{symbol}_{range_}'
    cached = _cache_get(ckey)
    if cached:
        return jsonify(cached)

    ohlc = _ohlc_data(symbol, range_)
    if not ohlc:
        return jsonify({'error': f'No data for {symbol}'}), 404

    result = {'symbol': symbol, 'range': range_, 'graphData': ohlc}
    _cache_set(ckey, result, ttl=STOCK_TTL)
    return jsonify(result)


@app.route('/api/portfolio/stock/<symbol>')
def portfolio_stock(symbol):
    range_ = request.args.get('range', '1M')
    ckey = f'portfolio_{symbol}_{range_}'
    cached = _cache_get(ckey)
    if cached:
        return jsonify(cached)

    ohlc = _ohlc_data(symbol, range_)
    price, change = _current_price(symbol)

    news = _fetch_news(symbol, 5)
    headlines = [n['headline'] for n in news]
    score, sentiment = _vader_sentiment(headlines)

    # Advanced ensemble prediction (LSTM/GRU + trees + trend) with XAI + action.
    pred = advanced_prediction(ohlc, sentiment_score=score, indicators=None)
    rec = pred['action']  # ensemble-driven Buy / Hold / Sell
    predicted, pred_conf = pred['predicted_price'], pred['confidence']
    usd = _usd_rate()

    info = PSX_COMPANY_MAP.get(symbol, {'name': symbol, 'sector': '', 'usd_exposure': 'medium'})
    usd_exposure = info.get('usd_exposure', 'medium')
    usd_impact = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}.get(usd_exposure, 'Medium')

    explanation = pred['explanation']

    timestamps = [e['timestamp'] for e in ohlc]
    indicators = _build_indicators(ohlc, timestamps)

    result = {
        'symbol': symbol,
        'name': info['name'],
        'sector': info.get('sector', ''),
        'current_price': round(price, 2),
        'change_percent': round(change, 2),
        'graphData': ohlc,
        'sentiment': sentiment,
        'sentiment_score': round(score, 3),
        'confidence': round(abs(score), 3),
        'recommendation': rec,
        'explanation': explanation,
        'predicted_price': predicted,
        'prediction_confidence': pred_conf,
        'predicted_return_pct': pred['predicted_return_pct'],
        'prediction': pred,  # full XAI breakdown: models, weights, feature importance
        'usd_rate': round(usd, 2),
        'usd_impact': usd_impact,
        'usd_exposure': usd_exposure,
        'top_news': news[:3],
        'indicators': indicators,
    }
    _cache_set(ckey, result, ttl=PORTFOLIO_TTL)
    return jsonify(result)


@app.route('/api/analysis/<symbol>')
def analysis(symbol):
    """Deep fused analysis: prediction + technical + sentiment + macro → verdict."""
    range_ = request.args.get('range', '3M')
    ckey = f'analysis_{symbol}_{range_}'
    cached = _cache_get(ckey)
    if cached:
        return jsonify(cached)

    ohlc = _ohlc_data(symbol, range_)
    if not ohlc:
        return jsonify({'error': f'No data for {symbol}'}), 404

    price, change = _current_price(symbol)
    news = _fetch_news(symbol, 10)
    score, _ = _vader_sentiment([n['headline'] for n in news])
    pred = advanced_prediction(ohlc, sentiment_score=score, indicators=None)

    info = PSX_COMPANY_MAP.get(symbol, {'name': symbol, 'sector': '', 'usd_exposure': 'medium'})
    deep = deep_analysis(ohlc, news, change, pred, info.get('usd_exposure', 'medium'))

    result = {
        'symbol': symbol,
        'name': info['name'],
        'sector': info.get('sector', ''),
        'current_price': round(price, 2),
        'change_percent': round(change, 2),
        'top_news': news[:5],
        **deep,
    }
    _cache_set(ckey, result, ttl=PORTFOLIO_TTL)
    return jsonify(result)


@app.route('/api/predict/<symbol>')
def predict(symbol):
    """Full ensemble prediction with XAI + Buy/Hold/Sell for one stock."""
    range_ = request.args.get('range', '3M')
    ckey = f'predict_{symbol}_{range_}'
    cached = _cache_get(ckey)
    if cached:
        return jsonify(cached)

    ohlc = _ohlc_data(symbol, range_)
    if not ohlc:
        return jsonify({'error': f'No data for {symbol}'}), 404

    price, change = _current_price(symbol)
    news = _fetch_news(symbol, 6)
    score, _ = _vader_sentiment([n['headline'] for n in news])
    pred = advanced_prediction(ohlc, sentiment_score=score, indicators=None)

    result = {
        'symbol': symbol,
        'current_price': round(price, 2),
        'change_percent': round(change, 2),
        'sentiment_score': round(score, 3),
        **pred,
    }
    _cache_set(ckey, result, ttl=PORTFOLIO_TTL)
    return jsonify(result)


@app.route('/api/sentiment/<symbol>')
def sentiment(symbol):
    ckey = f'sentiment_{symbol}'
    cached = _cache_get(ckey)
    if cached:
        return jsonify(cached)

    news = _fetch_news(symbol, 8)
    headlines = [n['headline'] for n in news]
    score, label = _vader_sentiment(headlines)

    news_with_sentiment = []
    for item in news:
        s = _vader.polarity_scores(item['headline'])['compound']
        sl = 'positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral'
        news_with_sentiment.append({**item, 'sentiment': sl, 'confidence': abs(s)})

    result = {
        'symbol': symbol,
        'sentiment': label,
        'sentiment_score': round(score, 3),
        'confidence': round(abs(score), 3),
        'top_news': news_with_sentiment,
    }
    _cache_set(ckey, result, ttl=NEWS_TTL)
    return jsonify(result)


@app.route('/api/newsfeed')
def newsfeed():
    cached = _cache_get('newsfeed')
    if cached:
        return jsonify(cached)

    all_news = []
    for sym in DEFAULT_SYMBOLS[:8]:
        news = _fetch_news(sym, 2)
        for item in news:
            s = _vader.polarity_scores(item['headline'])['compound']
            sl = 'positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral'
            all_news.append({
                **item,
                'symbol': sym,
                'sentiment': sl,
                'confidence': round(abs(s), 3),
            })

    # Sort by date descending
    all_news.sort(key=lambda x: x.get('published_at', ''), reverse=True)
    _cache_set('newsfeed', all_news, ttl=NEWS_TTL)
    return jsonify(all_news)


@app.route('/api/notifications')
def notifications():
    symbol = request.args.get('symbol', 'MEBL.KA')
    num = min(int(request.args.get('num_news', 1)), 5)
    news = _fetch_news(symbol, num)
    for item in news:
        s = _vader.polarity_scores(item['headline'])['compound']
        item['sentiment'] = 'positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral'
        item['confidence'] = round(abs(s), 3)
    return jsonify({'symbol': symbol, 'news': news[:num]})


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
