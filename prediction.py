"""
Advanced price-prediction engine for Stockfolio.

An ensemble of classical ML (Ridge, RandomForest, GradientBoosting) and — when
PyTorch is available — small sequence models (LSTM + GRU). Each model is scored
on a held-out validation tail and the ensemble weights it by inverse error, so
no single biased model dominates. Ships with an explainable-AI (XAI) breakdown
and a concrete Buy / Hold / Sell answer.

Predicts next-day *return* (more stationary than price), then maps back to price.
Degrades gracefully: with only numpy it still returns a momentum-based estimate;
torch models are optional.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    _SK = True
except Exception:
    _SK = False

try:
    import torch
    import torch.nn as nn
    _TORCH = True
    torch.set_num_threads(1)
except Exception:
    _TORCH = False


# ──────────────────────────────────────────────────────── feature engineering

_FEATURES = [
    'ret1', 'ret5', 'ret10', 'mom', 'vol10',
    'rsi', 'ema10_ratio', 'ema20_ratio', 'bb_pos', 'vol_chg',
]


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    out = np.full_like(closes, 50.0)
    for i in range(period, len(closes)):
        ag = gain[i - period + 1:i + 1].mean()
        al = loss[i - period + 1:i + 1].mean()
        rs = ag / al if al != 0 else 100.0
        out[i] = 100 - 100 / (1 + rs)
    return out


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def _build_matrix(ohlc: list[dict]):
    """Return (X, y, last_feature_row, closes) where y = next-day return."""
    closes = np.array([e['close'] for e in ohlc], dtype=float)
    vols = np.array([e.get('volume', 0) or 0 for e in ohlc], dtype=float)
    n = len(closes)

    rets = np.zeros(n)
    rets[1:] = closes[1:] / closes[:-1] - 1
    rsi = _rsi(closes) / 100.0
    ema10 = _ema(closes, 10)
    ema20 = _ema(closes, 20)
    bb_mid = np.convolve(closes, np.ones(20) / 20, mode='same')
    bb_std = np.array([closes[max(0, i - 19):i + 1].std() for i in range(n)])
    bb_lower = bb_mid - 2 * bb_std
    bb_upper = bb_mid + 2 * bb_std

    rows = []
    for i in range(n):
        ret1 = rets[i]
        ret5 = closes[i] / closes[i - 5] - 1 if i >= 5 else 0.0
        ret10 = closes[i] / closes[i - 10] - 1 if i >= 10 else 0.0
        mom = (ret5 + ret10) / 2
        vol10 = rets[max(0, i - 9):i + 1].std()
        band = bb_upper[i] - bb_lower[i]
        bb_pos = (closes[i] - bb_lower[i]) / band if band > 1e-9 else 0.5
        vol_chg = (vols[i] / vols[i - 1] - 1) if i >= 1 and vols[i - 1] > 0 else 0.0
        rows.append([
            ret1, ret5, ret10, mom, vol10,
            rsi[i], closes[i] / ema10[i] - 1, closes[i] / ema20[i] - 1,
            float(np.clip(bb_pos, -0.5, 1.5)), float(np.clip(vol_chg, -3, 3)),
        ])
    feats = np.array(rows, dtype=float)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # target: next-day return (align X[i] -> y = return from day i to i+1)
    X = feats[:-1]
    y = rets[1:]
    last_row = feats[-1]
    return X, y, last_row, closes


# ──────────────────────────────────────────────────────── sequence models

if _TORCH:
    class _SeqNet(nn.Module):
        def __init__(self, kind: str, hidden: int = 24):
            super().__init__()
            rnn = nn.LSTM if kind == 'lstm' else nn.GRU
            self.rnn = rnn(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    def _train_seq(returns: np.ndarray, kind: str, window: int = 20, epochs: int = 60):
        """Tiny LSTM/GRU on the return series → next-day return + val RMSE."""
        r = returns.astype(np.float32)
        if len(r) < window + 15:
            return None, None
        xs, ys = [], []
        for i in range(window, len(r)):
            xs.append(r[i - window:i])
            ys.append(r[i])
        X = torch.tensor(np.array(xs)).unsqueeze(-1)
        Y = torch.tensor(np.array(ys)).unsqueeze(-1)
        cut = int(len(X) * 0.8)
        net = _SeqNet(kind)
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(X[:cut]), Y[:cut])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(loss_fn(net(X[cut:]), Y[cut:])).item()) if len(X) > cut else float(loss.item())
            nxt = float(net(torch.tensor(r[-window:]).reshape(1, window, 1)).item())
        return nxt, val_rmse


# ──────────────────────────────────────────────────────── ensemble

def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _holt(closes: np.ndarray, alpha: float = 0.5, beta: float = 0.3):
    """Holt's double-exponential smoothing — a tiny, always-available trend model
    for small-scale trend understanding (level + trend, no heavy libraries).
    Returns one-step-ahead predicted closes and the next close."""
    n = len(closes)
    level = float(closes[0])
    trend = float(closes[1] - closes[0]) if n > 1 else 0.0
    preds = [level]
    for i in range(1, n):
        preds.append(level + trend)          # forecast made before seeing closes[i]
        prev_level = level
        level = alpha * closes[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array(preds), level + trend


def advanced_prediction(ohlc: list[dict], sentiment_score: float = 0.0,
                        indicators: dict | None = None) -> dict:
    """Full ensemble prediction + XAI + Buy/Hold/Sell. Never raises."""
    try:
        return _predict(ohlc, sentiment_score)
    except Exception as e:  # pragma: no cover
        last = ohlc[-1]['close'] if ohlc else 0.0
        return {
            'predicted_price': round(last, 2), 'predicted_return_pct': 0.0,
            'confidence': 0.4, 'action': 'Hold', 'recommendation': 'Hold',
            'models': {}, 'feature_importance': [], 'ensemble_weights': {},
            'explanation': 'Not enough data for a reliable prediction.',
            'error': str(e),
        }


def _predict(ohlc: list[dict], sentiment_score: float) -> dict:
    closes_all = [e['close'] for e in ohlc]
    last_close = float(closes_all[-1])
    if len(ohlc) < 25:
        # momentum fallback
        ret = (closes_all[-1] / closes_all[-5] - 1) / 4 if len(closes_all) >= 5 else 0.0
        return _assemble(last_close, ret, 0.45, {}, {}, [], sentiment_score,
                         'Short history — momentum-only estimate.')

    X, y, last_row, closes = _build_matrix(ohlc)
    cut = int(len(X) * 0.8)
    Xtr, Xval, ytr, yval = X[:cut], X[cut:], y[:cut], y[cut:]

    preds: dict[str, float] = {}
    rmses: dict[str, float] = {}
    importance = np.zeros(len(_FEATURES))
    imp_n = 0

    if _SK and len(Xtr) >= 10:
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xval_s, last_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform([last_row])
        defs = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=120, max_depth=6, random_state=0, n_jobs=1),
            'grad_boost': GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0),
        }
        for name, model in defs.items():
            try:
                model.fit(Xtr_s if name == 'ridge' else Xtr, ytr)
                xv = Xval_s if name == 'ridge' else Xval
                xl = last_s if name == 'ridge' else [last_row]
                rmses[name] = _rmse(model.predict(xv), yval) if len(Xval) else 0.02
                preds[name] = float(model.predict(xl)[0])
                if hasattr(model, 'feature_importances_'):
                    importance += model.feature_importances_
                    imp_n += 1
            except Exception:
                pass

    # Lightweight trend model (always available — small-scale trend understanding).
    try:
        hpreds, hnext = _holt(closes)
        trend_ret = hnext / closes[-1] - 1
        if len(y) > cut:
            vr = _rmse([hpreds[j + 1] / closes[j] - 1 for j in range(cut, len(y))],
                       [y[j] for j in range(cut, len(y))])
        else:
            vr = 0.02
        preds['trend'] = float(np.clip(trend_ret, -0.2, 0.2))
        rmses['trend'] = vr if vr and vr > 0 else 0.02
    except Exception:
        pass

    if _TORCH:
        for kind in ('lstm', 'gru'):
            nxt, vr = _train_seq(np.concatenate([[0.0], y]), kind)
            if nxt is not None:
                preds[kind] = nxt
                rmses[kind] = vr if vr and vr > 0 else 0.02

    if not preds:
        ret = float(np.mean(y[-5:])) if len(y) >= 5 else 0.0
        return _assemble(last_close, ret, 0.45, {}, {}, [], sentiment_score,
                         'Baseline estimate (ML libraries unavailable).')

    # inverse-error weighting (bias reduction)
    eps = 1e-4
    weights = {k: 1.0 / (rmses.get(k, 0.02) + eps) for k in preds}
    wsum = sum(weights.values())
    weights = {k: v / wsum for k, v in weights.items()}
    ens_ret = float(sum(weights[k] * preds[k] for k in preds))
    ens_ret = float(np.clip(ens_ret, -0.2, 0.2))

    # confidence: model agreement (low spread) + validation quality
    spread = float(np.std(list(preds.values()))) if len(preds) > 1 else 0.02
    avg_rmse = float(np.mean(list(rmses.values()))) if rmses else 0.02
    conf = 0.9 * math.exp(-spread * 40) * math.exp(-avg_rmse * 20)
    conf = float(np.clip(conf, 0.35, 0.95))

    # XAI: top feature drivers
    xai = []
    if imp_n:
        importance /= imp_n
        order = np.argsort(importance)[::-1][:4]
        xai = [{'feature': _pretty(_FEATURES[i]), 'weight': round(float(importance[i]), 3)} for i in order]

    models_out = {k: {'predicted_return_pct': round(preds[k] * 100, 2),
                      'weight': round(weights[k], 3),
                      'val_rmse': round(rmses.get(k, 0.0), 5)} for k in preds}

    return _assemble(last_close, ens_ret, conf, models_out, weights, xai, sentiment_score,
                     None, ohlc)


def _pretty(f: str) -> str:
    return {
        'ret1': 'Yesterday\'s move', 'ret5': '5-day momentum', 'ret10': '10-day momentum',
        'mom': 'Momentum', 'vol10': 'Volatility', 'rsi': 'RSI', 'ema10_ratio': 'Price vs EMA-10',
        'ema20_ratio': 'Price vs EMA-20', 'bb_pos': 'Bollinger position', 'vol_chg': 'Volume change',
    }.get(f, f)


def _assemble(last_close, ens_ret, conf, models_out, weights, xai, sentiment_score,
              note=None, ohlc=None):
    predicted_price = round(last_close * (1 + ens_ret), 2)
    exp_pct = round(ens_ret * 100, 2)

    # RSI/trend signal for the recommendation
    rsi_sig = 0.0
    if ohlc:
        closes = np.array([e['close'] for e in ohlc], dtype=float)
        rsi_now = _rsi(closes)[-1]
        if rsi_now < 30:
            rsi_sig = 0.4      # oversold → lean buy
        elif rsi_now > 70:
            rsi_sig = -0.4     # overbought → lean sell
        else:
            rsi_sig = (50 - rsi_now) / 100

    # combined signal: prediction + sentiment + RSI
    signal = (math.tanh(ens_ret * 30) * 0.55) + (float(np.clip(sentiment_score, -1, 1)) * 0.3) + (rsi_sig * 0.15)
    if signal > 0.33:
        action = 'Strong Buy'
    elif signal > 0.08:
        action = 'Buy'
    elif signal < -0.33:
        action = 'Strong Sell'
    elif signal < -0.08:
        action = 'Sell'
    else:
        action = 'Hold'

    dir_word = 'rise' if ens_ret > 0 else 'fall' if ens_ret < 0 else 'stay flat'
    sent_word = 'positive' if sentiment_score > 0.05 else 'negative' if sentiment_score < -0.05 else 'neutral'
    drivers = ', '.join(d['feature'] for d in xai[:3]) if xai else 'recent momentum'
    explanation = note or (
        f"Ensemble of {len(models_out)} models expects price to {dir_word} ~{abs(exp_pct):.2f}% next session "
        f"(to Rs {predicted_price}). News sentiment is {sent_word}. Key drivers: {drivers}. "
        f"Combined signal → {action}."
    )

    return {
        'predicted_price': predicted_price,
        'predicted_return_pct': exp_pct,
        'confidence': round(conf, 3),
        'action': action,
        'recommendation': action,
        'signal': round(float(signal), 3),
        'models': models_out,
        'ensemble_weights': {k: round(v, 3) for k, v in weights.items()},
        'feature_importance': xai,
        'explanation': explanation,
    }
