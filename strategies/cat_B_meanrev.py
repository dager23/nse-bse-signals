"""Category B: Mean Reversion (B1-B13)."""
import numpy as np
import pandas as pd

from utils import indicators as ta


def _state(entry, exit_, index, columns):
    st = pd.DataFrame(np.nan, index=index, columns=columns)
    st[entry] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def _hold_n(entry, n):
    """Enter on signal, hold n bars."""
    return entry.astype(float).rolling(n, min_periods=1).max().fillna(0)


def get_strategies():
    S = []

    # B1: Bollinger bounce (long lower band, exit mid)
    for n, k in [(20, 2.0), (20, 2.5), (10, 1.5)]:
        def fn(ctx, n=n, k=k):
            c = ctx["close"]
            mid, ub, lb, bw = ta.bollinger(c, n, k)
            return _state(c < lb, c > mid, c.index, c.columns)
        S.append(({"id": "B1", "name": "Bollinger Bounce", "category": "B-MeanRev",
                   "variant": f"BB({n},{k})"}, fn))

    # B2: BB squeeze -> breakout
    for q in [0.15, 0.25]:
        def fn(ctx, q=q):
            c = ctx["close"]
            mid, ub, lb, bw = ta.bollinger(c, 20, 2.0)
            squeeze = bw < bw.rolling(250).quantile(q)
            entry = squeeze.shift(1) & (c > ub)
            exit_ = c < mid
            return _state(entry, exit_, c.index, c.columns)
        S.append(({"id": "B2", "name": "BB Squeeze Breakout", "category": "B-MeanRev",
                   "variant": f"q{int(q*100)}%"}, fn))

    # B3: RSI oversold/overbought
    for n, lo, ex in [(14, 30, 50), (2, 10, 60), (14, 25, 55)]:
        def fn(ctx, n=n, lo=lo, ex=ex):
            c = ctx["close"]
            r = ta.rsi(c, n)
            return _state(r < lo, r > ex, c.index, c.columns)
        S.append(({"id": "B3", "name": "RSI Oversold", "category": "B-MeanRev",
                   "variant": f"RSI{n}<{lo}"}, fn))

    # B4: RSI bullish divergence (lower price low, higher RSI low), hold 10d
    def fn_b4(ctx):
        c = ctx["close"]
        r = ta.rsi(c, 14)
        price_ll = c.rolling(5).min() < c.shift(5).rolling(15).min()
        rsi_hl = r.rolling(5).min() > r.shift(5).rolling(15).min()
        entry = price_ll & rsi_hl & (r < 40)
        return _hold_n(entry, 10)
    S.append(({"id": "B4", "name": "RSI Divergence", "category": "B-MeanRev",
               "variant": "14d/hold10"}, fn_b4))

    # B5: Stochastic %K/%D cross in oversold zone
    for k_n in [14, 21]:
        def fn(ctx, k_n=k_n):
            h, l, c = ctx["high"], ctx["low"], ctx["close"]
            k, d = ta.stochastic(h, l, c, k_n)
            entry = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 25)
            exit_ = k > 80
            return _state(entry, exit_, c.index, c.columns)
        S.append(({"id": "B5", "name": "Stochastic Reversal", "category": "B-MeanRev",
                   "variant": f"Stoch{k_n}"}, fn))

    # B6: Stochastic RSI
    def fn_b6(ctx):
        c = ctx["close"]
        k, d = ta.stoch_rsi(c)
        entry = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 20)
        exit_ = k > 80
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "B6", "name": "Stochastic RSI", "category": "B-MeanRev",
               "variant": "14/14/3/3"}, fn_b6))

    # B7: CCI extremes
    for thr in [-100, -200]:
        def fn(ctx, thr=thr):
            h, l, c = ctx["high"], ctx["low"], ctx["close"]
            cci = ta.cci(h, l, c, 20)
            return _state(cci < thr, cci > 0, c.index, c.columns)
        S.append(({"id": "B7", "name": "CCI Extreme", "category": "B-MeanRev",
                   "variant": f"CCI<{thr}"}, fn))

    # B8: Williams %R
    def fn_b8(ctx):
        h, l, c = ctx["high"], ctx["low"], ctx["close"]
        wr = ta.williams_r(h, l, c, 14)
        return _state(wr < -80, wr > -20, c.index, c.columns)
    S.append(({"id": "B8", "name": "Williams %R", "category": "B-MeanRev",
               "variant": "14d 80/20"}, fn_b8))

    # B9: Z-score reversion
    for n, z_in, z_out in [(20, -2.0, 0.0), (50, -2.5, 0.0), (20, -1.5, 0.5)]:
        def fn(ctx, n=n, z_in=z_in, z_out=z_out):
            c = ctx["close"]
            z = ta.zscore(c, n)
            return _state(z < z_in, z > z_out, c.index, c.columns)
        S.append(({"id": "B9", "name": "Z-Score Reversion", "category": "B-MeanRev",
                   "variant": f"z{n}({z_in}/{z_out})"}, fn))

    # B10: Pairs trading via cointegration (yearly re-formation, sector pairs)
    def fn_b10(ctx):
        from itertools import combinations
        from statsmodels.tsa.stattools import coint
        from utils.sectors import SECTOR_MAP
        c = ctx["close"]
        logp = np.log(c)
        pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        years = sorted(set(c.index.year))
        sectors = {}
        for tkr, sec in SECTOR_MAP.items():
            if tkr in c.columns:
                sectors.setdefault(sec, []).append(tkr)
        for y in years[2:]:
            form = logp[(logp.index.year >= y - 2) & (logp.index.year < y)]
            trade_idx = c.index[c.index.year == y]
            if len(form) < 300 or len(trade_idx) == 0:
                continue
            pairs = []
            for sec, tkrs in sectors.items():
                for a, b in combinations(tkrs, 2):
                    fa, fb = form[a].dropna(), form[b].dropna()
                    common = fa.index.intersection(fb.index)
                    if len(common) < 300:
                        continue
                    try:
                        _, pval, _ = coint(fa[common], fb[common])
                    except Exception:
                        continue
                    if pval < 0.05:
                        beta = np.polyfit(fb[common], fa[common], 1)[0]
                        pairs.append((pval, a, b, beta))
            pairs = sorted(pairs)[:8]
            for _, a, b, beta in pairs:
                spread = logp[a] - beta * logp[b]
                mu = spread.rolling(60).mean()
                sd = spread.rolling(60).std()
                z = ((spread - mu) / sd).reindex(trade_idx)
                sig = pd.Series(np.nan, index=trade_idx)
                sig[z < -2] = 1.0    # long a, short b
                sig[z > 2] = -1.0
                sig[z.abs() < 0.25] = 0.0
                sig = sig.ffill().fillna(0)
                pos.loc[trade_idx, a] += sig
                pos.loc[trade_idx, b] -= sig * np.sign(beta)
        return pos.clip(-1, 1)
    S.append(({"id": "B10", "name": "Pairs Cointegration", "category": "B-MeanRev",
               "variant": "sector/EG/z2", "allow_short": True}, fn_b10))

    # B11: Ornstein-Uhlenbeck half-life filter + z reversion
    def fn_b11(ctx):
        c = ctx["close"]
        logp = np.log(c)
        dp = logp.diff()
        lag = logp.shift(1)
        # rolling OLS slope of dp on lagged level (250d) via cov/var
        cov = dp.rolling(250).cov(lag)
        var = lag.rolling(250).var()
        b = cov / var
        halflife = -np.log(2) / b
        z = ta.zscore(c, 30)
        entry = (z < -1.5) & (halflife > 2) & (halflife < 60)
        exit_ = z > 0
        st = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        st[entry] = 1.0
        st[exit_] = 0.0
        return st.ffill().fillna(0)
    S.append(({"id": "B11", "name": "OU Half-Life Reversion", "category": "B-MeanRev",
               "variant": "HL2-60/z30"}, fn_b11))

    # B12: Hurst exponent filter + z reversion (H computed every 5 bars)
    def fn_b12(ctx):
        c = ctx["close"]
        sub = c.iloc[::5]
        hurst = ta.rolling_hurst(sub, window=50, max_lag=15)  # 50 weekly-ish samples
        hurst = hurst.reindex(c.index).ffill()
        z = ta.zscore(c, 20)
        entry = (z < -2) & (hurst < 0.45)
        exit_ = z > 0
        st = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        st[entry] = 1.0
        st[exit_] = 0.0
        return st.ffill().fillna(0)
    S.append(({"id": "B12", "name": "Hurst-Filtered Reversion", "category": "B-MeanRev",
               "variant": "H<0.45/z20"}, fn_b12))

    # B13: DPO mean reversion
    def fn_b13(ctx):
        c = ctx["close"]
        d = ta.dpo(c, 20)
        sd = d.rolling(100).std()
        return _state(d < -1.5 * sd, d > 0, c.index, c.columns)
    S.append(({"id": "B13", "name": "DPO Reversion", "category": "B-MeanRev",
               "variant": "DPO20/1.5sd"}, fn_b13))

    return S
