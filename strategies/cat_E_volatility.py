"""Category E: Volatility-based strategies (E1-E8)."""
import numpy as np
import pandas as pd

from utils import indicators as ta


def _long(cond):
    return cond.astype(float)


def _state(entry, exit_, index, columns):
    st = pd.DataFrame(np.nan, index=index, columns=columns)
    st[entry] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def _all_stock_basket(c):
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(c.notna(), 0.0)


def get_strategies():
    S = []

    # E1: ATR breakout, hold N
    for k, hold in [(1.5, 10), (2.0, 5)]:
        def fn(ctx, k=k, hold=hold):
            c = ctx["close"]
            a = ta.atr(ctx["high"], ctx["low"], c, 14)
            entry = c > c.shift(1) + k * a.shift(1)
            return entry.astype(float).rolling(hold, min_periods=1).max().fillna(0)
        S.append(({"id": "E1", "name": "ATR Breakout", "category": "E-Volatility",
                   "variant": f"{k}xATR/hold{hold}"}, fn))

    # E2: ATR trailing stop (entry 20d breakout, exit below highest-close-since-entry - m*ATR)
    for m in [3.0, 2.0]:
        def fn(ctx, m=m):
            c = ctx["close"]
            a = ta.atr(ctx["high"], ctx["low"], c, 14).to_numpy(float)
            cv = c.to_numpy(float)
            hi20 = c.shift(1).rolling(20).max().to_numpy(float)
            T, N = cv.shape
            pos = np.zeros((T, N))
            hh = np.full(N, np.nan)
            inpos = np.zeros(N, dtype=bool)
            for t in range(1, T):
                entry = ~inpos & (cv[t] > hi20[t])
                inpos = inpos | entry
                hh = np.where(entry, cv[t], np.maximum(hh, cv[t]))
                stop = hh - m * a[t]
                ex = inpos & (cv[t] < stop)
                inpos = inpos & ~ex
                hh = np.where(inpos, hh, np.nan)
                pos[t] = np.where(np.isnan(cv[t]), 0, inpos.astype(float))
            return pd.DataFrame(pos, index=c.index, columns=c.columns)
        S.append(({"id": "E2", "name": "ATR Trailing Stop", "category": "E-Volatility",
                   "variant": f"{m}xATR trail"}, fn))

    # E3: Bollinger bandwidth expansion with direction
    def fn_e3(ctx):
        c = ctx["close"]
        mid, ub, lb, bw = ta.bollinger(c, 20, 2.0)
        return _long((bw > bw.shift(5)) & (c > mid))
    S.append(({"id": "E3", "name": "BB Width Expansion", "category": "E-Volatility",
               "variant": "bw rising"}, fn_e3))

    # E4: Implied (India VIX) vs realized vol premium
    def fn_e4(ctx):
        c = ctx["close"]
        vix = ctx["vix"].reindex(c.index).ffill()
        bench = ctx["benchmark"]
        rv = bench.pct_change().rolling(20).std() * np.sqrt(252) * 100
        rich = (vix > 1.15 * rv)  # implied rich vs realized -> harvest premium, long equity
        return _all_stock_basket(c).mul(rich.astype(float), axis=0)
    S.append(({"id": "E4", "name": "VIX vs Realized Vol", "category": "E-Volatility",
               "variant": "IV>1.15xRV"}, fn_e4))

    # E5: GARCH(1,1) vol targeting on equal-weight basket
    def fn_e5(ctx):
        from arch import arch_model
        bench = ctx["benchmark"]
        ret = bench.pct_change().dropna() * 100
        c = ctx["close"]
        fc = pd.Series(np.nan, index=ret.index)
        step = 60
        for i in range(750, len(ret), step):
            train = ret.iloc[i - 750:i]
            try:
                am = arch_model(train, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                f = res.forecast(horizon=step, reindex=False)
                sig = np.sqrt(f.variance.values[0])
                end = min(i + step, len(ret))
                fc.iloc[i:end] = sig[: end - i]
            except Exception:
                continue
        ann_fc = fc.reindex(c.index).ffill() * np.sqrt(252) / 100
        target = 0.15
        scale = (target / ann_fc).clip(0, 1.0)
        basket = _all_stock_basket(c)
        n = basket.sum(axis=1).clip(lower=1)
        return basket.div(n, axis=0).mul(scale.fillna(0), axis=0)
    S.append(({"id": "E5", "name": "GARCH Vol Targeting", "category": "E-Volatility",
               "variant": "target15%", "signals_are_weights": True}, fn_e5))

    # E6: Volatility clustering — crisis reversal (big down day in high-vol regime)
    def fn_e6(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        vol20 = ret.rolling(20).std()
        highvol = vol20 > vol20.rolling(250).quantile(0.9)
        bigdown = ret < -2 * vol20
        entry = highvol & bigdown
        return entry.astype(float).rolling(5, min_periods=1).max().fillna(0)
    S.append(({"id": "E6", "name": "Vol Clustering Reversal", "category": "E-Volatility",
               "variant": "2sd down/hold5"}, fn_e6))

    # E7: India VIX spike mean reversion (buy fear)
    for z_thr, hold in [(2.0, 10), (1.5, 20)]:
        def fn(ctx, z_thr=z_thr, hold=hold):
            c = ctx["close"]
            vix = ctx["vix"].reindex(c.index).ffill()
            z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
            spike = (z > z_thr).astype(float).rolling(hold, min_periods=1).max().fillna(0)
            return _all_stock_basket(c).mul(spike, axis=0)
        S.append(({"id": "E7", "name": "India VIX Spike Buy", "category": "E-Volatility",
                   "variant": f"z>{z_thr}/hold{hold}"}, fn))

    # E8: Variance-ratio regime switch (trend when VR>1, revert when VR<1)
    def fn_e8(ctx):
        c = ctx["close"]
        ret = np.log(c).diff()
        q = 5
        var1 = ret.rolling(250).var()
        varq = ret.rolling(q).sum().rolling(250).var()
        vr = varq / (q * var1)
        mom_sig = (c > c.shift(20)).astype(float)
        z = ta.zscore(c, 20)
        rev_sig = _state(z < -1.5, z > 0, c.index, c.columns)
        return mom_sig.where(vr > 1.05, rev_sig.where(vr < 0.95, 0.0))
    S.append(({"id": "E8", "name": "Variance Ratio Regime", "category": "E-Volatility",
               "variant": "VR(5,250)"}, fn_e8))

    return S
