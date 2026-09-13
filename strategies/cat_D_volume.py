"""Category D: Volume-based strategies (D1-D12)."""
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


def get_strategies():
    S = []

    # D1: OBV trend confirmation
    def fn_d1(ctx):
        c, v = ctx["close"], ctx["volume"]
        o = ta.obv(c, v)
        return _long((o > ta.sma(o, 20)) & (c > ta.sma(c, 20)))
    S.append(({"id": "D1", "name": "OBV Trend", "category": "D-Volume",
               "variant": "OBV>SMA20"}, fn_d1))

    # D2: OBV bullish divergence, hold 15
    def fn_d2(ctx):
        c, v = ctx["close"], ctx["volume"]
        o = ta.obv(c, v)
        price_ll = c.rolling(5).min() < c.shift(5).rolling(20).min()
        obv_hl = o.rolling(5).min() > o.shift(5).rolling(20).min()
        entry = price_ll & obv_hl
        return entry.astype(float).rolling(15, min_periods=1).max().fillna(0)
    S.append(({"id": "D2", "name": "OBV Divergence", "category": "D-Volume",
               "variant": "hold15"}, fn_d2))

    # D3: Rolling VWAP reversion (daily proxy for intraday VWAP; noted limitation)
    for n, dev in [(20, 0.03), (10, 0.02)]:
        def fn(ctx, n=n, dev=dev):
            c, v = ctx["close"], ctx["volume"]
            tp = (ctx["high"] + ctx["low"] + c) / 3
            vwap = (tp * v).rolling(n).sum() / v.rolling(n).sum()
            return _state(c < vwap * (1 - dev), c >= vwap, c.index, c.columns)
        S.append(({"id": "D3", "name": "VWAP Reversion (daily proxy)", "category": "D-Volume",
                   "variant": f"VWAP{n}/{int(dev*100)}%"}, fn))

    # D4: A/D line breakout
    def fn_d4(ctx):
        ad = ta.ad_line(ctx["high"], ctx["low"], ctx["close"], ctx["volume"])
        entry = ad > ad.shift(1).rolling(50).max()
        exit_ = ad < ad.shift(1).rolling(20).min()
        return _state(entry, exit_, ad.index, ad.columns)
    S.append(({"id": "D4", "name": "A/D Line Breakout", "category": "D-Volume",
               "variant": "50d high"}, fn_d4))

    # D5: Chaikin Money Flow
    for thr in [0.05, 0.10]:
        def fn(ctx, thr=thr):
            m = ta.cmf(ctx["high"], ctx["low"], ctx["close"], ctx["volume"], 20)
            return _state(m > thr, m < 0, m.index, m.columns)
        S.append(({"id": "D5", "name": "Chaikin Money Flow", "category": "D-Volume",
                   "variant": f"CMF>{thr}"}, fn))

    # D6: MFI reversion
    for lo, ex in [(20, 50), (25, 60)]:
        def fn(ctx, lo=lo, ex=ex):
            m = ta.mfi(ctx["high"], ctx["low"], ctx["close"], ctx["volume"], 14)
            return _state(m < lo, m > ex, m.index, m.columns)
        S.append(({"id": "D6", "name": "MFI Reversion", "category": "D-Volume",
                   "variant": f"MFI<{lo}"}, fn))

    # D7: Ease of Movement
    def fn_d7(ctx):
        e = ta.emv(ctx["high"], ctx["low"], ctx["volume"], 14)
        return _long(e > 0)
    S.append(({"id": "D7", "name": "Ease of Movement", "category": "D-Volume",
               "variant": "EMV14>0"}, fn_d7))

    # D8: Volume Point of Control (rolling 60d, 20 bins, evaluated every 5 bars)
    def fn_d8(ctx):
        c, v = ctx["close"], ctx["volume"]
        window, nbins, step = 60, 20, 5
        cv = c.to_numpy(float)
        vv = v.to_numpy(float)
        T, N = cv.shape
        vpoc = np.full((T, N), np.nan)
        for t in range(window, T, step):
            cs = cv[t - window:t]
            vs = vv[t - window:t]
            for j in range(N):
                col = cs[:, j]
                vol = vs[:, j]
                m = ~np.isnan(col) & ~np.isnan(vol)
                if m.sum() < 30:
                    continue
                hist, edges = np.histogram(col[m], bins=nbins, weights=vol[m])
                k = int(np.argmax(hist))
                vpoc[t, j] = (edges[k] + edges[k + 1]) / 2
        vp = pd.DataFrame(vpoc, index=c.index, columns=c.columns).ffill()
        return _state(c > vp * 1.02, c < vp * 0.98, c.index, c.columns)
    S.append(({"id": "D8", "name": "Volume Profile VPOC", "category": "D-Volume",
               "variant": "60d/20bin"}, fn_d8))

    # D9: Klinger oscillator
    def fn_d9(ctx):
        ko, sig = ta.klinger(ctx["high"], ctx["low"], ctx["close"], ctx["volume"])
        return _long(ko > sig)
    S.append(({"id": "D9", "name": "Klinger Oscillator", "category": "D-Volume",
               "variant": "34/55/13"}, fn_d9))

    # D10: Negative Volume Index (Fosback rule)
    def fn_d10(ctx):
        n = ta.nvi(ctx["close"], ctx["volume"])
        return _long(n > n.ewm(span=255, adjust=False).mean())
    S.append(({"id": "D10", "name": "Negative Volume Index", "category": "D-Volume",
               "variant": "NVI>EMA255"}, fn_d10))

    # D11: Volume spike + breakout
    for vmult in [2.0, 3.0]:
        def fn(ctx, vmult=vmult):
            c, v = ctx["close"], ctx["volume"]
            spike = v > vmult * v.rolling(20).mean()
            brk = c > c.shift(1).rolling(20).max()
            entry = spike & brk
            exit_ = c < c.shift(1).rolling(10).min()
            return _state(entry, exit_, c.index, c.columns)
        S.append(({"id": "D11", "name": "Volume Spike Breakout", "category": "D-Volume",
                   "variant": f"{vmult}x vol"}, fn))

    # D12: Volume Spread Analysis (simplified Wyckoff)
    def fn_d12(ctx):
        o, h, l, c, v = ctx["open"], ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
        spread = h - l
        avg_spread = spread.rolling(20).mean()
        avg_vol = v.rolling(20).mean()
        uptrend = c > ta.sma(c, 50)
        # "no supply": narrow spread, low volume, down bar in an uptrend
        no_supply = uptrend & (c < o) & (spread < 0.7 * avg_spread) & (v < 0.7 * avg_vol)
        # "upthrust": wide spread, high vol, close in bottom third of range
        upthrust = (spread > 1.5 * avg_spread) & (v > 1.5 * avg_vol) & \
                   ((c - l) < 0.33 * spread.replace(0, np.nan))
        return _state(no_supply, upthrust | ~uptrend, c.index, c.columns)
    S.append(({"id": "D12", "name": "VSA (Wyckoff)", "category": "D-Volume",
               "variant": "no-supply"}, fn_d12))

    return S
