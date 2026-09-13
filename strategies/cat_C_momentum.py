"""Category C: Momentum (C1-C14)."""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.sectors import SECTOR_MAP


def _long(cond):
    return cond.astype(float)


def _monthly_top_n(score, close, top_n, abs_filter=None):
    """Rank on last trading day of each month, hold top_n through next month."""
    month_end = pd.Series(close.index, index=close.index).groupby(
        [close.index.year, close.index.month]).transform("max") == close.index
    pos = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    ranks = score[month_end].rank(axis=1, ascending=False)
    sel = (ranks <= top_n).astype(float)
    if abs_filter is not None:
        sel = sel.where(abs_filter[month_end] > 0, 0.0)
    pos[month_end] = sel
    return pos.ffill().fillna(0)


def get_strategies():
    S = []

    # C1: MACD crossover
    for f, s_, g in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
        def fn(ctx, f=f, s_=s_, g=g):
            line, sig, hist = ta.macd(ctx["close"], f, s_, g)
            return _long(line > sig)
        S.append(({"id": "C1", "name": "MACD Crossover", "category": "C-Momentum",
                   "variant": f"MACD({f},{s_},{g})"}, fn))

    # C2: MACD histogram divergence (hist higher low, price lower low), hold 15d
    def fn_c2(ctx):
        c = ctx["close"]
        _, _, hist = ta.macd(c)
        price_ll = c.rolling(5).min() < c.shift(5).rolling(20).min()
        hist_hl = hist.rolling(5).min() > hist.shift(5).rolling(20).min()
        entry = price_ll & hist_hl & (hist < 0)
        return entry.astype(float).rolling(15, min_periods=1).max().fillna(0)
    S.append(({"id": "C2", "name": "MACD Hist Divergence", "category": "C-Momentum",
               "variant": "hold15"}, fn_c2))

    # C3: Rate of change threshold
    for n, thr in [(20, 0), (63, 5), (126, 10)]:
        def fn(ctx, n=n, thr=thr):
            return _long(ta.roc(ctx["close"], n) > thr)
        S.append(({"id": "C3", "name": "ROC Threshold", "category": "C-Momentum",
                   "variant": f"ROC{n}>{thr}"}, fn))

    # C4: Raw momentum
    for n in [10, 20]:
        def fn(ctx, n=n):
            c = ctx["close"]
            return _long(c > c.shift(n))
        S.append(({"id": "C4", "name": "Momentum", "category": "C-Momentum",
                   "variant": f"MOM{n}"}, fn))

    # C5: Sector rotation (top 2 sectors by 3m return, monthly)
    def fn_c5(ctx):
        c = ctx["close"]
        ret3m = c.pct_change(63, fill_method=None)
        sec = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in c.columns})
        sec_ret = ret3m.T.groupby(sec).mean().T
        month_end = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        top_sec = sec_ret[month_end].rank(axis=1, ascending=False) <= 2
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        sel = pd.DataFrame(0.0, index=top_sec.index, columns=c.columns)
        for t in c.columns:
            s = sec[t]
            if s in top_sec.columns:
                sel[t] = top_sec[s].astype(float)
        pos[month_end] = sel
        return pos.ffill().fillna(0)
    S.append(({"id": "C5", "name": "Sector Rotation", "category": "C-Momentum",
               "variant": "top2sec/3m"}, fn_c5))

    # C6: 52-week high proximity
    for near, far in [(0.05, 0.15), (0.02, 0.10)]:
        def fn(ctx, near=near, far=far):
            c = ctx["close"]
            hi52 = c.rolling(252).max()
            st = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
            st[c >= hi52 * (1 - near)] = 1.0
            st[c < hi52 * (1 - far)] = 0.0
            return st.ffill().fillna(0)
        S.append(({"id": "C6", "name": "52w High Momentum", "category": "C-Momentum",
                   "variant": f"within{int(near*100)}%"}, fn))

    # C7: Jegadeesh-Titman 12-1 momentum, top 10, monthly
    for top_n in [5, 10]:
        def fn(ctx, top_n=top_n):
            c = ctx["close"]
            mom = c.shift(21) / c.shift(252) - 1  # 12m return skipping last month
            return _monthly_top_n(mom, c, top_n)
        S.append(({"id": "C7", "name": "Jegadeesh-Titman 12-1", "category": "C-Momentum",
                   "variant": f"top{top_n}"}, fn))

    # C8: Dual momentum (relative top N + absolute > 0)
    for top_n in [5, 10]:
        def fn(ctx, top_n=top_n):
            c = ctx["close"]
            mom = c.pct_change(252, fill_method=None)
            return _monthly_top_n(mom, c, top_n, abs_filter=mom)
        S.append(({"id": "C8", "name": "Dual Momentum", "category": "C-Momentum",
                   "variant": f"top{top_n}/abs"}, fn))

    # C9: Faber TAA — hold stock when above its 10-month (210d) SMA, monthly check
    def fn_c9(ctx):
        c = ctx["close"]
        sma10m = ta.sma(c, 210)
        month_end = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        pos[month_end] = (c > sma10m)[month_end].astype(float)
        return pos.ffill().fillna(0)
    S.append(({"id": "C9", "name": "Faber TAA 10m SMA", "category": "C-Momentum",
               "variant": "10mSMA"}, fn_c9))

    # C10: Acceleration (momentum of momentum)
    def fn_c10(ctx):
        c = ctx["close"]
        r = ta.roc(c, 20)
        return _long((r > 0) & (r > r.shift(20)))
    S.append(({"id": "C10", "name": "Acceleration Factor", "category": "C-Momentum",
               "variant": "ROC20 accel"}, fn_c10))

    # C11: KST crossover
    def fn_c11(ctx):
        line, sig = ta.kst(ctx["close"])
        return _long(line > sig)
    S.append(({"id": "C11", "name": "KST Oscillator", "category": "C-Momentum",
               "variant": "Pring std"}, fn_c11))

    # C12: Coppock curve (monthly interval, classic buy signal)
    def fn_c12(ctx):
        c = ctx["close"]  # monthly bars
        cop = ta.coppock(c, wl=10, r1=14, r2=11)
        entry = (cop < 0) & (cop > cop.shift(1)) & (cop.shift(1) <= cop.shift(2))
        st = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        st[entry] = 1.0
        st[(cop > 0) & (cop < cop.shift(1))] = 0.0
        return st.ffill().fillna(0)
    S.append(({"id": "C12", "name": "Coppock Curve", "category": "C-Momentum",
               "variant": "monthly", "interval": "1mo"}, fn_c12))

    # C13: Elder Force Index
    for n in [13, 34]:
        def fn(ctx, n=n):
            return _long(ta.force_index(ctx["close"], ctx["volume"], n) > 0)
        S.append(({"id": "C13", "name": "Force Index", "category": "C-Momentum",
                   "variant": f"FI{n}"}, fn))

    # C14: TSI crossover
    def fn_c14(ctx):
        t = ta.tsi(ctx["close"])
        sig = t.ewm(span=13, adjust=False).mean()
        return _long(t > sig)
    S.append(({"id": "C14", "name": "True Strength Index", "category": "C-Momentum",
               "variant": "25/13"}, fn_c14))

    return S
