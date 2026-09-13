"""Category A: Trend Following (A1-A20)."""
import numpy as np
import pandas as pd

from utils import indicators as ta


def _long(cond):
    return cond.astype(float)


def _ls(cond_long, cond_short=None):
    """long/short state: +1 when long cond, -1 when short cond (else carry 0)."""
    if cond_short is None:
        return cond_long.astype(float) * 2 - 1
    return cond_long.astype(float) - cond_short.astype(float)


def get_strategies():
    S = []

    # A1: SMA crossover
    for fast, slow in [(20, 50), (50, 200), (10, 30)]:
        def fn(ctx, fast=fast, slow=slow):
            c = ctx["close"]
            return _long(ta.sma(c, fast) > ta.sma(c, slow))
        S.append(({"id": "A1", "name": "SMA Crossover", "category": "A-Trend",
                   "variant": f"SMA{fast}/{slow}"}, fn))

    # A2: EMA crossover
    for fast, slow in [(12, 26), (20, 50), (8, 21)]:
        def fn(ctx, fast=fast, slow=slow):
            c = ctx["close"]
            return _long(ta.ema(c, fast) > ta.ema(c, slow))
        S.append(({"id": "A2", "name": "EMA Crossover", "category": "A-Trend",
                   "variant": f"EMA{fast}/{slow}"}, fn))

    # A3: Triple EMA system 5/13/26
    for a, b, c_ in [(5, 13, 26), (8, 21, 55)]:
        def fn(ctx, a=a, b=b, c_=c_):
            c = ctx["close"]
            e1, e2, e3 = ta.ema(c, a), ta.ema(c, b), ta.ema(c, c_)
            return _long((e1 > e2) & (e2 > e3))
        S.append(({"id": "A3", "name": "Triple EMA", "category": "A-Trend",
                   "variant": f"TEMAx{a}/{b}/{c_}"}, fn))

    # A4: DEMA crossover
    for fast, slow in [(10, 30), (20, 50)]:
        def fn(ctx, fast=fast, slow=slow):
            c = ctx["close"]
            return _long(ta.dema(c, fast) > ta.dema(c, slow))
        S.append(({"id": "A4", "name": "DEMA Crossover", "category": "A-Trend",
                   "variant": f"DEMA{fast}/{slow}"}, fn))

    # A5: TEMA crossover
    for fast, slow in [(10, 30), (20, 50)]:
        def fn(ctx, fast=fast, slow=slow):
            c = ctx["close"]
            return _long(ta.tema(c, fast) > ta.tema(c, slow))
        S.append(({"id": "A5", "name": "TEMA Crossover", "category": "A-Trend",
                   "variant": f"TEMA{fast}/{slow}"}, fn))

    # A6: Hull MA direction change
    for n in [21, 50]:
        def fn(ctx, n=n):
            h = ta.hma(ctx["close"], n)
            return _long(h > h.shift(1))
        S.append(({"id": "A6", "name": "Hull MA Direction", "category": "A-Trend",
                   "variant": f"HMA{n}"}, fn))

    # A7: KAMA direction
    for er in [10, 21]:
        def fn(ctx, er=er):
            k = ta.kama(ctx["close"], er_n=er)
            return _long(ctx["close"] > k)
        S.append(({"id": "A7", "name": "KAMA Trend", "category": "A-Trend",
                   "variant": f"KAMA{er}"}, fn))

    # A8: VWMA crossover
    for fast, slow in [(10, 30), (20, 60)]:
        def fn(ctx, fast=fast, slow=slow):
            return _long(ta.vwma(ctx["close"], ctx["volume"], fast) >
                         ta.vwma(ctx["close"], ctx["volume"], slow))
        S.append(({"id": "A8", "name": "VWMA Crossover", "category": "A-Trend",
                   "variant": f"VWMA{fast}/{slow}"}, fn))

    # A9: Supertrend
    for n, m in [(10, 3.0), (7, 2.0), (14, 2.5)]:
        def fn(ctx, n=n, m=m):
            d = ta.supertrend(ctx["high"], ctx["low"], ctx["close"], n, m)
            return _long(d > 0)
        S.append(({"id": "A9", "name": "Supertrend", "category": "A-Trend",
                   "variant": f"ST({n},{m})"}, fn))

    # A10: Parabolic SAR
    for af, mx in [(0.02, 0.2), (0.01, 0.1)]:
        def fn(ctx, af=af, mx=mx):
            d = ta.psar(ctx["high"], ctx["low"], ctx["close"], af0=af, af_step=af, af_max=mx)
            return _long(d > 0)
        S.append(({"id": "A10", "name": "Parabolic SAR", "category": "A-Trend",
                   "variant": f"PSAR({af},{mx})"}, fn))

    # A11: Ichimoku full system
    def fn_a11(ctx):
        h, l, c = ctx["high"], ctx["low"], ctx["close"]
        tenkan, kijun, sa, sb = ta.ichimoku(h, l, c)
        cloud_top = pd.concat([sa, sb]).groupby(level=0).max().reindex(c.index)
        return _long((tenkan > kijun) & (c > cloud_top))
    S.append(({"id": "A11", "name": "Ichimoku Cloud", "category": "A-Trend",
               "variant": "9/26/52"}, fn_a11))

    # A12: ADX filter + DI direction
    for thr in [20, 25]:
        def fn(ctx, thr=thr):
            a, pdi, mdi = ta.adx(ctx["high"], ctx["low"], ctx["close"], 14)
            return _long((a > thr) & (pdi > mdi))
        S.append(({"id": "A12", "name": "ADX Trend Filter", "category": "A-Trend",
                   "variant": f"ADX>{thr}"}, fn))

    # A13: Aroon crossover
    for n in [25, 14]:
        def fn(ctx, n=n):
            up, dn = ta.aroon(ctx["high"], ctx["low"], n)
            return _long((up > 70) & (up > dn))
        S.append(({"id": "A13", "name": "Aroon Oscillator", "category": "A-Trend",
                   "variant": f"Aroon{n}"}, fn))

    # A14: Chandelier exit (long when close above chandelier stop)
    for n, m in [(22, 3.0), (22, 2.0)]:
        def fn(ctx, n=n, m=m):
            h, l, c = ctx["high"], ctx["low"], ctx["close"]
            stop = h.rolling(n).max() - m * ta.atr(h, l, c, n)
            # enter on 20d high breakout, exit when close < chandelier stop
            entry = c > c.shift(1).rolling(20).max()
            state = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
            state[entry] = 1.0
            state[c < stop] = 0.0
            return state.ffill().fillna(0)
        S.append(({"id": "A14", "name": "Chandelier Exit", "category": "A-Trend",
                   "variant": f"CE({n},{m})"}, fn))

    # A15: Donchian breakout (Turtle)
    for n_in, n_out in [(20, 10), (55, 20)]:
        def fn(ctx, n_in=n_in, n_out=n_out):
            c = ctx["close"]
            hi = ctx["high"].shift(1).rolling(n_in).max()
            lo = ctx["low"].shift(1).rolling(n_out).min()
            state = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
            state[c > hi] = 1.0
            state[c < lo] = 0.0
            return state.ffill().fillna(0)
        S.append(({"id": "A15", "name": "Donchian Breakout", "category": "A-Trend",
                   "variant": f"DC{n_in}/{n_out}"}, fn))

    # A16: Keltner channel breakout
    for k in [2.0, 1.5]:
        def fn(ctx, k=k):
            mid, ub, lb = ta.keltner(ctx["high"], ctx["low"], ctx["close"], 20, k)
            c = ctx["close"]
            state = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
            state[c > ub] = 1.0
            state[c < mid] = 0.0
            return state.ffill().fillna(0)
        S.append(({"id": "A16", "name": "Keltner Breakout", "category": "A-Trend",
                   "variant": f"KC(20,{k})"}, fn))

    # A17: Linear regression channel breakout
    def fn_a17(ctx):
        c = ctx["close"]
        ep, ub, lb = ta.linreg_channel(c, 100, 2.0)
        state = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        state[c > ub] = 1.0
        state[c < ep] = 0.0
        return state.ffill().fillna(0)
    S.append(({"id": "A17", "name": "LinReg Channel", "category": "A-Trend",
               "variant": "LR(100,2)"}, fn_a17))

    # A18: McGinley dynamic
    for n in [14, 30]:
        def fn(ctx, n=n):
            md = ta.mcginley(ctx["close"], n)
            return _long(ctx["close"] > md)
        S.append(({"id": "A18", "name": "McGinley Dynamic", "category": "A-Trend",
                   "variant": f"MD{n}"}, fn))

    # A19: Elder's Triple Screen (weekly trend via 65d EMA slope + daily force index dip)
    def fn_a19(ctx):
        c, v = ctx["close"], ctx["volume"]
        weekly_trend = ta.ema(c, 65)
        screen1 = weekly_trend > weekly_trend.shift(5)
        fi2 = ta.force_index(c, v, 2)
        screen2 = fi2 < 0  # pullback in uptrend
        entry = screen1 & screen2
        exit_ = ~screen1
        state = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        state[entry] = 1.0
        state[exit_] = 0.0
        return state.ffill().fillna(0)
    S.append(({"id": "A19", "name": "Elder Triple Screen", "category": "A-Trend",
               "variant": "EMA65+FI2"}, fn_a19))

    # A20: Heikin-Ashi color change
    for smooth in [1, 3]:
        def fn(ctx, smooth=smooth):
            ho, hc = ta.heikin_ashi(ctx["open"], ctx["high"], ctx["low"], ctx["close"])
            green = hc > ho
            if smooth > 1:
                green = green.rolling(smooth).min().astype(bool)  # N consecutive green
            return _long(green)
        S.append(({"id": "A20", "name": "Heikin-Ashi Trend", "category": "A-Trend",
                   "variant": f"HA{smooth}"}, fn))

    return S
