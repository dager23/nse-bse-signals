"""Category M: systematic parameter megagrids over classic signal families.

Every credible parameterization of the core signal types, generated
programmatically.  ~230 variants.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.composite import state, hold_n, monthly_topn, cross_up


def get_strategies():
    S = []

    # ---- M1: MA crossover megagrid: 5 MA types x 8 (fast,slow) pairs = 40
    MA_FN = {"sma": ta.sma, "ema": ta.ema, "wma": ta.wma, "dema": ta.dema, "hma": ta.hma}
    PAIRS = [(5, 20), (10, 30), (10, 50), (20, 50), (20, 100), (50, 100),
             (50, 200), (100, 200)]
    for mtype in MA_FN:
        for f, s_ in PAIRS:
            def fn(ctx, mtype=mtype, f=f, s_=s_):
                c = ctx["close"]
                ma = MA_FN[mtype]
                return (ma(c, f) > ma(c, s_)).astype(float)
            S.append(({"id": "M1", "name": "MA Cross Grid", "category": "M-Grids",
                       "variant": f"{mtype}{f}/{s_}"}, fn))

    # ---- M2: price vs single MA: 2 types x 6 lengths = 12
    for mtype in ["sma", "ema"]:
        for n in [10, 20, 50, 100, 150, 200]:
            def fn(ctx, mtype=mtype, n=n):
                c = ctx["close"]
                return (c > MA_FN[mtype](c, n)).astype(float)
            S.append(({"id": "M2", "name": "Price vs MA Grid", "category": "M-Grids",
                       "variant": f">{mtype}{n}"}, fn))

    # ---- M3: RSI zone grid: reversion 15 + momentum 3 = 18
    for n in [2, 5, 14]:
        for lo, hi in [(10, 50), (20, 55), (25, 60), (30, 70), (40, 60)]:
            def fn(ctx, n=n, lo=lo, hi=hi):
                c = ctx["close"]
                r = ta.rsi(c, n)
                return state(r < lo, r > hi, c.index, c.columns)
            S.append(({"id": "M3", "name": "RSI Zone Grid", "category": "M-Grids",
                       "variant": f"rsi{n} {lo}/{hi}"}, fn))
    for thr in [55, 60, 65]:
        def fn(ctx, thr=thr):
            c = ctx["close"]
            r = ta.rsi(c, 14)
            return state(cross_up(r, pd.DataFrame(thr, index=c.index, columns=c.columns)),
                         r < 45, c.index, c.columns)
        S.append(({"id": "M3", "name": "RSI Momentum Grid", "category": "M-Grids",
                   "variant": f"rsi14>{thr} mom"}, fn))

    # ---- M4: Donchian entry/exit grid: 4x4 = 16
    for ne in [10, 20, 55, 100]:
        for nx in [5, 10, 20, 50]:
            def fn(ctx, ne=ne, nx=nx):
                h, l, c = ctx["high"], ctx["low"], ctx["close"]
                hh = h.rolling(ne).max().shift(1)
                ll = l.rolling(nx).min().shift(1)
                return state(c > hh, c < ll, c.index, c.columns)
            S.append(({"id": "M4", "name": "Donchian Grid", "category": "M-Grids",
                       "variant": f"in{ne}/out{nx}"}, fn))

    # ---- M5: Bollinger grids: bounce 12 + breakout 12 = 24
    for n in [10, 20, 50]:
        for k in [1.5, 2.0, 2.5, 3.0]:
            def fn_b(ctx, n=n, k=k):
                c = ctx["close"]
                mid, up, lo_, _ = ta.bollinger(c, n, k)
                return state(c < lo_, c > mid, c.index, c.columns)
            S.append(({"id": "M5", "name": "BB Bounce Grid", "category": "M-Grids",
                       "variant": f"bounce n{n}k{k}"}, fn_b))

            def fn_k(ctx, n=n, k=k):
                c = ctx["close"]
                mid, up, lo_, _ = ta.bollinger(c, n, k)
                return state(c > up, c < mid, c.index, c.columns)
            S.append(({"id": "M5", "name": "BB Breakout Grid", "category": "M-Grids",
                       "variant": f"break n{n}k{k}"}, fn_k))

    # ---- M6: ROC threshold grid: 6 lookbacks x 3 thresholds = 18
    for n in [5, 10, 21, 63, 126, 252]:
        for thr in [0.0, 2.0, 5.0]:
            def fn(ctx, n=n, thr=thr):
                c = ctx["close"]
                r = ta.roc(c, n)
                return state(r > thr, r < 0, c.index, c.columns)
            S.append(({"id": "M6", "name": "ROC Grid", "category": "M-Grids",
                       "variant": f"roc{n}>{thr}"}, fn))

    # ---- M7: Stochastic zone grid: 3x3 = 9
    for kn in [5, 14, 21]:
        for lo, hi in [(20, 80), (10, 90), (30, 70)]:
            def fn(ctx, kn=kn, lo=lo, hi=hi):
                h, l, c = ctx["high"], ctx["low"], ctx["close"]
                k, d = ta.stochastic(h, l, c, kn)
                return state((k < lo) & (k > d), k > hi, c.index, c.columns)
            S.append(({"id": "M7", "name": "Stochastic Grid", "category": "M-Grids",
                       "variant": f"k{kn} {lo}/{hi}"}, fn))

    # ---- M8: Keltner grids: 2x3 breakout + 2x3 bounce = 12
    for n in [20, 50]:
        for k in [1.5, 2.0, 2.5]:
            def fn_a(ctx, n=n, k=k):
                h, l, c = ctx["high"], ctx["low"], ctx["close"]
                mid, up, lo_ = ta.keltner(h, l, c, n, k)
                return state(c > up, c < mid, c.index, c.columns)
            S.append(({"id": "M8", "name": "Keltner Break Grid", "category": "M-Grids",
                       "variant": f"break n{n}k{k}"}, fn_a))

            def fn_b(ctx, n=n, k=k):
                h, l, c = ctx["high"], ctx["low"], ctx["close"]
                mid, up, lo_ = ta.keltner(h, l, c, n, k)
                return state(c < lo_, c > mid, c.index, c.columns)
            S.append(({"id": "M8", "name": "Keltner Bounce Grid", "category": "M-Grids",
                       "variant": f"bounce n{n}k{k}"}, fn_b))

    # ---- M9: MACD parameter grid: 4 param sets x 2 rules = 8
    for f, sl, sg in [(8, 17, 9), (12, 26, 9), (5, 35, 5), (19, 39, 9)]:
        def fn_c(ctx, f=f, sl=sl, sg=sg):
            c = ctx["close"]
            line, sig, _ = ta.macd(c, f, sl, sg)
            return (line > sig).astype(float)
        S.append(({"id": "M9", "name": "MACD Grid", "category": "M-Grids",
                   "variant": f"cross {f}/{sl}/{sg}"}, fn_c))

        def fn_z(ctx, f=f, sl=sl, sg=sg):
            c = ctx["close"]
            line, sig, _ = ta.macd(c, f, sl, sg)
            return (line > 0).astype(float)
        S.append(({"id": "M9", "name": "MACD Zero Grid", "category": "M-Grids",
                   "variant": f"zero {f}/{sl}/{sg}"}, fn_z))

    # ---- M10: ATR breakout grid: 5 mults x 2 lookbacks = 10
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for lb in [5, 20]:
            def fn(ctx, k=k, lb=lb):
                h, l, c = ctx["high"], ctx["low"], ctx["close"]
                a = ta.atr(h, l, c, 14)
                ref = c.shift(lb)
                entry = c > ref + k * a
                exit_ = c < c.rolling(10).max() - 2 * a
                return state(entry, exit_, c.index, c.columns)
            S.append(({"id": "M10", "name": "ATR Breakout Grid", "category": "M-Grids",
                       "variant": f"{k}atr/{lb}d"}, fn))

    # ---- M11: 52w-high proximity grid: 4 entries x 2 exits = 8
    for prox in [0.0, 0.02, 0.05, 0.10]:
        for ex in ["dd20", "ma50"]:
            def fn(ctx, prox=prox, ex=ex):
                c = ctx["close"]
                hi = c.rolling(252).max()
                entry = c >= hi * (1 - prox)
                if ex == "dd20":
                    exit_ = c < hi * 0.80
                else:
                    exit_ = c < ta.sma(c, 50)
                return state(entry, exit_, c.index, c.columns)
            S.append(({"id": "M11", "name": "52wH Grid", "category": "M-Grids",
                       "variant": f"within{int(prox*100)}%/{ex}"}, fn))

    # ---- M12: vol-scaled cross-sectional momentum: 4 lb x 2 norm x 3 topk = 24
    for lb in [21, 63, 126, 252]:
        for norm in [False, True]:
            for k in [5, 10, 15]:
                def fn(ctx, lb=lb, norm=norm, k=k):
                    c = ctx["close"]
                    mom = c.pct_change(lb, fill_method=None)
                    if norm:
                        vol = c.pct_change(fill_method=None).rolling(lb).std()
                        mom = mom / vol.replace(0, np.nan)
                    return monthly_topn(c, mom, k)
                S.append(({"id": "M12", "name": "XS Momentum Grid", "category": "M-Grids",
                           "variant": f"lb{lb}{'v' if norm else ''}top{k}"}, fn))

    # ---- M13: fixed-holding sweep on 3 entry types x 4 holds = 12
    ENTRIES = {
        "goldcross": lambda ctx: cross_up(ta.sma(ctx["close"], 50), ta.sma(ctx["close"], 200)),
        "donch55": lambda ctx: ctx["close"] > ctx["high"].rolling(55).max().shift(1),
        "roc63": lambda ctx: cross_up(ta.roc(ctx["close"], 63),
                                      pd.DataFrame(0.0, index=ctx["close"].index,
                                                   columns=ctx["close"].columns)),
    }
    for ename, efn in ENTRIES.items():
        for hp in [5, 10, 21, 63]:
            def fn(ctx, efn=efn, hp=hp):
                return hold_n(efn(ctx).fillna(False), hp)
            S.append(({"id": "M13", "name": "Fixed-Hold Sweep", "category": "M-Grids",
                       "variant": f"{ename} h{hp}"}, fn))

    return S
