"""Category F: Pattern Recognition (F1-F13).

Candlestick patterns are vectorized; chart patterns (H&S, double bottom,
cup & handle) use pivot-point scans per column.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from utils import indicators as ta


def _hold_n(entry, n):
    return entry.astype(float).rolling(n, min_periods=1).max().fillna(0)


def _state(entry, exit_, index, columns):
    st = pd.DataFrame(np.nan, index=index, columns=columns)
    st[entry] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def _pivots(series, order=5):
    """Indices of pivot highs/lows for one clean numpy series."""
    hi = argrelextrema(series, np.greater_equal, order=order)[0]
    lo = argrelextrema(series, np.less_equal, order=order)[0]
    return hi, lo


def get_strategies():
    S = []

    # F1: Doji + bullish confirmation
    def fn_f1(ctx):
        o, h, l, c = ctx["open"], ctx["high"], ctx["low"], ctx["close"]
        rng = (h - l).replace(0, np.nan)
        doji = (c - o).abs() < 0.1 * rng
        confirm = c > h.shift(1)  # next bar closes above doji high
        downtrend = c < ta.sma(c, 20)
        entry = doji.shift(1) & confirm & downtrend.shift(1)
        return _hold_n(entry, 5)
    S.append(({"id": "F1", "name": "Doji Reversal", "category": "F-Patterns",
               "variant": "doji+conf/hold5"}, fn_f1))

    # F2: Bullish engulfing in downtrend
    def fn_f2(ctx):
        o, c = ctx["open"], ctx["close"]
        prev_red = c.shift(1) < o.shift(1)
        cur_green = c > o
        engulf = (o < c.shift(1)) & (c > o.shift(1))
        downtrend = c < ta.sma(c, 20)
        entry = prev_red & cur_green & engulf & downtrend
        return _hold_n(entry, 10)
    S.append(({"id": "F2", "name": "Bullish Engulfing", "category": "F-Patterns",
               "variant": "hold10"}, fn_f2))

    # F3: Hammer (long) / shooting star (exit)
    def fn_f3(ctx):
        o, h, l, c = ctx["open"], ctx["high"], ctx["low"], ctx["close"]
        body = (c - o).abs()
        lower = pd.concat([o, c]).groupby(level=0).min().reindex(c.index) - l
        upper = h - pd.concat([o, c]).groupby(level=0).max().reindex(c.index)
        rng = (h - l).replace(0, np.nan)
        hammer = (lower > 2 * body) & (upper < 0.3 * body.replace(0, np.nan)) & (c < ta.sma(c, 20))
        star = (upper > 2 * body) & (lower < 0.3 * body.replace(0, np.nan)) & (c > ta.sma(c, 20))
        return _state(hammer, star | (c < l.shift(1).rolling(10).min()), c.index, c.columns)
    S.append(({"id": "F3", "name": "Hammer/Shooting Star", "category": "F-Patterns",
               "variant": "2xbody"}, fn_f3))

    # F4: Morning star (3-bar reversal), hold 10
    def fn_f4(ctx):
        o, h, l, c = ctx["open"], ctx["high"], ctx["low"], ctx["close"]
        rng = (h - l).replace(0, np.nan)
        body = (c - o).abs()
        big_red = (o.shift(2) - c.shift(2)) > 0.6 * rng.shift(2)
        small_mid = body.shift(1) < 0.3 * rng.shift(1)
        big_green = (c - o) > 0.6 * rng
        closes_high = c > (o.shift(2) + c.shift(2)) / 2
        entry = big_red & small_mid & big_green & closes_high
        return _hold_n(entry, 10)
    S.append(({"id": "F4", "name": "Morning Star", "category": "F-Patterns",
               "variant": "hold10"}, fn_f4))

    # F5: Three white soldiers / three black crows
    def fn_f5(ctx):
        o, h, l, c = ctx["open"], ctx["high"], ctx["low"], ctx["close"]
        rng = (h - l).replace(0, np.nan)
        green = (c > o) & ((c - o) > 0.5 * rng)
        red = (c < o) & ((o - c) > 0.5 * rng)
        rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        soldiers = green & green.shift(1) & green.shift(2) & rising
        crows = red & red.shift(1) & red.shift(2) & falling
        return _state(soldiers, crows, c.index, c.columns)
    S.append(({"id": "F5", "name": "3 Soldiers/3 Crows", "category": "F-Patterns",
               "variant": "0.5body"}, fn_f5))

    # F6: Support/resistance breakout from pivot levels
    def fn_f6(ctx):
        c = ctx["close"]
        h, l = ctx["high"], ctx["low"]
        res = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        sup = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for tkr in c.columns:
            hv = h[tkr].to_numpy(float)
            lv = l[tkr].to_numpy(float)
            if np.isnan(hv).all():
                continue
            hi_idx, _ = _pivots(np.nan_to_num(hv, nan=-np.inf), order=10)
            _, lo_idx = _pivots(np.nan_to_num(lv, nan=np.inf), order=10)
            r = np.full(len(hv), np.nan)
            s = np.full(len(lv), np.nan)
            # level becomes known 10 bars after the pivot (confirmation lag)
            for i in hi_idx:
                if i + 10 < len(r):
                    r[i + 10] = hv[i]
            for i in lo_idx:
                if i + 10 < len(s):
                    s[i + 10] = lv[i]
            res[tkr] = pd.Series(r, index=c.index).ffill()
            sup[tkr] = pd.Series(s, index=c.index).ffill()
        entry = c > res * 1.01
        exit_ = c < sup * 0.99
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "F6", "name": "S/R Breakout", "category": "F-Patterns",
               "variant": "pivot10"}, fn_f6))

    # F7: Weekly pivot points (classic): long above P, exit below S1
    def fn_f7(ctx):
        c = ctx["close"]
        h, l = ctx["high"], ctx["low"]
        wk = c.index.to_period("W")
        hw = h.groupby(wk).max()
        lw = l.groupby(wk).min()
        cw = c.groupby(wk).last()
        piv = (hw + lw + cw) / 3
        s1 = 2 * piv - hw
        # previous week's pivot applied to current week
        piv_d = piv.shift(1).reindex(wk).set_axis(c.index)
        s1_d = s1.shift(1).reindex(wk).set_axis(c.index)
        return _state(c > piv_d, c < s1_d, c.index, c.columns)
    S.append(({"id": "F7", "name": "Weekly Pivot System", "category": "F-Patterns",
               "variant": "classic P/S1"}, fn_f7))

    # F8: Fibonacci retracement in uptrend
    def fn_f8(ctx):
        c = ctx["close"]
        hi60 = c.rolling(60).max()
        lo60 = c.rolling(60).min()
        retrace = (hi60 - c) / (hi60 - lo60).replace(0, np.nan)
        uptrend = ta.sma(c, 50) > ta.sma(c, 50).shift(10)
        turn_up = c > c.shift(3)
        entry = uptrend & turn_up & (retrace >= 0.382) & (retrace <= 0.618)
        exit_ = (retrace > 0.786) | (c >= hi60)
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "F8", "name": "Fibonacci Retracement", "category": "F-Patterns",
               "variant": "38-62%"}, fn_f8))

    # F9: Inverse head & shoulders (long-only variant of H&S)
    def fn_f9(ctx):
        c = ctx["close"]
        entry = pd.DataFrame(False, index=c.index, columns=c.columns)
        for tkr in c.columns:
            s = c[tkr].to_numpy(float)
            if np.isnan(s).all():
                continue
            filled = pd.Series(s).ffill().bfill().to_numpy()
            _, lo_idx = _pivots(filled, order=7)
            hi_idx, _ = _pivots(filled, order=7)
            for j in range(2, len(lo_idx)):
                l1, head, l2 = lo_idx[j - 2], lo_idx[j - 1], lo_idx[j]
                if not (filled[head] < filled[l1] and filled[head] < filled[l2]):
                    continue
                if abs(filled[l1] - filled[l2]) > 0.05 * filled[head]:
                    continue
                highs_between = [i for i in hi_idx if l1 < i < l2]
                if not highs_between:
                    continue
                neckline = max(filled[i] for i in highs_between)
                # breakout within 40 bars after right shoulder
                for t in range(l2 + 1, min(l2 + 40, len(filled))):
                    if filled[t] > neckline * 1.01:
                        entry.iloc[t, entry.columns.get_loc(tkr)] = True
                        break
        return _hold_n(entry, 20)
    S.append(({"id": "F9", "name": "Inverse H&S", "category": "F-Patterns",
               "variant": "pivot7/hold20"}, fn_f9))

    # F10: Double bottom breakout
    def fn_f10(ctx):
        c = ctx["close"]
        entry = pd.DataFrame(False, index=c.index, columns=c.columns)
        for tkr in c.columns:
            s = c[tkr].to_numpy(float)
            if np.isnan(s).all():
                continue
            filled = pd.Series(s).ffill().bfill().to_numpy()
            _, lo_idx = _pivots(filled, order=7)
            for j in range(1, len(lo_idx)):
                b1, b2 = lo_idx[j - 1], lo_idx[j]
                if b2 - b1 < 10 or b2 - b1 > 120:
                    continue
                if abs(filled[b1] - filled[b2]) > 0.02 * filled[b1]:
                    continue
                interim_high = filled[b1:b2 + 1].max()
                for t in range(b2 + 1, min(b2 + 40, len(filled))):
                    if filled[t] > interim_high * 1.01:
                        entry.iloc[t, entry.columns.get_loc(tkr)] = True
                        break
        return _hold_n(entry, 20)
    S.append(({"id": "F10", "name": "Double Bottom", "category": "F-Patterns",
               "variant": "2%/hold20"}, fn_f10))

    # F11: Triangle (range contraction) breakout
    def fn_f11(ctx):
        c, h, l = ctx["close"], ctx["high"], ctx["low"]
        rng20 = (h.rolling(20).max() - l.rolling(20).min())
        contracting = rng20 < 0.7 * rng20.shift(20)
        brk = c > h.shift(1).rolling(20).max()
        entry = contracting.shift(1) & brk
        exit_ = c < l.shift(1).rolling(10).min()
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "F11", "name": "Triangle Breakout", "category": "F-Patterns",
               "variant": "30% contraction"}, fn_f11))

    # F12: Cup and handle (heuristic)
    def fn_f12(ctx):
        c = ctx["close"]
        rim = c.shift(1).rolling(120).max()
        depth = (rim - c.rolling(120).min()) / rim
        cup = (depth > 0.15) & (depth < 0.5)
        near_rim = c > rim * 0.95
        # handle: shallow 5-15 bar pullback (<10%) after regaining the rim
        pull = (rim - c) / rim
        handle = (pull > 0.0) & (pull < 0.10) & near_rim.shift(5).fillna(False)
        brk = c > rim
        entry = cup.shift(10) & handle.shift(1) & brk
        return _hold_n(entry, 30)
    S.append(({"id": "F12", "name": "Cup and Handle", "category": "F-Patterns",
               "variant": "120d/hold30"}, fn_f12))

    # F13: Gap trading
    for mode in ["go", "fill"]:
        def fn(ctx, mode=mode):
            o, c, v, h = ctx["open"], ctx["close"], ctx["volume"], ctx["high"]
            gap = o / c.shift(1) - 1
            volc = v > 1.5 * v.rolling(20).mean()
            if mode == "go":
                entry = (gap > 0.02) & volc & (c > o)   # gap up holds -> momentum
            else:
                entry = (gap < -0.03)                    # big gap down -> fade
            return _hold_n(entry, 5)
        S.append(({"id": "F13", "name": "Gap Trading", "category": "F-Patterns",
                   "variant": f"gap-{mode}/hold5"}, fn))

    return S
