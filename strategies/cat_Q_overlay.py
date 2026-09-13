"""Category Q: risk overlays and ensembles on strong base signals.

12 base signals x overlay transforms (vol targeting, ATR stops, time stops,
profit targets, market gate, crash de-risk, VIX de-risk) + voting ensembles.
~90 variants.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.composite import state, monthly_topn, atr_trail

_B = {}


def bases(ctx):
    key = (ctx["universe"], ctx["interval"])
    if key in _B:
        return _B[key]
    h, l, c = ctx["high"], ctx["low"], ctx["close"]
    ret = c.pct_change(fill_method=None)
    macd_l, macd_s, _ = ta.macd(c)
    adx14, dip, dim = ta.adx(h, l, c, 14)
    out = {}
    # state-type bases (daily 0/1 per stock)
    out["sma50x200"] = ("state", (ta.sma(c, 50) > ta.sma(c, 200)).astype(float))
    out["ema20x50"] = ("state", (ta.ema(c, 20) > ta.ema(c, 50)).astype(float))
    out["donch55"] = ("state", state(c > h.rolling(55).max().shift(1),
                                     c < l.rolling(20).min().shift(1), c.index, c.columns))
    out["supertrend"] = ("state", (ta.supertrend(h, l, c, 10, 3.0) > 0).astype(float))
    out["kama"] = ("state", (c > ta.kama(c)).astype(float))
    out["macd"] = ("state", (macd_l > macd_s).astype(float))
    out["adxmacd"] = ("state", ((adx14 > 20) & (macd_l > macd_s) & (dip > dim)).astype(float))
    out["hi52"] = ("state", state(c >= c.rolling(252).max() * 0.95,
                                  c < ta.sma(c, 50), c.index, c.columns))
    # basket-type bases (monthly top-N)
    out["jt121"] = ("basket", monthly_topn(c, c.shift(21) / c.shift(252) - 1, 10))
    out["xsmom126"] = ("basket", monthly_topn(c, c.pct_change(126, fill_method=None), 10))
    out["lowvol"] = ("basket", monthly_topn(c, -ret.rolling(252).std(), 15))
    out["sharpemom"] = ("basket", monthly_topn(
        c, ret.rolling(126).mean() / ret.rolling(126).std().replace(0, np.nan), 10))
    _B[key] = out
    return out


def _port_weights(pos):
    n = (pos != 0).sum(axis=1).clip(lower=1)
    return pos.div(n, axis=0)


def apply_overlay(ctx, pos, kind, overlay):
    c = ctx["close"]
    bench = ctx["benchmark"]
    if overlay == "mktgate":
        gate = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
        return pos.mul(gate.astype(float), axis=0), False
    if overlay == "crash8":
        b10 = bench.pct_change(10).reindex(c.index)
        off = pd.Series(np.nan, index=c.index)
        off[b10 < -0.08] = 0.0
        off[b10 > -0.02] = 1.0
        return pos.mul(off.ffill().fillna(1.0), axis=0), False
    if overlay in ("volt10", "volt15"):
        target = 0.10 if overlay == "volt10" else 0.15
        w = _port_weights(pos)
        r = (w.shift(1) * c.pct_change(fill_method=None)).sum(axis=1)
        realized = r.rolling(63).std() * np.sqrt(252)
        scale = (target / realized.replace(0, np.nan)).clip(0, 1.5).shift(1).fillna(0)
        return w.mul(scale, axis=0), True
    if overlay == "vixhalf":
        vix = ctx["vix"]
        if vix is None:
            raise RuntimeError("no VIX")
        pct = vix.reindex(c.index).ffill().rolling(500, min_periods=250).rank(pct=True)
        scale = pd.Series(1.0, index=c.index).where(pct < 0.8, 0.5)
        w = _port_weights(pos)
        return w.mul(scale, axis=0), True
    if overlay == "time63":
        # cap holding at 63 bars per continuous position run
        arr = (pos > 0).to_numpy(bool)
        out = np.zeros_like(arr, dtype=float)
        cnt = np.zeros(arr.shape[1])
        for t in range(arr.shape[0]):
            cnt = np.where(arr[t], cnt + 1, 0)
            out[t] = (arr[t] & (cnt <= 63)).astype(float)
        return pd.DataFrame(out, index=pos.index, columns=pos.columns), False
    if overlay in ("atr2.5", "atr3.5"):
        mult = float(overlay[3:])
        entry = (pos > 0) & (pos.shift(1) <= 0)
        a = ta.atr(ctx["high"], ctx["low"], c, 14)
        trail = atr_trail(entry.fillna(False), c, a, mult)
        return ((pos > 0) & (trail > 0)).astype(float), False
    if overlay == "profit25":
        # exit a run once unrealized gain exceeds 25%; re-enter on fresh signal
        p = (pos > 0).to_numpy(bool)
        px = c.to_numpy(float)
        out = np.zeros_like(p, dtype=float)
        entry_px = np.full(p.shape[1], np.nan)
        taken = np.zeros(p.shape[1], dtype=bool)
        for t in range(p.shape[0]):
            fresh = p[t] & (~p[t - 1] if t > 0 else True)
            entry_px = np.where(fresh, px[t], entry_px)
            taken = np.where(fresh, False, taken)
            gain = px[t] / entry_px - 1
            hit = p[t] & ~taken & (gain > 0.25)
            taken = taken | hit
            out[t] = (p[t] & ~taken).astype(float)
            taken = np.where(~p[t], False, taken)
        return pd.DataFrame(out, index=pos.index, columns=pos.columns), False
    raise ValueError(overlay)


STATE_OVERLAYS = ["volt10", "volt15", "atr2.5", "atr3.5", "time63", "profit25",
                  "mktgate", "crash8"]
BASKET_OVERLAYS = ["volt10", "volt15", "mktgate", "crash8", "vixhalf"]


def get_strategies():
    S = []
    NAMES = ["sma50x200", "ema20x50", "donch55", "supertrend", "kama", "macd",
             "adxmacd", "hi52", "jt121", "xsmom126", "lowvol", "sharpemom"]
    for i, bname in enumerate(NAMES, 1):
        overlays = None  # decided lazily from base kind

        def make(bname):
            def gen(ctx, overlay):
                kind, pos = bases(ctx)[bname]
                newpos, is_w = apply_overlay(ctx, pos, kind, overlay)
                return newpos, is_w
            return gen
        gen = make(bname)
        for overlay in STATE_OVERLAYS + ["vixhalf"]:
            def fn(ctx, gen=gen, overlay=overlay):
                newpos, is_w = gen(ctx, overlay)
                fn.__dict__["_w"] = is_w
                return newpos
            # overlay validity is resolved at signal time; weights flagged below
            is_weights = overlay in ("volt10", "volt15", "vixhalf")
            meta = {"id": f"Q{i}", "name": f"Overlay:{bname}", "category": "Q-Overlay",
                    "variant": f"{bname}|{overlay}"}
            if is_weights:
                meta["signals_are_weights"] = True
            # skip invalid combos: per-name stops need per-name state entries
            if overlay in ("atr2.5", "atr3.5", "time63", "profit25"):
                # applied to both kinds; for baskets they emulate stop-outs
                pass
            S.append((meta, fn))

    # Q13: voting ensembles over the 8 state bases
    for need in [3, 4, 6]:
        def fn(ctx, need=need):
            b = bases(ctx)
            votes = sum((b[n][1] > 0).astype(int) for n in
                        ["sma50x200", "ema20x50", "donch55", "supertrend",
                         "kama", "macd", "adxmacd", "hi52"])
            return (votes >= need).astype(float)
        S.append(({"id": "Q13", "name": "Signal Voting", "category": "Q-Overlay",
                   "variant": f">={need} of 8"}, fn))

    # Q14: breadth-of-signals timing (market-level vote)
    for need in [0.4, 0.5]:
        def fn(ctx, need=need):
            c = ctx["close"]
            b = bases(ctx)
            votes = sum((b[n][1] > 0).astype(int) for n in
                        ["sma50x200", "ema20x50", "macd", "hi52"])
            frac = (votes >= 2).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
            gate = (frac > need).astype(float)
            return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
                c.notna(), 0.0).mul(gate, axis=0)
        S.append(({"id": "Q14", "name": "Signal Breadth Timing", "category": "Q-Overlay",
                   "variant": f"frac>{need}"}, fn))

    return S
