"""Atom library for the mass screener: precomputed primitive signals.

Everything is computed ONCE on the daily Nifty-50 panel and shared across
hundreds of thousands of composed strategies.  Three atom classes:

  MA[key]     -> moving-average level DataFrames (for crossover states)
  OSC[key]    -> bounded/continuous oscillator DataFrames (threshold rules)
  BOOL[key]   -> ready boolean condition DataFrames (conjunction rules)
  GATE[key]   -> date-level boolean Series (market regime gates)
  RANK[key]   -> month-end cross-sectional factor rank matrices (blends)

All causal: only rolling/expanding transforms of past data.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta


def build_atoms(ctx):
    o, h, l, c, v = ctx["open"], ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
    bench, vix = ctx["benchmark"], ctx["vix"]
    ret = c.pct_change(fill_method=None)

    # ---------- moving averages ----------
    MA = {}
    for n in [3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]:
        MA[f"sma{n}"] = ta.sma(c, n)
        MA[f"ema{n}"] = ta.ema(c, n)
    for n in [20, 50]:
        MA[f"dema{n}"] = ta.dema(c, n)
        MA[f"tema{n}"] = ta.tema(c, n)
    MA["kama"] = ta.kama(c)

    # ---------- oscillators / continuous scores ----------
    OSC = {}
    for n in [2, 5, 7, 14, 21, 30]:
        OSC[f"rsi{n}"] = ta.rsi(c, n)
    for n in [5, 14, 21]:
        k, d = ta.stochastic(h, l, c, n)
        OSC[f"stoch{n}"] = k
    for n in [14, 20, 50]:
        OSC[f"cci{n}"] = ta.cci(h, l, c, n)
    OSC["willr14"] = ta.williams_r(h, l, c, 14)
    OSC["mfi14"] = ta.mfi(h, l, c, v, 14)
    OSC["tsi"] = ta.tsi(c)
    for n in [5, 10, 21, 63, 126, 252]:
        OSC[f"roc{n}"] = ta.roc(c, n)
    for n in [10, 20, 50]:
        OSC[f"z{n}"] = ta.zscore(c, n)
    for n, k_ in [(20, 2.0), (50, 2.0)]:
        mid, up, lo_, _ = ta.bollinger(c, n, k_)
        OSC[f"pctb{n}"] = (c - lo_) / (up - lo_).replace(0, np.nan)
    for n in [20, 55, 100]:
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        OSC[f"dpos{n}"] = (c - ll) / (hh - ll).replace(0, np.nan)  # channel position
    OSC["volr20"] = v / v.rolling(20).mean()
    OSC["atrp14"] = ta.atr(h, l, c, 14) / c
    OSC["cmf20"] = ta.cmf(h, l, c, v, 20)
    OSC["hi52d"] = c / c.rolling(252).max()
    OSC["vol20"] = ret.rolling(20).std()
    OSC["clv20"] = (((c - l) - (h - c)) / (h - l).replace(0, np.nan)).rolling(20).mean()
    OSC["updn20"] = (v.where(c > c.shift(1), 0.0).rolling(20).sum()
                     / v.where(c < c.shift(1), 0.0).rolling(20).sum().replace(0, np.nan))

    # ---------- ready booleans ----------
    BOOL = {}
    macd_l, macd_s, _ = ta.macd(c)
    adx14, dip, dim = ta.adx(h, l, c, 14)
    BOOL["macd_up"] = (macd_l > macd_s).fillna(False)
    BOOL["macd_pos"] = (macd_l > 0).fillna(False)
    BOOL["adx_trend"] = ((adx14 > 22) & (dip > dim)).fillna(False)
    BOOL["obv_up"] = (ta.obv(c, v).diff(20) > 0).fillna(False)
    BOOL["vspike"] = (v > 2 * v.rolling(20).mean()).fillna(False)
    BOOL["gapup"] = ((o / c.shift(1) - 1) > 0.01).fillna(False)
    BOOL["gapdn"] = ((o / c.shift(1) - 1) < -0.015).fillna(False)
    BOOL["nr7brk"] = (((h - l) == (h - l).rolling(7).min()).shift(1)
                      & (c > h.shift(1))).fillna(False)
    BOOL["up3"] = ((c > c.shift(1)).rolling(3).sum() == 3).fillna(False)
    BOOL["dn3"] = ((c < c.shift(1)).rolling(3).sum() == 3).fillna(False)
    BOOL["hh20"] = (c >= h.rolling(20).max().shift(1)).fillna(False)
    BOOL["ll20"] = (c <= l.rolling(20).min().shift(1)).fillna(False)
    BOOL["quiet"] = (OSC["vol20"] < OSC["vol20"].rolling(252).median()).fillna(False)
    BOOL["insideday"] = ((h < h.shift(1)) & (l > l.shift(1))).fillna(False)

    # ---------- market gates (date-level Series) ----------
    GATE = {"none": None}
    b200 = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
    b50 = (bench > bench.rolling(50).mean()).reindex(c.index).fillna(False)
    GATE["mkt200"] = b200
    GATE["mkt50"] = b50
    breadth = (c > MA["sma200"]).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
    GATE["breadth50"] = breadth > 0.5
    GATE["breadth30"] = breadth > 0.3
    bmom = bench.pct_change(63).reindex(c.index)
    GATE["bmom_pos"] = (bmom > 0).fillna(False)
    if vix is not None:
        vp = vix.reindex(c.index).ffill().rolling(500, min_periods=250).rank(pct=True)
        GATE["vix_low"] = (vp < 0.7).fillna(False)
        GATE["vix_high"] = (vp > 0.8).fillna(False)
    bdd = (bench / bench.rolling(252).max() - 1).reindex(c.index).fillna(0)
    GATE["nocrash"] = bdd > -0.12

    # ---------- month-end factor ranks (for cross-sectional blends) ----------
    me_mask = (pd.Series(c.index, index=c.index).groupby(
        [c.index.year, c.index.month]).transform("max") == c.index)
    me_idx = c.index[me_mask]
    factors_raw = {
        "mom121": c.shift(21) / c.shift(252) - 1,
        "mom6": c.shift(21) / c.shift(126) - 1,
        "mom3": c.pct_change(63, fill_method=None),
        "lowvol": -ret.rolling(252).std(),
        "rev1m": -c.pct_change(21, fill_method=None),
        "liq": (c * v).rolling(63).mean(),
        "onight": (o / c.shift(1) - 1).rolling(63).sum(),
        "smom": ret.rolling(126).mean() / ret.rolling(126).std().replace(0, np.nan),
        "hi52": c / c.rolling(252).max(),
        "lowbeta": None,  # filled below
    }
    bret = bench.pct_change().reindex(c.index).fillna(0)
    bm = bret.rolling(252).mean()
    cov = ret.mul(bret, axis=0).rolling(252).mean() - ret.rolling(252).mean().mul(bm, axis=0)
    var = (bret ** 2).rolling(252).mean() - bm ** 2
    factors_raw["lowbeta"] = -cov.div(var.replace(0, np.nan), axis=0)
    RANK = {k: df.loc[me_idx].rank(axis=1, pct=True).to_numpy(np.float32)
            for k, df in factors_raw.items()}

    return {"MA": MA, "OSC": OSC, "BOOL": BOOL, "GATE": GATE,
            "RANK": RANK, "me_idx": me_idx, "close": c, "ret": ret}
