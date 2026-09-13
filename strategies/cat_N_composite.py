"""Category N: combinatorial strategy composition — 10 entries x 6 regime
gates x 6 exits = 360 variants.

All primitives are precomputed once per context and cached; each variant is
then a cheap boolean combination fed to the shared state machine.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.composite import state, hold_n, atr_trail, cross_up

_P = {}


def primitives(ctx):
    key = (ctx["universe"], ctx["interval"])
    if key in _P:
        return _P[key]
    o, h, l, c, v = ctx["open"], ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
    bench = ctx["benchmark"]
    r14 = ta.rsi(c, 14)
    r2 = ta.rsi(c, 2)
    e20, e50 = ta.ema(c, 20), ta.ema(c, 50)
    macd_l, macd_s, _ = ta.macd(c)
    mid, up, lo_, _bw = ta.bollinger(c, 20, 2.0)
    a14 = ta.atr(h, l, c, 14)
    adx14, dip, dim = ta.adx(h, l, c, 14)
    st_dir = ta.supertrend(h, l, c, 10, 3.0)
    vol20 = c.pct_change(fill_method=None).rolling(20).std()
    vol_med = vol20.rolling(252).median()
    breadth = (c > ta.sma(c, 200)).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
    bench_up = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
    gap = o / c.shift(1) - 1

    F = pd.DataFrame(False, index=c.index, columns=c.columns)

    entries = {
        "emaX": cross_up(e20, e50).fillna(False),
        "donch20": (c > h.rolling(20).max().shift(1)).fillna(False),
        "rsi14lo": ((r14 < 30) & (r14.shift(1) >= 30)).fillna(False),
        "rsi2lo": ((r2 < 10) & (r2.shift(1) >= 10)).fillna(False),
        "bbtouch": ((c < lo_) & (c.shift(1) >= lo_.shift(1))).fillna(False),
        "macdX": cross_up(macd_l, macd_s).fillna(False),
        "roc21": ((ta.roc(c, 21) > 5) & (ta.roc(c, 21).shift(1) <= 5)).fillna(False),
        "hi20": ((c >= c.rolling(252).max() * 0.98)
                 & (c.shift(1) < c.rolling(252).max().shift(1) * 0.98)).fillna(False),
        "stflip": ((st_dir > 0) & (st_dir.shift(1) <= 0)).fillna(False),
        "gapdn": ((gap < -0.02)).fillna(False),
    }
    gates = {
        "none": None,
        "adx20": (adx14 > 20).fillna(False),
        "mktup": F.add(bench_up, axis=0).astype(bool),
        "quiet": (vol20 < vol_med).fillna(False),
        "active": (vol20 > vol_med).fillna(False),
        "breadth": F.add(breadth > 0.5, axis=0).astype(bool),
    }
    exit_data = {
        "e20": e20, "e50": e50, "r14": r14, "atr": a14, "close": c, "low": l,
        "mid": mid,
    }
    _P[key] = (entries, gates, exit_data)
    return _P[key]


EXITS = ["inv", "t10", "t21", "atr2.5", "rsi70", "lo10"]


def build(ctx, ename, gname, xname):
    entries, gates, ed = primitives(ctx)
    c = ed["close"]
    entry = entries[ename]
    if gates[gname] is not None:
        entry = entry & gates[gname]
    if xname == "t10":
        return hold_n(entry, 10)
    if xname == "t21":
        return hold_n(entry, 21)
    if xname == "atr2.5":
        return atr_trail(entry, c, ed["atr"], 2.5)
    if xname == "rsi70":
        return state(entry, ed["r14"] > 70, c.index, c.columns)
    if xname == "lo10":
        return state(entry, c < ed["low"].rolling(10).min().shift(1), c.index, c.columns)
    # "inv": a sensible inverse per entry family
    inv_map = {
        "emaX": ed["e20"] < ed["e50"],
        "donch20": c < ed["low"].rolling(20).min().shift(1),
        "rsi14lo": ed["r14"] > 55,
        "rsi2lo": ed["r14"] > 60,
        "bbtouch": c > ed["mid"],
        "macdX": None,  # handled below
        "roc21": None,
        "hi20": c < ed["e50"],
        "stflip": None,
        "gapdn": ed["r14"] > 55,
    }
    inv = inv_map[ename]
    if inv is None:
        inv = c < ed["e50"]
    return state(entry, inv.fillna(False), c.index, c.columns)


def get_strategies():
    S = []
    ENTRY_NAMES = ["emaX", "donch20", "rsi14lo", "rsi2lo", "bbtouch", "macdX",
                   "roc21", "hi20", "stflip", "gapdn"]
    GATE_NAMES = ["none", "adx20", "mktup", "quiet", "active", "breadth"]
    for i, ename in enumerate(ENTRY_NAMES, 1):
        for gname in GATE_NAMES:
            for xname in EXITS:
                def fn(ctx, ename=ename, gname=gname, xname=xname):
                    return build(ctx, ename, gname, xname)
                S.append(({"id": f"N{i}", "name": f"Compose:{ename}",
                           "category": "N-Compose",
                           "variant": f"{ename}|{gname}|{xname}"}, fn))
    return S
