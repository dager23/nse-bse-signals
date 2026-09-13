"""Category R: screen survivors promoted to the full engine.

Interprets the parseable spec strings emitted by screener.py and reconstructs
each strategy's positions on the full context (any universe/interval).
"""
import os
import re

import numpy as np
import pandas as pd

import config
from utils.screen_atoms import build_atoms

_ATOMS = {}
_OSCQ = {}

WIN_START, WIN_END = "2010-07-01", "2025-07-01"


def atoms_for(ctx):
    key = (ctx["universe"], ctx["interval"])
    if key not in _ATOMS:
        _ATOMS[key] = build_atoms(ctx)
        oscq = {}
        for k, v in _ATOMS[key]["OSC"].items():
            w = v.loc[WIN_START:WIN_END].to_numpy()
            oscq[k] = np.nanpercentile(w, [10, 25, 40, 60, 75, 90])
        _OSCQ[key] = oscq
    return _ATOMS[key], _OSCQ[key]


def _state(entry, exit_, c):
    st = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    st[entry & ~exit_] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def _hold(entry, n):
    return entry.astype(float).rolling(n, min_periods=1).max().fillna(0)


def _gate(atoms, entry, gname, c):
    if gname in ("none", "or"):
        return entry
    if gname == "quietX":
        return entry & atoms["BOOL"]["quiet"]
    if gname == "activeX":
        return entry & ~atoms["BOOL"]["quiet"]
    g = atoms["GATE"].get(gname)
    if g is None:
        return entry
    return entry.mul(g, axis=0).astype(bool)


def _boolmap(atoms, c):
    B = dict(atoms["BOOL"])
    for n in [20, 50, 100, 200]:
        B[f"c>sma{n}"] = (c > atoms["MA"][f"sma{n}"]).fillna(False)
    for n in [10, 20, 50]:
        B[f"c>ema{n}"] = (c > atoms["MA"][f"ema{n}"]).fillna(False)
    for x in [5, 8, 12]:
        B[f"trail{x}"] = (c < c.rolling(20).max() * (1 - x / 100)).fillna(False)
    return B


def _exit_of(atoms, xname, c):
    if xname == "rsi70":
        return (atoms["OSC"]["rsi14"] > 70).fillna(False)
    if xname.startswith("trail"):
        return (c < c.rolling(20).max() * (1 - int(xname[5:]) / 100)).fillna(False)
    if xname == "inv50":
        return (c < atoms["MA"]["ema50"]).fillna(False)
    raise ValueError(xname)


def build_positions(ctx, spec):
    atoms, oscq = atoms_for(ctx)
    c = ctx["close"]
    f = spec.split("|")
    fam = f[0]

    if fam == "V":
        pair, gname, xname = f[1], f[2], f[3]
        fast, slow = pair.split(">")
        above = (atoms["MA"][fast] > atoms["MA"][slow]).fillna(False)
        gated = _gate(atoms, above, gname, c)
        if xname == "inv":
            return gated.astype(float)
        fresh = gated & ~gated.shift(1).fillna(False)
        if xname == "rsi70":
            return _state(fresh, _exit_of(atoms, "rsi70", c) | ~above, c)
        return _hold(fresh, 21)

    if fam == "T":
        oname, cin, cout, gname, style = f[1], f[2], f[3], f[4], f[5]
        osc = atoms["OSC"][oname]
        tin = float(cin[1:])
        tout = float(cout[1:])
        entry = (osc < tin) if cin[0] == "<" else (osc > tin)
        exit_ = (osc > tout) if cout[0] == ">" else (osc < tout)
        entry = _gate(atoms, entry.fillna(False), gname, c)
        exit_ = exit_.fillna(False)
        return _state(entry, exit_, c) if style == "state" else _hold(entry, 10)

    if fam == "U":
        B = _boolmap(atoms, c)
        if len(f) == 5 and f[3] == "or":
            return _hold(B[f[1]] | B[f[2]], 10)
        conj_names, gname, xname = f[1].split("&"), f[2], f[3]
        entry = B[conj_names[0]]
        for nm in conj_names[1:]:
            entry = entry & B[nm]
        entry = _gate(atoms, entry, gname, c)
        if xname == "rsi70":
            return _state(entry, _exit_of(atoms, "rsi70", c), c)
        return _hold(entry, int(xname[4:]))

    if fam == "W":
        B = _boolmap(atoms, c)
        conds, gname, xname = f[1].split("&"), f[2], f[3]
        entry = None
        for d in conds:
            m = re.match(r"^(\w+)([<>])q(\d)$", d)
            if m:
                on, op, qi = m.group(1), m.group(2), int(m.group(3))
                thr = oscq[on][qi]
                cond = (atoms["OSC"][on] < thr) if op == "<" else (atoms["OSC"][on] > thr)
                cond = cond.fillna(False)
            else:
                cond = B[d]
            entry = cond if entry is None else (entry & cond)
        entry = _gate(atoms, entry, gname, c)
        if xname.startswith("hold"):
            return _hold(entry, int(xname[4:]))
        return _state(entry, _exit_of(atoms, xname, c), c)

    if fam == "S":
        blend, topk = f[1], int(f[2][3:])
        me_idx = atoms["me_idx"]
        score = None
        for term in blend.split("+"):
            nm, w = term.split("*")
            r = pd.DataFrame(atoms["RANK"][nm], index=me_idx,
                             columns=c.columns)
            score = r * float(w) if score is None else score + r * float(w)
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for d in me_idx:
            row = score.loc[d].dropna()
            if len(row) < topk:
                continue
            sel = row.nlargest(topk).index
            out = pd.Series(0.0, index=c.columns)
            out[list(sel)] = 1.0
            pos.loc[d] = out
        return pos.ffill().fillna(0)

    raise ValueError(fam)


def get_strategies():
    path = os.path.join(config.RESULTS_DIR, "promoted_specs.csv")
    if not os.path.exists(path):
        return []
    specs = pd.read_csv(path)
    S = []
    for _, r in specs.iterrows():
        def fn(ctx, spec=r.spec):
            return build_positions(ctx, spec)
        S.append(({"id": f"R-{r.family}", "name": f"Promoted:{r.family}",
                   "category": "R-Promoted", "variant": r.spec}, fn))
    return S
