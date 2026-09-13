"""Wave 4 mass screener: evaluates 100k+ composed strategy configurations with
a fast vectorized engine (close-to-close, turnover costs, 15Y window).

Families:
  V  MA-crossover space           (~16k)   "V|ema10>sma50|<gate>|<exit>"
  T  oscillator threshold sweeps  (~4.6k)  "T|rsi5|<20|>60|<gate>|<style>"
  U  boolean conjunctions         (~6.3k)  "U|hh20&obv_up|<gate>|<exit>"
  W  random expression search     (60k)    "W|rsi14<30&clv20>0.2|<gate>|<exit>"
  S  factor-blend space           (~14.7k) "S|mom121*0.4+lowvol*0.6|top10"

Every configuration's spec string is fully parseable so survivors can be
reconstructed exactly for the full multi-timeframe engine.

Usage: python screener.py --families V,T,U,S,W
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

import config
from run_all import get_ctx
from utils.screen_atoms import build_atoms

OUT_DIR = os.path.join(config.RESULTS_DIR, "screen")
os.makedirs(OUT_DIR, exist_ok=True)

COST = config.TRANSACTION_COST + config.SLIPPAGE
ANN = 252
RF = config.RISK_FREE_RATE

WIN_START, WIN_END = "2010-07-01", "2025-07-01"


# ------------------------------------------------------------------ fast eval
class Evaluator:
    def __init__(self, ctx):
        c = ctx["close"].loc[WIN_START:WIN_END]
        self.index = c.index
        self.columns = c.columns
        self.close = c.to_numpy(np.float64)
        self.alive = ~np.isnan(self.close)
        r = c.pct_change(fill_method=None).to_numpy(np.float64)
        self.ret = np.nan_to_num(r, nan=0.0)
        self.T, self.N = self.close.shape

    def slice(self, df):
        return df.loc[WIN_START:WIN_END].to_numpy()

    def evaluate(self, pos):
        """pos: T x N ndarray of desired position at close (0/1). Returns metrics."""
        pos = np.where(self.alive, pos, 0.0)
        held = np.vstack([np.zeros((1, self.N)), pos[:-1]])
        n_active = held.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = held / np.where(n_active > 0, n_active, 1)[:, None]
        gross = (w * self.ret).sum(axis=1)
        turnover = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, self.N)))).sum(axis=1)
        r = gross - turnover * COST
        eq = np.cumprod(1 + r)
        if eq[-1] <= 0:
            return None
        yrs = self.T / ANN
        cagr = eq[-1] ** (1 / yrs) - 1
        sd = r.std()
        sharpe = (r.mean() * ANN - RF) / (sd * np.sqrt(ANN)) if sd > 0 else np.nan
        peak = np.maximum.accumulate(eq)
        mdd = (eq / peak - 1).min()
        exposure = (held.sum(axis=1) > 0).mean()
        return {"sharpe": sharpe, "cagr": cagr, "maxdd": mdd,
                "exposure": exposure, "turn_yr": turnover.sum() / yrs}


def state_from(entry, exit_):
    """Vectorized long/flat state machine on boolean ndarrays."""
    T, N = entry.shape
    m = np.where(entry & ~exit_, 1, np.where(exit_, 0, -1)).astype(np.int8)
    set_mask = m >= 0
    rows = np.arange(T)[:, None]
    idx = np.where(set_mask, rows, 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    vals = np.where(set_mask, m, 0).astype(np.float32)
    return np.take_along_axis(vals, idx, axis=0)


def hold_n_np(entry, n):
    return pd.DataFrame(entry).rolling(n, min_periods=1).max().fillna(0).to_numpy()


# ------------------------------------------------------------------ atoms -> arrays
class Arsenal:
    """Numpy views of the atom library, pre-sliced to the screen window."""

    def __init__(self, ctx):
        atoms = build_atoms(ctx)
        ev = Evaluator(ctx)
        self.ev = ev
        sl = slice(None)
        idx = atoms["close"].index
        mask = (idx >= WIN_START) & (idx <= WIN_END)
        self.MA = {k: v.to_numpy(np.float64)[mask] for k, v in atoms["MA"].items()}
        self.OSC = {k: v.to_numpy(np.float64)[mask] for k, v in atoms["OSC"].items()}
        self.BOOL = {k: v.to_numpy(bool)[mask] for k, v in atoms["BOOL"].items()}
        self.GATE = {k: (None if g is None else g.to_numpy(bool)[mask])
                     for k, g in atoms["GATE"].items()}
        self.close = ev.close
        # MA-derived boolean states for conjunctions
        for n in [20, 50, 100, 200]:
            self.BOOL[f"c>sma{n}"] = self.close > self.MA[f"sma{n}"]
        for n in [10, 20, 50]:
            self.BOOL[f"c>ema{n}"] = self.close > self.MA[f"ema{n}"]
        # trail exits
        cdf = pd.DataFrame(self.close)
        for x in [5, 8, 12]:
            self.BOOL[f"trail{x}"] = (cdf < cdf.rolling(20).max()
                                      * (1 - x / 100)).fillna(False).to_numpy()
        # month-end machinery for S family
        me_idx = atoms["me_idx"]
        self.me_rows = np.searchsorted(idx[mask], me_idx[me_idx >= idx[mask][0]])
        keep = (me_idx >= idx[mask][0])
        self.RANK = {k: v[-keep.sum():] for k, v in atoms["RANK"].items()}

    def gate_apply(self, entry, gname):
        g = self.GATE[gname]
        return entry if g is None else entry & g[:, None]

    def exit_of(self, xname):
        if xname == "rsi70":
            return self.OSC["rsi14"] > 70
        if xname.startswith("trail"):
            return self.BOOL[xname]
        raise ValueError(xname)

    def positions_from_topk(self, scores_me, k):
        """scores_me: (n_me x N) blend scores at month-ends -> daily 0/1 pos."""
        T, N = self.close.shape
        pos = np.zeros((T, N), dtype=np.float32)
        sel = np.zeros((scores_me.shape[0], N), dtype=np.float32)
        for i in range(scores_me.shape[0]):
            row = scores_me[i]
            valid = ~np.isnan(row)
            if valid.sum() < k:
                continue
            top = np.argpartition(np.where(valid, -row, np.inf), k)[:k]
            sel[i, top] = 1.0
        # forward-fill month-end selections to daily rows
        me = self.me_rows
        for i in range(len(me)):
            a = me[i]
            b = me[i + 1] if i + 1 < len(me) else T
            pos[a:b] = sel[i]
        return pos


# ------------------------------------------------------------------ generators
def gen_V(ars):
    keys = list(ars.MA)
    def length(k):
        num = "".join(ch for ch in k if ch.isdigit())
        return int(num) if num else 10  # kama ~ medium
    GATES = ["none", "mkt200", "breadth50", "vix_low", "nocrash", "bmom_pos",
             "mkt50", "breadth30", "vix_high", "quietX", "activeX"]
    EXITS = ["inv", "rsi70", "hold21"]
    quiet = ars.BOOL["quiet"]
    for fast, slow in itertools.combinations(keys, 2):
        lf, ls = length(fast), length(slow)
        if lf == ls:
            continue
        if lf > ls:
            fast, slow = slow, fast
        above = ars.MA[fast] > ars.MA[slow]
        for gname in GATES:
            if gname == "quietX":
                gated = above & quiet
            elif gname == "activeX":
                gated = above & ~quiet
            else:
                gated = ars.gate_apply(above, gname)
            for xname in EXITS:
                if xname == "inv":
                    pos = gated.astype(np.float32)
                elif xname == "rsi70":
                    pos = state_from(gated & ~np.roll(gated, 1, axis=0),
                                     ars.exit_of("rsi70") | ~above)
                else:
                    pos = hold_n_np(gated & ~np.roll(gated, 1, axis=0), 21)
                yield "V", f"V|{fast}>{slow}|{gname}|{xname}", pos


def gen_T(ars):
    SPECS = {  # osc: (rev_in, rev_out, mom_in, mom_out)
        "rsi2": ([2, 5, 10, 15], [50, 60, 70, 80], [90, 95], [50]),
        "rsi5": ([10, 15, 20, 30], [50, 60, 70, 80], [80, 90], [50]),
        "rsi14": ([20, 25, 30, 35], [50, 55, 60, 70], [60, 65, 70], [45, 50]),
        "rsi21": ([25, 30, 35, 40], [50, 55, 60, 65], [60, 65], [45]),
        "stoch5": ([5, 10, 20, 30], [50, 70, 80, 90], [80, 90], [50]),
        "stoch14": ([10, 20, 30], [50, 70, 80], [80], [50]),
        "cci20": ([-200, -150, -100, -50], [0, 50, 100, 150], [100, 150], [0]),
        "willr14": ([-95, -90, -80], [-50, -30, -20], [-20, -10], [-50]),
        "mfi14": ([10, 20, 30], [50, 60, 70], [70, 80], [50]),
        "z20": ([-3, -2.5, -2, -1.5], [-0.5, 0, 0.5, 1], [1.5, 2], [0]),
        "z50": ([-3, -2.5, -2], [-0.5, 0, 0.5], [2], [0]),
        "pctb20": ([-0.1, 0, 0.1, 0.2], [0.5, 0.7, 0.9, 1.0], [1.0, 1.1], [0.5]),
        "dpos55": ([0, 0.05, 0.1], [0.5, 0.7, 0.9], [0.9, 0.95, 1.0], [0.5]),
        "dpos20": ([0, 0.1], [0.5, 0.8], [0.95, 1.0], [0.5]),
        "roc21": ([-15, -10, -8], [0, 5], [5, 8, 10], [0, -2]),
        "roc63": ([-25, -20, -15], [0, 5], [8, 12, 15], [0]),
        "tsi": ([-40, -30, -20], [0, 10], [15, 25], [0]),
        "cmf20": ([-0.25, -0.15], [0, 0.1], [0.1, 0.2], [0]),
        "clv20": ([-0.4, -0.3], [0, 0.1], [0.15, 0.25], [0]),
        "updn20": ([0.4, 0.6], [1.0, 1.2], [1.3, 1.6], [1.0]),
        "hi52d": ([0.6, 0.7], [0.85, 0.95], [0.95, 0.98, 1.0], [0.85]),
        "volr20": (None, None, [2, 3], [1]),
        "mom_stoch21": None,
    }
    GATES = ["none", "mkt200", "breadth50"]
    for oname, spec in SPECS.items():
        if oname not in ars.OSC or spec is None:
            continue
        osc = ars.OSC[oname]
        rev_in, rev_out, mom_in, mom_out = spec
        combos = []
        if rev_in:
            combos += [("<", a, ">", b) for a in rev_in for b in rev_out if b > a]
        if mom_in:
            combos += [(">", a, "<", b) for a in mom_in for b in mom_out if b < a]
        for cin, tin, cout, tout in combos:
            entry = (osc < tin) if cin == "<" else (osc > tin)
            exit_ = (osc > tout) if cout == ">" else (osc < tout)
            entry = np.nan_to_num(entry, nan=False)
            exit_ = np.nan_to_num(exit_, nan=False)
            for gname in GATES:
                e = ars.gate_apply(entry, gname)
                for style in ["state", "hold10"]:
                    pos = state_from(e, exit_) if style == "state" else hold_n_np(e, 10)
                    yield "T", f"T|{oname}|{cin}{tin}|{cout}{tout}|{gname}|{style}", pos


def gen_U(ars, rng):
    bools = {k: v for k, v in ars.BOOL.items() if not k.startswith("trail")}
    names = sorted(bools)
    GATES = ["none", "mkt200", "breadth50"]
    EXITS = ["hold5", "hold10", "hold21", "rsi70"]
    for a, b in itertools.combinations(names, 2):
        conj = bools[a] & bools[b]
        for gname in GATES:
            e = ars.gate_apply(conj, gname)
            for xname in EXITS:
                if xname == "rsi70":
                    pos = state_from(e, ars.exit_of("rsi70"))
                else:
                    pos = hold_n_np(e, int(xname[4:]))
                yield "U", f"U|{a}&{b}|{gname}|{xname}", pos
    # sampled triples
    for _ in range(1500):
        a, b, c_ = rng.choice(names, 3, replace=False)
        conj = bools[a] & bools[b] & bools[c_]
        for xname in ["hold10", "rsi70"]:
            pos = (hold_n_np(conj, 10) if xname == "hold10"
                   else state_from(conj, ars.exit_of("rsi70")))
            yield "U", f"U|{a}&{b}&{c_}|none|{xname}", pos
    # OR pairs
    for a, b in itertools.combinations(names, 2):
        pos = hold_n_np(bools[a] | bools[b], 10)
        yield "U", f"U|{a}|{b}|or|hold10", pos


def gen_W(ars, rng, n=70000):
    """Random expression search: entry = conjunction of 2-3 conditions."""
    osc_names = sorted(ars.OSC)
    bool_names = sorted(ars.BOOL)
    gate_names = sorted(k for k in ars.GATE)
    exits = ["hold5", "hold10", "hold21", "hold42", "rsi70", "trail8", "trail12", "inv50"]
    # canonical condition builders: osc vs its own quantile thresholds
    osc_q = {k: np.nanpercentile(v, [10, 25, 40, 60, 75, 90]) for k, v in ars.OSC.items()}
    for i in range(n):
        k = rng.integers(2, 4)
        conds, desc = [], []
        for _ in range(k):
            if rng.random() < 0.45:
                bn = bool_names[rng.integers(len(bool_names))]
                conds.append(ars.BOOL[bn])
                desc.append(bn)
            else:
                on = osc_names[rng.integers(len(osc_names))]
                qi = rng.integers(6)
                thr = osc_q[on][qi]
                if rng.random() < 0.5:
                    conds.append(np.nan_to_num(ars.OSC[on] < thr, nan=False))
                    desc.append(f"{on}<q{qi}")
                else:
                    conds.append(np.nan_to_num(ars.OSC[on] > thr, nan=False))
                    desc.append(f"{on}>q{qi}")
        entry = conds[0]
        for c_ in conds[1:]:
            entry = entry & c_
        gname = gate_names[rng.integers(len(gate_names))]
        entry = ars.gate_apply(entry, gname)
        xname = exits[rng.integers(len(exits))]
        if xname.startswith("hold"):
            pos = hold_n_np(entry, int(xname[4:]))
        elif xname == "rsi70":
            pos = state_from(entry, ars.exit_of("rsi70"))
        elif xname.startswith("trail"):
            pos = state_from(entry, ars.exit_of(xname))
        else:  # inv50: exit below ema50
            pos = state_from(entry, ~ars.BOOL["c>ema50"])
        yield "W", f"W|{'&'.join(desc)}|{gname}|{xname}", pos


def gen_S(ars, rng):
    names = sorted(ars.RANK)
    R = {k: v for k, v in ars.RANK.items()}
    def blend(ws):  # ws: dict name->w
        sc = None
        for nm, w in ws.items():
            sc = R[nm] * w if sc is None else sc + R[nm] * w
        return sc
    def emit(ws, k):
        spec = "+".join(f"{nm}*{w:.2f}" for nm, w in ws.items())
        pos = ars.positions_from_topk(blend(ws), k)
        return "S", f"S|{spec}|top{k}", pos
    # pairs
    for a, b in itertools.combinations(names, 2):
        for wa in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for k in [5, 10, 15]:
                yield emit({a: wa, b: 1 - wa}, k)
    # triples
    for tri in itertools.combinations(names, 3):
        for ws in [(1/3, 1/3, 1/3), (0.5, 0.3, 0.2), (0.2, 0.5, 0.3), (0.6, 0.2, 0.2)]:
            for k in [5, 10, 15]:
                yield emit(dict(zip(tri, ws)), k)
    # quads (equal + 2 random draws)
    for quad in itertools.combinations(names, 4):
        yield emit(dict(zip(quad, [0.25] * 4)), 10)
        for _ in range(2):
            w = rng.dirichlet([1, 1, 1, 1])
            yield emit(dict(zip(quad, w)), 10)
    # random full-width draws
    for _ in range(12000):
        w = rng.dirichlet(np.ones(len(names)) * 0.5)
        keep = w > 0.05
        if keep.sum() < 2:
            continue
        ws = {n_: float(w_) for n_, w_, kp in zip(names, w, keep) if kp}
        tot = sum(ws.values())
        ws = {n_: w_ / tot for n_, w_ in ws.items()}
        k = int(rng.choice([5, 10, 15]))
        yield emit(ws, k)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="V,T,U,S,W")
    args = ap.parse_args()
    fams = args.families.split(",")

    print("building atoms...", flush=True)
    t0 = time.time()
    ctx = get_ctx("nifty50", "1d")
    ars = Arsenal(ctx)
    ev = ars.ev
    print(f"atoms ready in {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(42)
    GENS = {"V": lambda: gen_V(ars), "T": lambda: gen_T(ars),
            "U": lambda: gen_U(ars, rng), "S": lambda: gen_S(ars, rng),
            "W": lambda: gen_W(ars, rng)}
    for fam in fams:
        fam = fam.strip().upper()
        rows, part, n = [], 0, 0
        t1 = time.time()
        for family, spec, pos in GENS[fam]():
            m = ev.evaluate(pos)
            n += 1
            if m is not None:
                m.update({"family": family, "spec": spec})
                rows.append(m)
            if len(rows) >= 20000:
                pd.DataFrame(rows).to_parquet(
                    os.path.join(OUT_DIR, f"{fam}_part{part}.parquet"))
                part += 1
                rows = []
            if n % 5000 == 0:
                print(f"[{fam}] {n} screened ({time.time()-t1:.0f}s)", flush=True)
        if rows:
            pd.DataFrame(rows).to_parquet(
                os.path.join(OUT_DIR, f"{fam}_part{part}.parquet"))
        print(f"[{fam}] DONE {n} configs in {time.time()-t1:.0f}s", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
